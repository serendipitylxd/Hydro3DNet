import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import torch_scatter
except Exception as e:
    # Incase someone doesn't want to use dynamic pillar vfe and hasn't installed torch_scatter
    pass

from .vfe_template import VFETemplate
from .dynamic_pillar_vfe import PFNLayerV2


class HydroAugmentedVFE(VFETemplate):
    def __init__(self, model_cfg, num_point_features, voxel_size, grid_size, point_cloud_range, **kwargs):
        super().__init__(model_cfg=model_cfg)

        self.use_norm = self.model_cfg.USE_NORM
        self.with_distance = self.model_cfg.WITH_DISTANCE
        self.use_absolute_xyz = self.model_cfg.USE_ABSLOTE_XYZ
        self.use_dynamic_water_level_boundary_points = getattr(self.model_cfg, 'USE_DYNAMIC_WATER_LEVEL_BOUNDARY_POINTS', False)
        self.water_range = self.model_cfg.WATER_RANGE  # [x_min, y_min, z_min, x_max, y_max, z_max]

        self.water_pillar = self.model_cfg.WATER_PILLAR  # [pillar_x, pillar_y]

        #Check whether the water area can be divided evenly by the Pillar size
        water_x_min, water_y_min, _, water_x_max, water_y_max, _ = self.water_range
        water_x_size = water_x_max - water_x_min
        water_y_size = water_y_max - water_y_min
        pillar_x, pillar_y = self.water_pillar
        tol = 1e-6
        x_ratio = water_x_size / pillar_x
        y_ratio = water_y_size / pillar_y

        if abs(x_ratio - round(x_ratio)) > tol or abs(y_ratio - round(y_ratio)) > tol:
            raise ValueError(f"The Water range must be divisible by the pillar size. "
                             f"Current water area size: ({water_x_size}, {water_y_size}), "
                             f"pillar size: ({pillar_x}, {pillar_y}), "
                             f"x_ratio: {x_ratio}, y_ratio: {y_ratio}")

        num_point_features += 6 if self.use_absolute_xyz else 3
        if self.with_distance:
            num_point_features += 1

        self.num_filters = self.model_cfg.NUM_FILTERS
        assert len(self.num_filters) > 0
        num_filters = [num_point_features] + list(self.num_filters)

        pfn_layers = []
        for i in range(len(num_filters) - 1):
            in_filters = num_filters[i]
            out_filters = num_filters[i + 1]
            pfn_layers.append(
                PFNLayerV2(in_filters, out_filters, self.use_norm, last_layer=(i >= len(num_filters) - 2))
            )
        self.pfn_layers = nn.ModuleList(pfn_layers)

        self.voxel_x = voxel_size[0]
        self.voxel_y = voxel_size[1]
        self.voxel_z = voxel_size[2]
        self.x_offset = self.voxel_x / 2 + point_cloud_range[0]
        self.y_offset = self.voxel_y / 2 + point_cloud_range[1]
        self.z_offset = self.voxel_z / 2 + point_cloud_range[2]

        self.scale_xyz = grid_size[0] * grid_size[1] * grid_size[2]
        self.scale_yz = grid_size[1] * grid_size[2]
        self.scale_z = grid_size[2]

        self.grid_size = torch.tensor(grid_size).cuda()
        self.voxel_size = torch.tensor(voxel_size).cuda()
        self.point_cloud_range = torch.tensor(point_cloud_range).cuda()

    def get_output_feature_dim(self):
        return self.num_filters[-1]

    def forward(self, batch_dict, **kwargs):
        points = batch_dict['points'] # (batch_idx, x, y, z, i, e)
        if self.use_dynamic_water_level_boundary_points:
            water_level = batch_dict['water_level']  # (bs)
            #print("water_level.shape before adding water points",water_level.shape)
            #################### new dynamic water level boundary poin ####################
            # Water Area Parameters
            (water_x_min, water_y_min, _,
             water_x_max, water_y_max, _) = self.water_range
            pillar_x, pillar_y = self.water_pillar

            # Filter the points within the XY range of the water area
            in_water_mask = (
                (points[:, 1] >= water_x_min) &
                (points[:, 1] < water_x_max) &
                (points[:, 2] >= water_y_min) &
                (points[:, 2] < water_y_max))
            water_points = points[in_water_mask]

            if water_points.shape[0] > 0:
                # Calculate the water_pillar index to which each point belongs
                water_xy = water_points[:, [1, 2]]  # [x, y]
                pillar_x_idx = ((water_xy[:, 0] - water_x_min) // pillar_x).long()
                pillar_y_idx = ((water_xy[:, 1] - water_y_min) // pillar_y).long()

                # Calculate the globally unique identifier (considering batch)
                max_x = int((water_x_max - water_x_min) // pillar_x)
                max_y = int((water_y_max - water_y_min) // pillar_y)
                # Floating-point error correction (make sure tolerance covers if the boundary is exactly not divisible)
                if abs(max_x * pillar_x + water_x_min - water_x_max) > 1e-6:
                    max_x += 1
                if abs(max_y * pillar_y + water_y_min - water_y_max) > 1e-6:
                    max_y += 1
                water_coords = water_points[:, 0].long() * (max_x * max_y) + \
                               pillar_x_idx * max_y + pillar_y_idx

                # Count the points of each pillar
                unq_coords, unq_inv, unq_cnt = torch.unique(
                    water_coords, return_inverse=True, return_counts=True)

                # Filter out pillars with points greater than or equal to 50
                mask = unq_cnt >= 50
                #mask = unq_cnt >= 5
                if mask.any():
                    # Calculate the mean values of XY
                    xy_mean = torch_scatter.scatter_mean(water_xy, unq_inv, dim=0)
                    selected_xy = xy_mean[mask]

                    # Obtain the water level value corresponding to the batch
                    batch_ids = unq_coords[mask] // (max_x * max_y)

                    # GPU security check
                    assert batch_ids.max() < water_level.shape[0], \
                        f"batch_ids.max()={batch_ids.max().item()} >= water_level.shape[0]={water_level.shape[0]}"

                    z_values = water_level[batch_ids].view(-1, 1)
                    #print("water_level z_values", z_values)
                    # Build new point features
                    new_xyz = torch.cat([selected_xy, z_values], dim=1)
                    other_feats = torch_scatter.scatter_mean(
                        water_points[:, 4:], unq_inv, dim=0)[mask]

                    # Combination New Points [batch_idx, x, y, z,...]
                    new_points = torch.cat([
                        batch_ids.float().unsqueeze(1),
                        new_xyz,
                        other_feats
                    ], dim=1)

                    # Merge into the original point cloud
                    points = torch.cat([points, new_points], dim=0)
            batch_dict['enhanced_points'] = points
        #################### ends adding a new dot ####################
        # print(" Points.shape after adding water points ",points.shape)


        points_coords = torch.floor((points[:, [1,2,3]] - self.point_cloud_range[[0,1,2]]) / self.voxel_size[[0,1,2]]).int()
        mask = ((points_coords >= 0) & (points_coords < self.grid_size[[0,1,2]])).all(dim=1)
        points = points[mask]
        points_coords = points_coords[mask]
        points_xyz = points[:, [1, 2, 3]].contiguous()

        merge_coords = points[:, 0].int() * self.scale_xyz + \
                       points_coords[:, 0] * self.scale_yz + \
                       points_coords[:, 1] * self.scale_z + \
                       points_coords[:, 2]

        unq_coords, unq_inv, unq_cnt = torch.unique(merge_coords, return_inverse=True, return_counts=True, dim=0)

        points_mean = torch_scatter.scatter_mean(points_xyz, unq_inv, dim=0)
        f_cluster = points_xyz - points_mean[unq_inv, :]

        f_center = torch.zeros_like(points_xyz)
        f_center[:, 0] = points_xyz[:, 0] - (points_coords[:, 0].to(points_xyz.dtype) * self.voxel_x + self.x_offset)
        f_center[:, 1] = points_xyz[:, 1] - (points_coords[:, 1].to(points_xyz.dtype) * self.voxel_y + self.y_offset)
        # f_center[:, 2] = points_xyz[:, 2] - self.z_offset
        f_center[:, 2] = points_xyz[:, 2] - (points_coords[:, 2].to(points_xyz.dtype) * self.voxel_z + self.z_offset)

        if self.use_absolute_xyz:
            features = [points[:, 1:], f_cluster, f_center]
        else:
            features = [points[:, 4:], f_cluster, f_center]

        if self.with_distance:
            points_dist = torch.norm(points[:, 1:4], 2, dim=1, keepdim=True)
            features.append(points_dist)
        features = torch.cat(features, dim=-1)

        for pfn in self.pfn_layers:
            features = pfn(features, unq_inv)

        # generate voxel coordinates
        unq_coords = unq_coords.int()
        voxel_coords = torch.stack((unq_coords // self.scale_xyz,
                                    (unq_coords % self.scale_xyz) // self.scale_yz,
                                    (unq_coords % self.scale_yz) // self.scale_z,
                                    unq_coords % self.scale_z), dim=1)
        voxel_coords = voxel_coords[:, [0, 3, 2, 1]]

        batch_dict['pillar_features'] = batch_dict['voxel_features'] = features
        batch_dict['voxel_coords'] = voxel_coords


        return batch_dict
