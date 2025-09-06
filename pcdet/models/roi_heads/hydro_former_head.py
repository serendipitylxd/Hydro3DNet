from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from .roi_head_template import RoIHeadTemplate
from ...utils import common_utils, loss_utils

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        scale = torch.cat([avg_out, max_out], dim=1)
        scale = self.conv(scale)
        return self.sigmoid(scale)

class HydroFormerHead(RoIHeadTemplate):
    def __init__(self, input_channels, model_cfg, num_class=1, **kwargs):
        super().__init__(num_class=num_class, model_cfg=model_cfg)
        self.model_cfg = model_cfg
        self.point_cloud_range = kwargs['point_cloud_range']
        self.voxel_size = kwargs['voxel_size']


        GRID_SIZE = self.model_cfg.ROI_GRID_POOL.GRID_SIZE
        pre_channel = self.model_cfg.ROI_GRID_POOL.IN_CHANNEL * GRID_SIZE * GRID_SIZE

        shared_fc_list = []
        for k in range(0, self.model_cfg.SHARED_FC.__len__()):
            shared_fc_list.extend([
                nn.Conv1d(pre_channel, self.model_cfg.SHARED_FC[k], kernel_size=1, bias=False),
                nn.BatchNorm1d(self.model_cfg.SHARED_FC[k]),
                nn.ReLU()
            ])
            pre_channel = self.model_cfg.SHARED_FC[k]

            if k != self.model_cfg.SHARED_FC.__len__() - 1 and self.model_cfg.DP_RATIO > 0:
                shared_fc_list.append(nn.Dropout(self.model_cfg.DP_RATIO))

        self.shared_fc_layer = nn.Sequential(*shared_fc_list)

        self.iou_layers = self.make_fc_layers(
            input_channels=pre_channel, output_channels=1, fc_list=self.model_cfg.IOU_FC
        )
        self.init_weights(weight_init='xavier')

        if torch.__version__ >= '1.3':
            self.affine_grid = partial(F.affine_grid, align_corners=True)
            self.grid_sample = partial(F.grid_sample, align_corners=True)
        else:
            self.affine_grid = F.affine_grid
            self.grid_sample = F.grid_sample

        # Get the correct number of input channels
        self.grid_pool_in_channels = self.model_cfg.ROI_GRID_POOL.IN_CHANNEL


        # Add spatial attention
        self.spatial_attention = SpatialAttention()

        self.use_enhanced_points = self.model_cfg.get('USE_ENHANCED_POINTS', False)

        # Add point cloud processing parameters
        self.num_points_per_roi = self.model_cfg.get('NUM_POINTS_PER_ROI', 512)

        # Add point cloud processing parameters
        self.point_mlp = nn.Sequential(
            #nn.Linear(4, 64),
            nn.Linear(3, 64),  # Suppose each point has xyz
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )

        # Point Cloud Attention
        # self.point_attention = nn.MultiheadAttention(embed_dim=128, num_heads=4)
        self.point_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=128, nhead=4, dim_feedforward=256, batch_first=True
            ),
            num_layers = 1
            #num_layers = 4
        )

        # Multimodal fusion module
        self.fuse_conv = nn.Sequential(
            nn.Conv2d(self.grid_pool_in_channels + 128, self.grid_pool_in_channels, 1),
            nn.BatchNorm2d(self.grid_pool_in_channels),
            nn.ReLU()
        )

    def roi_point_pool(self, batch_dict):
        """
        Extract the point cloud within the ROI and extract the features, and add the water area boundary information
        """
        batch_size = batch_dict['batch_size']
        rois = batch_dict['rois'].detach()  # (B, N, 7)
        if self.use_enhanced_points:
            points = batch_dict['enhanced_points']  # (N_points, 4) [bs_idx, x, y, z]
        else:
            points = batch_dict['points']  # (N_points, 4) [bs_idx, x, y, z]

        #water_level = batch_dict['water_level']  # (batch_size)
        #print("water_level.shape",water_level.shape, ". water_level",water_level)
        #self.water_range = self.model_cfg.WATER_RANGE  # [x_min, y_min, z_min, x_max, y_max, z_max]

        pooled_points_list = []
        for b_id in range(batch_size):
            batch_mask = points[:, 0] == b_id
            batch_points = points[batch_mask][:, 1:4]  # (N, 3)
            #current_water_level = water_level[b_id].item()# The water level value of the current batch

            for roi in rois[b_id]:
                center = roi[:3]
                dims = roi[3:6]
                angle = roi[6]

                # Convert to the ROI coordinate system
                rel_pos = batch_points - center
                cosa, sina = torch.cos(angle), torch.sin(angle)
                rot_x = rel_pos[:, 0] * cosa + rel_pos[:, 1] * (-sina)
                rot_y = rel_pos[:, 0] * sina + rel_pos[:, 1] * cosa

                # Check whether the point is within the ROI
                mask_x = torch.abs(rot_x) < dims[0] / 2
                mask_y = torch.abs(rot_y) < dims[1] / 2
                mask_z = torch.abs(rel_pos[:, 2] - center[2]) < dims[2] / 2
                mask = mask_x & mask_y & mask_z
                roi_points = batch_points[mask]

                # Sample or fill to a fixed number
                num_points = roi_points.shape[0]
                if num_points > self.num_points_per_roi:
                    idx = torch.randperm(num_points)[:self.num_points_per_roi]
                    roi_points = roi_points[idx]
                elif num_points < self.num_points_per_roi:
                    zeros = torch.zeros((self.num_points_per_roi - num_points, 3),
                                        dtype=roi_points.dtype, device=roi_points.device)
                    roi_points = torch.cat([roi_points, zeros], dim=0)


                pooled_points_list.append(roi_points.unsqueeze(0))

        pooled_points = torch.cat(pooled_points_list, dim=0)  # (B*N, num_points, 4)
        return pooled_points

    def init_weights(self, weight_init='xavier'):
        if weight_init == 'kaiming':
            init_func = nn.init.kaiming_normal_
        elif weight_init == 'xavier':
            init_func = nn.init.xavier_normal_
        elif weight_init == 'normal':
            init_func = nn.init.normal_
        else:
            raise NotImplementedError

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                if weight_init == 'normal':
                    init_func(m.weight, mean=0, std=0.001)
                else:
                    init_func(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def roi_grid_pool(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size:
                rois: (B, num_rois, 7 + C)
                spatial_features_2d: (B, C, H, W)
        Returns:

        """
        batch_size = batch_dict['batch_size']
        rois = batch_dict['rois'].detach()
        spatial_features_2d = batch_dict['spatial_features_2d'].detach()
        height, width = spatial_features_2d.size(2), spatial_features_2d.size(3)

        # dataset_cfg = batch_dict['dataset_cfg']
        # min_x = dataset_cfg.POINT_CLOUD_RANGE[0]
        # min_y = dataset_cfg.POINT_CLOUD_RANGE[1]
        # voxel_size_x = dataset_cfg.DATA_PROCESSOR[-1].VOXEL_SIZE[0]
        # voxel_size_y = dataset_cfg.DATA_PROCESSOR[-1].VOXEL_SIZE[1]
        min_x = self.point_cloud_range[0]
        min_y = self.point_cloud_range[1]
        voxel_size_x = self.voxel_size[0]
        voxel_size_y = self.voxel_size[1]
        down_sample_ratio = self.model_cfg.ROI_GRID_POOL.DOWNSAMPLE_RATIO

        pooled_features_list = []
        torch.backends.cudnn.enabled = False
        for b_id in range(batch_size):
            # Map global boxes coordinates to feature map coordinates
            x1 = (rois[b_id, :, 0] - rois[b_id, :, 3] / 2 - min_x) / (voxel_size_x * down_sample_ratio)
            x2 = (rois[b_id, :, 0] + rois[b_id, :, 3] / 2 - min_x) / (voxel_size_x * down_sample_ratio)
            y1 = (rois[b_id, :, 1] - rois[b_id, :, 4] / 2 - min_y) / (voxel_size_y * down_sample_ratio)
            y2 = (rois[b_id, :, 1] + rois[b_id, :, 4] / 2 - min_y) / (voxel_size_y * down_sample_ratio)

            angle, _ = common_utils.check_numpy_to_torch(rois[b_id, :, 6])

            cosa = torch.cos(angle)
            sina = torch.sin(angle)

            theta = torch.stack((
                (x2 - x1) / (width - 1) * cosa, (x2 - x1) / (width - 1) * (-sina), (x1 + x2 - width + 1) / (width - 1),
                (y2 - y1) / (height - 1) * sina, (y2 - y1) / (height - 1) * cosa, (y1 + y2 - height + 1) / (height - 1)
            ), dim=1).view(-1, 2, 3).float()

            grid_size = self.model_cfg.ROI_GRID_POOL.GRID_SIZE
            grid = self.affine_grid(
                theta,
                torch.Size((rois.size(1), spatial_features_2d.size(1), grid_size, grid_size))
            )

            pooled_features = self.grid_sample(
                spatial_features_2d[b_id].unsqueeze(0).expand(rois.size(1), spatial_features_2d.size(1), height, width),
                grid
            )

            pooled_features_list.append(pooled_features)

        torch.backends.cudnn.enabled = True
        pooled_features = torch.cat(pooled_features_list, dim=0)

        return pooled_features



    def forward(self, batch_dict):
        """
        :param input_data: input dict
        :return:
        """
        targets_dict = self.proposal_layer(
            batch_dict, nms_config=self.model_cfg.NMS_CONFIG['TRAIN' if self.training else 'TEST']
        )
        if self.training:
            targets_dict = self.assign_targets(batch_dict)
            batch_dict['rois'] = targets_dict['rois']
            batch_dict['roi_labels'] = targets_dict['roi_labels']

        # RoI aware pooling
        pooled_features = self.roi_grid_pool(batch_dict)  # (BxN, C, 7, 7)
        #print("pooled_features:", pooled_features.shape)

        # Spatial Attention
        spatial_att = self.spatial_attention(pooled_features)
        pooled_features = pooled_features * spatial_att

        # Obtain ROI point cloud features
        pooled_points = self.roi_point_pool(batch_dict)  # (B*N, P, 3)

        # Point Cloud Feature Enhancement
        B_N, P, C = pooled_points.shape  # (B*N, P, 4)
        point_features = pooled_points.view(-1, C)  # (B*N*P, 4)
        point_features = self.point_mlp(point_features)  # (B*N*P, 128)
        point_features = point_features.view(B_N, P, -1)  # (B*N, P, 128)

        # Use TransformerEncoder
        point_features = self.point_encoder(point_features)  # (B*N, P, 128)
        point_features = point_features.mean(dim=1)  # (B*N, 128)

        # Feature Fusion
        point_features = point_features.view(-1, 128, 1, 1).expand(-1, -1, 7, 7)
        #print("point_features:", point_features.shape)
        #print("pooled_features:", pooled_features.shape)
        fused_features = torch.cat([pooled_features, point_features], dim=1)# (B*N, self.grid_pool_in_channels + 128, 7, 7)
        #print("fused_features:", fused_features.shape)
        fused_features = self.fuse_conv(fused_features)  # (B*N, self.grid_pool_in_channels, 7, 7)
        #print("fused_features:", fused_features.shape)

        batch_size_rcnn = fused_features.shape[0]

        shared_features = self.shared_fc_layer(fused_features.view(batch_size_rcnn, -1, 1))
        rcnn_iou = self.iou_layers(shared_features).transpose(1, 2).contiguous().squeeze(dim=1)  # (B*N, 1)

        if not self.training:
            batch_dict['batch_cls_preds'] = rcnn_iou.view(batch_dict['batch_size'], -1, rcnn_iou.shape[-1])
            batch_dict['batch_box_preds'] = batch_dict['rois']
            batch_dict['cls_preds_normalized'] = False
        else:
            targets_dict['rcnn_iou'] = rcnn_iou

            self.forward_ret_dict = targets_dict

        return batch_dict

    def get_loss(self, tb_dict=None):
        tb_dict = {} if tb_dict is None else tb_dict
        rcnn_loss = 0
        rcnn_loss_cls, cls_tb_dict = self.get_box_iou_layer_loss(self.forward_ret_dict)
        rcnn_loss += rcnn_loss_cls
        tb_dict.update(cls_tb_dict)

        tb_dict['rcnn_loss'] = rcnn_loss.item()
        return rcnn_loss, tb_dict

    def get_box_iou_layer_loss(self, forward_ret_dict):
        loss_cfgs = self.model_cfg.LOSS_CONFIG
        rcnn_iou = forward_ret_dict['rcnn_iou']
        rcnn_iou_labels = forward_ret_dict['rcnn_cls_labels'].view(-1)
        rcnn_iou_flat = rcnn_iou.view(-1)
        if loss_cfgs.IOU_LOSS == 'BinaryCrossEntropy':
            batch_loss_iou = nn.functional.binary_cross_entropy_with_logits(
                rcnn_iou_flat,
                rcnn_iou_labels.float(), reduction='none'
            )
        elif loss_cfgs.IOU_LOSS == 'L2':
            batch_loss_iou = nn.functional.mse_loss(rcnn_iou_flat, rcnn_iou_labels, reduction='none')
        elif loss_cfgs.IOU_LOSS == 'smoothL1':
            diff = rcnn_iou_flat - rcnn_iou_labels
            batch_loss_iou = loss_utils.WeightedSmoothL1Loss.smooth_l1_loss(diff, 1.0 / 9.0)
        elif loss_cfgs.IOU_LOSS == 'focalbce':
            batch_loss_iou = loss_utils.sigmoid_focal_cls_loss(rcnn_iou_flat, rcnn_iou_labels)
        else:
            raise NotImplementedError

        iou_valid_mask = (rcnn_iou_labels >= 0).float()
        rcnn_loss_iou = (batch_loss_iou * iou_valid_mask).sum() / torch.clamp(iou_valid_mask.sum(), min=1.0)

        rcnn_loss_iou = rcnn_loss_iou * loss_cfgs.LOSS_WEIGHTS['rcnn_iou_weight']
        tb_dict = {'rcnn_loss_iou': rcnn_loss_iou.item()}
        return rcnn_loss_iou, tb_dict
