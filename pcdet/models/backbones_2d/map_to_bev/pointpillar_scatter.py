import torch
import torch.nn as nn
import torch.nn.functional as F

#import torch_scatter
class PointPillarScatter(nn.Module):
    def __init__(self, model_cfg, grid_size, **kwargs):
        super().__init__()

        self.model_cfg = model_cfg
        self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES
        self.nx, self.ny, self.nz = grid_size
        assert self.nz == 1

    def forward(self, batch_dict, **kwargs):
        pillar_features, coords = batch_dict['pillar_features'], batch_dict['voxel_coords']
        batch_spatial_features = []
        batch_size = coords[:, 0].max().int().item() + 1
        for batch_idx in range(batch_size):
            spatial_feature = torch.zeros(
                self.num_bev_features,
                self.nz * self.nx * self.ny,
                dtype=pillar_features.dtype,
                device=pillar_features.device)

            batch_mask = coords[:, 0] == batch_idx
            this_coords = coords[batch_mask, :]
            indices = this_coords[:, 1] + this_coords[:, 2] * self.nx + this_coords[:, 3]
            indices = indices.type(torch.long)
            pillars = pillar_features[batch_mask, :]
            pillars = pillars.t()
            spatial_feature[:, indices] = pillars
            batch_spatial_features.append(spatial_feature)

        batch_spatial_features = torch.stack(batch_spatial_features, 0)
        batch_spatial_features = batch_spatial_features.view(batch_size, self.num_bev_features * self.nz, self.ny, self.nx)
        batch_dict['spatial_features'] = batch_spatial_features
        return batch_dict


class PointPillarScatter3d(nn.Module):
    def __init__(self, model_cfg, grid_size, **kwargs):
        super().__init__()
        
        self.model_cfg = model_cfg
        self.nx, self.ny, self.nz = self.model_cfg.INPUT_SHAPE
        self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES
        self.num_bev_features_before_compression = self.model_cfg.NUM_BEV_FEATURES // self.nz

    def forward(self, batch_dict, **kwargs):
        pillar_features, coords = batch_dict['pillar_features'], batch_dict['voxel_coords']
        
        batch_spatial_features = []
        batch_size = coords[:, 0].max().int().item() + 1
        for batch_idx in range(batch_size):
            spatial_feature = torch.zeros(
                self.num_bev_features_before_compression,
                self.nz * self.nx * self.ny,
                dtype=pillar_features.dtype,
                device=pillar_features.device)

            batch_mask = coords[:, 0] == batch_idx
            this_coords = coords[batch_mask, :]
            indices = this_coords[:, 1] * self.ny * self.nx + this_coords[:, 2] * self.nx + this_coords[:, 3]
            indices = indices.type(torch.long)
            pillars = pillar_features[batch_mask, :]
            pillars = pillars.t()
            spatial_feature[:, indices] = pillars
            batch_spatial_features.append(spatial_feature)

        batch_spatial_features = torch.stack(batch_spatial_features, 0)
        batch_spatial_features = batch_spatial_features.view(batch_size, self.num_bev_features_before_compression * self.nz, self.ny, self.nx)
        batch_dict['spatial_features'] = batch_spatial_features

        voxel_features = batch_dict['voxel_features']
        voxel_coords = batch_dict['voxel_coords']

        # 确保坐标不超出BEV特征图尺寸
        bev_shape = [800, 800]  # 根据实际配置调整
        assert (voxel_coords[:, 2] < bev_shape[0]).all(), "BEV X索引越界！"
        assert (voxel_coords[:, 3] < bev_shape[1]).all(), "BEV Y索引越界！"

        return batch_dict
        
class PointPillarScatter_spa(nn.Module):
    def __init__(self, model_cfg, grid_size, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES
        self.spa = self.model_cfg.get('IS_SPA', True)

        self.nx, self.ny, self.nz = grid_size
        assert self.nz == 1

        if self.spa:
            self.pillar_attention_rope = PillarAttentionWithRoPE(
                num_pillars=self.ny * self.nx,
                pillar_dim=self.num_bev_features,
                hidden_dim=self.num_bev_features
            )

    def forward(self, batch_dict, **kwargs):
        pillar_features, coords = batch_dict['pillar_features'], batch_dict['voxel_coords']
        batch_spatial_features = []
        batch_size = batch_dict['batch_size']

        for batch_idx in range(batch_size):
            spatial_feature = torch.zeros(
                self.num_bev_features,
                self.nz * self.nx * self.ny,
                dtype=pillar_features.dtype,
                device=pillar_features.device
            )

            sparsity_mask = torch.zeros(
                (self.ny * self.nx),
                dtype=torch.bool,
                device=coords.device
            )

            batch_mask = coords[:, 0] == batch_idx
            this_coords = coords[batch_mask, :]

            if this_coords.shape[0] > 0:
                indices = this_coords[:, 1] + this_coords[:, 2] * self.nx + this_coords[:, 3]
                indices = indices.long()

                pillars = pillar_features[batch_mask, :].t()
                spatial_feature[:, indices] = pillars
                sparsity_mask[indices] = True

            if self.spa:
                spatial_feature = self.pillar_attention_rope(
                    spatial_feature.permute(1, 0),  # (H*W, C)
                    sparsity_mask,
                    self.ny,
                    self.nx
                )

                if spatial_feature is None:
                    raise RuntimeError("pillar_attention_rope returned None, please check its forward() return value.")

                batch_spatial_features.append(spatial_feature.permute(1, 0))
            else:
                batch_spatial_features.append(spatial_feature)

        batch_spatial_features = torch.stack(batch_spatial_features, 0)
        batch_spatial_features = batch_spatial_features.view(
            batch_size,
            self.num_bev_features * self.nz,
            self.ny,
            self.nx
        )

        batch_dict['spatial_features'] = batch_spatial_features
        batch_dict['spatial_features_stride'] = 1
        return batch_dict


class PillarAttentionWithRoPE(nn.Module):
    def __init__(self, num_pillars, pillar_dim, hidden_dim):
        super(PillarAttentionWithRoPE, self).__init__()
        self.num_pillars = num_pillars
        self.pillar_dim = pillar_dim
        self.hidden_dim = hidden_dim

        self.q_mlp = nn.Linear(pillar_dim, hidden_dim)
        self.k_mlp = nn.Linear(pillar_dim, hidden_dim)
        self.v_mlp = nn.Linear(pillar_dim, hidden_dim)

        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, pillar_dim)
        )

        self.norm1 = nn.LayerNorm(pillar_dim)
        self.norm2 = nn.LayerNorm(pillar_dim)

    def apply_rope(self, x, H, W):
        """
        x: (H, W, C)
        """
        device = x.device
        half_dim = x.shape[-1] // 2

        if half_dim == 0:
            return x

        theta = torch.arange(half_dim, device=device, dtype=torch.float32) / half_dim
        theta = 10000 ** (-theta)

        h_coords = torch.linspace(-1, 1, H, device=device)
        w_coords = torch.linspace(-1, 1, W, device=device)

        # 对 torch 新版本更稳
        meshgrid = torch.stack(torch.meshgrid(h_coords, w_coords, indexing='ij'), dim=-1)  # (H, W, 2)

        h_sin = torch.sin(meshgrid[..., 0:1] * theta)
        h_cos = torch.cos(meshgrid[..., 0:1] * theta)
        w_sin = torch.sin(meshgrid[..., 1:2] * theta)
        w_cos = torch.cos(meshgrid[..., 1:2] * theta)

        x1 = x[..., :half_dim]
        x2 = x[..., half_dim:half_dim * 2]

        x_rot = torch.cat([
            x1 * h_cos * w_cos - x2 * h_sin * w_sin,
            x1 * h_sin * w_sin + x2 * h_cos * w_cos
        ], dim=-1)

        if x.shape[-1] > 2 * half_dim:
            x_rot = torch.cat([x_rot, x[..., 2 * half_dim:]], dim=-1)

        return x_rot

    def forward(self, pillar_features, sparsity_mask, H, W):
        """
        pillar_features: (H*W, C)
        sparsity_mask:   (H*W,)
        """
        pillar_features_2d = pillar_features.view(H, W, -1)
        rope_features = self.apply_rope(pillar_features_2d, H, W)
        pillar_features = (pillar_features_2d + rope_features).view(H * W, -1)

        # 如果当前 batch 没有非空 pillar，直接返回原始特征
        if sparsity_mask.sum() == 0:
            return pillar_features

        non_empty_pillars_raw = pillar_features[sparsity_mask]     # (P, C)
        non_empty_pillars = self.norm1(non_empty_pillars_raw)

        Q = self.q_mlp(non_empty_pillars)
        K = self.k_mlp(non_empty_pillars)
        V = self.v_mlp(non_empty_pillars)

        attention_scores = torch.matmul(Q, K.transpose(-2, -1))
        attention_weights = F.softmax(attention_scores, dim=-1)
        attended_features = torch.matmul(attention_weights, V)

        attended_features = attended_features + non_empty_pillars_raw
        updated_pillars = self.ffn(self.norm2(attended_features))
        updated_pillars = updated_pillars + attended_features

        output_pillars = pillar_features.clone()
        output_pillars[sparsity_mask] = updated_pillars

        return output_pillars