#!/usr/bin/python3
# _*_coding: utf-8 _*_

import numpy as np
import torch
import torch.nn as nn
from sklearn.cluster import DBSCAN
from sklearn.neighbors import KDTree

from ...utils.spconv_utils import spconv


def post_act_block(in_channels, out_channels, kernel_size, indice_key=None, stride=1, padding=0,
                   conv_type='subm', norm_fn=None):
    if conv_type == 'subm':
        conv = spconv.SubMConv3d(in_channels, out_channels, kernel_size, bias=False, indice_key=indice_key)
    elif conv_type == 'spconv':
        conv = spconv.SparseConv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                                   bias=False, indice_key=indice_key)
    elif conv_type == 'inverseconv':
        conv = spconv.SparseInverseConv3d(in_channels, out_channels, kernel_size, indice_key=indice_key, bias=False)
    else:
        raise NotImplementedError

    m = spconv.SparseSequential(
        conv,
        norm_fn(out_channels),
        nn.ReLU(),
    )
    return m


class CQCA_cfa(nn.Module):
    """
    Speed-up version for TROUT adaptation.

    Main changes:
    1) optional BYPASS_DBSCAN: directly returns zero image features + original points.
    2) optional MAX_DBSCAN_POINTS: randomly subsample points before DBSCAN.
    3) vectorized cluster-density counting (remove per-point python loops).
    4) robust empty-point handling.
    """
    def __init__(self, model_cfg, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg

        self.dbscan_map_w = self.model_cfg.DBSCAN_MAP_W
        self.dbscan_map_h = self.model_cfg.DBSCAN_MAP_H
        self.dbscan_feature = self.model_cfg.DBSCAN_FEATURE
        self.dbscan_v = self.model_cfg.DBSCAN_V
        self.dbscan_y = self.model_cfg.DBSCAN_Y
        self.resolution = self.model_cfg.RESOLUTION
        self.dbscan_eps = self.model_cfg.DBSCAN_EPS
        self.dbscan_sample = self.model_cfg.DBSCAN_SAMPLE
        self.point_x = self.model_cfg.POINTX
        self.point_y = self.model_cfg.POINTY

        self.bypass_dbscan = self.model_cfg.get('BYPASS_DBSCAN', False)
        self.max_dbscan_points = self.model_cfg.get('MAX_DBSCAN_POINTS', -1)
        self.max_cluster_size = self.model_cfg.get('MAX_CLUSTER_SIZE', 100)
        self.keep_noise_when_empty = self.model_cfg.get('KEEP_NOISE_WHEN_EMPTY', True)

        self.conv = nn.Sequential(
            nn.Conv2d(self.dbscan_feature, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16, eps=1e-3, momentum=0.01),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32, eps=1e-3, momentum=0.01),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64, eps=1e-3, momentum=0.01),
            nn.ReLU(),
        )

    def _empty_map(self, device):
        return torch.zeros(
            (self.dbscan_feature, self.dbscan_map_w, self.dbscan_map_h),
            dtype=torch.float32,
            device=device
        )

    def _subsample_points(self, point):
        if self.max_dbscan_points is None or self.max_dbscan_points <= 0:
            return point
        if point.shape[0] <= self.max_dbscan_points:
            return point
        idx = torch.randperm(point.shape[0], device=point.device)[:self.max_dbscan_points]
        return point[idx]

    def generate_map(self, points_xyv, eps, samples):
        orig_device = points_xyv.device

        if points_xyv.shape[0] == 0:
            empty_labels = torch.empty((0,), dtype=torch.long, device=orig_device)
            return self._empty_map(orig_device), empty_labels

        points_np = points_xyv.detach().cpu().numpy().astype(np.float32, copy=False)
        labels_np = DBSCAN(eps=eps, min_samples=samples).fit_predict(points_np)

        density_np = np.zeros(labels_np.shape[0], dtype=np.float32)
        valid_mask = labels_np != -1
        if valid_mask.any():
            uniq, counts = np.unique(labels_np[valid_mask], return_counts=True)
            label2count = dict(zip(uniq.tolist(), counts.tolist()))
            density_np[valid_mask] = np.array([label2count[int(lb)] for lb in labels_np[valid_mask]], dtype=np.float32)

            too_dense_mask = valid_mask & (density_np > float(self.max_cluster_size))
            labels_np[too_dense_mask] = -1

        dbscan_keep = labels_np > 0
        bev_map = np.zeros((self.dbscan_map_w, self.dbscan_map_h, self.dbscan_feature), dtype=np.float32)

        if dbscan_keep.any():
            kept_points = points_np[dbscan_keep]
            kept_density = density_np[dbscan_keep]
            kept_labels = labels_np[dbscan_keep].astype(np.float32)
            dbscan_points = np.concatenate(
                [kept_points, kept_density[:, None], kept_labels[:, None]], axis=1
            )

            bev_x = (dbscan_points[:, 0] / self.resolution).astype(np.int64) - 1
            bev_y = ((dbscan_points[:, 1] + self.dbscan_y) / self.resolution).astype(np.int64) - 1
            bev_x = np.clip(bev_x, 0, self.dbscan_map_h - 1)
            bev_y = np.clip(bev_y, 0, self.dbscan_map_w - 1)
            bev_map[bev_y, bev_x] = dbscan_points[:, 2:]

        bev_map_t = torch.from_numpy(bev_map).permute(2, 0, 1).to(orig_device)
        labels_t = torch.from_numpy(labels_np).long().to(orig_device)
        return bev_map_t, labels_t

    def forward(self, batch_dict, **kwargs):
        points = batch_dict['points']
        batch_size = batch_dict['batch_size']

        if self.bypass_dbscan:
            spatial_features_img = torch.zeros(
                (batch_size, 64, self.dbscan_map_w, self.dbscan_map_h),
                dtype=torch.float32,
                device=points.device
            )
            batch_dict['spatial_features_img'] = spatial_features_img
            batch_dict['cluster_points'] = points
            return batch_dict

        dbscan_maps = []
        cluster_points = []
        feature_index = self.dbscan_v

        for batch_idx in range(batch_size):
            batch_mask = points[:, 0] == batch_idx
            point = points[batch_mask]

            if point.shape[0] == 0:
                dbscan_maps.append(self._empty_map(points.device))
                continue

            point = self._subsample_points(point)
            point_xyv = torch.cat([point[:, 1:3], point[:, feature_index].reshape(-1, 1)], dim=1)
            db_map, db_labels = self.generate_map(point_xyv, self.dbscan_eps, self.dbscan_sample)
            dbscan_maps.append(db_map)

            keep_mask = db_labels != -1
            if keep_mask.sum() == 0:
                final_points = point if self.keep_noise_when_empty else point[:0]
            else:
                final_points = point[keep_mask]
            cluster_points.append(final_points)

        dbscan_map2 = torch.stack(dbscan_maps, 0)
        cluster_points = torch.cat(cluster_points, 0) if len(cluster_points) > 0 else points[:0]
        dbscan_map2 = dbscan_map2.view(batch_size, self.dbscan_feature, self.dbscan_map_w, self.dbscan_map_h)
        out = self.conv(dbscan_map2)

        batch_dict['spatial_features_img'] = out
        batch_dict['cluster_points'] = cluster_points
        return batch_dict


class AdaptiveDBSCAN:
    def __init__(self, min_samples=5):
        self.min_samples = min_samples

    def fit(self, points):
        self.points = points
        self.X = points[:, 0:2]
        X = points[:, 0:2]
        self.labels = np.zeros(len(X))
        self.cluster_idx = 0
        self.kd_tree = KDTree(X)
        self.alte = 2.5
        for i in range(len(X)):
            if self.labels[i] == 0:
                if self.expand_cluster(i):
                    self.cluster_idx += 1

    def expand_cluster(self, idx):
        neighbors = self.query_neighbors(idx)
        if len(neighbors) < self.min_samples:
            self.labels[idx] = -1
            return False
        else:
            self.labels[idx] = self.cluster_idx
            for neighbor_idx in neighbors:
                if self.labels[neighbor_idx] == 0:
                    self.labels[neighbor_idx] = self.cluster_idx
                    neighbor_neighbors = self.query_neighbors(neighbor_idx)
                    if len(neighbor_neighbors) >= self.min_samples:
                        neighbors = np.append(neighbors, neighbor_neighbors)
            return True

    def query_neighbors(self, idx):
        eps = self.alte * np.linalg.norm(self.X[idx]) * np.tan(np.pi / 180 * 0.75)
        if eps < self.alte * 0.2:
            eps = self.alte * 0.2
        neighbors = self.kd_tree.query_radius([self.X[idx]], r=eps)[0]
        return neighbors