import torch
from pcdet.ops.roiaware_pool3d import roiaware_pool3d_utils

def cal_scores_by_npoints(cls_scores, iou_scores, num_points_in_gt, cls_thresh=10, iou_thresh=100):
    assert iou_thresh >= cls_thresh
    alpha = torch.zeros(cls_scores.shape, dtype=torch.float32).cuda()
    alpha[num_points_in_gt <= cls_thresh] = 0
    alpha[num_points_in_gt >= iou_thresh] = 1
    mask = (num_points_in_gt > cls_thresh) & (num_points_in_gt < iou_thresh)
    alpha[mask] = (num_points_in_gt[mask] - cls_thresh) / (iou_thresh - cls_thresh)
    return (1 - alpha) * cls_scores + alpha * iou_scores

def set_nms_score_by_class(iou_preds, cls_preds, label_preds, class_names, score_by_class):
    nms_scores = torch.zeros_like(iou_preds)
    for i in range(len(class_names)):
        mask = (label_preds == (i + 1))
        score_type = score_by_class[class_names[i]]
        if score_type == 'iou':
            nms_scores[mask] = iou_preds[mask]
        elif score_type == 'cls':
            nms_scores[mask] = cls_preds[mask]
        else:
            raise NotImplementedError
    return nms_scores

def get_nms_scores(iou_preds, cls_preds, box_preds, label_preds, points, method,
                   score_cfg=None, score_by_class=None, class_names=None):
    if method == 'iou' or method is None:
        return iou_preds
    elif method == 'cls':
        return cls_preds
    elif method == 'weighted_iou_cls':
        return score_cfg.iou * iou_preds + score_cfg.cls * cls_preds
    elif method == 'num_pts_iou_cls':
        # NOTE: box_preds.device.index == batch_index
        point_mask = (points[:, 0] == box_preds.device.index)
        batch_points = points[point_mask][:, 1:4]
        num_pts_in_gt = roiaware_pool3d_utils.points_in_boxes_cpu(
            batch_points.cpu(), box_preds[:, 0:7].cpu()
        ).sum(dim=1).float().cuda()
        return cal_scores_by_npoints(cls_preds, iou_preds, num_pts_in_gt,
                                     cls_thresh=score_cfg.cls, iou_thresh=score_cfg.iou)
    elif method == 'score_by_class':
        return set_nms_score_by_class(iou_preds, cls_preds, label_preds, class_names, score_by_class)
    else:
        raise NotImplementedError(f"Unsupported score fusion method: {method}")
