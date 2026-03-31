import torch
import torch.nn as nn
from pcdet.ops.rotated_iou import cal_iou_3d


def reduce_loss(loss, reduction):
    if reduction == 'none':
        return loss
    elif reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    else:
        raise ValueError(f'Invalid reduction: {reduction}')


def axis_aligned_bbox_overlaps_3d(bboxes1, bboxes2, mode='iou', is_aligned=False, eps=1e-6):
    """
    bboxes1, bboxes2: (..., m, 6) / (..., n, 6)
    format: [x1, y1, z1, x2, y2, z2]
    """
    assert mode in ['iou', 'giou'], f'Unsupported mode {mode}'
    assert bboxes1.size(-1) == 6 or bboxes1.size(0) == 0
    assert bboxes2.size(-1) == 6 or bboxes2.size(0) == 0

    rows = bboxes1.size(-2)
    cols = bboxes2.size(-2)
    if is_aligned:
        assert rows == cols

    if rows * cols == 0:
        if is_aligned:
            return bboxes1.new_empty(bboxes1.shape[:-2] + (rows,))
        else:
            return bboxes1.new_empty(bboxes1.shape[:-2] + (rows, cols))

    area1 = (
        (bboxes1[..., 3] - bboxes1[..., 0]) *
        (bboxes1[..., 4] - bboxes1[..., 1]) *
        (bboxes1[..., 5] - bboxes1[..., 2])
    )
    area2 = (
        (bboxes2[..., 3] - bboxes2[..., 0]) *
        (bboxes2[..., 4] - bboxes2[..., 1]) *
        (bboxes2[..., 5] - bboxes2[..., 2])
    )

    if is_aligned:
        lt = torch.max(bboxes1[..., :3], bboxes2[..., :3])   # (..., rows, 3)
        rb = torch.min(bboxes1[..., 3:], bboxes2[..., 3:])   # (..., rows, 3)

        wh = (rb - lt).clamp(min=0)
        overlap = wh[..., 0] * wh[..., 1] * wh[..., 2]

        if mode in ['iou', 'giou']:
            union = area1 + area2 - overlap
        else:
            union = area1

        if mode == 'giou':
            enclosed_lt = torch.min(bboxes1[..., :3], bboxes2[..., :3])
            enclosed_rb = torch.max(bboxes1[..., 3:], bboxes2[..., 3:])
    else:
        lt = torch.max(bboxes1[..., :, None, :3], bboxes2[..., None, :, :3])   # (..., rows, cols, 3)
        rb = torch.min(bboxes1[..., :, None, 3:], bboxes2[..., None, :, 3:])   # (..., rows, cols, 3)

        wh = (rb - lt).clamp(min=0)
        overlap = wh[..., 0] * wh[..., 1] * wh[..., 2]

        if mode in ['iou', 'giou']:
            union = area1[..., None] + area2[..., None, :] - overlap
        else:
            union = area1[..., None]

        if mode == 'giou':
            enclosed_lt = torch.min(bboxes1[..., :, None, :3], bboxes2[..., None, :, :3])
            enclosed_rb = torch.max(bboxes1[..., :, None, 3:], bboxes2[..., None, :, 3:])

    union = torch.clamp(union, min=eps)
    ious = overlap / union

    if mode == 'iou':
        return ious

    enclose_wh = (enclosed_rb - enclosed_lt).clamp(min=0)
    enclose_area = enclose_wh[..., 0] * enclose_wh[..., 1] * enclose_wh[..., 2]
    enclose_area = torch.clamp(enclose_area, min=eps)
    gious = ious - (enclose_area - union) / enclose_area
    return gious


class AxisAlignedBboxOverlaps3D(object):
    def __call__(self, bboxes1, bboxes2, mode='iou', is_aligned=False):
        assert bboxes1.size(-1) == bboxes2.size(-1) == 6
        return axis_aligned_bbox_overlaps_3d(
            bboxes1, bboxes2, mode=mode, is_aligned=is_aligned
        )

    def __repr__(self):
        return self.__class__.__name__ + '()'


def iou_3d_loss(pred, target, weight=None, reduction='mean', avg_factor=None):
    iou_loss = 1 - cal_iou_3d(pred[None, ...], target[None, ...])

    if weight is not None:
        iou_loss = iou_loss * weight

    if avg_factor is None:
        iou_loss = reduce_loss(iou_loss, reduction)
    else:
        if reduction == 'mean':
            iou_loss = iou_loss.sum() / avg_factor
        elif reduction != 'none':
            raise ValueError('avg_factor can not be used with reduction="sum"')

    return iou_loss


def axis_aligned_iou_loss(pred, target, weight=None, reduction='mean', avg_factor=None):
    def _transform(bbox):
        return torch.stack((
            bbox[..., 0] - bbox[..., 3] / 2,
            bbox[..., 1] - bbox[..., 4] / 2,
            bbox[..., 2] - bbox[..., 5] / 2,
            bbox[..., 0] + bbox[..., 3] / 2,
            bbox[..., 1] + bbox[..., 4] / 2,
            bbox[..., 2] + bbox[..., 5] / 2,
        ), dim=-1)

    axis_aligned_iou = AxisAlignedBboxOverlaps3D()(
        _transform(pred), _transform(target), is_aligned=True
    )
    iou_loss = 1 - axis_aligned_iou

    if weight is not None:
        iou_loss = iou_loss * weight

    if avg_factor is None:
        iou_loss = reduce_loss(iou_loss, reduction)
    else:
        if reduction == 'mean':
            iou_loss = iou_loss.sum() / avg_factor
        elif reduction != 'none':
            raise ValueError('avg_factor can not be used with reduction="sum"')

    return iou_loss


class IoU3DMixin(nn.Module):
    def __init__(self, loss_function, reduction='mean', loss_weight=1.0):
        super().__init__()
        self.loss_function = loss_function
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self, pred, target, weight=None, avg_factor=None, reduction_override=None, **kwargs):
        if weight is not None and not torch.any(weight > 0):
            return pred.sum() * weight.sum()

        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = reduction_override if reduction_override else self.reduction

        if weight is not None and weight.dim() > 1:
            weight = weight.mean(-1)

        loss = self.loss_weight * self.loss_function(
            pred, target, weight, reduction=reduction, avg_factor=avg_factor, **kwargs
        )
        return loss


class IoU3DLoss(IoU3DMixin):
    def __init__(self, with_yaw=True, **kwargs):
        loss_function = iou_3d_loss if with_yaw else axis_aligned_iou_loss
        super().__init__(loss_function=loss_function, **kwargs)