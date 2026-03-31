import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import box_utils
from pcdet.ops.iou3d_nms import iou3d_nms_utils


class SigmoidFocalClassificationLoss(nn.Module):
    """
    Sigmoid focal cross entropy loss.
    """

    def __init__(self, gamma: float = 2.0, alpha: float = 0.25):
        """
        Args:
            gamma: Weighting parameter to balance loss for hard and easy examples.
            alpha: Weighting parameter to balance loss for positive and negative examples.
        """
        super(SigmoidFocalClassificationLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    @staticmethod
    def sigmoid_cross_entropy_with_logits(input: torch.Tensor, target: torch.Tensor):
        """ PyTorch Implementation for tf.nn.sigmoid_cross_entropy_with_logits:
            max(x, 0) - x * z + log(1 + exp(-abs(x))) in
            https://www.tensorflow.org/api_docs/python/tf/nn/sigmoid_cross_entropy_with_logits

        Args:
            input: (B, #anchors, #classes) float tensor.
                Predicted logits for each class
            target: (B, #anchors, #classes) float tensor.
                One-hot encoded classification targets

        Returns:
            loss: (B, #anchors, #classes) float tensor.
                Sigmoid cross entropy loss without reduction
        """
        loss = torch.clamp(input, min=0) - input * target + \
               torch.log1p(torch.exp(-torch.abs(input)))
        return loss

    def forward(self, input: torch.Tensor, target: torch.Tensor, weights: torch.Tensor):
        """
        Args:
            input: (B, #anchors, #classes) float tensor.
                Predicted logits for each class
            target: (B, #anchors, #classes) float tensor.
                One-hot encoded classification targets
            weights: (B, #anchors) float tensor.
                Anchor-wise weights.

        Returns:
            weighted_loss: (B, #anchors, #classes) float tensor after weighting.
        """
        pred_sigmoid = torch.sigmoid(input)
        if self.alpha > 0:
            alpha_weight = target * self.alpha + (1 - target) * (1 - self.alpha)
        else:
            alpha_weight = 1
        pt = target * (1.0 - pred_sigmoid) + (1.0 - target) * pred_sigmoid
        focal_weight = alpha_weight * torch.pow(pt, self.gamma)

        bce_loss = self.sigmoid_cross_entropy_with_logits(input, target)

        loss = focal_weight * bce_loss

        if weights.shape.__len__() == 2 or \
                (weights.shape.__len__() == 1 and target.shape.__len__() == 2):
            weights = weights.unsqueeze(-1)

        assert weights.shape.__len__() == loss.shape.__len__()

        return loss * weights


class WeightedSmoothL1Loss(nn.Module):
    """
    Code-wise Weighted Smooth L1 Loss modified based on fvcore.nn.smooth_l1_loss
    https://github.com/facebookresearch/fvcore/blob/master/fvcore/nn/smooth_l1_loss.py
                  | 0.5 * x ** 2 / beta   if abs(x) < beta
    smoothl1(x) = |
                  | abs(x) - 0.5 * beta   otherwise,
    where x = input - target.
    """
    def __init__(self, beta: float = 1.0 / 9.0, code_weights: list = None):
        """
        Args:
            beta: Scalar float.
                L1 to L2 change point.
                For beta values < 1e-5, L1 loss is computed.
            code_weights: (#codes) float list if not None.
                Code-wise weights.
        """
        super(WeightedSmoothL1Loss, self).__init__()
        self.beta = beta
        if code_weights is not None:
            self.code_weights = np.array(code_weights, dtype=np.float32)
            self.code_weights = torch.from_numpy(self.code_weights).cuda()

    @staticmethod
    def smooth_l1_loss(diff, beta):
        if beta < 1e-5:
            loss = torch.abs(diff)
        else:
            n = torch.abs(diff)
            loss = torch.where(n < beta, 0.5 * n ** 2 / beta, n - 0.5 * beta)

        return loss

    def forward(self, input: torch.Tensor, target: torch.Tensor, weights: torch.Tensor = None):
        """
        Args:
            input: (B, #anchors, #codes) float tensor.
                Ecoded predicted locations of objects.
            target: (B, #anchors, #codes) float tensor.
                Regression targets.
            weights: (B, #anchors) float tensor if not None.

        Returns:
            loss: (B, #anchors) float tensor.
                Weighted smooth l1 loss without reduction.
        """
        target = torch.where(torch.isnan(target), input, target)  # ignore nan targets

        diff = input - target
        # code-wise weighting
        if self.code_weights is not None:
            diff = diff * self.code_weights.view(1, 1, -1)

        loss = self.smooth_l1_loss(diff, self.beta)

        # anchor-wise weighting
        if weights is not None:
            assert weights.shape[0] == loss.shape[0] and weights.shape[1] == loss.shape[1]
            loss = loss * weights.unsqueeze(-1)

        return loss


class WeightedL1Loss(nn.Module):
    def __init__(self, code_weights: list = None):
        """
        Args:
            code_weights: (#codes) float list if not None.
                Code-wise weights.
        """
        super(WeightedL1Loss, self).__init__()
        if code_weights is not None:
            self.code_weights = np.array(code_weights, dtype=np.float32)
            self.code_weights = torch.from_numpy(self.code_weights).cuda()

    @torch.cuda.amp.custom_fwd(cast_inputs=torch.float16)
    def forward(self, input: torch.Tensor, target: torch.Tensor, weights: torch.Tensor = None):
        """
        Args:
            input: (B, #anchors, #codes) float tensor.
                Ecoded predicted locations of objects.
            target: (B, #anchors, #codes) float tensor.
                Regression targets.
            weights: (B, #anchors) float tensor if not None.

        Returns:
            loss: (B, #anchors) float tensor.
                Weighted smooth l1 loss without reduction.
        """
        target = torch.where(torch.isnan(target), input, target)  # ignore nan targets

        diff = input - target
        # code-wise weighting
        if self.code_weights is not None:
            diff = diff * self.code_weights.view(1, 1, -1)

        loss = torch.abs(diff)

        # anchor-wise weighting
        if weights is not None:
            assert weights.shape[0] == loss.shape[0] and weights.shape[1] == loss.shape[1]
            loss = loss * weights.unsqueeze(-1)

        return loss


class WeightedCrossEntropyLoss(nn.Module):
    """
    Transform input to fit the fomation of PyTorch offical cross entropy loss
    with anchor-wise weighting.
    """
    def __init__(self):
        super(WeightedCrossEntropyLoss, self).__init__()

    def forward(self, input: torch.Tensor, target: torch.Tensor, weights: torch.Tensor):
        """
        Args:
            input: (B, #anchors, #classes) float tensor.
                Predited logits for each class.
            target: (B, #anchors, #classes) float tensor.
                One-hot classification targets.
            weights: (B, #anchors) float tensor.
                Anchor-wise weights.

        Returns:
            loss: (B, #anchors) float tensor.
                Weighted cross entropy loss without reduction
        """
        input = input.permute(0, 2, 1)
        target = target.argmax(dim=-1)
        loss = F.cross_entropy(input, target, reduction='none') * weights
        return loss


def get_corner_loss_lidar(pred_bbox3d: torch.Tensor, gt_bbox3d: torch.Tensor):
    """
    Args:
        pred_bbox3d: (N, 7) float Tensor.
        gt_bbox3d: (N, 7) float Tensor.

    Returns:
        corner_loss: (N) float Tensor.
    """
    assert pred_bbox3d.shape[0] == gt_bbox3d.shape[0]

    pred_box_corners = box_utils.boxes_to_corners_3d(pred_bbox3d)
    gt_box_corners = box_utils.boxes_to_corners_3d(gt_bbox3d)

    gt_bbox3d_flip = gt_bbox3d.clone()
    gt_bbox3d_flip[:, 6] += np.pi
    gt_box_corners_flip = box_utils.boxes_to_corners_3d(gt_bbox3d_flip)
    # (N, 8)
    corner_dist = torch.min(torch.norm(pred_box_corners - gt_box_corners, dim=2),
                            torch.norm(pred_box_corners - gt_box_corners_flip, dim=2))
    # (N, 8)
    corner_loss = WeightedSmoothL1Loss.smooth_l1_loss(corner_dist, beta=1.0)

    return corner_loss.mean(dim=1)


def compute_fg_mask(gt_boxes2d, shape, downsample_factor=1, device=torch.device("cpu")):
    """
    Compute foreground mask for images
    Args:
        gt_boxes2d: (B, N, 4), 2D box labels
        shape: torch.Size or tuple, Foreground mask desired shape
        downsample_factor: int, Downsample factor for image
        device: torch.device, Foreground mask desired device
    Returns:
        fg_mask (shape), Foreground mask
    """
    fg_mask = torch.zeros(shape, dtype=torch.bool, device=device)

    # Set box corners
    gt_boxes2d /= downsample_factor
    gt_boxes2d[:, :, :2] = torch.floor(gt_boxes2d[:, :, :2])
    gt_boxes2d[:, :, 2:] = torch.ceil(gt_boxes2d[:, :, 2:])
    gt_boxes2d = gt_boxes2d.long()

    # Set all values within each box to True
    B, N = gt_boxes2d.shape[:2]
    for b in range(B):
        for n in range(N):
            u1, v1, u2, v2 = gt_boxes2d[b, n]
            fg_mask[b, v1:v2, u1:u2] = True

    return fg_mask


def neg_loss_cornernet(pred, gt, mask=None):
    """
    Refer to https://github.com/tianweiy/CenterPoint.
    Modified focal loss. Exactly the same as CornerNet. Runs faster and costs a little bit more memory
    Args:
        pred: (batch x c x h x w)
        gt: (batch x c x h x w)
        mask: (batch x h x w)
    Returns:
    """
    pos_inds = gt.eq(1).float()
    neg_inds = gt.lt(1).float()

    neg_weights = torch.pow(1 - gt, 4)

    loss = 0

    pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
    neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds

    if mask is not None:
        mask = mask[:, None, :, :].float()
        pos_loss = pos_loss * mask
        neg_loss = neg_loss * mask
        num_pos = (pos_inds.float() * mask).sum()
    else:
        num_pos = pos_inds.float().sum()

    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()

    if num_pos == 0:
        loss = loss - neg_loss
    else:
        loss = loss - (pos_loss + neg_loss) / num_pos
    return loss


def neg_loss_sparse(pred, gt):
    """
    Refer to https://github.com/tianweiy/CenterPoint.
    Modified focal loss. Exactly the same as CornerNet. Runs faster and costs a little bit more memory
    Args:
        pred: (batch x c x n)
        gt: (batch x c x n)
    Returns:
    """
    pos_inds = gt.eq(1).float()
    neg_inds = gt.lt(1).float()

    neg_weights = torch.pow(1 - gt, 4)

    loss = 0

    pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
    neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds

    num_pos = pos_inds.float().sum()

    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()

    if num_pos == 0:
        loss = loss - neg_loss
    else:
        loss = loss - (pos_loss + neg_loss) / num_pos
    return loss


class FocalLossCenterNet(nn.Module):
    """
    Refer to https://github.com/tianweiy/CenterPoint
    """
    def __init__(self):
        super(FocalLossCenterNet, self).__init__()
        self.neg_loss = neg_loss_cornernet

    def forward(self, out, target, mask=None):
        return self.neg_loss(out, target, mask=mask)


def _reg_loss(regr, gt_regr, mask):
    """
    Refer to https://github.com/tianweiy/CenterPoint
    L1 regression loss
    Args:
        regr (batch x max_objects x dim)
        gt_regr (batch x max_objects x dim)
        mask (batch x max_objects)
    Returns:
    """
    num = mask.float().sum()
    mask = mask.unsqueeze(2).expand_as(gt_regr).float()
    isnotnan = (~ torch.isnan(gt_regr)).float()
    mask *= isnotnan
    regr = regr * mask
    gt_regr = gt_regr * mask

    loss = torch.abs(regr - gt_regr)
    loss = loss.transpose(2, 0)

    loss = torch.sum(loss, dim=2)
    loss = torch.sum(loss, dim=1)
    # else:
    #  # D x M x B
    #  loss = loss.reshape(loss.shape[0], -1)

    # loss = loss / (num + 1e-4)
    loss = loss / torch.clamp_min(num, min=1.0)
    # import pdb; pdb.set_trace()
    return loss


def _gather_feat(feat, ind, mask=None):
    dim  = feat.size(2)
    ind  = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    feat = feat.gather(1, ind)
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat


def _transpose_and_gather_feat(feat, ind):
    feat = feat.permute(0, 2, 3, 1).contiguous()
    feat = feat.view(feat.size(0), -1, feat.size(3))
    feat = _gather_feat(feat, ind)
    return feat


class RegLossCenterNet(nn.Module):
    """
    Refer to https://github.com/tianweiy/CenterPoint
    """

    def __init__(self):
        super(RegLossCenterNet, self).__init__()

    def forward(self, output, mask, ind=None, target=None):
        """
        Args:
            output: (batch x dim x h x w) or (batch x max_objects)
            mask: (batch x max_objects)
            ind: (batch x max_objects)
            target: (batch x max_objects x dim)
        Returns:
        """
        if ind is None:
            pred = output
        else:
            pred = _transpose_and_gather_feat(output, ind)
        loss = _reg_loss(pred, target, mask)
        return loss


class FocalLossSparse(nn.Module):
    """
    Refer to https://github.com/tianweiy/CenterPoint
    """
    def __init__(self):
        super(FocalLossSparse, self).__init__()
        self.neg_loss = neg_loss_sparse

    def forward(self, out, target):
        return self.neg_loss(out, target)


class RegLossSparse(nn.Module):
    """
    Refer to https://github.com/tianweiy/CenterPoint
    """

    def __init__(self):
        super(RegLossSparse, self).__init__()

    def forward(self, output, mask, ind=None, target=None, batch_index=None):
        """
        Args:
            output: (N x dim)
            mask: (batch x max_objects)
            ind: (batch x max_objects)
            target: (batch x max_objects x dim)
        Returns:
        """

        pred = []
        batch_size = mask.shape[0]
        for bs_idx in range(batch_size):
            batch_inds = batch_index==bs_idx
            pred.append(output[batch_inds][ind[bs_idx]])
        pred = torch.stack(pred)

        loss = _reg_loss(pred, target, mask)
        return loss


class IouLossSparse(nn.Module):
    '''IouLoss loss for an output tensor
    Arguments:
        output (batch x dim x h x w)
        mask (batch x max_objects)
        ind (batch x max_objects)
        target (batch x max_objects x dim)
    '''

    def __init__(self):
        super(IouLossSparse, self).__init__()

    def forward(self, iou_pred, mask, ind, box_pred, box_gt, batch_index):
        if mask.sum() == 0:
            return iou_pred.new_zeros((1))
        batch_size = mask.shape[0]
        mask = mask.bool()

        loss = 0
        for bs_idx in range(batch_size):
            batch_inds = batch_index==bs_idx
            pred = iou_pred[batch_inds][ind[bs_idx]][mask[bs_idx]]
            pred_box = box_pred[batch_inds][ind[bs_idx]][mask[bs_idx]]
            target = iou3d_nms_utils.boxes_aligned_iou3d_gpu(pred_box, box_gt[bs_idx])
            target = 2 * target - 1
            loss += F.l1_loss(pred, target, reduction='sum')

        loss = loss / (mask.sum() + 1e-4)
        return loss


class IouRegLossSparse(nn.Module):
    '''Distance IoU loss for output boxes
        Arguments:
            output (batch x dim x h x w)
            mask (batch x max_objects)
            ind (batch x max_objects)
            target (batch x max_objects x dim)
    '''

    def __init__(self, type="DIoU"):
        super(IouRegLossSparse, self).__init__()

    def center_to_corner2d(self, center, dim):
        corners_norm = torch.tensor([[-0.5, -0.5], [-0.5, 0.5], [0.5, 0.5], [0.5, -0.5]],
                                    dtype=torch.float32, device=dim.device)
        corners = dim.view([-1, 1, 2]) * corners_norm.view([1, 4, 2])
        corners = corners + center.view(-1, 1, 2)
        return corners

    def bbox3d_iou_func(self, pred_boxes, gt_boxes):
        assert pred_boxes.shape[0] == gt_boxes.shape[0]

        qcorners = self.center_to_corner2d(pred_boxes[:, :2], pred_boxes[:, 3:5])
        gcorners = self.center_to_corner2d(gt_boxes[:, :2], gt_boxes[:, 3:5])

        inter_max_xy = torch.minimum(qcorners[:, 2], gcorners[:, 2])
        inter_min_xy = torch.maximum(qcorners[:, 0], gcorners[:, 0])
        out_max_xy = torch.maximum(qcorners[:, 2], gcorners[:, 2])
        out_min_xy = torch.minimum(qcorners[:, 0], gcorners[:, 0])

        # calculate area
        volume_pred_boxes = pred_boxes[:, 3] * pred_boxes[:, 4] * pred_boxes[:, 5]
        volume_gt_boxes = gt_boxes[:, 3] * gt_boxes[:, 4] * gt_boxes[:, 5]

        inter_h = torch.minimum(pred_boxes[:, 2] + 0.5 * pred_boxes[:, 5], gt_boxes[:, 2] + 0.5 * gt_boxes[:, 5]) - \
                torch.maximum(pred_boxes[:, 2] - 0.5 * pred_boxes[:, 5], gt_boxes[:, 2] - 0.5 * gt_boxes[:, 5])
        inter_h = torch.clamp(inter_h, min=0)

        inter = torch.clamp((inter_max_xy - inter_min_xy), min=0)
        volume_inter = inter[:, 0] * inter[:, 1] * inter_h
        volume_union = volume_gt_boxes + volume_pred_boxes - volume_inter

        # boxes_iou3d_gpu(pred_boxes, gt_boxes)
        inter_diag = torch.pow(gt_boxes[:, 0:3] - pred_boxes[:, 0:3], 2).sum(-1)

        outer_h = torch.maximum(gt_boxes[:, 2] + 0.5 * gt_boxes[:, 5], pred_boxes[:, 2] + 0.5 * pred_boxes[:, 5]) - \
                torch.minimum(gt_boxes[:, 2] - 0.5 * gt_boxes[:, 5], pred_boxes[:, 2] - 0.5 * pred_boxes[:, 5])
        outer_h = torch.clamp(outer_h, min=0)
        outer = torch.clamp((out_max_xy - out_min_xy), min=0)
        outer_diag = outer[:, 0] ** 2 + outer[:, 1] ** 2 + outer_h ** 2

        dious = volume_inter / volume_union - inter_diag / outer_diag
        dious = torch.clamp(dious, min=-1.0, max=1.0)

        return dious

    def forward(self, box_pred, mask, ind, box_gt, batch_index):
        if mask.sum() == 0:
            return box_pred.new_zeros((1))
        mask = mask.bool()
        batch_size = mask.shape[0]

        loss = 0
        for bs_idx in range(batch_size):
            batch_inds = batch_index==bs_idx
            pred_box = box_pred[batch_inds][ind[bs_idx]]
            iou = self.bbox3d_iou_func(pred_box[mask[bs_idx]], box_gt[bs_idx])
            loss += (1. - iou).sum()

        loss =  loss / (mask.sum() + 1e-4)
        return loss


class L1Loss(nn.Module):
    def __init__(self):
        super(L1Loss, self).__init__()

    def forward(self, pred, target):
        if target.numel() == 0:
            return pred.sum() * 0
        assert pred.size() == target.size()
        loss = torch.abs(pred - target)
        return loss


def calculate_iou_loss_transfusionhead(iou_preds, batch_box_preds, gt_boxes, weights, num_pos):
    """
    Args:
        iou_preds: (batch x 1 x proposal)
        batch_box_preds: (batch x proposal x 7)
        # gt_boxes: (batch x N, 7 or 9)
        gt_boxes: (batch x proposal x 7)
        weights:
        num_pos: int
    Returns:
    """
    # iou_target = iou3d_nms_utils.paired_boxes_iou3d_gpu(batch_box_preds.reshape(-1, 7), gt_boxes.reshape(-1, 7))
    # # iou_target = iou3d_nms_utils.boxes_iou3d_gpu(selected_box_preds[:, 0:7].clone(), gt_boxes[mask][:, 0:7].clone()).diag()
    # iou_target = iou_target * 2 - 1  # [0, 1] ==> [-1, 1]

    # # print(selected_iou_preds.view(-1), iou_target)
    # valid_index = torch.nonzero(weights[:, :, 0].view(-1)).squeeze(-1)
    # loss = F.l1_loss(iou_preds.view(-1)[valid_index], iou_target[valid_index], reduction='sum')
    # # loss = loss / torch.clamp(torch.FloatTensor([iou_preds.shape[0] * iou_preds.shape[2]])[0].to(iou_target.device), min=1e-4)
    # loss = loss / max(num_pos, 1)
    # return loss

    iou_target = iou3d_nms_utils.boxes_aligned_iou3d_gpu(batch_box_preds.reshape(-1, 7), gt_boxes.reshape(-1, 7)).view(-1)
    # valid_index = torch.nonzero(iou_target).squeeze(-1)
    valid_index = torch.nonzero(iou_target * weights[:, :, 0].view(-1)).squeeze(-1)
    num_pos = valid_index.shape[0]

    iou_target = iou_target * 2 - 1  # [0, 1] ==> [-1, 1]

    loss = F.l1_loss(iou_preds.view(-1)[valid_index], iou_target[valid_index], reduction='sum')
    loss = loss / max(num_pos, 1)
    return loss


def calculate_iou_reg_loss_transfusionhead(batch_box_preds, gt_boxes, weights, num_pos):

    valid_index = torch.nonzero(weights[:, :, 0].view(-1)).squeeze(-1)
    iou = box_utils.bbox3d_overlaps_diou(batch_box_preds.reshape(-1, 7)[valid_index], gt_boxes.reshape(-1, 7)[valid_index])
    revalid_index = torch.nonzero(iou > 0).squeeze(-1)
    iou = iou[revalid_index]
    num_pos = revalid_index.shape[0]
    loss = (1.0 - iou).sum() / max(num_pos, 1)

    return loss


class GaussianFocalLoss(nn.Module):
    """GaussianFocalLoss is a variant of focal loss.

    More details can be found in the `paper
    <https://arxiv.org/abs/1808.01244>`_
    Code is modified from `kp_utils.py
    <https://github.com/princeton-vl/CornerNet/blob/master/models/py_utils/kp_utils.py#L152>`_  # noqa: E501
    Please notice that the target in GaussianFocalLoss is a gaussian heatmap,
    not 0/1 binary target.

    Args:
        alpha (float): Power of prediction.
        gamma (float): Power of target for negative samples.
        reduction (str): Options are "none", "mean" and "sum".
        loss_weight (float): Loss weight of current loss.
    """

    def __init__(self,
                 alpha=2.0,
                 gamma=4.0):
        super(GaussianFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred, target):
        eps = 1e-12
        pos_weights = target.eq(1)
        neg_weights = (1 - target).pow(self.gamma)
        pos_loss = -(pred + eps).log() * (1 - pred).pow(self.alpha) * pos_weights
        neg_loss = -(1 - pred + eps).log() * pred.pow(self.alpha) * neg_weights

        return pos_loss + neg_loss


def calculate_iou_loss_centerhead(iou_preds, batch_box_preds, mask, ind, gt_boxes):
    """
    Args:
        iou_preds: (batch x 1 x h x w)
        batch_box_preds: (batch x (7 or 9) x h x w)
        mask: (batch x max_objects)
        ind: (batch x max_objects)
        gt_boxes: (batch x N, 7 or 9)
    Returns:
    """
    if mask.sum() == 0:
        return iou_preds.new_zeros((1))

    mask = mask.bool()
    selected_iou_preds = _transpose_and_gather_feat(iou_preds, ind)[mask]

    selected_box_preds = _transpose_and_gather_feat(batch_box_preds, ind)[mask]
    iou_target = iou3d_nms_utils.paired_boxes_iou3d_gpu(selected_box_preds[:, 0:7], gt_boxes[mask][:, 0:7])
    # iou_target = iou3d_nms_utils.boxes_iou3d_gpu(selected_box_preds[:, 0:7].clone(), gt_boxes[mask][:, 0:7].clone()).diag()
    iou_target = iou_target * 2 - 1  # [0, 1] ==> [-1, 1]

    # print(selected_iou_preds.view(-1), iou_target)
    loss = F.l1_loss(selected_iou_preds.view(-1), iou_target, reduction='sum')
    loss = loss / torch.clamp(mask.sum(), min=1e-4)
    return loss


def calculate_iou_reg_loss_centerhead(batch_box_preds, mask, ind, gt_boxes):
    if mask.sum() == 0:
        return batch_box_preds.new_zeros((1))

    mask = mask.bool()

    selected_box_preds = _transpose_and_gather_feat(batch_box_preds, ind)

    iou = box_utils.bbox3d_overlaps_diou(selected_box_preds[mask][:, 0:7], gt_boxes[mask][:, 0:7])

    loss = (1.0 - iou).sum() / torch.clamp(mask.sum(), min=1e-4)
    return loss


# Code for SAFDNet


def focal_loss_sparse(pred, target):
    pos_inds = target.eq(1).float()
    neg_inds = target.lt(1).float()

    neg_weights = torch.pow(1 - target, 4)
    pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
    neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds

    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()

    loss = 0
    num_pos = pos_inds.float().sum()
    if num_pos == 0:
        loss = loss - neg_loss
    else:
        loss = loss - (pos_loss + neg_loss) / num_pos
    return loss


def reg_loss_sparse(pred, mask, ind, target, spatial_indices):
    if sum([m.sum() for m in mask]) == 0:
        return pred.sum() * 0.0

    pred = torch.cat([pred[spatial_indices[:, 0] == bidx][ind[bidx]] for bidx in range(len(mask))], dim=0)
    target = torch.cat([t for t in target], dim=0)
    mask = torch.cat([m for m in mask], dim=0)
    num = mask.sum().float()

    mask = mask.unsqueeze(1).expand_as(target).float()
    isnotnan = (~ torch.isnan(target)).float()
    mask *= isnotnan
    pred = pred * mask
    target = target * mask

    loss = torch.abs(pred - target)
    loss = torch.sum(loss, dim=0)
    loss = loss / torch.clamp_min(num, min=1.0)
    return loss


def iou_loss_sparse(iou_pred, mask, ind, box_pred, box_gt, spatial_indices):
    loss = 0
    for bidx in range(len(mask)):
        batch_inds = spatial_indices[:, 0] == bidx
        pred_iou = iou_pred[batch_inds][ind[bidx]][mask[bidx]]
        pred_box = box_pred[batch_inds][ind[bidx]][mask[bidx]]
        gt_box = box_gt[bidx][mask[bidx]]
        target_iou = iou3d_nms_utils.boxes_aligned_iou3d_gpu(pred_box[:, :7], gt_box[:, :7])
        target_iou = 2 * target_iou - 1
        loss += F.l1_loss(pred_iou, target_iou, reduction='sum')
    loss = loss / torch.clamp(sum([m.sum() for m in mask]).float(), min=1e-4)
    return loss


def iou_loss_sparse_transfusionhead(iou_preds, box_preds, box_targets, weights):
    iou_target = iou3d_nms_utils.boxes_aligned_iou3d_gpu(box_preds.reshape(-1, 7), box_targets.reshape(-1, 7)).view(-1)
    valid_index = torch.nonzero(iou_target * weights[:, :, 0].view(-1)).squeeze(-1)
    iou_target = iou_target * 2 - 1  # [0, 1] ==> [-1, 1]

    num_pos = valid_index.shape[0]
    loss = F.l1_loss(iou_preds.view(-1)[valid_index], iou_target[valid_index], reduction='sum')
    loss = loss / max(num_pos, 1)
    return loss


def iou_reg_loss_sparse_transfusionhead(box_preds, box_targets, weights):
    valid_index = torch.nonzero(weights[:, :, 0].view(-1)).squeeze(-1)
    iou = box_utils.bbox3d_overlaps_diou(box_preds.reshape(-1, 7)[valid_index], box_targets.reshape(-1, 7)[valid_index])
    valid_index = torch.nonzero(iou > 0).squeeze(-1)
    iou = iou[valid_index]
    num_pos = valid_index.shape[0]
    loss = (1.0 - iou).sum() / max(num_pos, 1)
    return loss

class SigmoidQualityFocalClassificationLoss(nn.Module):
    """
    Sigmoid focal cross entropy loss.
    """

    def __init__(self, gamma: float = 2.0, alpha: float = 0.25):
        """
        Args:
            gamma: Weighting parameter to balance loss for hard and easy examples.
        """
        super(SigmoidQualityFocalClassificationLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, input: torch.Tensor, target: torch.Tensor, weights: torch.Tensor):
        # alpha_weight = target * self.alpha + (1 - target) * (1 - self.alpha)
        func = F.binary_cross_entropy_with_logits
        # negatives are supervised by 0 quality score
        pred_sigmoid = torch.sigmoid(input)
        scale_factor = pred_sigmoid
        zerolabel = scale_factor.new_zeros(input.shape)
        loss = func(input, zerolabel, reduction='none') * scale_factor.pow(self.gamma)

        # positives are supervised by bbox quality (IoU) score
        positives = target > 0
        scale_factor = target[positives] - pred_sigmoid[positives]
        loss[positives] = func(input[positives], target[positives], reduction='none') * scale_factor.abs().pow(self.gamma)

        if weights.shape.__len__() == 2 or \
                (weights.shape.__len__() == 1 and target.shape.__len__() == 2):
            weights = weights.unsqueeze(-1)

        assert weights.shape.__len__() == loss.shape.__len__()

        return loss * weights * self.alpha


class DistributionFocalLoss(nn.Module):
    """
    Sigmoid focal cross entropy loss.
    """

    def __init__(self, loss_weight: float = 0.25):
        """
        Args:
            gamma: Weighting parameter to balance loss for hard and easy examples.
        """
        super(DistributionFocalLoss, self).__init__()
        self.loss_weight = loss_weight

    def forward(self, input: torch.Tensor, target: torch.Tensor, weights: torch.Tensor):
        # alpha_weight = target * self.alpha + (1 - target) * (1 - self.alpha)
        dis_left = target.long()
        dis_right = dis_left + 1
        weight_left = dis_right.float() - target
        weight_right = target - dis_left.float()
        # ipdb.set_trace()
        loss = F.cross_entropy(input, dis_left, reduction='none') * weight_left \
               + F.cross_entropy(input, dis_right, reduction='none') * weight_right

        return loss * weights * self.loss_weight


# Code for SparseDynamicAssign
class DynamicPositiveMask(nn.Module):
    def __init__(self, cls_weight=1, reg_weight=2, voxel_size=[0.8, 0.8]) -> None:
        super().__init__()
        self.cls_weight = cls_weight
        self.reg_weight = reg_weight
        self.voxel_size = voxel_size

    def cls_cost(self, pred_cls, pos_mask):
        cls_score = torch.max(pred_cls * pos_mask, dim=-1)[0]
        cls_cost = 1 - cls_score

        return cls_cost

    def rwiou_cost(self, pred_reg, gt_reg, mask, r_factor=0.2):
        isnotnan = (~ torch.isnan(gt_reg)).float().all(dim=-1)

        u, rdiou = box_utils.get_rwiou(pred_reg, gt_reg, r_factor, self.voxel_size)

        focal_reg_loss = 1 - torch.clamp(rdiou, min=0, max=1.0) + u
        rdiou_loss_src = focal_reg_loss * mask * isnotnan

        return rdiou_loss_src, u

    def gaussian_heatmap(self, distances, radius=2, normalize=True, eps=0.01):
        sigma = (2 * radius + 1) / 6
        h = torch.exp(-(distances) / (2 * sigma * sigma))
        if normalize:
            h = h / (h.max() + eps)
        return h

    def forward(self, pred_cls, target_cls, pred_reg, gt_reg, masks, iou_target, r_factor=0.5):
        with torch.no_grad():
            cls_cost = self.cls_cost(pred_cls, target_cls)  # [bs, max_num_boxes, dynamic_pos_num]
            reg_cost, u = self.rwiou_cost(pred_reg, gt_reg, masks, r_factor)  # [bs, max_num_boxes, dynamic_pos_num]

        positive_masks = target_cls.new_zeros(*masks.shape)
        # positive_masks = self.gaussian_heatmap(distances)

        all_cost = self.cls_weight * cls_cost * masks + self.reg_weight * reg_cost + (1 - masks.float()) * 100

        sort_cost, local_sort_inds = torch.sort(all_cost, dim=-1)
        # sort_cost, local_sort_inds = torch.sort(masks, dim=-1)

        obj_positive_nums = torch.sum(iou_target, dim=-1).clamp(1).int()  # [bs, max_num_boxes]

        for batch_id in range(pred_cls.shape[0]):
            box_num = (torch.sum(masks[batch_id], -1) > 0).sum()
            tmp_positive_nums = obj_positive_nums[batch_id]
            for box_id in range(box_num):
                local_pos_inds = local_sort_inds[batch_id][box_id][:tmp_positive_nums[box_id]]
                positive_masks[batch_id][box_id] = iou_target[batch_id][box_id]
                positive_masks[batch_id][box_id][local_pos_inds] = 1

        return positive_masks * masks


class UpFormerL1Loss(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, regr, gt_regr, mask, **kwargs):
        """
        L1 regression loss
        Args:
            regr (num_objects x dim)
            gt_regr (num_objects x dim)
            mask (num_objects)
        Returns:
        """
        num = mask.float().sum()
        mask = mask.unsqueeze(-1).expand_as(gt_regr).float()
        isnotnan = (~ torch.isnan(gt_regr)).float()
        mask *= isnotnan
        regr = regr * mask
        gt_regr = gt_regr * mask

        loss = torch.abs(regr - gt_regr)

        loss = loss.sum(0)
        # else:
        #  # D x M x B
        #  loss = loss.reshape(loss.shape[0], -1)

        # loss = loss / (num + 1e-4)
        loss = loss / torch.clamp_min(num, min=1.0)
        # import pdb; pdb.set_trace()
        return loss.sum()


class SlotFormerIoULoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, pred_ious, gt_ious, mask):
        """
        IoU loss
        Args:
            pred_ious (num_objects)
            gt_ious (num_objects)
            mask (num_objects)
        Returns:
        """
        pred_ious = pred_ious[mask.bool()]
        gt_ious = gt_ious[mask.bool()]

        loss = F.l1_loss(pred_ious, gt_ious, reduction='sum')

        loss = loss / torch.clamp_min(mask.sum(), min=1.0)

        return loss


class RWIoULoss(nn.Module):
    def __init__(self, voxel_size) -> None:
        super().__init__()
        self.voxel_size = voxel_size

    def forward(self, pred_reg, gt_reg, mask, r_factor=0.5):
        isnotnan = (~ torch.isnan(gt_reg)).float().all(dim=-1)
        pred_boxes = pred_reg[mask]
        gt_boxes = gt_reg[mask]
        u, rwiou = box_utils.get_rwiou(pred_reg, gt_reg, r_factor, self.voxel_size)

        focal_reg_loss = 1 - torch.clamp(rwiou, min=0, max=1.0) + u
        rwiou_loss_src = focal_reg_loss * mask * isnotnan

        num = (mask * isnotnan).float().sum()
        rwiou_loss_src = rwiou_loss_src.sum() / torch.clamp(num, min=1.0)

        return rwiou_loss_src


class DCDetIoULoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, pred_ious, gt_ious, mask):
        """
        IoU loss
        Args:
            pred_ious (num_objects)
            gt_ious (num_objects)
            mask (num_objects)
        Returns:
        """
        pred_ious = pred_ious[mask.bool()]
        gt_ious = gt_ious[mask.bool()]

        loss = F.l1_loss(pred_ious, gt_ious, reduction='sum')

        loss = loss / torch.clamp_min(mask.sum(), min=1.0)
        # import pdb; pdb.set_trace()
        return loss


def quality_focal_loss(pred, target, beta=2.0, threshold=0.01):
    r"""Quality Focal Loss (QFL) is from `Generalized Focal Loss: Learning
    Qualified and Distributed Bounding Boxes for Dense Object Detection
    <https://arxiv.org/abs/2006.04388>`_.

    Args:
        pred (torch.Tensor): Predicted joint representation of classification
            and quality (IoU) estimation with shape (N, C), C is the number of
            classes.
        target (tuple([torch.Tensor])): Target category label with shape (N,)
            and target quality label with shape (N,).
        beta (float): The beta parameter for calculating the modulating factor.
            Defaults to 2.0.

    Returns:
        torch.Tensor: Loss tensor with shape (N,).
    """
    pred_sigmoid = pred.sigmoid()
    scale_factor = pred_sigmoid
    zerolabel = scale_factor.new_zeros(pred.shape)
    loss = F.binary_cross_entropy_with_logits(
        pred, zerolabel, reduction="none"
    ) * scale_factor.pow(beta)

    pos = target >= threshold
    scale_factor = target[pos] - pred_sigmoid[pos]

    loss[pos] = F.binary_cross_entropy_with_logits(
        pred[pos], target[pos], reduction="none"
    ) * scale_factor.abs().pow(beta)

    loss = loss.sum() / pos.sum()

    return loss


### cagroup loss utils

def reduce_loss(loss, reduction):
    """Reduce loss as specified.
    Args:
        loss (Tensor): Elementwise loss tensor.
        reduction (str): Options are "none", "mean" and "sum".
    Return:
        Tensor: Reduced loss tensor.
    """
    reduction_enum = F._Reduction.get_enum(reduction)
    # none: 0, elementwise_mean:1, sum: 2
    if reduction_enum == 0:
        return loss
    elif reduction_enum == 1:
        return loss.mean()
    elif reduction_enum == 2:
        return loss.sum()


def _expand_onehot_labels(labels, label_weights, label_channels, ignore_index):
    """Expand onehot labels to match the size of prediction."""
    bin_labels = labels.new_full((labels.size(0), label_channels), 0)
    valid_mask = (labels >= 0) & (labels != ignore_index)
    inds = torch.nonzero(
        valid_mask & (labels < label_channels), as_tuple=False)

    if inds.numel() > 0:
        bin_labels[inds, labels[inds]] = 1

    valid_mask = valid_mask.view(-1, 1).expand(labels.size(0),
                                               label_channels).float()
    if label_weights is None:
        bin_label_weights = valid_mask
    else:
        bin_label_weights = label_weights.view(-1, 1).repeat(1, label_channels)
        bin_label_weights *= valid_mask

    return bin_labels, bin_label_weights,


def binary_cross_entropy(pred, label, weight=None, reduction='mean', avg_factor=None, class_weight=None,
                         ignore_index=-100):
    if pred.dim() != label.dim():
        label, weight, valid_mask = _expand_onehot_labels(
            label, weight, pred.size(-1), ignore_index)
    else:
        # should mask out the ignored elements
        valid_mask = ((label >= 0) & (label != ignore_index)).float()
        if weight is not None:
            # The inplace writing method will have a mismatched broadcast
            # shape error if the weight and valid_mask dimensions
            # are inconsistent such as (B,N,1) and (B,N,C).
            weight = weight * valid_mask
        else:
            weight = valid_mask

    if avg_factor is None and reduction == 'mean':
        avg_factor = valid_mask.sum().item()

    # weighted element-wise losses
    weight = weight.float()
    loss = F.binary_cross_entropy_with_logits(
        pred, label.float(), pos_weight=class_weight, reduction='none')
    # do the reduction for the weighted loss
    loss = loss * weight
    if avg_factor is None:
        loss = reduce_loss(loss, reduction)
    else:
        if reduction == 'mean':
            eps = torch.finfo(torch.float32).eps
            loss = loss.sum() / (avg_factor + eps)
        elif reduction != 'none':
            raise ValueError('avg_factor can not be used with reduction="sum"')

    return loss


class CrossEntropy(nn.Module):
    def __init__(self,
                 use_sigmoid=True,
                 reduction='mean',
                 class_weight=None,
                 loss_weight=1.0,
                 ignore_index=-100,
                 ) -> None:
        super(CrossEntropy, self).__init__()
        assert use_sigmoid, "Now we only support sigmoid implementation."
        self.reduction = reduction
        self.class_weight = class_weight
        self.loss_weight = loss_weight
        self.ignore_index = ignore_index
        if use_sigmoid:
            self.cls_criterion = binary_cross_entropy

    def forward(self, cls_score, label, weight=None, avg_factor=None, reduction_override=None, ignore_index=None,
                **kwargs):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if ignore_index is None:
            ignore_index = self.ignore_index

        if self.class_weight is not None:
            class_weight = cls_score.new_tensor(
                self.class_weight, device=cls_score.device)
        else:
            class_weight = None

        loss_cls = self.loss_weight * self.cls_criterion(
            cls_score,
            label,
            weight,
            class_weight=class_weight,
            reduction=reduction,
            avg_factor=avg_factor,
            ignore_index=ignore_index,
            **kwargs)
        return loss_cls

def weight_reduce_loss(loss, weight=None, reduction='mean', avg_factor=None):
    """Apply element-wise weight and reduce loss.

    Args:
        loss (Tensor): Element-wise loss.
        weight (Tensor): Element-wise weights.
        reduction (str): Same as built-in losses of PyTorch.
        avg_factor (float): Avarage factor when computing the mean of losses.

    Returns:
        Tensor: Processed loss values.
    """
    # if weight is specified, apply element-wise weight
    if weight is not None:
        loss = loss * weight

    # if avg_factor is not specified, just reduce the loss
    if avg_factor is None:
        loss = reduce_loss(loss, reduction)
    else:
        # if reduction is mean, then average the loss by avg_factor
        if reduction == 'mean':
            loss = loss.sum() / avg_factor
        # if reduction is 'none', then do nothing, otherwise raise an error
        elif reduction != 'none':
            raise ValueError('avg_factor can not be used with reduction="sum"')
    return loss

def py_sigmoid_focal_loss(pred,
                          target,
                          weight=None,
                          gamma=2.0,
                          alpha=0.25,
                          reduction='mean',
                          avg_factor=None):
    """PyTorch version of `Focal Loss <https://arxiv.org/abs/1708.02002>`_.

    Args:
        pred (torch.Tensor): The prediction with shape (N, C), C is the
            number of classes
        target (torch.Tensor): The learning label of the prediction.
        weight (torch.Tensor, optional): Sample-wise loss weight.
        gamma (float, optional): The gamma for calculating the modulating
            factor. Defaults to 2.0.
        alpha (float, optional): A balanced form for Focal Loss.
            Defaults to 0.25.
        reduction (str, optional): The method used to reduce the loss into
            a scalar. Defaults to 'mean'.
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.
    """
    pred_sigmoid = pred.sigmoid()
    target = target.type_as(pred)
    pt = (1 - pred_sigmoid) * target + pred_sigmoid * (1 - target)
    focal_weight = (alpha * target + (1 - alpha) *
                    (1 - target)) * pt.pow(gamma)
    loss = F.binary_cross_entropy_with_logits(
        pred, target, reduction='none') * focal_weight
    if weight is not None:
        if weight.shape != loss.shape:
            if weight.size(0) == loss.size(0):
                # For most cases, weight is of shape (num_priors, ),
                #  which means it does not have the second axis num_class
                weight = weight.view(-1, 1)
            else:
                # Sometimes, weight per anchor per class is also needed. e.g.
                #  in FSAF. But it may be flattened of shape
                #  (num_priors x num_class, ), while loss is still of shape
                #  (num_priors, num_class).
                assert weight.numel() == loss.numel()
                weight = weight.view(loss.size(0), -1)
        assert weight.ndim == loss.ndim
    loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
    return loss

def py_sigmoid_focal_loss(pred,
                          target,
                          weight=None,
                          gamma=2.0,
                          alpha=0.25,
                          reduction='mean',
                          avg_factor=None):
    """PyTorch version of `Focal Loss <https://arxiv.org/abs/1708.02002>`_.

    Args:
        pred (torch.Tensor): The prediction with shape (N, C), C is the
            number of classes
        target (torch.Tensor): The learning label of the prediction.
        weight (torch.Tensor, optional): Sample-wise loss weight.
        gamma (float, optional): The gamma for calculating the modulating
            factor. Defaults to 2.0.
        alpha (float, optional): A balanced form for Focal Loss.
            Defaults to 0.25.
        reduction (str, optional): The method used to reduce the loss into
            a scalar. Defaults to 'mean'.
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.
    """
    pred_sigmoid = pred.sigmoid()
    target = target.type_as(pred)
    pt = (1 - pred_sigmoid) * target + pred_sigmoid * (1 - target)
    focal_weight = (alpha * target + (1 - alpha) *
                    (1 - target)) * pt.pow(gamma)
    loss = F.binary_cross_entropy_with_logits(
        pred, target, reduction='none') * focal_weight
    if weight is not None:
        if weight.shape != loss.shape:
            if weight.size(0) == loss.size(0):
                # For most cases, weight is of shape (num_priors, ),
                #  which means it does not have the second axis num_class
                weight = weight.view(-1, 1)
            else:
                # Sometimes, weight per anchor per class is also needed. e.g.
                #  in FSAF. But it may be flattened of shape
                #  (num_priors x num_class, ), while loss is still of shape
                #  (num_priors, num_class).
                assert weight.numel() == loss.numel()
                weight = weight.view(loss.size(0), -1)
        assert weight.ndim == loss.ndim
    loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
    return loss

class FocalLoss(nn.Module):

    def __init__(self,
                 use_sigmoid=True,
                 gamma=2.0,
                 alpha=0.25,
                 reduction='mean',
                 loss_weight=1.0):
        """`Focal Loss <https://arxiv.org/abs/1708.02002>`_

        Args:
            use_sigmoid (bool, optional): Whether to the prediction is
                used for sigmoid or softmax. Defaults to True.
            gamma (float, optional): The gamma for calculating the modulating
                factor. Defaults to 2.0.
            alpha (float, optional): A balanced form for Focal Loss.
                Defaults to 0.25.
            reduction (str, optional): The method used to reduce the loss into
                a scalar. Defaults to 'mean'. Options are "none", "mean" and
                "sum".
            loss_weight (float, optional): Weight of loss. Defaults to 1.0.
        """
        super(FocalLoss, self).__init__()
        assert use_sigmoid is True, 'Only sigmoid focal loss supported now.'
        self.use_sigmoid = use_sigmoid
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        """Forward function.

        Args:
            pred (torch.Tensor): The prediction.
            target (torch.Tensor): The learning label of the prediction.
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Options are "none", "mean" and "sum".

        Returns:
            torch.Tensor: The calculated loss
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if self.use_sigmoid:
            # we dont support cuda implementation now
            num_classes = pred.size(1)
            # -1 denotes backgroud
            # TODO(lihe): check this
            target[target < 0] = num_classes
            target = F.one_hot(target.long(), num_classes=num_classes + 1)
            target = target[:, :num_classes]
            calculate_loss_func = py_sigmoid_focal_loss

            loss_cls = self.loss_weight * calculate_loss_func(
                pred,
                target,
                weight,
                gamma=self.gamma,
                alpha=self.alpha,
                reduction=reduction,
                avg_factor=avg_factor)

        else:
            raise NotImplementedError
        return loss_cls


def smooth_l1_loss(pred, target, weight=None, beta=1.0, reduction='mean', avg_factor=None):
    """Smooth L1 loss.

    Args:
        pred (torch.Tensor): The prediction.
        target (torch.Tensor): The learning target of the prediction.
        beta (float, optional): The threshold in the piecewise function.
            Defaults to 1.0.

    Returns:
        torch.Tensor: Calculated loss
    """
    assert beta > 0
    assert pred.size() == target.size() and target.numel() > 0
    diff = torch.abs(pred - target)
    loss = torch.where(diff < beta, 0.5 * diff * diff / beta,
                       diff - 0.5 * beta)

    # if weight is specified, apply element-wise weight
    if weight is not None:
        loss = loss * weight

    # if avg_factor is not specified, just reduce the loss
    if avg_factor is None:
        loss = reduce_loss(loss, reduction)
    else:
        # if reduction is mean, then average the loss by avg_factor
        if reduction == 'mean':
            loss = loss.sum() / avg_factor
        # if reduction is 'none', then do nothing, otherwise raise an error
        elif reduction != 'none':
            raise ValueError('avg_factor can not be used with reduction="sum"')
    return loss


class SmoothL1Loss(nn.Module):
    """Smooth L1 loss.

    Args:
        beta (float, optional): The threshold in the piecewise function.
            Defaults to 1.0.
        reduction (str, optional): The method to reduce the loss.
            Options are "none", "mean" and "sum". Defaults to "mean".
        loss_weight (float, optional): The weight of loss.
    """

    def __init__(self, beta=1.0, reduction='mean', loss_weight=1.0):
        super(SmoothL1Loss, self).__init__()
        self.beta = beta
        self.reduction = reduction  ## 'sum'
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        """Forward function.

        Args:
            pred (torch.Tensor): The prediction.
            target (torch.Tensor): The learning target of the prediction.
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        loss_bbox = self.loss_weight * smooth_l1_loss(
            pred,
            target,
            weight,
            beta=self.beta,
            reduction=reduction,  ## 'sum'
            avg_factor=avg_factor,
            **kwargs)
        return loss_bbox