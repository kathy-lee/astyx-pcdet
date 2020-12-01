import torch
import torch.nn as nn

from ...utils import box_utils
from .point_head_template import PointHeadTemplate
from ...ops.roiaware_pool3d import roiaware_pool3d_utils

class SoftmaxFocalClassificationLoss(nn.Module):
    """
    Softmax focal cross entropy loss.
    """

    def __init__(self, gamma: float = 2.0, alpha: float = 0.25):
        """
        Args:
            gamma: Weighting parameter to balance loss for hard and easy examples.
            alpha: Weighting parameter to balance loss for positive and negative examples.
        """
        super(SoftmaxFocalClassificationLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

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
        pred_sigmoid = torch.softmax(input)
        alpha_weight = target * self.alpha + (1 - target) * (1 - self.alpha)
        pt = target * (1.0 - pred_sigmoid) + (1.0 - target) * pred_sigmoid
        focal_weight = alpha_weight * torch.pow(pt, self.gamma)

        bce_loss = torch.nn.functional.cross_entropy(input, target)

        loss = focal_weight * bce_loss

        if weights.shape.__len__() == 2 or \
                (weights.shape.__len__() == 1 and target.shape.__len__() == 2):
            weights = weights.unsqueeze(-1)

        assert weights.shape.__len__() == loss.shape.__len__()

        return loss * weights


class PointSegHead(PointHeadTemplate):
    """
    A simple point-based segmentation head, which are used for PV-RCNN keypoint segmentaion.
    Reference Paper: https://arxiv.org/abs/1912.13192
    PV-RCNN: Point-Voxel Feature Set Abstraction for 3D Object Detection
    """
    def __init__(self, num_class, input_channels, model_cfg, **kwargs):
        super().__init__(model_cfg=model_cfg, num_class=num_class)
        self.cls_layers = self.make_fc_layers(
            fc_cfg=self.model_cfg.CLS_FC,
            input_channels=input_channels,
            output_channels=num_class
        )

    def assign_targets(self, input_dict):
        """
        Args:
            input_dict:
                point_features: (N1 + N2 + N3 + ..., C)
                batch_size:
                point_coords: (N1 + N2 + N3 + ..., 4) [bs_idx, x, y, z]
                gt_boxes (optional): (B, M, 8)
        Returns:
            point_cls_labels: (N1 + N2 + N3 + ...), long type, 0:background, -1:ignored
            point_part_labels: (N1 + N2 + N3 + ..., 3)
        """
        point_coords = input_dict['point_coords']
        gt_boxes = input_dict['gt_boxes']
        assert gt_boxes.shape.__len__() == 3, 'gt_boxes.shape=%s' % str(gt_boxes.shape)
        assert point_coords.shape.__len__() in [2], 'points.shape=%s' % str(point_coords.shape)

        batch_size = gt_boxes.shape[0]
        extend_gt_boxes = box_utils.enlarge_box3d(
            gt_boxes.view(-1, gt_boxes.shape[-1]), extra_width=self.model_cfg.TARGET_CONFIG.GT_EXTRA_WIDTH
        ).view(batch_size, -1, gt_boxes.shape[-1])
        targets_dict = self.assign_stack_targets(
            points=point_coords, gt_boxes=gt_boxes, extend_gt_boxes=extend_gt_boxes,
            set_ignore_flag=True, use_ball_constraint=False,
            ret_part_labels=False
        )

        return targets_dict

    def get_loss(self, tb_dict=None):
        tb_dict = {} if tb_dict is None else tb_dict
        point_loss_cls, tb_dict_1 = self.get_cls_layer_loss()

        point_loss = point_loss_cls
        tb_dict.update(tb_dict_1)
        return point_loss, tb_dict

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size:
                point_features: (N1 + N2 + N3 + ..., C) or (B, N, C)
                point_features_before_fusion: (N1 + N2 + N3 + ..., C)
                point_coords: (N1 + N2 + N3 + ..., 4) [bs_idx, x, y, z]
                point_labels (optional): (N1 + N2 + N3 + ...)
                gt_boxes (optional): (B, M, 8)
        Returns:
            batch_dict:
                point_cls_scores: (N1 + N2 + N3 + ..., 1)
                point_part_offset: (N1 + N2 + N3 + ..., 3)
        """
        if self.model_cfg.get('USE_POINT_FEATURES_BEFORE_FUSION', False):
            point_features = batch_dict['point_features_before_fusion']
        else:
            point_features = batch_dict['point_features']
        point_cls_preds = self.cls_layers(point_features)  # (total_points, num_class)

        ret_dict = {
            'point_cls_preds': point_cls_preds,
        }

        point_cls_scores = torch.softmax(point_cls_preds)
        batch_dict['point_cls_scores'], _ = point_cls_scores.max(dim=-1)

        if self.training:
            targets_dict = self.assign_targets(batch_dict)
            ret_dict['point_cls_labels'] = targets_dict['point_cls_labels']
        self.forward_ret_dict = ret_dict
        print(f'after assign points cls: ')
        print(ret_dict['point_class_labels'])
        return batch_dict

    def build_losses(self, losses_cfg):
        self.add_module(
            'cls_loss_func',
            SoftmaxFocalClassificationLoss(alpha=0.25, gamma=2.0)
        )

    # def assign_point_targets(self, points, gt_boxes, extend_gt_boxes=None,
    #                          ret_box_labels=False, ret_part_labels=False,
    #                          set_ignore_flag=True, use_ball_constraint=False, central_radius=2.0):
    #     """
    #         Args:
    #             points: (N1 + N2 + N3 + ..., 4) [bs_idx, x, y, z]
    #             gt_boxes: (B, M, 8)
    #             extend_gt_boxes: [B, M, 8]
    #             ret_box_labels:
    #             ret_part_labels:
    #             set_ignore_flag:
    #             use_ball_constraint:
    #             central_radius:
    #
    #         Returns:
    #             point_cls_labels: (N1 + N2 + N3 + ...), long type, 0:background, -1:ignored
    #
    #         """
    #     assert len(points.shape) == 2 and points.shape[1] == 4, 'points.shape=%s' % str(points.shape)
    #     assert len(gt_boxes.shape) == 3 and gt_boxes.shape[2] == 8, 'gt_boxes.shape=%s' % str(gt_boxes.shape)
    #     assert extend_gt_boxes is None or len(extend_gt_boxes.shape) == 3 and extend_gt_boxes.shape[2] == 8, \
    #         'extend_gt_boxes.shape=%s' % str(extend_gt_boxes.shape)
    #     assert set_ignore_flag != use_ball_constraint, 'Choose one only!'
    #     batch_size = gt_boxes.shape[0]
    #     bs_idx = points[:, 0]
    #     print(points.shape)
    #     print(bs_idx.shape)
    #     point_cls_labels = points.new_zeros(points.shape[0]).long()
    #     for k in range(batch_size):
    #         bs_mask = (bs_idx == k)
    #         points_single = points[bs_mask][:, 1:4]
    #         point_cls_labels_single = point_cls_labels.new_zeros(bs_mask.sum())
    #         box_idxs_of_pts = roiaware_pool3d_utils.points_in_boxes_gpu(
    #             points_single.unsqueeze(dim=0), gt_boxes[k:k + 1, :, 0:7].contiguous()
    #         ).long().squeeze(dim=0)
    #         box_fg_flag = (box_idxs_of_pts >= 0)
    #         #box_fg_type = self.assign_pt_type(box_idxs_of_pts) # assign point type
    #         if set_ignore_flag:
    #             extend_box_idxs_of_pts = roiaware_pool3d_utils.points_in_boxes_gpu(
    #                 points_single.unsqueeze(dim=0), extend_gt_boxes[k:k + 1, :, 0:7].contiguous()
    #             ).long().squeeze(dim=0)
    #             fg_flag = box_fg_flag
    #             ignore_flag = fg_flag ^ (extend_box_idxs_of_pts >= 0)
    #             point_cls_labels_single[ignore_flag] = -1
    #         elif use_ball_constraint:
    #             box_centers = gt_boxes[k][box_idxs_of_pts][:, 0:3].clone()
    #             box_centers[:, 2] += gt_boxes[k][box_idxs_of_pts][:, 5] / 2
    #             ball_flag = ((box_centers - points_single).norm(dim=1) < central_radius)
    #             fg_flag = box_fg_flag & ball_flag
    #         else:
    #             raise NotImplementedError
    #
    #         gt_box_of_fg_points = gt_boxes[k][box_idxs_of_pts[fg_flag]]
    #         point_cls_labels_single[fg_flag] = 1 if self.num_class == 1 else gt_box_of_fg_points[:, -1].long()
    #         #point_cls_labels_single[fg_flag] = box_fg_type[fg_flag]
    #         point_cls_labels[bs_mask] = point_cls_labels_single
    #
    #     targets_dict = {
    #         'point_cls_labels': point_cls_labels
    #     }
    #     return targets_dict