"""
Hungarian matcher + set-prediction loss for Mask2Former.

Forked from transformers 5.14.1 `models/mask2former/modeling_mask2former.py`
(sample_point .. Mask2FormerLoss), itself adapted from facebookresearch/Mask2Former. Changes vs. upstream:
  - plain nn.Module, no requires_backends / accelerate (single-process training only);
  - empty-target guards: a slice with zero instances (common at volume edges) yields empty matcher
    indices and zero-valued mask losses instead of running grid_sample on a 0-sized batch;
  - Mask2FormerLoss applies the class/mask/dice weights itself (upstream did it in
    Mask2FormerForUniversalSegmentation.get_loss_dict).
"""
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torch import Tensor, nn

from emrmodel.mask2former.config import M2FConfig


# Adapted from https://github.com/facebookresearch/detectron2/blob/main/projects/PointRend/point_rend/point_features.py
def sample_point(input_features: Tensor, point_coordinates: Tensor, add_dim=False, **kwargs) -> Tensor:
    """grid_sample wrapper taking [0, 1]-normalized (B, P, 2) point coordinates."""
    if point_coordinates.dim() == 3:
        add_dim = True
        point_coordinates = point_coordinates.unsqueeze(2)

    point_features = torch.nn.functional.grid_sample(input_features, 2.0 * point_coordinates - 1.0, **kwargs)
    if add_dim:
        point_features = point_features.squeeze(3)
    return point_features


def dice_loss(inputs: Tensor, labels: Tensor, num_masks: int) -> Tensor:
    probs = inputs.sigmoid().flatten(1)
    numerator = 2 * (probs * labels).sum(-1)
    denominator = probs.sum(-1) + labels.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_masks


def sigmoid_cross_entropy_loss(inputs: Tensor, labels: Tensor, num_masks: int) -> Tensor:
    cross_entropy_loss = nn.functional.binary_cross_entropy_with_logits(inputs, labels, reduction="none")
    return cross_entropy_loss.mean(1).sum() / num_masks


def pair_wise_dice_loss(inputs: Tensor, labels: Tensor) -> Tensor:
    inputs = inputs.sigmoid().flatten(1)
    numerator = 2 * torch.matmul(inputs, labels.T)
    denominator = inputs.sum(-1)[:, None] + labels.sum(-1)[None, :]
    return 1 - (numerator + 1) / (denominator + 1)


def pair_wise_sigmoid_cross_entropy_loss(inputs: Tensor, labels: Tensor) -> Tensor:
    height_and_width = inputs.shape[1]
    cross_entropy_loss_pos = nn.functional.binary_cross_entropy_with_logits(inputs, torch.ones_like(inputs), reduction="none")
    cross_entropy_loss_neg = nn.functional.binary_cross_entropy_with_logits(inputs, torch.zeros_like(inputs), reduction="none")
    loss_pos = torch.matmul(cross_entropy_loss_pos / height_and_width, labels.T)
    loss_neg = torch.matmul(cross_entropy_loss_neg / height_and_width, (1 - labels).T)
    return loss_pos + loss_neg


class Mask2FormerHungarianMatcher(nn.Module):
    """1-to-1 assignment between queries and ground-truth instances; unmatched queries become "no object"."""

    def __init__(self, cost_class: float = 1.0, cost_mask: float = 1.0, cost_dice: float = 1.0, num_points: int = 12544):
        super().__init__()
        if cost_class == 0 and cost_mask == 0 and cost_dice == 0:
            raise ValueError("All costs can't be 0")
        self.num_points = num_points
        self.cost_class = cost_class
        self.cost_mask = cost_mask
        self.cost_dice = cost_dice

    @torch.no_grad()
    def forward(self, masks_queries_logits: Tensor, class_queries_logits: Tensor, mask_labels: list, class_labels: list):
        """
        masks_queries_logits (B, Q, h, w), class_queries_logits (B, Q, num_labels + 1),
        mask_labels: list of (N_i, H, W), class_labels: list of (N_i,).
        Returns a list (len B) of (query_idx, target_idx) int64 tensor pairs.
        """
        indices = []
        batch_size = masks_queries_logits.shape[0]
        for i in range(batch_size):
            if len(class_labels[i]) == 0:
                indices.append((np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)))
                continue

            pred_probs = class_queries_logits[i].softmax(-1)
            pred_mask = masks_queries_logits[i]

            # classification cost approximated as -p[target class]; the constant 1 doesn't change the matching
            cost_class = -pred_probs[:, class_labels[i]]
            target_mask = mask_labels[i].to(pred_mask)[:, None]
            pred_mask = pred_mask[:, None]

            # same K random points for every (prediction, target) pair
            point_coordinates = torch.rand(1, self.num_points, 2, device=pred_mask.device)
            target_coordinates = point_coordinates.repeat(target_mask.shape[0], 1, 1)
            target_mask = sample_point(target_mask, target_coordinates, align_corners=False).squeeze(1)
            pred_coordinates = point_coordinates.repeat(pred_mask.shape[0], 1, 1)
            pred_mask = sample_point(pred_mask, pred_coordinates, align_corners=False).squeeze(1)

            cost_mask = pair_wise_sigmoid_cross_entropy_loss(pred_mask, target_mask)
            cost_dice = pair_wise_dice_loss(pred_mask, target_mask)
            cost_matrix = self.cost_mask * cost_mask + self.cost_class * cost_class + self.cost_dice * cost_dice
            # avoid "cost matrix is infeasible" from inf/nan entries
            cost_matrix = torch.minimum(cost_matrix, torch.tensor(1e10))
            cost_matrix = torch.maximum(cost_matrix, torch.tensor(-1e10))
            cost_matrix = torch.nan_to_num(cost_matrix, 0)
            indices.append(linear_sum_assignment(cost_matrix.cpu()))

        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


class Mask2FormerLoss(nn.Module):
    """DETR-style set loss: Hungarian-match, then supervise class (CE) and mask (point-sampled BCE + dice) per pair."""

    def __init__(self, config: M2FConfig):
        super().__init__()
        self.num_labels = config.num_labels
        self.weight_dict = {
            "loss_cross_entropy": config.class_weight,
            "loss_mask": config.mask_weight,
            "loss_dice": config.dice_weight,
        }

        self.eos_coef = config.no_object_weight
        empty_weight = torch.ones(self.num_labels + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer("empty_weight", empty_weight)

        self.num_points = config.train_num_points
        self.oversample_ratio = config.oversample_ratio
        self.importance_sample_ratio = config.importance_sample_ratio

        self.matcher = Mask2FormerHungarianMatcher(
            cost_class=config.class_weight,
            cost_dice=config.dice_weight,
            cost_mask=config.mask_weight,
            num_points=self.num_points,
        )

    def _max_by_axis(self, sizes):
        maxes = sizes[0]
        for sublist in sizes[1:]:
            for index, item in enumerate(sublist):
                maxes[index] = max(maxes[index], item)
        return maxes

    def _pad_images_to_max_in_batch(self, tensors):
        max_size = self._max_by_axis([list(tensor.shape) for tensor in tensors])
        batch_shape = [len(tensors)] + max_size
        batch_size, _, height, width = batch_shape
        padded_tensors = torch.zeros(batch_shape, dtype=tensors[0].dtype, device=tensors[0].device)
        padding_masks = torch.ones((batch_size, height, width), dtype=torch.bool, device=tensors[0].device)
        for tensor, padded_tensor, padding_mask in zip(tensors, padded_tensors, padding_masks):
            padded_tensor[: tensor.shape[0], : tensor.shape[1], : tensor.shape[2]].copy_(tensor)
            padding_mask[: tensor.shape[1], : tensor.shape[2]] = False
        return padded_tensors, padding_masks

    def loss_labels(self, class_queries_logits: Tensor, class_labels: list, indices) -> dict:
        batch_size, num_queries, _ = class_queries_logits.shape
        idx = self._get_predictions_permutation_indices(indices)
        target_classes_o = torch.cat([target[j] for target, (_, j) in zip(class_labels, indices)])
        target_classes = torch.full(
            (batch_size, num_queries), fill_value=self.num_labels, dtype=torch.int64, device=class_queries_logits.device
        )
        target_classes[idx] = target_classes_o
        loss_ce = nn.functional.cross_entropy(class_queries_logits.transpose(1, 2), target_classes, weight=self.empty_weight)
        return {"loss_cross_entropy": loss_ce}

    def loss_masks(self, masks_queries_logits: Tensor, mask_labels: list, indices, num_masks) -> dict:
        src_idx = self._get_predictions_permutation_indices(indices)
        tgt_idx = self._get_targets_permutation_indices(indices)
        pred_masks = masks_queries_logits[src_idx]

        if pred_masks.shape[0] == 0:
            # no GT instance anywhere in the batch: nothing to supervise, keep the graph connected
            zero = masks_queries_logits.sum() * 0.0
            return {"loss_mask": zero, "loss_dice": zero}

        target_masks, _ = self._pad_images_to_max_in_batch(mask_labels)
        target_masks = target_masks[tgt_idx]

        # no need to upsample predictions: points are in normalized coordinates
        pred_masks = pred_masks[:, None]
        target_masks = target_masks[:, None]

        with torch.no_grad():
            point_coordinates = self.sample_points_using_uncertainty(
                pred_masks,
                self.calculate_uncertainty,
                self.num_points,
                self.oversample_ratio,
                self.importance_sample_ratio,
            )
            point_labels = sample_point(target_masks, point_coordinates, align_corners=False).squeeze(1)

        point_logits = sample_point(pred_masks, point_coordinates, align_corners=False).squeeze(1)

        return {
            "loss_mask": sigmoid_cross_entropy_loss(point_logits, point_labels, num_masks),
            "loss_dice": dice_loss(point_logits, point_labels, num_masks),
        }

    def _get_predictions_permutation_indices(self, indices):
        batch_indices = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        predictions_indices = torch.cat([src for (src, _) in indices])
        return batch_indices, predictions_indices

    def _get_targets_permutation_indices(self, indices):
        batch_indices = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        target_indices = torch.cat([tgt for (_, tgt) in indices])
        return batch_indices, target_indices

    def calculate_uncertainty(self, logits: Tensor) -> Tensor:
        """Uncertainty = -|logit|: points closest to the decision boundary score highest."""
        return -(torch.abs(logits))

    def sample_points_using_uncertainty(self, logits, uncertainty_function, num_points, oversample_ratio, importance_sample_ratio):
        num_boxes = logits.shape[0]
        num_points_sampled = int(num_points * oversample_ratio)

        point_coordinates = torch.rand(num_boxes, num_points_sampled, 2, device=logits.device)
        point_logits = sample_point(logits, point_coordinates, align_corners=False)
        point_uncertainties = uncertainty_function(point_logits)

        num_uncertain_points = int(importance_sample_ratio * num_points)
        num_random_points = num_points - num_uncertain_points

        idx = torch.topk(point_uncertainties[:, 0, :], k=num_uncertain_points, dim=1)[1]
        shift = num_points_sampled * torch.arange(num_boxes, dtype=torch.long, device=logits.device)
        idx += shift[:, None]
        point_coordinates = point_coordinates.view(-1, 2)[idx.view(-1), :].view(num_boxes, num_uncertain_points, 2)

        if num_random_points > 0:
            point_coordinates = torch.cat(
                [point_coordinates, torch.rand(num_boxes, num_random_points, 2, device=logits.device)], dim=1
            )
        return point_coordinates

    def get_num_masks(self, class_labels: list, device) -> Tensor:
        num_masks = sum(len(classes) for classes in class_labels)
        num_masks = torch.as_tensor(num_masks, dtype=torch.float, device=device)
        return torch.clamp(num_masks, min=1)

    def _unweighted(self, masks_queries_logits, class_queries_logits, mask_labels, class_labels, num_masks):
        indices = self.matcher(masks_queries_logits, class_queries_logits, mask_labels, class_labels)
        return {
            **self.loss_masks(masks_queries_logits, mask_labels, indices, num_masks),
            **self.loss_labels(class_queries_logits, class_labels, indices),
        }

    def forward(self, masks_queries_logits: Tensor, class_queries_logits: Tensor, mask_labels: list,
                class_labels: list, auxiliary_predictions: list = None) -> dict:
        """
        Returns the *weighted* losses: loss_cross_entropy / loss_mask / loss_dice for the final decoder layer, plus
        the same three suffixed _0, _1, ... per auxiliary (intermediate) layer. Their plain sum is the training loss.
        """
        num_masks = self.get_num_masks(class_labels, device=masks_queries_logits.device)
        losses = self._unweighted(masks_queries_logits, class_queries_logits, mask_labels, class_labels, num_masks)
        if auxiliary_predictions is not None:
            for idx, aux_outputs in enumerate(auxiliary_predictions):
                aux = self._unweighted(aux_outputs["masks_queries_logits"], aux_outputs["class_queries_logits"],
                                       mask_labels, class_labels, num_masks)
                losses.update({f"{key}_{idx}": value for key, value in aux.items()})

        for key, weight in self.weight_dict.items():
            for loss_key in losses:
                if loss_key.startswith(key):
                    losses[loss_key] = losses[loss_key] * weight
        return losses
