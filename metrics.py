# metrics.py
import torch
import numpy as np

class emrMetrics:
    def __init__(self, threshold=0.5):
        """
        Metrics for 2.5D instance segmentation.

        Args:
            threshold (float): Probability threshold for binarizing predicted masks.
        """
        self.threshold = threshold

    @staticmethod
    def dice_coefficient(pred_mask, true_mask, eps=1e-6):
        """
        Computes Dice coefficient between two binary masks.
        """
        intersection = (pred_mask & true_mask).float().sum((1, 2))
        union = pred_mask.float().sum((1, 2)) + true_mask.float().sum((1, 2))
        dice = (2.0 * intersection + eps) / (union + eps)
        return dice.mean().item()

    @staticmethod
    def iou(pred_mask, true_mask, eps=1e-6):
        """
        Computes Intersection over Union between two binary masks.
        """
        intersection = (pred_mask & true_mask).float().sum((1, 2))
        union = (pred_mask | true_mask).float().sum((1, 2))
        iou = (intersection + eps) / (union + eps)
        return iou.mean().item()

    def evaluate_batch(self, preds, targets):
        """
        Evaluates Dice, IoU, Precision, Recall for one batch.
        Args:
            preds (list[dict]): Model predictions.
            targets (list[dict]): Ground truth annotations.
        """
        batch_results = []
        for pred, target in zip(preds, targets):
            # Convert predicted masks: [N_pred, 1, H, W] → [N_pred, H, W]
            pred_masks = (pred["masks"] > self.threshold).squeeze(1)
            true_masks = target["masks"] > 0

            # Match instances greedily (best Dice per GT)
            dice_scores, iou_scores = [], []
            for gt_mask in true_masks:
                if pred_masks.numel() == 0:
                    dice_scores.append(0.0)
                    iou_scores.append(0.0)
                    continue
                scores = [self.dice_coefficient(pm, gt_mask) for pm in pred_masks]
                dice_scores.append(max(scores))
                iou_scores.append(max([self.iou(pm, gt_mask) for pm in pred_masks]))

            batch_results.append({
                "dice_mean": np.mean(dice_scores),
                "iou_mean": np.mean(iou_scores),
                "num_gt": len(true_masks)
            })
        return batch_results
