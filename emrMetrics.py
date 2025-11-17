# metrics.py

'''
https://www.sciencedirect.com/science/article/pii/S0925231225002565?via%3Dihub
- jaccard index / IoU - corresponds to the number of pixels at the intersection of binary masks divided by the number of pixels at the union of the masks.
- overlap threshold (alpha)- IoU threshold value 
- (mean) average Precision - It measures the precision of the predicted instance masks at various overlap thresholds.
- Difference in Count (DiC) [5]: DiC quantifies the counting accuracy of a model by measuring the difference between the number of predicted instances and the actual number of ground-truth instances.
- Instance Precision - InsPr quantifies precision by calculating the ratio of correctly predicted instances to the total number of predicted instances. It is defined as the quotient of the number of true positive instance predictions divided by the total number of predictions. 
- INstance Recall - InsRe is defined as the quotient of the number of true positive instance predictions divided by the total number of ground truth instances.
- Instance F1 - harmonic mean of instance precision and instance recall.
'''
import os
import torch
import numpy as np
from scipy.optimize import linear_sum_assignment
from typing import List, Dict, Tuple, Optional


class emrMetrics:
    """
    Metrics class for Mask R-CNN instance segmentation.
    - Computes IoU using chunked boolean logic (CPU-side by default to avoid GPU OOM)
    - Uses Hungarian matching for optimal 1-to-1 assignment
    - Tracks per-threshold TP/FP/FN, matched IoU and Dice
    - Keeps running sums for global IoU/Dice to avoid storing huge lists
    """

    def __init__(self, overlap_thresholds: List[float] = [0.5, 0.75, 0.9], iou_chunk_size: int = 32):
        self.overlap_thresholds = sorted(overlap_thresholds)
        self.iou_chunk_size = int(iou_chunk_size)
        self.reset()

    def reset(self):
        # Per-threshold statistics
        self.stats_per_threshold = {
            alpha: {
                'true_positives': 0,
                'false_positives': 0,
                'false_negatives': 0,
                'matched_ious': [],    # small lists of matched IoUs (scalars)
                'matched_dice': []     # small lists of matched Dice values (scalars)
            }
            for alpha in self.overlap_thresholds
        }

        # Running (global) statistics to avoid large memory
        self.global_matched_iou_sum = 0.0
        self.global_matched_iou_count = 0
        self.global_matched_dice_sum = 0.0
        self.global_matched_dice_count = 0

        # Counts
        self.total_predictions = 0
        self.total_ground_truths = 0
        self.count_differences: List[int] = []

    # ------------------------- IoU computation -------------------------
    def compute_pairwise_iou(
        self,
        masks1: torch.Tensor,
        masks2: torch.Tensor,
        chunk_size: Optional[int] = None
    ) -> torch.Tensor:
        """
        Compute pairwise IoU between two sets of binary masks in a memory-safe way.

        - Moves computation to CPU to avoid allocating large broadcast tensors on GPU.
        - Uses chunking over predictions to limit peak memory: result is [N1, N2].

        Args:
            masks1: [N1, H, W] or [N1, 1, H, W] (torch.Tensor)
            masks2: [N2, H, W] or [N2, 1, H, W]
            chunk_size: number of predictions to process at once (defaults to self.iou_chunk_size)

        Returns:
            iou_matrix: torch.Tensor on CPU of shape [N1, N2]
        """
        if chunk_size is None:
            chunk_size = self.iou_chunk_size

        # Normalize shapes and move to CPU
        if masks1.dim() == 4:
            masks1 = masks1.squeeze(1)
        if masks2.dim() == 4:
            masks2 = masks2.squeeze(1)

        # If tensors are on GPU, move to CPU for IoU computation to avoid OOM
        device1 = masks1.device
        device2 = masks2.device

        m1 = masks1.detach().cpu()
        m2 = masks2.detach().cpu()

        # Binarize (bool)
        m1 = m1 > 0.5
        m2 = m2 > 0.5

        N1 = 0 if m1.numel() == 0 else m1.shape[0]
        N2 = 0 if m2.numel() == 0 else m2.shape[0]

        if N1 == 0 or N2 == 0:
            return torch.zeros((N1, N2), dtype=torch.float32)

        # Precompute areas of m2 (shape [N2]) and keep as float
        area2 = m2.view(N2, -1).sum(dim=1).float()  # [N2]

        ious = []  # list of chunks
        for i in range(0, N1, chunk_size):
            chunk = m1[i:i+chunk_size]                  # [C, H, W]
            C = chunk.shape[0]

            # chunk: [C, H, W] -> [C, 1, H, W] ; m2: [N2, H, W] -> [1, N2, H, W]
            # This creates an intermediate [C, N2, H, W] boolean array on CPU (C small)
            inter = (chunk[:, None, :, :] & m2[None, :, :, :]).view(C, N2, -1).sum(dim=2).float()  # [C, N2]

            area1_chunk = chunk.view(C, -1).sum(dim=1).float().unsqueeze(1)  # [C, 1]
            union = area1_chunk + area2.unsqueeze(0) - inter
            chunk_iou = inter / (union + 1e-6)
            ious.append(chunk_iou)

        iou_matrix = torch.cat(ious, dim=0)  # [N1, N2] on CPU (float)

        # If original tensors were on CUDA and user expects CUDA, we return CPU tensor (safe). Caller may move to device if needed.
        return iou_matrix

    # ------------------------- Hungarian matching -------------------------
    def hungarian_match(self, iou_matrix: torch.Tensor, alpha: float) -> List[Tuple[int, int]]:
        """
        Perform optimal 1-to-1 matching using Hungarian algorithm.
        iou_matrix is expected on CPU (or convertible to CPU).
        Only matches with IoU >= alpha are returned.
        """
        if iou_matrix.numel() == 0:
            return []

        # Convert to cost matrix (minimize cost = 1 - IoU)
        cost = (1.0 - iou_matrix).numpy()
        row_idx, col_idx = linear_sum_assignment(cost)

        matches: List[Tuple[int, int]] = []
        for r, c in zip(row_idx, col_idx):
            if iou_matrix[r, c].item() >= alpha:
                matches.append((int(r), int(c)))
        return matches

    # ------------------------- Update (per batch) -------------------------
    def update(self, preds_batch: List[Dict], targets_batch: List[Dict]):
        """
        Update metrics with a batch of predictions and targets.

        preds_batch and targets_batch are lists (length B) of dicts containing 'masks' -> tensor
        """
        for preds, targets in zip(preds_batch, targets_batch):
            pred_masks = preds.get('masks', torch.empty((0, 1, 1, 1)))
            target_masks = targets.get('masks', torch.empty((0, 1, 1, 1)))

            n_pred = pred_masks.shape[0]
            n_gt = target_masks.shape[0]

            self.total_predictions += n_pred
            self.total_ground_truths += n_gt
            self.count_differences.append(abs(n_pred - n_gt))

            # Handle degenerate cases quickly
            if n_pred == 0 or n_gt == 0:
                for alpha in self.overlap_thresholds:
                    stats = self.stats_per_threshold[alpha]
                    stats['false_positives'] += n_pred
                    stats['false_negatives'] += n_gt
                continue

            # Compute IoU matrix on CPU in chunks to keep memory small
            iou_matrix = self.compute_pairwise_iou(pred_masks, target_masks)

            # For global metrics we consider matches at alpha=0.0 (best match regardless of threshold)
            global_matches = self.hungarian_match(iou_matrix, alpha=0.0)
            for p, g in global_matches:
                iou_val = iou_matrix[p, g].item()
                dice_val = (2.0 * iou_val) / (1.0 + iou_val + 1e-6)
                self.global_matched_iou_sum += float(iou_val)
                self.global_matched_iou_count += 1
                self.global_matched_dice_sum += float(dice_val)
                self.global_matched_dice_count += 1

            # Per-threshold stats (use Hungarian matching per threshold)
            for alpha in self.overlap_thresholds:
                matches = self.hungarian_match(iou_matrix, alpha)

                tp = len(matches)
                fp = n_pred - tp
                fn = n_gt - tp

                stats = self.stats_per_threshold[alpha]
                stats['true_positives'] += tp
                stats['false_positives'] += fp
                stats['false_negatives'] += fn

                for p, g in matches:
                    iou_val = iou_matrix[p, g].item()
                    dice_val = (2.0 * iou_val) / (1.0 + iou_val + 1e-6)
                    stats['matched_ious'].append(float(iou_val))
                    stats['matched_dice'].append(float(dice_val))
            
            torch.cuda.empty_cache()

    # ------------------------- Compute final metrics -------------------------
    def compute(self) -> Dict[str, float]:
        metrics: Dict[str, float] = {}

        # Global Jaccard / mean IoU across globally matched pairs
        if self.global_matched_iou_count > 0:
            metrics['jaccard_index'] = self.global_matched_iou_sum / self.global_matched_iou_count
        else:
            metrics['jaccard_index'] = 0.0

        # Global Dice across matched pairs
        if self.global_matched_dice_count > 0:
            metrics['dice_score'] = self.global_matched_dice_sum / self.global_matched_dice_count
        else:
            metrics['dice_score'] = 0.0

        metrics['mean_difference_in_count'] = (
            float(np.mean(self.count_differences)) if self.count_differences else 0.0
        )

        # Per-threshold metrics
        ap_values: List[float] = []
        for alpha in self.overlap_thresholds:
            s = self.stats_per_threshold[alpha]
            tp = s['true_positives']
            fp = s['false_positives']
            fn = s['false_negatives']

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

            mean_iou = float(np.mean(s['matched_ious'])) if s['matched_ious'] else 0.0
            mean_dice = float(np.mean(s['matched_dice'])) if s['matched_dice'] else 0.0

            ap = precision  # precision at threshold used as AP proxy (consistent with previous design)
            ap_values.append(ap)

            key = f"@{alpha:.2f}".replace('.', '_')
            metrics[f'instance_precision{key}'] = precision
            metrics[f'instance_recall{key}'] = recall
            metrics[f'instance_f1{key}'] = f1
            metrics[f'mean_iou{key}'] = mean_iou
            metrics[f'mean_dice{key}'] = mean_dice
            metrics[f'ap{key}'] = ap

        metrics['mean_average_precision'] = float(np.mean(ap_values)) if ap_values else 0.0
        metrics['total_predictions'] = int(self.total_predictions)
        metrics['total_ground_truths'] = int(self.total_ground_truths)

        return metrics

    # ------------------------- Utilities -------------------------
    def save(self, path: str = "metrics_summary.txt") -> None:
        """Save __str__ output to a file path."""
        path = os.path.join("results", path)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w',encoding='utf-8') as f:
            f.write(str(self))

    def __str__(self) -> str:
        results = self.compute()
        lines: List[str] = ["Instance Segmentation Metrics Summary:\n", "=" * 60]

        lines.append(f"Jaccard Index (IoU):        {results['jaccard_index']:.4f}")
        lines.append(f"Dice Score (global):        {results['dice_score']:.4f}")
        lines.append(f"Difference in Count (DiC):  {results['mean_difference_in_count']:.2f}")
        lines.append(f"Mean Average Precision:     {results['mean_average_precision']:.4f}\n")

        for alpha in self.overlap_thresholds:
            key = f"@{alpha:.2f}".replace('.', '_')
            lines.append(f"Metrics at α = {alpha:.2f}:")
            lines.append(f"  Instance Precision:  {results[f'instance_precision{key}']:.4f}")
            lines.append(f"  Instance Recall:     {results[f'instance_recall{key}']:.4f}")
            lines.append(f"  Instance F1:         {results[f'instance_f1{key}']:.4f}")
            lines.append(f"  Mean IoU (matched):  {results[f'mean_iou{key}']:.4f}")
            lines.append(f"  Mean Dice (matched): {results[f'mean_dice{key}']:.4f}\n")

        lines.append(f"Total Predictions:     {results['total_predictions']}")
        lines.append(f"Total Ground Truths:   {results['total_ground_truths']}")

        return "\n".join(lines)

