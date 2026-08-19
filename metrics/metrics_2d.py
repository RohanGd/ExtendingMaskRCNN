# metrics/metrics_2d.py
"""
Per-2D-slice / per-batch instance segmentation metrics for Mask R-CNN.
See metrics/metrics_volume.py for the companion per-3D-volume suite.

References:
https://www.sciencedirect.com/science/article/pii/S0925231225002565?via%3Dihub
- jaccard index / IoU, overlap threshold (alpha), (mean) average precision,
  Difference in Count (DiC), Instance Precision/Recall/F1.
https://public.celltrackingchallenge.net/documents/SEG.pdf
- SEG score (majority-overlap rule; here computed per-slice -- see
  metrics_volume.emrMetricsVolume.seg_score() for the true, volume-level CTC SEG).
Kumar et al. 2017 -- Aggregated Jaccard Index (AJI).
Kirillov et al. 2019 -- Panoptic Quality (PQ = SQ x RQ).
"""
import numpy as np
import torch
from typing import List, Dict, Optional

from metrics import math_utils


class emrMetrics2D:
    """
    2D per-slice/per-batch instance segmentation metrics.
    - Chunked CPU pairwise IoU (memory-safe), Hungarian 1-to-1 matching.
    - Per-threshold TP/FP/FN, matched IoU/Dice, true COCO-style AP, PQ (SQ/RQ).
    - Global AJI accumulated as a running (C, U) ratio-of-sums (not averaged per-image).
    - Per-slice SEG (CTC majority-overlap rule).
    """

    def __init__(self, overlap_thresholds: List[float] = [0.5, 0.75, 0.9], iou_chunk_size: int = 32):
        self.overlap_thresholds = sorted(overlap_thresholds)
        self.iou_chunk_size = int(iou_chunk_size)
        self.reset()

    def reset(self):
        self.stats_per_threshold = {
            alpha: {
                'true_positives': 0,
                'false_positives': 0,
                'false_negatives': 0,
                'matched_ious': [],
                'matched_dice': [],
                'pq_iou_sum': 0.0,
                'ap_records': [],  # List[(score, is_tp)]
            }
            for alpha in self.overlap_thresholds
        }

        self.global_matched_iou_sum = 0.0
        self.global_matched_iou_count = 0
        self.global_matched_dice_sum = 0.0
        self.global_matched_dice_count = 0

        self.total_predictions = 0
        self.total_ground_truths = 0
        self.count_differences: List[int] = []
        self.seg_scores = list()

        self.aji_c = 0.0
        self.aji_u = 0.0

    # ------------------------- Update (per batch) -------------------------
    def update(self, preds_batch: List[Dict], targets_batch: List[Dict]):
        """
        preds_batch/targets_batch: lists (length B) of dicts with 'masks' -> tensor
        ([N,H,W] or [N,1,H,W]); preds additionally carry 'scores' -> tensor [N].
        """
        for preds, targets in zip(preds_batch, targets_batch):
            pred_masks = preds.get('masks', torch.empty((0, 1, 1, 1)))
            target_masks = targets.get('masks', torch.empty((0, 1, 1, 1)))
            pred_scores = preds.get('scores', torch.empty((pred_masks.shape[0],)))

            n_pred = pred_masks.shape[0]
            n_gt = target_masks.shape[0]

            self.total_predictions += n_pred
            self.total_ground_truths += n_gt
            self.count_differences.append(abs(n_pred - n_gt))

            if n_pred == 0 or n_gt == 0:
                for alpha in self.overlap_thresholds:
                    stats = self.stats_per_threshold[alpha]
                    stats['false_positives'] += n_pred
                    stats['false_negatives'] += n_gt
                    for p in range(n_pred):
                        stats['ap_records'].append((float(pred_scores[p].item()), False))
                if n_pred == 0 and n_gt > 0:
                    self.aji_u += float(target_masks.bool().sum().item())
                elif n_gt == 0 and n_pred > 0:
                    self.aji_u += float((pred_masks > 0.5).bool().sum().item())
                continue

            # rows = GT, cols = predictions (shared convention, see math_utils docstring)
            iou_matrix = math_utils.pairwise_iou_masks(target_masks, pred_masks, chunk_size=self.iou_chunk_size)

            # Global (alpha=0) match for running IoU/Dice sums
            global_matches = math_utils.hungarian_match(iou_matrix, alpha=0.0)
            for g, p in global_matches:
                iou_val = iou_matrix[g, p].item()
                dice_val = math_utils.dice_from_iou(iou_val)
                self.global_matched_iou_sum += float(iou_val)
                self.global_matched_iou_count += 1
                self.global_matched_dice_sum += float(dice_val)
                self.global_matched_dice_count += 1

            # Per-threshold stats
            for alpha in self.overlap_thresholds:
                matches = math_utils.hungarian_match(iou_matrix, alpha)
                matched_preds = {p for _, p in matches}

                tp = len(matches)
                fp = n_pred - tp
                fn = n_gt - tp

                stats = self.stats_per_threshold[alpha]
                stats['true_positives'] += tp
                stats['false_positives'] += fp
                stats['false_negatives'] += fn

                for g, p in matches:
                    iou_val = iou_matrix[g, p].item()
                    stats['matched_ious'].append(float(iou_val))
                    stats['matched_dice'].append(math_utils.dice_from_iou(iou_val))
                    stats['pq_iou_sum'] += float(iou_val)

                for p in range(n_pred):
                    stats['ap_records'].append((float(pred_scores[p].item()), p in matched_preds))

            # AJI (global ratio-of-sums accumulation)
            c, u = math_utils.aji_intersection_union_masks(target_masks, pred_masks)
            self.aji_c += c
            self.aji_u += u

            # SEG (per-slice, CTC majority-overlap rule)
            self.seg_scores.append(math_utils.seg_score_single_frame(target_masks, pred_masks))

    # ------------------------- Compute final metrics -------------------------
    def compute(self) -> Dict[str, float]:
        metrics: Dict[str, float] = {}

        metrics['jaccard_index'] = (
            self.global_matched_iou_sum / self.global_matched_iou_count
            if self.global_matched_iou_count > 0 else 0.0
        )
        metrics['dice_score'] = (
            self.global_matched_dice_sum / self.global_matched_dice_count
            if self.global_matched_dice_count > 0 else 0.0
        )
        metrics['mean_difference_in_count'] = (
            float(np.mean(self.count_differences)) if self.count_differences else 0.0
        )
        metrics['aji'] = math_utils.aggregated_jaccard_index(self.aji_c, self.aji_u)

        ap_values: List[float] = []
        for alpha in self.overlap_thresholds:
            s = self.stats_per_threshold[alpha]
            tp, fp, fn = s['true_positives'], s['false_positives'], s['false_negatives']

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

            mean_iou = float(np.mean(s['matched_ious'])) if s['matched_ious'] else 0.0
            mean_dice = float(np.mean(s['matched_dice'])) if s['matched_dice'] else 0.0

            ap = math_utils.coco_style_ap(s['ap_records'], self.total_ground_truths)
            ap_values.append(ap)

            sq, rq, pq = math_utils.sq_rq_pq(tp, fp, fn, s['pq_iou_sum'])
            acc = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0

            key = f"@{alpha:.2f}".replace('.', '_')
            metrics[f'instance_precision{key}'] = precision
            metrics[f'instance_recall{key}'] = recall
            metrics[f'instance_f1{key}'] = f1
            metrics[f'mean_iou{key}'] = mean_iou
            metrics[f'mean_dice{key}'] = mean_dice
            metrics[f'ap{key}'] = ap
            metrics[f'sq{key}'] = sq
            metrics[f'rq{key}'] = rq
            metrics[f'pq{key}'] = pq
            metrics[f'acc{key}'] = acc

        metrics['mean_average_precision'] = float(np.mean(ap_values)) if ap_values else 0.0
        metrics['total_predictions'] = int(self.total_predictions)
        metrics['total_ground_truths'] = int(self.total_ground_truths)

        seg_scores = np.array(self.seg_scores) if self.seg_scores else np.array([0.0])
        metrics["SEG_score"] = float(seg_scores.mean())
        metrics["SEG_score_median"] = float(np.median(seg_scores))

        return metrics

    # ------------------------- Utilities -------------------------
    def save(self, path: str = "metrics_summary.txt") -> None:
        with open(path, 'w', encoding='utf-8') as f:
            f.write(str(self))

    def __str__(self) -> str:
        results = self.compute()
        lines: List[str] = ["2D Per-Slice Instance Segmentation Metrics Summary:\n", "=" * 60]

        lines.append(f"Jaccard Index (IoU):        {results['jaccard_index']:.4f}")
        lines.append(f"Dice Score (global):        {results['dice_score']:.4f}")
        lines.append(f"AJI (global):                {results['aji']:.4f}")
        lines.append(f"Difference in Count (DiC):  {results['mean_difference_in_count']:.2f}")
        lines.append(f"Mean Average Precision:     {results['mean_average_precision']:.4f}\n")

        for alpha in self.overlap_thresholds:
            key = f"@{alpha:.2f}".replace('.', '_')
            lines.append(f"Metrics at @ = {alpha:.2f}:")
            lines.append(f"  Instance Precision:  {results[f'instance_precision{key}']:.4f}")
            lines.append(f"  Instance Recall:     {results[f'instance_recall{key}']:.4f}")
            lines.append(f"  Instance F1:         {results[f'instance_f1{key}']:.4f}")
            lines.append(f"  Mean IoU (matched):  {results[f'mean_iou{key}']:.4f}")
            lines.append(f"  Mean Dice (matched): {results[f'mean_dice{key}']:.4f}")
            lines.append(f"  AP:                  {results[f'ap{key}']:.4f}")
            lines.append(f"  SQ / RQ / PQ:        {results[f'sq{key}']:.4f} / {results[f'rq{key}']:.4f} / {results[f'pq{key}']:.4f}")
            lines.append(f"  Acc:                 {results[f'acc{key}']:.4f}\n")

        lines.append(f"Total Predictions:     {results['total_predictions']}")
        lines.append(f"Total Ground Truths:   {results['total_ground_truths']}")
        lines.append(f"SEG_score mean:        {results['SEG_score']}")
        lines.append(f"SEG_score median:      {results['SEG_score_median']}")

        return "\n".join(lines)
