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
import torch
import numpy as np

import torch
import torch.nn.functional as F
import torch.nn.functional as F
from typing import List, Tuple, Dict, Optional
import numpy as np

import torch
import torch.nn.functional as F
from typing import List, Tuple, Dict, Optional
import numpy as np


class emrMetrics:
    """
    Metrics class for Mask R-CNN instance segmentation.
    Computes IoU/Jaccard Index and instance-level metrics with overlap threshold matching.
    """
    
    def __init__(self, overlap_thresholds: List[float] = [0.5, 0.75, 0.9]):
        """
        Args:
            overlap_thresholds: List of IoU/alpha thresholds for matching (default: [0.5, 0.75, 0.9])
        """
        self.overlap_thresholds = sorted(overlap_thresholds)
        self.reset()
    
    def reset(self):
        """Reset all accumulated metrics"""
        # Per-threshold statistics
        self.stats_per_threshold = {
            alpha: {
                'true_positives': 0,
                'false_positives': 0,
                'false_negatives': 0,
                'matched_ious': []
            } for alpha in self.overlap_thresholds
        }
        
        # Overall statistics
        self.all_ious = []
        self.total_predictions = 0
        self.total_ground_truths = 0
        self.count_differences = []
    
    def compute_pairwise_iou(self, masks1: torch.Tensor, masks2: torch.Tensor) -> torch.Tensor:
        """
        Compute pairwise IoU/Jaccard Index between two sets of masks.
        
        IoU = |A ∩ B| / |A ∪ B|
        
        Args:
            masks1: [N1, H, W] or [N1, 1, H, W]
            masks2: [N2, H, W] or [N2, 1, H, W]
            
        Returns:
            iou_matrix: [N1, N2] pairwise IoU scores
        """
        # Ensure masks are binary and squeeze channel dimension if present
        if masks1.dim() == 4:
            masks1 = masks1.squeeze(1)
        if masks2.dim() == 4:
            masks2 = masks2.squeeze(1)
        
        masks1 = (masks1 > 0.5).float()
        masks2 = (masks2 > 0.5).float()
        
        N1, N2 = masks1.shape[0], masks2.shape[0]
        
        # Flatten spatial dimensions
        masks1_flat = masks1.view(N1, -1)  # [N1, H*W]
        masks2_flat = masks2.view(N2, -1)  # [N2, H*W]
        
        # Compute intersection: [N1, N2]
        intersection = torch.matmul(masks1_flat, masks2_flat.t())
        
        # Compute areas
        area1 = masks1_flat.sum(dim=1, keepdim=True)  # [N1, 1]
        area2 = masks2_flat.sum(dim=1, keepdim=True)  # [N2, 1]
        
        # Compute union: [N1, N2]
        union = area1 + area2.t() - intersection
        
        # Compute IoU, avoiding division by zero
        iou = intersection / (union + 1e-6)
        
        return iou
    
    def match_predictions_at_threshold(
        self, 
        preds: torch.Tensor, 
        targets: torch.Tensor,
        alpha: float
    ) -> Tuple[List[Tuple[int, int]], torch.Tensor]:
        """
        Match predictions to ground truth using overlap threshold (alpha).
        
        Args:
            preds: [N_pred, 1, H, W] or [N_pred, H, W]
            targets: [N_gt, H, W]
            alpha: Overlap threshold (IoU threshold)
            
        Returns:
            matches: List of (pred_idx, gt_idx) pairs that exceed threshold
            iou_matrix: [N_pred, N_gt] IoU matrix
        """
        if preds.shape[0] == 0 or targets.shape[0] == 0:
            return [], torch.zeros((preds.shape[0], targets.shape[0]))
        
        # Compute pairwise IoU
        iou_matrix = self.compute_pairwise_iou(preds, targets)  # [N_pred, N_gt]
        
        # Greedy matching: assign each prediction to highest IoU ground truth
        matches = []
        matched_gt = set()
        
        # Sort predictions by max IoU in descending order
        max_ious, _ = iou_matrix.max(dim=1)
        sorted_pred_indices = torch.argsort(max_ious, descending=True)
        
        for pred_idx in sorted_pred_indices.tolist():
            # Find best unmatched ground truth that exceeds threshold
            best_iou = 0
            best_gt_idx = -1
            
            for gt_idx in range(targets.shape[0]):
                if gt_idx not in matched_gt:
                    iou_val = iou_matrix[pred_idx, gt_idx].item()
                    if iou_val > best_iou and iou_val >= alpha:
                        best_iou = iou_val
                        best_gt_idx = gt_idx
            
            if best_gt_idx >= 0:
                matches.append((pred_idx, best_gt_idx))
                matched_gt.add(best_gt_idx)
        
        return matches, iou_matrix
    
    def update(self, preds_batch: List[Dict], targets_batch: List[Dict]):
        """
        Update metrics with a batch of predictions and targets.
        
        Args:
            preds_batch: List of B dicts, each containing 'masks' key with [N_pred_i, 1, H, W] tensor
            targets_batch: List of B dicts, each containing 'masks' key with [N_gt_i, H, W] tensor
        """
        for preds, targets in zip(preds_batch, targets_batch):
            # Extract masks from dictionaries
            pred_masks = preds.get('masks', torch.empty(0, 1, 1, 1))  # Default empty if no masks
            target_masks = targets.get('masks', torch.empty(0, 1, 1))  # Default empty if no masks
            
            n_pred = pred_masks.shape[0]
            n_gt = target_masks.shape[0]
            
            # Update overall counts
            self.total_predictions += n_pred
            self.total_ground_truths += n_gt
            
            # Compute Difference in Count (DiC)
            self.count_differences.append(abs(n_pred - n_gt))
            
            # Process each overlap threshold
            for alpha in self.overlap_thresholds:
                # Match predictions to ground truth at this threshold
                matches, iou_matrix = self.match_predictions_at_threshold(pred_masks, target_masks, alpha)
                
                # True Positives: successfully matched instances
                tp = len(matches)
                
                # False Positives: predictions that couldn't be matched
                fp = n_pred - tp
                
                # False Negatives: ground truths that couldn't be matched
                fn = n_gt - tp
                
                # Update statistics
                stats = self.stats_per_threshold[alpha]
                stats['true_positives'] += tp
                stats['false_positives'] += fp
                stats['false_negatives'] += fn
                
                # Store matched IoUs
                for pred_idx, gt_idx in matches:
                    iou_val = iou_matrix[pred_idx, gt_idx].item()
                    stats['matched_ious'].append(iou_val)
            
            # Store all IoU values (for overall Jaccard Index)
            if n_pred > 0 and n_gt > 0:
                _, iou_matrix = self.match_predictions_at_threshold(pred_masks, target_masks, 0.0)
                self.all_ious.extend(iou_matrix.flatten().tolist())
    
    def compute(self) -> Dict[str, float]:
        """
        Compute final metrics.
        
        Returns:
            Dictionary containing:
                - jaccard_index: Mean IoU over all pairwise comparisons
                - mean_average_precision (mAP): Mean AP across all thresholds
                - ap@{alpha}: Average Precision at each threshold
                - difference_in_count (DiC): Mean absolute difference in instance counts
                - instance_precision@{alpha}: Instance Precision at each threshold
                - instance_recall@{alpha}: Instance Recall at each threshold
                - instance_f1@{alpha}: Instance F1 Score at each threshold
        """
        metrics = {}
        
        # 1. Jaccard Index / IoU
        if len(self.all_ious) > 0:
            metrics['jaccard_index'] = np.mean(self.all_ious)
        else:
            metrics['jaccard_index'] = 0.0
        
        # 2. Difference in Count (DiC)
        if len(self.count_differences) > 0:
            metrics['difference_in_count'] = np.mean(self.count_differences)
        else:
            metrics['difference_in_count'] = 0.0
        
        # 3. Per-threshold metrics
        ap_values = []
        
        for alpha in self.overlap_thresholds:
            stats = self.stats_per_threshold[alpha]
            tp = stats['true_positives']
            fp = stats['false_positives']
            fn = stats['false_negatives']
            
            # Instance Precision: TP / (TP + FP)
            if tp + fp > 0:
                precision = tp / (tp + fp)
            else:
                precision = 0.0
            
            # Instance Recall: TP / (TP + FN)
            if tp + fn > 0:
                recall = tp / (tp + fn)
            else:
                recall = 0.0
            
            # Instance F1 Score: Harmonic mean of precision and recall
            if precision + recall > 0:
                f1 = 2 * (precision * recall) / (precision + recall)
            else:
                f1 = 0.0
            
            # Average Precision at this threshold (same as precision for single threshold)
            ap = precision
            ap_values.append(ap)
            
            # Store per-threshold metrics
            threshold_str = f"@{alpha:.2f}".replace('.', '_')
            metrics[f'instance_precision{threshold_str}'] = precision
            metrics[f'instance_recall{threshold_str}'] = recall
            metrics[f'instance_f1{threshold_str}'] = f1
            metrics[f'ap{threshold_str}'] = ap
            
            # Also store mean IoU for matched instances at this threshold
            if len(stats['matched_ious']) > 0:
                metrics[f'mean_iou{threshold_str}'] = np.mean(stats['matched_ious'])
            else:
                metrics[f'mean_iou{threshold_str}'] = 0.0
        
        # 4. Mean Average Precision (mAP) - average AP across all thresholds
        if len(ap_values) > 0:
            metrics['mean_average_precision'] = np.mean(ap_values)
        else:
            metrics['mean_average_precision'] = 0.0
        
        # Additional summary statistics
        metrics['total_predictions'] = self.total_predictions
        metrics['total_ground_truths'] = self.total_ground_truths
        
        return metrics
    
    def __str__(self) -> str:
        """
        Get a formatted string summary of the metrics.
        
        Returns:
            Formatted string with all metrics
        """
        results = self.compute()
        
        lines = ["Instance Segmentation Metrics Summary:"]
        lines.append("=" * 60)
        
        # Overall metrics
        lines.append(f"Jaccard Index (IoU):        {results['jaccard_index']:.4f}")
        lines.append(f"Difference in Count (DiC):  {results['difference_in_count']:.2f}")
        lines.append(f"Mean Average Precision:     {results['mean_average_precision']:.4f}")
        lines.append("")
        
        # Per-threshold metrics
        for alpha in self.overlap_thresholds:
            threshold_str = f"@{alpha:.2f}".replace('.', '_')
            lines.append(f"Metrics at α = {alpha:.2f}:")
            lines.append(f"  Instance Precision:  {results[f'instance_precision{threshold_str}']:.4f}")
            lines.append(f"  Instance Recall:     {results[f'instance_recall{threshold_str}']:.4f}")
            lines.append(f"  Instance F1:         {results[f'instance_f1{threshold_str}']:.4f}")
            lines.append(f"  Average Precision:   {results[f'ap{threshold_str}']:.4f}")
            lines.append(f"  Mean IoU (matched):  {results[f'mean_iou{threshold_str}']:.4f}")
            lines.append("")
        
        lines.append(f"Total Predictions:     {results['total_predictions']}")
        lines.append(f"Total Ground Truths:   {results['total_ground_truths']}")
        
        results = self.compute()
        print("\nDictionary format:")
        for key, value in sorted(results.items()):
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")

        return "\n".join(lines)

# class emrMetrics:
#     """
#     Computes instance segmentation metrics for 2D or 2.5D biomedical images.

#     Metrics:
#     - IoU / Jaccard Index
#     - mean Average Precision (mAP) at different IoU thresholds
#     - Difference in Count (DiC)
#     - Instance Precision (InsPr)
#     - Instance Recall (InsRe)
#     - Instance F1 Score (InsF1)
#     """

#     def __init__(self, iou_thresholds=None):
#         """
#         Args:
#             iou_thresholds (list, optional): List of IoU thresholds for mAP.
#                                              Default: [0.5, 0.75]
#         """
#         if iou_thresholds is None:
#             iou_thresholds = [0.5, 0.75]
#         self.iou_thresholds = iou_thresholds
#         self.reset()

#     def reset(self):
#         """Resets all accumulators."""
#         self.all_iou = []
#         self.tp, self.fp, self.fn = 0, 0, 0
#         self.total_pred, self.total_gt = 0, 0
#         self.ap_per_thresh = {thr: [] for thr in self.iou_thresholds}

#     @staticmethod
#     def compute_iou(mask1, mask2):
#         """Computes IoU between two binary masks."""
#         intersection = torch.logical_and(mask1, mask2).float().sum()
#         union = torch.logical_or(mask1, mask2).float().sum()
#         if union == 0:
#             return 0.0
#         return (intersection / union).item()

#     def update(self, preds, targets):
#         """
#         Updates metrics based on one batch of predictions and targets.
        
#         Args:
#             preds (list[dict]): Model predictions. Each dict contains 'masks' [N_pred, 1, H, W].
#             targets (list[dict]): Ground truth. Each dict contains 'masks' [N_gt, H, W].
#         """
#         for pred, target in zip(preds, targets):
#             if len(pred["masks"]) == 0 and len(target["masks"]) == 0:
#                 continue

#             pred_masks = (pred["masks"] > 0.5).squeeze(1)  # [N_pred, H, W]
#             gt_masks = target["masks"]                     # [N_gt, H, W]

#             num_pred = len(pred_masks)
#             num_gt = len(gt_masks)

#             self.total_pred += num_pred
#             self.total_gt += num_gt

#             if num_gt == 0:
#                 self.fp += num_pred
#                 continue

#             # Compute IoU matrix [N_pred, N_gt]
#             iou_matrix = torch.zeros((num_pred, num_gt))
#             for i, pm in enumerate(pred_masks):
#                 for j, gm in enumerate(gt_masks):
#                     iou_matrix[i, j] = self.compute_iou(pm, gm)

#             # Only include IoU of true positive matches (best IoU per GT or per Pred)
#             max_iou_per_pred = iou_matrix.max(dim=1).values
#             self.all_iou.extend(max_iou_per_pred[max_iou_per_pred > 0].tolist())


#             # For each IoU threshold, compute TP/FP/FN
#             for thr in self.iou_thresholds:
#                 matches = (iou_matrix >= thr).float()
#                 tp = (max_iou_per_pred >= thr).sum().item()
#                 fp = len(max_iou_per_pred) - tp
#                 fn = len(gt_masks) - tp

#                 self.tp += tp
#                 self.fp += fp
#                 self.fn += fn

#                 precision = tp / (tp + fp + 1e-6)
#                 recall = tp / (tp + fn + 1e-6)
#                 self.ap_per_thresh[thr].append(precision)

#     def compute(self):
#         """Computes aggregated metrics across all batches."""
#         mean_iou = torch.tensor(self.all_iou).mean().item() if len(self.all_iou) else 0.0
#         mean_ap = {thr: torch.tensor(v).mean().item() if v else 0.0
#                    for thr, v in self.ap_per_thresh.items()}
#         mAP = sum(mean_ap.values()) / len(mean_ap) if len(mean_ap) > 0 else 0.0

#         # Instance-level metrics
#         InsPr = self.tp / (self.tp + self.fp + 1e-6)
#         InsRe = self.tp / (self.tp + self.fn + 1e-6)
#         InsF1 = 2 * InsPr * InsRe / (InsPr + InsRe + 1e-6)
#         DiC = self.total_pred - self.total_gt

#         return {
#             "Mean IoU": mean_iou,
#             "mAP": mAP,
#             "DiC": DiC,
#             "Instance Precision": InsPr,
#             "Instance Recall": InsRe,
#             "Instance F1": InsF1,
#         }

#     def __str__(self):
#         metrics = self.compute()
#         msg = "\n".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
#         return f"EMR Metrics:\n{msg}"



