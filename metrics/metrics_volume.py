# metrics/metrics_volume.py
"""
Per-3D-volume instance segmentation metrics, operating on the linked 3D
instance-labeled volume pairs SEG_helper_functions.make_files_for_SEG produces:
  exp_dir/01_GT/SEG/man_seg{T}.tif   -- ground truth, instance-labeled (Z,H,W)
  exp_dir/01_RES/mask{T}.tif          -- prediction, instance-labeled (Z,H,W)
  exp_dir/01_RES/mask{T}.scores.json  -- {linked_instance_id: aggregated_score}, optional

This is where SEG is properly defined (a per-video/per-volume metric per the CTC
spec, see https://public.celltrackingchallenge.net/documents/SEG.pdf) -- unlike the
2D-slice approximation in metrics/metrics_2d.py.

Absorbs the former utils/metrics2diou.py (mean_semantic_3d_iou) and
utils/metrics_acc_X.py (acc_at_threshold) functionality.
"""
import os
import glob
import json
import numpy as np
import tifffile
from typing import Dict, List, Optional

from metrics import math_utils


class emrMetricsVolume:
    def __init__(self, exp_dir: str, overlap_thresholds: List[float] = [0.5, 0.75, 0.9],
                 acc_thresholds: List[float] = [0.3, 0.5]):
        self.exp_dir = exp_dir
        self.overlap_thresholds = sorted(overlap_thresholds)
        self.acc_thresholds = sorted(acc_thresholds)
        self.gt_dir = os.path.join(exp_dir, "01_GT", "SEG")
        self.res_dir = os.path.join(exp_dir, "01_RES")

    # ------------------------- volume pair discovery -------------------------
    def _iter_volume_pairs(self):
        gt_paths = sorted(glob.glob(os.path.join(self.gt_dir, "man_seg*.tif")))
        if not gt_paths:
            raise FileNotFoundError(f"No man_seg*.tif files found in {self.gt_dir}")

        for gt_path in gt_paths:
            T = os.path.basename(gt_path)[len("man_seg"):-len(".tif")]
            res_path = os.path.join(self.res_dir, f"mask{T}.tif")
            if not os.path.exists(res_path):
                raise FileNotFoundError(
                    f"Missing prediction volume for T={T}: expected {res_path} "
                    f"(matching GT volume {gt_path})"
                )
            scores_path = os.path.join(self.res_dir, f"mask{T}.scores.json")
            scores = None
            if os.path.exists(scores_path):
                with open(scores_path, encoding="utf-8") as f:
                    scores = {int(k): v for k, v in json.load(f).items()}

            gt_vol = tifffile.imread(gt_path)
            res_vol = tifffile.imread(res_path)
            yield T, gt_vol, res_vol, scores

    # ------------------------- semantic 3D IoU -------------------------
    def mean_semantic_3d_iou(self) -> float:
        ious = []
        for _, gt_vol, res_vol, _ in self._iter_volume_pairs():
            gt = gt_vol > 0
            pred = res_vol > 0
            if gt.shape != pred.shape:
                raise ValueError(f"Shape mismatch: GT {gt.shape} vs Pred {pred.shape}")
            intersection = np.logical_and(gt, pred).sum()
            union = np.logical_or(gt, pred).sum()
            ious.append(1.0 if union == 0 else intersection / union)
        return float(np.mean(ious)) if ious else 0.0

    # ------------------------- SEG -------------------------
    def seg_score(self) -> Dict[str, float]:
        """Cheap, standalone: only what training_loop.py's per-epoch validation needs."""
        scores = [math_utils.seg_score_volume(gt_vol, res_vol)
                  for _, gt_vol, res_vol, _ in self._iter_volume_pairs()]
        if not scores:
            return {"mean": 0.0, "median": 0.0}
        return {"mean": float(np.mean(scores)), "median": float(np.median(scores))}

    # ------------------------- AJI -------------------------
    def aji(self) -> float:
        C, U = 0.0, 0.0
        for _, gt_vol, res_vol, _ in self._iter_volume_pairs():
            c, u = math_utils.aji_intersection_union_labels(gt_vol, res_vol)
            C += c
            U += u
        return math_utils.aggregated_jaccard_index(C, U)

    # ------------------------- PQ / Acc@X / AP (share per-threshold Hungarian counts) -------------------------
    def _per_threshold_counts_and_ap_records(self):
        """
        One pass over all volume pairs, computing (for each overlap threshold) the
        accumulated (tp, fp, fn, iou_sum) PQ counts (Acc@X reuses these directly) and
        the (score, is_tp) AP records, plus the total 3D GT instance count.
        """
        counts = {alpha: [0, 0, 0, 0.0] for alpha in self.overlap_thresholds}  # tp,fp,fn,iou_sum
        ap_records = {alpha: [] for alpha in self.overlap_thresholds}
        total_gt = 0
        num_volumes = 0

        for T, gt_vol, res_vol, scores in self._iter_volume_pairs():
            num_volumes += 1
            iou_matrix, gt_ids, pred_ids = math_utils.pairwise_iou_labels(gt_vol, res_vol)
            total_gt += len(gt_ids)

            for alpha in self.overlap_thresholds:
                tp, fp, fn, iou_sum = math_utils.panoptic_quality_counts(iou_matrix, alpha)
                c = counts[alpha]
                c[0] += tp
                c[1] += fp
                c[2] += fn
                c[3] += iou_sum

                if scores is not None:
                    matches = math_utils.hungarian_match(iou_matrix, alpha)
                    matched_pred_idx = {p for _, p in matches}
                    for p_idx, pred_id in enumerate(pred_ids):
                        score = scores.get(int(pred_id))
                        if score is None:
                            continue
                        ap_records[alpha].append((score, p_idx in matched_pred_idx))

        return counts, ap_records, total_gt, num_volumes

    def panoptic_quality(self) -> Dict[float, Dict[str, float]]:
        counts, _, _, _ = self._per_threshold_counts_and_ap_records()
        out = {}
        for alpha, (tp, fp, fn, iou_sum) in counts.items():
            sq, rq, pq = math_utils.sq_rq_pq(tp, fp, fn, iou_sum)
            out[alpha] = {"tp": tp, "fp": fp, "fn": fn, "sq": sq, "rq": rq, "pq": pq}
        return out

    def acc_at_threshold(self, threshold: float, counts=None) -> float:
        if counts is None:
            counts, _, _, _ = self._per_threshold_counts_and_ap_records()
        if threshold not in counts:
            # threshold not part of overlap_thresholds -- compute standalone
            tp_total = fp_total = fn_total = 0
            for _, gt_vol, res_vol, _ in self._iter_volume_pairs():
                iou_matrix, _, _ = math_utils.pairwise_iou_labels(gt_vol, res_vol)
                tp, fp, fn, _ = math_utils.panoptic_quality_counts(iou_matrix, threshold)
                tp_total += tp
                fp_total += fp
                fn_total += fn
        else:
            tp_total, fp_total, fn_total, _ = counts[threshold]
        denom = tp_total + fp_total + fn_total
        return tp_total / denom if denom > 0 else 0.0

    def average_precision(self) -> Dict[float, float]:
        _, ap_records, total_gt, _ = self._per_threshold_counts_and_ap_records()
        return {alpha: math_utils.coco_style_ap(records, total_gt)
                for alpha, records in ap_records.items()}

    # ------------------------- compute all -------------------------
    def compute(self) -> Dict[str, float]:
        metrics: Dict[str, float] = {}

        seg = self.seg_score()
        metrics["volume_SEG_score"] = seg["mean"]
        metrics["volume_SEG_score_median"] = seg["median"]
        metrics["volume_mean_semantic_3d_iou"] = self.mean_semantic_3d_iou()
        metrics["volume_aji"] = self.aji()

        counts, ap_records, total_gt, num_volumes = self._per_threshold_counts_and_ap_records()
        metrics["total_gt_instances_3d"] = total_gt

        ap_values = []
        total_pred = 0
        for alpha in self.overlap_thresholds:
            tp, fp, fn, iou_sum = counts[alpha]
            sq, rq, pq = math_utils.sq_rq_pq(tp, fp, fn, iou_sum)
            ap = math_utils.coco_style_ap(ap_records[alpha], total_gt)
            ap_values.append(ap)
            total_pred = tp + fp  # same across thresholds up to matching, keep last

            key = f"@{alpha:.2f}".replace('.', '_')
            metrics[f"volume_sq{key}"] = sq
            metrics[f"volume_rq{key}"] = rq
            metrics[f"volume_pq{key}"] = pq
            metrics[f"volume_ap{key}"] = ap

        metrics["volume_mean_average_precision"] = float(np.mean(ap_values)) if ap_values else 0.0
        metrics["total_pred_instances_3d"] = total_pred

        for threshold in self.acc_thresholds:
            key = f"@{threshold:.2f}".replace('.', '_')
            metrics[f"volume_acc{key}"] = self.acc_at_threshold(threshold, counts=counts if threshold in counts else None)

        metrics["num_volumes"] = num_volumes

        return metrics

    # ------------------------- Utilities -------------------------
    def save(self, path: str) -> None:
        with open(path, 'w', encoding='utf-8') as f:
            f.write(str(self))

    def __str__(self) -> str:
        results = self.compute()
        lines: List[str] = ["Volume-Level (3D) Instance Segmentation Metrics Summary:\n", "=" * 60]

        lines.append(f"Number of volumes:          {results['num_volumes']}")
        lines.append(f"SEG_score mean:              {results['volume_SEG_score']:.4f}")
        lines.append(f"SEG_score median:            {results['volume_SEG_score_median']:.4f}")
        lines.append(f"Mean Semantic 3D IoU:        {results['volume_mean_semantic_3d_iou']:.4f}")
        lines.append(f"AJI (global):                {results['volume_aji']:.4f}")
        lines.append(f"Mean Average Precision:      {results['volume_mean_average_precision']:.4f}\n")

        for alpha in self.overlap_thresholds:
            key = f"@{alpha:.2f}".replace('.', '_')
            lines.append(f"Metrics at @ = {alpha:.2f}:")
            lines.append(f"  SQ / RQ / PQ:  {results[f'volume_sq{key}']:.4f} / {results[f'volume_rq{key}']:.4f} / {results[f'volume_pq{key}']:.4f}")
            lines.append(f"  AP:            {results[f'volume_ap{key}']:.4f}\n")

        for threshold in self.acc_thresholds:
            key = f"@{threshold:.2f}".replace('.', '_')
            lines.append(f"Acc@{threshold:.2f}: {results[f'volume_acc{key}']:.4f}")

        lines.append(f"\nTotal GT instances (3D):    {results['total_gt_instances_3d']}")
        lines.append(f"Total Pred instances (3D):  {results['total_pred_instances_3d']}")

        return "\n".join(lines)
