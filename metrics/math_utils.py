"""
Shared, granularity-agnostic math primitives used by both metrics/metrics_2d.py
(per-slice/per-batch) and metrics/metrics_volume.py (per-3D-volume).

Convention used throughout this module and its callers: wherever a pairwise IoU
matrix is built, rows = ground-truth instances, cols = predicted instances
(iou_matrix.shape == [n_gt, n_pred]). hungarian_match/panoptic_quality_counts
are written against that convention.
"""
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from typing import List, Tuple, Optional


# ------------------------- pairwise IoU -------------------------

def pairwise_iou_masks(masks1: torch.Tensor, masks2: torch.Tensor, chunk_size: int = 32) -> torch.Tensor:
    """
    Chunked, CPU-side pairwise IoU between two sets of binary masks (memory-safe for
    large mask counts/resolutions -- avoids allocating a full [N1,N2,H,W] broadcast
    tensor on GPU).

    masks1: [N1, H, W] or [N1, 1, H, W]
    masks2: [N2, H, W] or [N2, 1, H, W]
    Returns: [N1, N2] float32 CPU tensor.
    """
    if masks1.dim() == 4:
        masks1 = masks1.squeeze(1)
    if masks2.dim() == 4:
        masks2 = masks2.squeeze(1)

    m1 = masks1.detach().cpu() > 0.5
    m2 = masks2.detach().cpu() > 0.5

    N1 = 0 if m1.numel() == 0 else m1.shape[0]
    N2 = 0 if m2.numel() == 0 else m2.shape[0]

    if N1 == 0 or N2 == 0:
        return torch.zeros((N1, N2), dtype=torch.float32)

    area2 = m2.view(N2, -1).sum(dim=1).float()

    ious = []
    for i in range(0, N1, chunk_size):
        chunk = m1[i:i + chunk_size]
        C = chunk.shape[0]

        inter = (chunk[:, None, :, :] & m2[None, :, :, :]).view(C, N2, -1).sum(dim=2).float()
        area1_chunk = chunk.view(C, -1).sum(dim=1).float().unsqueeze(1)
        union = area1_chunk + area2.unsqueeze(0) - inter
        ious.append(inter / (union + 1e-6))

    return torch.cat(ious, dim=0)


def _pairwise_label_stats(label_vol1: np.ndarray, label_vol2: np.ndarray):
    """
    Returns (inter[n1,n2], area1[n1], area2[n2], ids1, ids2) for two integer-labeled
    arrays (0 = background), via a compact-index bincount trick so this stays O(voxels)
    rather than O(n1*n2) Python loops over potentially large 3D volumes.
    """
    v1 = np.asarray(label_vol1).ravel()
    v2 = np.asarray(label_vol2).ravel()

    ids1 = np.unique(v1)
    ids1 = ids1[ids1 != 0]
    ids2 = np.unique(v2)
    ids2 = ids2[ids2 != 0]
    n1, n2 = len(ids1), len(ids2)

    if n1 == 0 or n2 == 0:
        return (np.zeros((n1, n2), dtype=np.float64),
                np.zeros(n1, dtype=np.float64), np.zeros(n2, dtype=np.float64),
                ids1, ids2)

    remap1 = np.full(int(v1.max()) + 1, -1, dtype=np.int64)
    remap1[ids1] = np.arange(n1)
    remap2 = np.full(int(v2.max()) + 1, -1, dtype=np.int64)
    remap2[ids2] = np.arange(n2)

    idx1 = remap1[v1]
    idx2 = remap2[v2]

    area1 = np.bincount(idx1[idx1 >= 0], minlength=n1).astype(np.float64)
    area2 = np.bincount(idx2[idx2 >= 0], minlength=n2).astype(np.float64)

    mask = (idx1 >= 0) & (idx2 >= 0)
    combined = idx1[mask] * n2 + idx2[mask]
    inter = np.bincount(combined, minlength=n1 * n2).astype(np.float64).reshape(n1, n2)

    return inter, area1, area2, ids1, ids2


def pairwise_iou_labels(gt_label_vol: np.ndarray, pred_label_vol: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Pairwise IoU between two integer-labeled arrays/volumes (0 = background).
    Returns (iou[n_gt, n_pred], gt_ids, pred_ids).
    """
    inter, area_gt, area_pred, gt_ids, pred_ids = _pairwise_label_stats(gt_label_vol, pred_label_vol)
    union = area_gt[:, None] + area_pred[None, :] - inter
    iou = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
    return iou, gt_ids, pred_ids


# ------------------------- Hungarian matching -------------------------

def hungarian_match(iou_matrix, alpha: float) -> List[Tuple[int, int]]:
    """
    Optimal 1-to-1 matching via the Hungarian algorithm. Accepts a torch.Tensor or
    np.ndarray IoU matrix. Only matches with IoU >= alpha are returned.
    Row/col meaning is caller-defined (this function is symmetric in that choice).
    """
    if hasattr(iou_matrix, "detach"):
        iou_np = iou_matrix.detach().cpu().numpy()
    else:
        iou_np = np.asarray(iou_matrix)

    if iou_np.size == 0:
        return []

    cost = 1.0 - iou_np
    row_idx, col_idx = linear_sum_assignment(cost)

    matches: List[Tuple[int, int]] = []
    for r, c in zip(row_idx, col_idx):
        if iou_np[r, c] >= alpha:
            matches.append((int(r), int(c)))
    return matches


def jaccard(mask_r, mask_s) -> float:
    inter = (mask_r & mask_s).sum().item()
    union = (mask_r | mask_s).sum().item()
    return inter / union if union > 0 else 0.0


def dice_from_iou(iou: float) -> float:
    return (2.0 * iou) / (1.0 + iou + 1e-6)


# ------------------------- SEG (CTC) -------------------------
# https://public.celltrackingchallenge.net/documents/SEG.pdf
# J(S,R) = |R∩S|/|R∪S|; R and S match iff |R∩S| > 0.5*|R| (at most one S can match a
# given R -- two disjoint predicted instances can't both cover >50% of R); unmatched
# R scores 0; SEG = mean of J over all reference (GT) objects.

def seg_score_single_frame(gt_masks, pred_masks) -> float:
    """2D-slice SEG (majority-overlap rule), gt_masks/pred_masks: lists/tensors of
    per-instance boolean masks for a single image."""
    if len(gt_masks) == 0:
        return 0.0

    gt_masks = [m.bool() for m in gt_masks]
    pred_masks = [(m > 0.5).bool() for m in pred_masks]

    scores = []
    for R in gt_masks:
        R_area = R.sum().item()
        best_S = None
        best_inter = 0
        for S in pred_masks:
            inter = (R & S).sum().item()
            if inter > 0.5 * R_area and inter > best_inter:
                best_inter = inter
                best_S = S
        scores.append(0.0 if best_S is None else jaccard(R, best_S))

    return sum(scores) / len(scores)


def seg_score_volume(gt_label_vol: np.ndarray, pred_label_vol: np.ndarray) -> float:
    """True CTC SEG on a full (Z,H,W) (or 2D) integer-labeled volume pair."""
    inter, area_gt, area_pred, gt_ids, pred_ids = _pairwise_label_stats(gt_label_vol, pred_label_vol)
    n_gt = len(gt_ids)
    if n_gt == 0 or len(pred_ids) == 0:
        return 0.0

    scores = []
    for i in range(n_gt):
        row = inter[i]
        R_area = area_gt[i]
        candidate = row > 0.5 * R_area
        if not np.any(candidate):
            scores.append(0.0)
            continue
        j = int(np.argmax(np.where(candidate, row, -1)))
        union = R_area + area_pred[j] - row[j]
        scores.append(float(row[j] / union) if union > 0 else 0.0)

    return float(np.mean(scores))


# ------------------------- Aggregated Jaccard Index (Kumar et al. 2017) -------------------------
# For each GT object, take its best-IoU-matching predicted object (argmax IoU, not the
# SEG majority rule); accumulate C += |G∩P_best|, U += |G∪P_best|. Predicted objects
# never chosen as any GT's best match have their full area added to U. AJI = C/U.
# Returned as raw (C, U) since AJI is a global ratio-of-sums that must be accumulated
# across the whole eval set before dividing once, not averaged per-image/per-volume.

def aji_intersection_union_masks(gt_masks: torch.Tensor, pred_masks: torch.Tensor) -> Tuple[float, float]:
    gt = gt_masks.squeeze(1) if gt_masks.dim() == 4 else gt_masks
    pred = pred_masks.squeeze(1) if pred_masks.dim() == 4 else pred_masks
    gt = gt.detach().cpu().bool()
    pred = pred.detach().cpu().bool()

    n_gt, n_pred = gt.shape[0], pred.shape[0]
    if n_gt == 0:
        return 0.0, float(pred.sum().item()) if n_pred > 0 else 0.0
    if n_pred == 0:
        return 0.0, float(gt.sum().item())

    iou_matrix = pairwise_iou_masks(gt, pred)  # [n_gt, n_pred]
    area_gt = gt.view(n_gt, -1).sum(dim=1).float()
    area_pred = pred.view(n_pred, -1).sum(dim=1).float()

    best_j = iou_matrix.argmax(dim=1)
    used_pred = set()

    C, U = 0.0, 0.0
    for i in range(n_gt):
        j = int(best_j[i].item())
        used_pred.add(j)
        inter_ij = (gt[i] & pred[j]).sum().item()
        C += inter_ij
        U += area_gt[i].item() + area_pred[j].item() - inter_ij

    for j in set(range(n_pred)) - used_pred:
        U += area_pred[j].item()

    return float(C), float(U)


def aji_intersection_union_labels(gt_label_vol: np.ndarray, pred_label_vol: np.ndarray) -> Tuple[float, float]:
    inter, area_gt, area_pred, gt_ids, pred_ids = _pairwise_label_stats(gt_label_vol, pred_label_vol)
    n_gt, n_pred = len(gt_ids), len(pred_ids)

    if n_gt == 0:
        return 0.0, float(area_pred.sum()) if n_pred > 0 else 0.0
    if n_pred == 0:
        return 0.0, float(area_gt.sum())

    union = area_gt[:, None] + area_pred[None, :] - inter
    iou = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)

    best_j = np.argmax(iou, axis=1)
    rows = np.arange(n_gt)
    C = float(inter[rows, best_j].sum())
    U = float(union[rows, best_j].sum())

    unmatched = sorted(set(range(n_pred)) - set(best_j.tolist()))
    if unmatched:
        U += float(area_pred[unmatched].sum())

    return C, U


def aggregated_jaccard_index(C: float, U: float) -> float:
    return C / U if U > 0 else 0.0


# ------------------------- Panoptic Quality (Kirillov et al. 2019) -------------------------
# Match GT/pred via IoU>=threshold (unique/unambiguous above 0.5, matches PQ's original
# definition); TP=matches, FP=unmatched pred, FN=unmatched GT; SQ=mean matched IoU;
# RQ=TP/(TP+0.5FP+0.5FN); PQ=SQ*RQ. Returned as raw counts (tp,fp,fn,iou_sum), since
# SQ/RQ must be computed from sums accumulated across the whole eval set, not averaged
# per-image/per-volume PQ values.

def panoptic_quality_counts(iou_matrix, threshold: float = 0.5) -> Tuple[int, int, int, float]:
    """iou_matrix: [n_gt, n_pred] (torch.Tensor or np.ndarray). Returns (tp, fp, fn, iou_sum)."""
    n_gt, n_pred = iou_matrix.shape
    matches = hungarian_match(iou_matrix, threshold)
    tp = len(matches)
    fp = n_pred - tp
    fn = n_gt - tp

    iou_np = iou_matrix.detach().cpu().numpy() if hasattr(iou_matrix, "detach") else np.asarray(iou_matrix)
    iou_sum = float(sum(iou_np[r, c] for r, c in matches))
    return tp, fp, fn, iou_sum


def sq_rq_pq(tp: int, fp: int, fn: int, iou_sum: float) -> Tuple[float, float, float]:
    sq = iou_sum / tp if tp > 0 else 0.0
    rq = tp / (tp + 0.5 * fp + 0.5 * fn) if (tp + fp + fn) > 0 else 0.0
    return sq, rq, sq * rq


# ------------------------- COCO-style Average Precision -------------------------

def coco_style_ap(records: List[Tuple[float, bool]], total_gt_count: int) -> float:
    """
    records: list of (confidence_score, is_true_positive) for every prediction in the
    eval set at a fixed IoU threshold. total_gt_count: total GT instance count in the
    eval set (recall denominator). Single-class AP (this codebase is single-class
    nuclei/cell segmentation) via COCO's 101-point recall-interpolated integration.
    """
    if total_gt_count == 0 or len(records) == 0:
        return 0.0

    records = sorted(records, key=lambda r: r[0], reverse=True)
    tp_cum, fp_cum = 0, 0
    precisions, recalls = [], []
    for _, is_tp in records:
        if is_tp:
            tp_cum += 1
        else:
            fp_cum += 1
        precisions.append(tp_cum / (tp_cum + fp_cum))
        recalls.append(tp_cum / total_gt_count)

    precisions = np.array(precisions)
    recalls = np.array(recalls)

    # monotonic non-increasing precision envelope
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])

    interpolated = []
    for rt in np.linspace(0.0, 1.0, 101):
        idxs = np.where(recalls >= rt)[0]
        interpolated.append(precisions[idxs].max() if len(idxs) > 0 else 0.0)

    return float(np.mean(interpolated))
