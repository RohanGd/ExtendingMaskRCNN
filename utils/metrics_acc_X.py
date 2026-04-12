import os
import numpy as np
import tifffile
from scipy.optimize import linear_sum_assignment

def load_mask(path: str) -> np.ndarray:
    mask = tifffile.imread(path)
    if mask.ndim == 3 and mask.shape[0] == 1:
        mask = mask[0]
    return mask

def drop_dim_by_coloring(mask: np.ndarray) -> np.ndarray:
    """Convert N-instance masks to single labeled mask (0=background)."""
    if mask.ndim == 3:  # (num_instances, H, W)
        new_mask = np.zeros(mask.shape[1:], dtype=np.int32)
        for c in range(mask.shape[0]):
            new_mask[mask[c] != 0] = c + 1
        return new_mask
    return mask

def compute_iou_matrix(gt_mask, pred_mask):
    gt_ids = np.unique(gt_mask)
    gt_ids = gt_ids[gt_ids != 0]
    pred_ids = np.unique(pred_mask)
    pred_ids = pred_ids[pred_ids != 0]

    iou_matrix = np.zeros((len(gt_ids), len(pred_ids)))
    for i, g in enumerate(gt_ids):
        gt_binary = gt_mask == g
        for j, p in enumerate(pred_ids):
            pred_binary = pred_mask == p
            inter = np.logical_and(gt_binary, pred_binary).sum()
            union = np.logical_or(gt_binary, pred_binary).sum()
            iou_matrix[i, j] = inter / union if union > 0 else 0.0
    return iou_matrix, gt_ids, pred_ids

def compute_TP_FP_FN(gt_mask, pred_mask, threshold=0.5):
    iou_matrix, gt_ids, pred_ids = compute_iou_matrix(gt_mask, pred_mask)

    if len(gt_ids) == 0:
        TP = 0
        FP = len(pred_ids)
        FN = 0
        return TP, FP, FN
    if len(pred_ids) == 0:
        TP = 0
        FP = 0
        FN = len(gt_ids)
        return TP, FP, FN

    # Hungarian matching
    cost_matrix = 1 - iou_matrix
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    matched_gt = set()
    matched_pred = set()
    for i, j in zip(row_ind, col_ind):
        if iou_matrix[i, j] >= threshold:
            matched_gt.add(i)
            matched_pred.add(j)

    TP = len(matched_gt)
    FP = len(pred_ids) - len(matched_pred)
    FN = len(gt_ids) - len(matched_gt)

    return TP, FP, FN

def evaluate_2d_dataset(gt_dir, res_dir, threshold=0.5):
    gt_files = sorted([f for f in os.listdir(gt_dir) if f.endswith('.tif')])
    res_files = sorted([f for f in os.listdir(res_dir) if f.endswith('.tif')])

    results = []
    for gt_name, res_name in zip(gt_files, res_files):
        gt_mask = drop_dim_by_coloring(load_mask(os.path.join(gt_dir, gt_name)))
        pred_mask = drop_dim_by_coloring(load_mask(os.path.join(res_dir, res_name)))

        TP, FP, FN = compute_TP_FP_FN(gt_mask, pred_mask, threshold)
        results.append({"file": gt_name, "TP": TP, "FP": FP, "FN": FN})
    return results


files = [
'channelFusion_20260328_214504',
'earlyFusion_GlobalGaussian_20260328_214504',
'earlyFusion_GlobalPerFPN_20260328_215310',
'earlyFusion_Pixel_20260328_215331',
'earlyFusion_Windowed_20260328_220034',
'lateFusion_conv3d_20260328_221338',
'lateFusion_mean_20260328_222135',
'lateFusion_onlyCenter_20260328_222937',
'lateFusion_SE_20260328_223711',
"data/n1",
"data/SIM+_n3/channelFusion_20260327_090820",
"data/SIM+_n3/earlyFusion_GlobalGaussian_20260327_090820",
"data/SIM+_n3/lateFusion_onlyCenter_20260327_091016",
"data/12spheroids_Low_n5/channelFusion_20260330_163145",
"data/12spheroids_Low_n5/earlyFusion_GlobalGaussian_20260330_174953",
"data/12spheroids_Low_n5/lateFusion_onlyCenter_20260330_191518",
"data/SIM+_n7/channelFusion_20260326_183827",
"data/SIM+_n7/earlyFusion_GlobalGaussian_20260326_205246",
"data/SIM+_n7/lateFusion_onlyCenter_20260327_141812"
]

for i, file in enumerate(files):
    if i < 16:
        continue
    print(file)
    
    gt_dir = f"test_dir/{i+1}_GT/SEG"
    res_dir = f"test_dir/{i+1}_RES"

    results = evaluate_2d_dataset(gt_dir, res_dir, threshold=0.5)
    # for r in results:
    #     print(r)

    TP_total = sum(r["TP"] for r in results)
    FP_total = sum(r["FP"] for r in results)
    FN_total = sum(r["FN"] for r in results)
    Acc = TP_total / (TP_total + FP_total + FN_total)
    print(TP_total, FP_total, FN_total)
    print("Accuracy: ", Acc, " for file: ", file)