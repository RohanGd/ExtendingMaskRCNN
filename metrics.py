import sys
import os
import tifffile
import numpy as np

def load_volume(path: str) -> np.ndarray:
    """Load a 3D multipage TIFF volume"""
    return tifffile.imread(path)

def semantic_3d_iou(gt_vol: np.ndarray, res_vol: np.ndarray) -> float:
    """Compute semantic 3D IoU between two volumes"""
    gt = gt_vol > 0
    pred = res_vol > 0

    if gt.shape != pred.shape:
        raise ValueError(f"Shape mismatch: GT {gt.shape} vs Pred {pred.shape}")

    intersection = np.logical_and(gt, pred).sum()
    union = np.logical_or(gt, pred).sum()

    if union == 0:
        return 1.0  # both empty
    return intersection / union

def main():
    if len(sys.argv) < 2:
        print("Usage: python compute_3d_iou.py <experiment_dir>")
        sys.exit(1)

    exp_dir = sys.argv[1]
    gt_dir = os.path.join(exp_dir, "01_GT/SEG")
    res_dir = os.path.join(exp_dir, "01_RES")

    gt_paths = sorted([os.path.join(gt_dir, f) for f in os.listdir(gt_dir) if f.endswith(".tif")])
    res_paths = sorted([os.path.join(res_dir, f) for f in os.listdir(res_dir) if f.endswith(".tif")])

    assert len(gt_paths) == len(res_paths), "Mismatch in number of GT vs RES volumes"

    ious = []

    for gt_path, res_path in zip(gt_paths, res_paths):
        gt_vol = load_volume(gt_path)
        res_vol = load_volume(res_path)

        iou = semantic_3d_iou(gt_vol, res_vol)
        ious.append(iou)
        print(f"{os.path.basename(gt_path)} IoU: {iou:.4f}")

    mean_iou = np.mean(ious)
    print(f"\nMean 3D semantic IoU across all volumes: {mean_iou:.4f}")

if __name__ == "__main__":
    main()

