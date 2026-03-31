import sys
import os
import tifffile
import numpy as np

data_dir = sys.argv[1]
dir_id = sys.argv[2]

gt_dir = os.path.join(data_dir, f"{dir_id}_GT/SEG")
res_dir = os.path.join(data_dir, f"{dir_id}_RES")

gts = sorted([x for x in os.listdir(gt_dir) if x.endswith('.tif')])
ress = sorted([x for x in os.listdir(res_dir) if x.endswith('.tif')])

def iou(gt, res):
    gt = gt.astype(bool)
    res = res.astype(bool)

    union = np.logical_or(gt, res).sum()
    if union == 0:
        return 1.0  # both empty → perfect match

    return np.logical_and(gt, res).sum() / union


ious = []
for gt_name, res_name in zip(gts, ress):
    gt = tifffile.imread(os.path.join(gt_dir, gt_name))
    res = tifffile.imread(os.path.join(res_dir, res_name))

    ious.append(iou(gt, res))
    # print(gt_name, res_name)

print(sum(ious) / len(ious))

