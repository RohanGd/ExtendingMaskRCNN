import numpy as np

from skimage.measure import label
import os, shutil
import tifffile
from collections import defaultdict
import warnings
from scipy.optimize import linear_sum_assignment
import torchvision, torch


def save_preds(preds, save_dir):
    """
    Saves the predicted mask from a model into the specified save dir as a .tif file.
    
    :param preds: preds["masks"] (num_preds, 1, H, W)
    :param save_dir: path to save, generally exp_dir/pred_masks
    """
    warnings.filterwarnings("ignore", message=".*writing zero-size array to nonconformant TIFF")
    num_files = len(os.listdir(save_dir))
    for i, pred in enumerate(preds): # Batch size
        pred_mask = pred["masks"] # torch.tensor
        pred_mask = (pred_mask > 0.5).bool()
        # this is of shape: [num_instances, 1, H, W]
        save_name = save_dir + f"/{str(num_files + i).zfill(5)}.tif"
        tifffile.imwrite(save_name, pred_mask.detach().cpu().numpy().astype(np.uint16))



def open_binary_slice(file_path: str):
    if file_path.endswith(".npz"):
        data = np.load(file_path, allow_pickle=True)
        data = data["masks"]  # (N, H, W)
        return np.any(data, axis=0).astype(np.uint16)

    elif file_path.endswith(".tif"):
        data = tifffile.imread(file_path)
        return np.any(data, axis=0).astype(np.uint16)
    
def load_binary_pred_slice(pred_path):
    pred = tifffile.imread(pred_path)
    if pred.ndim == 4:
        pred = pred[:, 0]
    # pred: (N, 1, H, W)
    return pred.astype(np.uint8)

def binary_OR(array):
    return np.any(array, axis=(0))


def binary_slices_to_instance_volume(binary_slices):
    """
    binary_slices: list of (H, W) arrays, ordered by Z
    returns: (Z, H, W) labeled volume
    """
    binary_3d = np.stack(binary_slices, axis=0)
    instance_3d = label(binary_3d, connectivity=3)
    return instance_3d.astype(np.uint16)

"""
This is missing a save_preds function as in the orignal file. However, its implementation was a little weird. 
So my test_dataloader has shuffle = False, so in testing_loop.py it just saves each prediction as 00000.tif, 00001.tif and as you see it also uses the same recolor function. Then later in make_files_for_SEG I was renaming them as per the file names in the dataaset/-/test/masks, since shuffle =False, this was working. How would I incorporate this, in your new code.

"""

def save_renamed_preds(target_masks_file_paths, pred_masks_dir, output_dir):
    """
    Saves prediction masks into a new directory with names matching dataset/test/masks.

    Transforms naming from:
        pred_masks/: 01.tif, 02.tif, 03.tif, ...
    to:
        output_dir/: 0000_000.tif, 0000_001.tif, 0000_002.tif, ...

    Assumes test dataloader has shuffle=False.

    :param target_masks_file_paths: list of files in dataset/test/masks
    :param pred_masks_dir: directory containing predicted masks
    :param output_dir: new directory to save renamed predictions
    """
    os.makedirs(output_dir, exist_ok=True)

    pred_masks_file_paths = sorted(os.listdir(pred_masks_dir))
    target_masks_file_paths = sorted(target_masks_file_paths)

    assert len(target_masks_file_paths) == len(pred_masks_file_paths)

    for i in range(len(pred_masks_file_paths)):
        target_name = target_masks_file_paths[i].replace(".npz", ".tif")
        src_path = os.path.join(pred_masks_dir, pred_masks_file_paths[i])
        dst_path = os.path.join(output_dir, target_name)
        shutil.copy(src_path, dst_path)



def compute_iou_matrix(masks1, masks2):
    n1, n2 = len(masks1), len(masks2)
    iou = np.zeros((n1, n2), dtype=np.float32)

    for i in range(n1):
        m1 = masks1[i]
        for j in range(n2):
            m2 = masks2[j]

            inter = np.logical_and(m1, m2).sum()
            union = np.logical_or(m1, m2).sum()

            if union > 0:
                iou[i, j] = inter / union

    return iou


def hungarian_matching_across(slice_paths, iou_threshold=0.5):
    slices = [load_binary_pred_slice(p) for p in slice_paths]

    Z = len(slices)
    H, W = slices[0].shape[1:]

    volume = np.zeros((Z, H, W), dtype=np.int32)

    global_id = 1

    # --- initialize first slice ---
    prev_masks = slices[0]
    prev_ids = {}

    for i, m in enumerate(prev_masks):
        prev_ids[i] = global_id
        volume[0][m.astype(bool)] = global_id
        global_id += 1

    # --- process remaining slices ---
    for z in range(1, Z):
        curr_masks = slices[z]

        if len(prev_masks) == 0:
            # all new
            prev_ids = {}
            for j, m in enumerate(curr_masks):
                prev_ids[j] = global_id
                volume[z][m.astype(bool)] = global_id
                global_id += 1
            prev_masks = curr_masks
            continue

        if len(curr_masks) == 0:
            prev_masks = curr_masks
            prev_ids = {}
            continue

        iou = compute_iou_matrix(prev_masks, curr_masks)

        cost = 1 - iou
        row_ind, col_ind = linear_sum_assignment(cost)

        curr_ids = {}

        # --- matched pairs ---
        for i, j in zip(row_ind, col_ind):
            if iou[i, j] >= iou_threshold:
                curr_ids[j] = prev_ids[i]

        # --- unmatched → new IDs ---
        for j in range(len(curr_masks)):
            if j not in curr_ids:
                curr_ids[j] = global_id
                global_id += 1

        # --- write to volume ---
        for j, m in enumerate(curr_masks):
            volume[z][m.astype(bool)] = curr_ids[j]

        # update
        prev_masks = curr_masks
        prev_ids = curr_ids

    return volume

def make_files_for_SEG(exp_dir, target_masks_dir, pred_masks_dir):
    """
    Creates:
      exp_dir/01_GT/SEG/man_segT.tif
      exp_dir/01_RES/maskT.tif
    """


    target_masks_file_paths = sorted(os.listdir(target_masks_dir))
    new_output_dir = os.path.join(exp_dir, "renamed_preds")
    save_renamed_preds(target_masks_file_paths, pred_masks_dir, output_dir=new_output_dir)
    pred_masks_dir = new_output_dir

    gt_out = os.path.join(exp_dir, "01_GT", "SEG")
    res_out = os.path.join(exp_dir, "01_RES")
    os.makedirs(gt_out, exist_ok=True)
    os.makedirs(res_out, exist_ok=True)


    # -------------------------
    # Step 1: ordered GT slices
    # -------------------------
    gt_files = sorted([
        f for f in os.listdir(target_masks_dir)
        if f.endswith(".npz")
    ])

    # -------------------------
    # Step 2: ordered preds
    # -------------------------
    pred_files = sorted([
        f for f in os.listdir(pred_masks_dir)
        if f.endswith(".tif")
    ])

    assert len(gt_files) == len(pred_files), "GT / pred count mismatch"

    # -------------------------
    # Step 3: map index → (T, Z)
    # -------------------------
    slices_by_T = defaultdict(list)

    for idx, gt_name in enumerate(gt_files):
        T, Z = gt_name.replace(".npz", "").split("_")
        pred_path = os.path.join(pred_masks_dir, pred_files[idx])
        slices_by_T[T].append((int(Z), pred_path))

    # -------------------------
    # Step 4: per-T processing
    # -------------------------
    for T in sorted(slices_by_T.keys()):

        # hungarian matching
        all_slice_paths = sorted(slices_by_T[T])
        all_slice_paths = [i[1] for i in all_slice_paths]
        inst_3d = hungarian_matching_across(all_slice_paths)
        # ---- Predictions ----
        # binary_slices = []
        # for _, pred_path in sorted(slices_by_T[T]):
        #     b = load_binary_pred_slice(pred_path)
        #     b = binary_OR(b)
        #     binary_slices.append(b)
        #     print(b.shape)

        # binary_3d = np.stack(binary_slices, axis=0)
        # inst_3d = label(binary_3d, connectivity=1)

        tifffile.imwrite(
            os.path.join(res_out, f"mask{T}.tif"),
            inst_3d.astype(np.uint16)
        )

        # ---- GT (copy or reconstruct) ----
        # safest option: reuse original GT volumes if available
        # otherwise reconstruct binary → CC same as preds
        gt_slices = []
        for gt_name in gt_files:
            if gt_name.startswith(T + "_"):
                gt_path = os.path.join(target_masks_dir, gt_name)
                data = np.load(gt_path, allow_pickle=True)
                gt_slices.append(data["orignal_mask"])

        # gt_3d = label(np.stack(gt_slices, axis=0), connectivity=1)
        gt_3d = np.stack(gt_slices)

        # gt_3d = torch.from_numpy(gt_3d)
        # resize = torchvision.transforms.Resize(
        #     size=(256, 256),
        # )
        # gt_3d:torch.Tensor = resize(gt_3d)
        # gt_3d = gt_3d.detach().cpu().numpy()
        tifffile.imwrite(
            os.path.join(gt_out, f"man_seg{T}.tif"),
            gt_3d.astype(np.uint16)
        )
