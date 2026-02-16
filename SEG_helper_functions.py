import numpy as np

from skimage.measure import label
import os, shutil
import tifffile
from collections import defaultdict
import warnings



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
    # pred: (N, 1, H, W)
    binary = np.any(pred, axis=(0, 1))
    return binary.astype(np.uint8)



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

        # ---- Predictions ----
        binary_slices = []
        for _, pred_path in sorted(slices_by_T[T]):
            b = load_binary_pred_slice(pred_path)
            binary_slices.append(b)

        binary_3d = np.stack(binary_slices, axis=0)
        inst_3d = label(binary_3d, connectivity=1)

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
                gt_slices.append(np.any(data["masks"], axis=0))

        gt_3d = label(np.stack(gt_slices, axis=0), connectivity=1)

        tifffile.imwrite(
            os.path.join(gt_out, f"man_seg{T}.tif"),
            gt_3d.astype(np.uint16)
        )
