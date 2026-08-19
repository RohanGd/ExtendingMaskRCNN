import numpy as np
import json

from skimage.measure import label
import os, shutil
import tifffile
from collections import defaultdict
import warnings
from scipy.optimize import linear_sum_assignment
import torchvision, torch


def save_preds(preds, save_dir, save_scores=True):
    """
    Saves the predicted mask from a model into the specified save dir as a .tif file.
    When save_scores is True, also writes a sibling {idx}.scores.npy per slice
    (flat [num_instances] float32, same order as the tif's instance axis) so
    per-instance confidence can be propagated through hungarian_matching_across
    into volume-level average precision.

    :param preds: preds["masks"] (num_preds, 1, H, W), preds["scores"] (num_preds,)
    :param save_dir: path to save, generally exp_dir/pred_masks
    """
    warnings.filterwarnings("ignore", message=".*writing zero-size array to nonconformant TIFF")
    num_files = len([f for f in os.listdir(save_dir) if f.endswith(".tif")])
    for i, pred in enumerate(preds): # Batch size
        pred_mask = pred["masks"] # torch.tensor
        pred_mask = (pred_mask > 0.5).bool()
        # this is of shape: [num_instances, 1, H, W]
        save_name = save_dir + f"/{str(num_files + i).zfill(5)}.tif"
        tifffile.imwrite(save_name, pred_mask.detach().cpu().numpy().astype(np.uint16))

        if save_scores:
            scores = pred.get("scores", torch.empty(0))
            scores_name = save_dir + f"/{str(num_files + i).zfill(5)}.scores.npy"
            np.save(scores_name, scores.detach().cpu().numpy().astype(np.float32))



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

def load_slice_scores(pred_path):
    """Loads the sibling {name}.scores.npy for a {name}.tif pred slice, if present.
    Returns a flat [N] float array (N = that slice's instance count), or None."""
    scores_path = pred_path[:-len(".tif")] + ".scores.npy" if pred_path.endswith(".tif") else None
    if scores_path is None or not os.path.exists(scores_path):
        return None
    return np.load(scores_path)

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

    pred_masks_file_paths = sorted(f for f in os.listdir(pred_masks_dir) if f.endswith(".tif"))
    target_masks_file_paths = sorted(target_masks_file_paths)

    assert len(target_masks_file_paths) == len(pred_masks_file_paths)

    for i in range(len(pred_masks_file_paths)):
        target_name = target_masks_file_paths[i].replace(".npz", ".tif")
        src_path = os.path.join(pred_masks_dir, pred_masks_file_paths[i])
        dst_path = os.path.join(output_dir, target_name)
        shutil.copy(src_path, dst_path)

        scores_src_path = src_path[:-len(".tif")] + ".scores.npy"
        if os.path.exists(scores_src_path):
            scores_dst_path = os.path.join(output_dir, target_name.replace(".tif", ".scores.npy"))
            shutil.copy(scores_src_path, scores_dst_path)



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


def hungarian_matching_across(slice_paths, iou_threshold=0.5, score_agg="mean"):
    """
    Links per-slice predicted instances into persistent 3D instance ids via
    Hungarian IoU matching between consecutive Z slices.

    Also propagates each 2D detection's confidence score (from the sibling
    {name}.scores.npy files, if present) into a per-linked-3D-instance aggregated
    score, needed for volume-level average precision (2D scores alone don't survive
    the Z-linking step otherwise). score_agg: "mean" or "max".

    Returns: (volume [Z,H,W] int32 instance-labeled, linked_scores: Dict[global_id, float]).
    linked_scores is empty if no sibling .scores.npy files were found.
    """
    assert score_agg in ("mean", "max"), f"score_agg must be 'mean' or 'max', got {score_agg!r}"

    slices = [load_binary_pred_slice(p) for p in slice_paths]
    slice_scores = [load_slice_scores(p) for p in slice_paths]

    # Drop all-zero-pixel "instances" (a mask that never exceeds the >0.5 threshold
    # anywhere -- common with a weak/untrained model at a permissive box_score_thresh)
    # before assigning ids: otherwise they'd consume a global id and a recorded score
    # while painting zero pixels into the volume, leaving mask{T}.scores.json with ids
    # that never actually appear in mask{T}.tif.
    for z in range(len(slices)):
        masks = slices[z]
        if masks.shape[0] == 0:
            continue
        nonempty = masks.reshape(masks.shape[0], -1).any(axis=1)
        slices[z] = masks[nonempty]
        if slice_scores[z] is not None:
            slice_scores[z] = slice_scores[z][nonempty]

    Z = len(slices)
    H, W = slices[0].shape[1:]

    volume = np.zeros((Z, H, W), dtype=np.int32)

    global_id = 1
    id_to_scores = defaultdict(list)

    def _record_scores(ids_for_slice, scores_for_slice):
        if scores_for_slice is None:
            return
        for local_idx, gid in ids_for_slice.items():
            id_to_scores[gid].append(float(scores_for_slice[local_idx]))

    # --- initialize first slice ---
    prev_masks = slices[0]
    prev_ids = {}

    for i, m in enumerate(prev_masks):
        prev_ids[i] = global_id
        volume[0][m.astype(bool)] = global_id
        global_id += 1
    _record_scores(prev_ids, slice_scores[0])

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
            _record_scores(prev_ids, slice_scores[z])
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
        # Note: if two predicted instances overlap heavily within this slice, a
        # later instance's pixels overwrite an earlier one's here (paint order =
        # enumeration order) -- pre-existing behavior, not changed by score
        # propagation. Rare in practice, but can leave a linked id in
        # id_to_scores/linked_scores with zero painted pixels in this slice if it
        # was fully overwritten and had no other slice to appear in.
        for j, m in enumerate(curr_masks):
            volume[z][m.astype(bool)] = curr_ids[j]
        _record_scores(curr_ids, slice_scores[z])

        # update
        prev_masks = curr_masks
        prev_ids = curr_ids

    agg_fn = np.mean if score_agg == "mean" else np.max
    linked_scores = {gid: float(agg_fn(scores)) for gid, scores in id_to_scores.items()}

    return volume, linked_scores

def make_files_for_SEG(exp_dir, target_masks_dir, pred_masks_dir, score_agg="mean"):
    """
    Creates:
      exp_dir/01_GT/SEG/man_segT.tif
      exp_dir/01_RES/maskT.tif
      exp_dir/01_RES/maskT.scores.json   ({linked_instance_id: aggregated_score}, if
                                           the pred slices had sibling .scores.npy files)
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
    # Files are flat slice ids (e.g. "000000.npz"); the mapping from a flat id
    # to its (volume, local-slice-index) lives in metadata.json, since volumes
    # no longer have a consistent slice count baked into the filename.
    metadata_path = os.path.join(os.path.dirname(target_masks_dir), "metadata.json")
    with open(metadata_path, encoding="utf-8") as fp:
        metadata = json.load(fp)

    idx_to_volume = {}
    for T, ids in metadata.items():
        for z, file_idx in enumerate(ids):
            idx_to_volume[file_idx] = (T, z)

    slices_by_T = defaultdict(list)

    for idx, gt_name in enumerate(gt_files):
        file_idx = int(gt_name.replace(".npz", ""))
        T, Z = idx_to_volume[file_idx]
        pred_path = os.path.join(pred_masks_dir, pred_files[idx])
        slices_by_T[T].append((Z, pred_path))

    # -------------------------
    # Step 4: per-T processing
    # -------------------------
    for T in sorted(slices_by_T.keys()):
        # CTC convention requires zero-padded frame numbers (man_seg0000.tif,
        # mask0000.tif, ...); num_digits=4 here to match how SEGMeasure is invoked
        # elsewhere (`./SEGMeasure <dir> 01 4`) -- the binary reports "No ground
        # truth object found!" if this padding doesn't match its num_digits arg.
        T_padded = f"{int(T):04d}"

        # hungarian matching
        all_slice_paths = sorted(slices_by_T[T])
        all_slice_paths = [i[1] for i in all_slice_paths]
        inst_3d, linked_scores = hungarian_matching_across(all_slice_paths, score_agg=score_agg)
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
            os.path.join(res_out, f"mask{T_padded}.tif"),
            inst_3d.astype(np.uint16)
        )

        if linked_scores:
            with open(os.path.join(res_out, f"mask{T_padded}.scores.json"), "w", encoding="utf-8") as f:
                json.dump({str(gid): score for gid, score in linked_scores.items()}, f)

        # ---- GT (copy or reconstruct) ----
        # safest option: reuse original GT volumes if available
        # otherwise reconstruct binary → CC same as preds
        gt_name_by_idx = {int(f.replace(".npz", "")): f for f in gt_files}
        gt_slices = []
        for file_idx in metadata[T]:
            if file_idx not in gt_name_by_idx:
                continue
            gt_path = os.path.join(target_masks_dir, gt_name_by_idx[file_idx])
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
            os.path.join(gt_out, f"man_seg{T_padded}.tif"),
            gt_3d.astype(np.uint16)
        )
