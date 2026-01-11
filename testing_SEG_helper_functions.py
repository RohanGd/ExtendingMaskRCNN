import torch, numpy
import os, tifffile

def save_preds(preds, save_dir):
    """
    Saves the predicted mask from a model into the specified save dir as a .tif file.
    
    :param preds: preds["masks"] (num_preds, 1, H, W)
    :param save_dir: path to save, generally exp_dir/pred_masks
    """
    num_files = len(os.listdir(save_dir))
    for i, pred in enumerate(preds): # Batch size
        pred_mask = pred["masks"]
        pred_mask = (pred_mask > 0.5).bool().to(torch.int32)
        # this is of shape: [num_instances, 1, H, W]
        save_name = save_dir + f"/{num_files + i + 1}.tif"
        # now assign a grey value to each instance
        recolored_pred_mask = combine_binary_masks_into_grey(pred_mask)
        tifffile.imwrite(save_name, recolored_pred_mask.detach().cpu().numpy())

def combine_binary_masks_into_grey(binary_mask):
    """
    Converts a binary mask [num_instances, H, W] into a grey value [H, W] with a unique grey value per binary instance mask
    
    :param binary_mask: torch.Tensor, size=(num_instances, H, W)
    :returns recolored_grey_mask: torch.Tensor, size(H, W), dtype=torch.dtype(uint16)
    """
    if binary_mask.ndim == 4:
        num_instances, _, H, W = binary_mask.shape
    elif binary_mask.ndim == 3:
        num_instances, H, W = binary_mask.shape
    else:
        raise(Exception("Invalid dim size"))

    # Use int32 for operations, convert to uint16 at the end
    recolored_grey_mask = torch.zeros((H, W), device="cuda", dtype=torch.int32)
    for instance_id in range(num_instances):
        instance_grey_value = int(254 / num_instances) * (1 + instance_id)
        binary_mask_instances = binary_mask[instance_id][0] if binary_mask.ndim == 4 else binary_mask[instance_id] # H, W
        # assigniung a unique grey value for every binary mask
        recolored_grey_mask[binary_mask_instances != 0] = instance_grey_value
    
    # Convert to uint16 before returning
    return recolored_grey_mask.to(torch.uint16)

def rename_saved_preds(target_masks_file_paths, pred_masks_dir):
    """
    Renames the files in exp_dir/pred_masks to follow the naming convention in dataset/test/masks
    This transforms the naming in pred_masks/ from  01.tif, 02.tif,, 03.tif, .. to 0000_000.tif, 0000_001.tif, 0000_002.tif, ..
    This works because the test_dataloader has shuffle set to false and thus the files are laoded in order and fed to the model in the same order.
    :param target_masks_file_paths: files in dataset/test/masks
    :param pred_masks_dir: exp_dir/pred_masks
    """
    pred_masks_file_paths = sorted(os.listdir(pred_masks_dir))
    target_masks_file_paths = sorted(target_masks_file_paths)
    assert(len(target_masks_file_paths) == len(pred_masks_file_paths))

    for i in range(len(pred_masks_file_paths)):
        img_path = target_masks_file_paths[i]
        mask_path = pred_masks_file_paths[i]
        os.rename(os.path.join(pred_masks_dir, mask_path), os.path.join(pred_masks_dir, img_path.replace(".npz", ".tif")))


def group_by_volume_and_save(save_dir, source_dir, masks_file_paths, is_gt):
    v = 3
    # Given a list of file paths, group the paths according to the first v number of characters into a seperate list.
    if is_gt:
        num_volumes = len([f for f in masks_file_paths if f.endswith("000.npz")])
    else:
        num_volumes = len([f for f in masks_file_paths if f.endswith("000.tif")])

    volume_indices = [str(x).zfill(v) for x in range(num_volumes)] # 000, 001, 002, 003, 004, ...
    for volume_id in volume_indices:
        volume_slices = [os.path.join(source_dir, f) for f in masks_file_paths if f.startswith("0" + volume_id)]
        if is_gt:
            save_path = os.path.join(save_dir, f"man_seg{volume_id}.tif")
        else:
            save_path = os.path.join(save_dir, f"mask{volume_id}.tif")
        combine_slices_into_volume(volume_slices, save_path)


def combine_slices_into_volume(slice_file_paths: list[str], save_path: str):
    """
    Stack 2D TIF slices into a 3D TIF volume file.
    
    Args:
        slice_file_paths: List of paths to .tif files (2D slices)
        save_dir: Directory path where the 3D volume will be saved
    """
    
    slices = []
    reference_shape = None
    
    for i, file_path in enumerate(slice_file_paths):
        # Read TIF file
        if file_path.endswith(".npz"):
            img_array = numpy.load(file_path)
            img_array = img_array["masks"]
            img_array = combine_binary_masks_into_grey(img_array)
            img_array = img_array.detach().cpu().numpy()
        else:
            img_array = tifffile.imread(file_path)
        

        # Validate dimensions
        if reference_shape is None:
            reference_shape = img_array.shape
        elif img_array.shape != reference_shape:
            raise ValueError(
                f"All slices must have shape {img_array.shape}, expected {reference_shape} , see reference volume_slice: {file_path}"
            )
        
        slices.append(img_array)
    
    # Stack into 3D volume (slices, height, width)
    volume = numpy.stack(slices, axis=0)
    tifffile.imwrite(save_path, volume, compression='deflate')
    

def make_files_for_SEG(exp_dir, target_masks_dir, pred_masks_dir):
    """
    Follows the naming convention for SEG cli tool on the celltrackingchallenge.com to create dirs to store the gt masks and the predictions.
    
    :param exp_dir: Path of the experiment folder
    :param target_masks_dir: datasets/test/masks
    :param pred_masks_dir: exp_dir/pred_masks
    """
    target_masks_file_paths = sorted(os.listdir(target_masks_dir))
    rename_saved_preds(target_masks_file_paths, pred_masks_dir)
    pred_masks_file_paths = sorted(os.listdir(pred_masks_dir))

    gt_save_dir = os.path.join(exp_dir, "01_GT", "SEG")
    res_save_dir = os.path.join(exp_dir, "01_RES")
    os.makedirs(gt_save_dir, exist_ok=True)
    os.makedirs(res_save_dir, exist_ok=True)

    group_by_volume_and_save(save_dir=gt_save_dir, source_dir=target_masks_dir, masks_file_paths=target_masks_file_paths, is_gt=True)
    group_by_volume_and_save(save_dir=res_save_dir, source_dir=pred_masks_dir, masks_file_paths=pred_masks_file_paths, is_gt=False)


