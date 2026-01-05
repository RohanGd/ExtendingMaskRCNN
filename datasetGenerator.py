import torch
import numpy as np
import os
from random import shuffle
import tifffile as tiff
import torch.nn.functional as F
from datetime import datetime



def main():
    # dataset_path = "data/Fluo-N3DH-CHO"
    dataset_path = "data/Fluo-N3DH-SIM+"

    rusure = input(f"WARNING: ARE YOU SURE YOU WANT TO RESHUFFLE TRAIN TEST AND VALIDATION SPLITS? Y/N\nFor dataset: {dataset_path}\n Enter Y/N:    ")
    if rusure != "Y":
        exit()

    save_dir = create_new_dir_struct(dataset_path)

    img_paths = get_file_paths(dataset_path=dataset_path, type="imgs")
    mask_paths = get_file_paths(dataset_path=dataset_path, type="masks")

    train_paths, test_paths, val_paths = train_test_val_split_on_paths(img_paths, mask_paths)

    # val_paths = val_paths[:2]
    print("-"*50,"\nCREATING VAL DATASET")
    create_dataset(val_paths, save_dir, type_="val")
    print("-"*50,"\nCREATING TEST DATASET")
    create_dataset(test_paths, save_dir, type_="test")
    print("-"*50,"\nCREATING TRAIN DATASET")
    create_dataset(train_paths, save_dir, type_="train")


def create_new_dir_struct(dataset_path:str):
    new_path = "datasets/" + dataset_path.split("/")[-1]
    if not os.path.exists(new_path):
        subdirs = ["train/imgs", "train/masks", "test/imgs", "test/masks", "val/imgs", "val/masks"]
        for subdir in subdirs:
            os.makedirs(os.path.join(new_path, subdir))
    return new_path


def get_file_paths(dataset_path:str, type:str):
    """Returns file paths for all imgs or masks inside the dataset folder. Expects folder structure:
    dataset:
        - /01
        - /01_ERR_SEG
        - /02
        - /02_ERR_SEG

    Args:
        dataset_path (str): path of the dataset folder (data/Fluo-N3DH-SIM+)
        type (str): "imgs" or "masks"
    """
    paths = list()
    for fol_name in ["01", "02"]:
        if type == "imgs":
            fol_paths = [os.path.join(dataset_path, fol_name, x) for x in os.listdir(os.path.join(dataset_path, fol_name))]
        elif type == "masks":
            fol_paths =[os.path.join(dataset_path, fol_name+"_ERR_SEG", x) for x in os.listdir(os.path.join(dataset_path,  fol_name+"_ERR_SEG"))]
        else:
            raise 'Specify type either "imgs" or "masks".'
        paths.extend(fol_paths)
    return sorted(paths)


def train_test_val_split_on_paths(img_paths:str, mask_paths:str, split=[0.6, 0.3, 0.1]):
    """Shuffles the list of img_paths and mask_paths and then splits the lists into train, test and val in the given split ratio.

    Args:
        imgs_paths (str)
        masks_paths (str)
        split (list, optional): train, test, val splits. Defaults to [0.6, 0.3, 0.1].

    Returns:
        train_paths (list): list of tuples of img_path and corresponding mask_path
        test_paths (list)
        val_paths (list)
    """
    assert len(img_paths) == len(mask_paths), "Length mismatch of img_paths and mask_paths in create_splits_on_file_paths"
    paired_data_label = dict()
    n = len(img_paths)
    for i in range(n):
        pair = (img_paths[i], mask_paths[i])
        paired_data_label.update({i: pair})
    new_indices = list(range(n))
    shuffle(new_indices)
    train_indices = new_indices[0 : int(n*split[0])]
    test_indices = new_indices[int(n*split[0]): int(n*(split[0] + split[1]))]
    val_indices = new_indices[int(n*(split[0] + split[1])) : ]

    def pluck(indices):
        target_paths = list()
        for index in indices:
            target_paths.append(paired_data_label[index])
        return target_paths

    train_paths, test_paths, val_paths = pluck(train_indices), pluck(test_indices), pluck(val_indices)

    return train_paths, test_paths, val_paths


def create_dataset(file_paths:str, save_dir:str, type_:str):
    print(os.cpu_count())
    epoch0 = datetime.now()
    for idx, path_pair in enumerate(file_paths):
        run1_start = datetime.now()
        make(path_pair, idx, save_dir, type_)
        run1_time = datetime.now() - run1_start
        total_time_elapsed = datetime.now() - epoch0
        remaining_items = len(file_paths) - (idx + 1)
        avg_time_per_item = total_time_elapsed / (idx + 1)
        estimated_time_remaining = avg_time_per_item * remaining_items
        print(f"PROGRESS: {idx+1} / {len(file_paths)} | ETA: {estimated_time_remaining} | TFO: {run1_time}")


def resize_with_padding(img, target_size=512, is_mask=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_slices, height, width = img.shape

    if isinstance(img, np.ndarray):
        if is_mask:
            img = torch.from_numpy(img).long().to(device)  # Keep as integer
        else:
            img = torch.from_numpy(img).float().to(device)
    
    # If already target size, return as is
    if height == target_size and width == target_size:
        return img
    
    # Calculate scaling factor to fit within target size
    ratio = min(target_size / width, target_size / height)
    new_h = int(height * ratio)
    new_w = int(width * ratio)
    
    # Add batch and channel dimensions for interpolate: (N, C, H, W)
    img_reshaped = img.unsqueeze(1).float()  # Need float for interpolation
    
    # Resize - use nearest for masks to preserve IDs
    img_resized = F.interpolate(
        img_reshaped, 
        size=(new_h, new_w), 
        mode='nearest' if is_mask else 'bilinear',  # Nearest for masks!
        align_corners=False if not is_mask else None
    )
    
    # Calculate padding (left, right, top, bottom)
    pad_left = (target_size - new_w) // 2
    pad_right = target_size - new_w - pad_left
    pad_top = (target_size - new_h) // 2
    pad_bottom = target_size - new_h - pad_top
    
    # Apply padding
    img_padded = F.pad(
        img_resized, 
        (pad_left, pad_right, pad_top, pad_bottom), 
        mode='constant', 
        value=0
    )
    
    # Remove channel dimension and return
    img_padded = img_padded.squeeze(1)
    
    # Convert back to long if mask
    if is_mask:
        img_padded = img_padded.long()
    
    del img, img_reshaped, img_resized
    return img_padded.cpu()


def get_target_from_mask(mask, image_id):
    """
    mask: torch.Tensor of shape [H, W]
    returns: target dict for Mask R-CNN
    """

    # mask = torch.from_numpy(mask) if isinstance(mask, np.ndarray) else mask

    # get unique object ids (excluding background)
    obj_ids = torch.unique(mask)
    obj_ids = obj_ids[obj_ids != 0]

    # create binary masks for each object id
    masks = (mask[None, :, :] == obj_ids[:, None, None]).to(torch.uint8)  # [N, H, W]

    boxes = []
    for m in masks:
        pos = torch.where(m)
        if len(pos[0]) == 0:
            continue
        xmin, xmax = pos[1].min(), pos[1].max()
        ymin, ymax = pos[0].min(), pos[0].max()

        if xmax <= xmin or ymax <= ymin:
            continue  # this gave error in training (basically empty boxes if xmin == xmax or ymin == ymax)
        boxes.append([xmin, ymin, xmax, ymax])

    if len(boxes) == 0:
        boxes = torch.zeros((0, 4), dtype=torch.float32)
        labels = torch.zeros((0,), dtype=torch.int64)
        masks = torch.zeros((0, *mask.shape), dtype=torch.uint8)
    else:
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.ones((len(boxes),), dtype=torch.int64)

    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    iscrowd = torch.zeros((len(labels),), dtype=torch.int64)

    # target = {
    #     "boxes": boxes.numpy(),
    #     "labels": labels.numpy(),
    #     "masks": masks.numpy(),
    #     "image_id": np.array([image_id]),
    #     "area": area.numpy(),
    #     "iscrowd": iscrowd.numpy()
    # }
    target = {
        "boxes": boxes,
        "labels": labels,
        "masks": masks,
        "image_id": torch.tensor([image_id]),
        "area": area,
        "iscrowd": iscrowd
    }
    
    
    return target
    

def make(path_pair, idx, save_dir, type_):
    img_path, mask_path = path_pair
    _img = tiff.imread(img_path)
    _mask = tiff.imread(mask_path)
    assert _img.shape[0] == _mask.shape[0], f"Mismatch between number of slices of mask and image for {img_path} and {mask_path}"
    _img = resize_with_padding(_img, is_mask=False)
    _mask = resize_with_padding(_mask, is_mask=True)
    volume_depth = _img.shape[0]

    # bottleneck here, in saving individual slices
    _img_cpu = _img.cpu().numpy()
    _mask_cpu = _mask
       
    for slice_idx in range(_img_cpu.shape[0]):
        # if slice_idx % 20 == 0:
        #     time.sleep(2)
        img_slice = _img_cpu[slice_idx].copy()
        mask_slice = _mask_cpu[slice_idx]
        
        save_slice_worker((img_slice, mask_slice, idx, slice_idx, save_dir, type_, volume_depth))


def save_slice_worker(args):
    img_slice, mask_slice, idx, slice_idx, save_dir, type_, volume_depth = args
    save_as_2d_slice(slice_data=img_slice, volume_idx=idx, slice_idx=slice_idx, 
                     save_dir=save_dir, type_=type_)
    target = get_target_from_mask(mask=mask_slice, image_id=idx * volume_depth + slice_idx)
    save_target(target, volume_idx=idx, slice_idx=slice_idx, type_=type_, save_dir=save_dir)


def save_as_2d_slice(slice_data, volume_idx, slice_idx, save_dir, type_):
    v, s = 4, 3
    # Create subdirectory per volume
    filepath = f"{save_dir}/{type_}/imgs/{str(volume_idx).zfill(v)}_{str(slice_idx).zfill(s)}.npy"
    np.save(filepath, slice_data)


def save_target(target, volume_idx, slice_idx, type_, save_dir):
    v, s = 4, 3
    # Create subdirectory per volume
    filepath = f"{save_dir}/{type_}/masks/{str(volume_idx).zfill(v)}_{str(slice_idx).zfill(s)}.npz"
    target = {
        'boxes': target['boxes'].cpu().numpy(),
        'labels': target['labels'].cpu().numpy(),
        'masks': target['masks'].cpu().numpy(),
        'image_id': target['image_id'].cpu().numpy(),
        'area': target['area'].cpu().numpy(),
        'iscrowd': target['iscrowd'].cpu().numpy()
    }
    np.savez_compressed(filepath, **target)

if __name__ == "__main__":
    main()