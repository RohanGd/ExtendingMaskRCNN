import os
import tifffile as tiff
import cv2
import numpy as np
import torch
from random import shuffle
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed

# combine all 01 and 02 folders into one dataset and corresponding truth labels as well.
# Then create a random test and train and validation split. 
# Find out if you need the entirety of the volume. Keep say only 5 empty slices.
# TODO (NOT IMPORTANT): Use multiprocessing to create this faster

def get_data_paths(dataset_path:str, type:str):
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
    return paths


def create_new_dir_struct(dataset_path:str):
    new_path = "datasets/" + dataset_path.split("/")[-1]
    if not os.path.exists(new_path):
        subdirs = ["train/imgs", "train/masks", "test/imgs", "test/masks", "val/imgs", "val/masks"]
        for subdir in subdirs:
            os.makedirs(os.path.join(new_path, subdir))
    return new_path


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


def pad_to_shape(img, target_h, target_w):
    h, w = img.shape[1:]
    dh = target_h - h
    dw = target_w - w

    top = dh // 2
    bottom = dh - top
    left = dw // 2
    right = dw - left
    img_stack = []
    for slice_id in range(img.shape[0]):
        slice = img[slice_id]
        img_stack.append(cv2.copyMakeBorder(slice, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0))
    return np.stack(img_stack)

# def pad_to_shape(img, target_h, target_w):
#     n, h, w = img.shape
#     pad_h = target_h - h
#     pad_w = target_w - w

#     pads = (
#         (0, 0),
#         (pad_h//2, pad_h - pad_h//2),
#         (pad_w//2, pad_w - pad_w//2),
#     )
#     return np.pad(img, pads, mode="constant")



def get_max_shape(paths):
    max_width = 0
    max_height = 0
    for split_paths in paths:
        for idx, path in enumerate(split_paths):
            img_path, mask_path = path
            img_height, img_width = tiff.TiffFile(img_path).pages[0].shape
            mask_height, mask_width = tiff.TiffFile(mask_path).pages[0].shape
            max_width = max(max_width, img_width, mask_width)
            max_height = max(max_height, img_height, mask_height)
    return max_height, max_width

def get_target_from_mask(mask, image_id):
    """
    mask: torch.Tensor of shape [H, W]
    returns: target dict for Mask R-CNN
    """

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

    target = {
        "boxes": boxes,
        "labels": labels,
        "masks": masks,
        "image_id": torch.tensor([image_id]),
        "area": area,
        "iscrowd": iscrowd
    }

    return target
    

def make(path_pair, idx, save_dir, type_, max_h, max_w, v):
    img_path, mask_path = path_pair
    _img = tiff.imread(img_path)
    _mask = tiff.imread(mask_path)
    assert _img.shape[0] == _mask.shape[0], f"Mismatch between number of slices of mask and image for {img_path} and {mask_path}"
    s = len(str(_img.shape[0]-1))
    _img = pad_to_shape(_img, max_h, max_w)
    _mask = pad_to_shape(_mask, max_h, max_w)

    for slice_idx in range(_img.shape[0]):
        img_slice = _img[slice_idx]
        mask_slice = _mask[slice_idx]
        mask_slice = torch.as_tensor(mask_slice, dtype=torch.float64)
        save_as_2d_slice(slice_data=img_slice, volume_idx=idx, slice_idx=slice_idx, save_dir=save_dir, type_=type_, v=v, s=s)
        target = get_target_from_mask(mask=mask_slice, image_id=idx * _img.shape[0] + slice_idx)
        save_target(target, volume_idx=idx, slice_idx=slice_idx, type_=type_, save_dir=save_dir, v=v, s=s)

def create_dataset(file_paths:str, save_dir:str, type_:str, max_h:int, max_w:int):
    num_files = len(file_paths)
    v = len(str(num_files - 1))

    print(os.cpu_count())
    with ProcessPoolExecutor(max_workers=3) as executor:
        futures = []
        for idx, path_pair in enumerate(file_paths):
            futures.append(executor.submit(make, path_pair, idx, save_dir, type_, max_h, max_w, v))
        for future in as_completed(futures):
            print("ok")


def save_as_2d_slice(slice_data, volume_idx, slice_idx, save_dir, type_, v, s):
    """
    type_: "imgs" or "masks"
    """
    v, s = 4, 3
    torch.save(torch.as_tensor(slice_data, dtype=torch.float32), f"{save_dir}/{type_}/imgs/{str(volume_idx).zfill(v)}_{str(slice_idx).zfill(s)}.pt")
    pass


def save_target(target, volume_idx, slice_idx, type_, save_dir, v, s):
    v, s = 4, 3
    torch.save(target, f"{save_dir}/{type_}/masks/{str(volume_idx).zfill(v)}_{str(slice_idx).zfill(s)}.pt")



####################################################################################
####################################################################################
####################################################################################
####################################################################################
####################################################################################
####################################################################################

def main():
    dataset_path = "data/Fluo-N3DH-CHO"
    # dataset_path = "data/Fluo-N3DH-SIM+"

    rusure = input(f"WARNING: ARE YOU SURE YOU WANT TO RESHUFFLE TRAIN TEST AND VALIDATION SPLITS? Y/N\nFor dataset: {dataset_path}\n Enter Y/N:    ")
    if rusure != "Y":
        exit()

    start_time = datetime.now()
    # imgs_paths, masks_paths = combine_01_02_get_paths(dataset_path)
    img_paths = get_data_paths(dataset_path=dataset_path, type="imgs")
    mask_paths = get_data_paths(dataset_path=dataset_path, type="masks")
    print(len(img_paths), len(mask_paths))
    train_paths, test_paths, val_paths = train_test_val_split_on_paths(img_paths, mask_paths)
    max_h, max_w = get_max_shape((train_paths, test_paths, val_paths))
    save_dir = create_new_dir_struct(dataset_path)

    print(len(val_paths), len(train_paths), len(test_paths), max_h, max_w)
    create_dataset(train_paths, save_dir, type_="train", max_h=max_h, max_w=max_w)
    create_dataset(test_paths, save_dir, type_="test", max_h=max_h, max_w=max_w)
    create_dataset(val_paths, save_dir, type_="val", max_h=max_h, max_w=max_w)

    print(len(os.listdir(save_dir + "/train/imgs")))
    print(len(os.listdir(save_dir + "/test/imgs")))
    print(len(os.listdir(save_dir + "/val/imgs")))
    print()
    print(len(os.listdir(save_dir + "/train/masks")))
    print(len(os.listdir(save_dir + "/test/masks")))
    print(len(os.listdir(save_dir + "/val/masks")))

    end_time = datetime.now()
    print(f"TOTAL TIME TAKEN: {end_time - start_time}")


if __name__ == "__main__":
    main()