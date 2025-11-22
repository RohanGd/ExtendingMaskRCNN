import os
import tifffile
import cv2
import numpy as np

dataset_path = "Fluo-N3DH-SIM+"

# combine all 01 and 02 folders into one dataset and corresponding truth labels as well.
# Then create a random test and train and validation split. 
# Find out if you need the entirety of the volume. Keep say only 5 empty slices.

def combine_01_02_get_paths(dataset_path):    
    imgs_paths, masks_paths = list(), list()
    for fol_name in ["01", "02"]:
        imgs_01 = [os.path.join(dataset_path, fol_name, x) for x in os.listdir(os.path.join(dataset_path, fol_name))]
        masks_01 =[os.path.join(dataset_path, "{fol}_GT".format(fol=fol_name), "SEG", x) for x in os.listdir(os.path.join(dataset_path, "{fol}_GT".format(fol=fol_name), "SEG"))]

        imgs_paths.extend(imgs_01)
        masks_paths.extend(masks_01)

    return imgs_paths, masks_paths

def create_new_dir_struct(dataset_path):
    new_path = dataset_path + "_joined"
    if not os.path.exists(new_path):
        subdirs = ["train/imgs", "train/masks", "test/imgs", "test/masks", "val/imgs", "val/masks"]
        for subdir in subdirs:
            os.makedirs(os.path.join(new_path, subdir))
    return new_path


def study_empty_slice_counts(file_paths):
    good_slice_start_counts, good_slice_end_counts = list(), list()
    for img_path in file_paths:
        img = tifffile.imread(img_path)
        good_slice_start = 0
        good_slice_end = 0
        for slice_id, slice in enumerate(img):
            if slice.any():
                # good slice
                if good_slice_start == 0:
                    good_slice_start = slice_id
                good_slice_end = slice_id
        good_slice_start_counts.append(good_slice_start)
        good_slice_end_counts.append(good_slice_end)

    avg = list(map(lambda x: sum(x)/len(x), (good_slice_start_counts, good_slice_end_counts)))
    print(avg)
    value_counts = list()
    # Option 3: List comprehension (cleaner)
    from pprint import pprint
    pprint([{val: x.count(val) for val in set(x)} for x in (good_slice_start_counts, good_slice_end_counts)])


def create_splits_on_file_paths(imgs_paths, masks_paths, split=[0.6, 0.3, 0.1]):
    assert len(imgs_paths) == len(masks_paths), "Length mismatch of imgs_paths and masks_paths in create_splits_on_file_paths"
    paired_data_label = dict()
    n = len(imgs_paths)
    for i in range(n):
        pair = (imgs_paths[i], masks_paths[i])
        paired_data_label.update({i: pair})
    from random import shuffle
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


def get_max_shape(file_paths):
    max_width = 0
    max_height = 0
    for idx, path in enumerate(file_paths):
        img_path, mask_path = path
        img_height, img_width = tifffile.TiffFile(img_path).pages[0].shape
        mask_height, mask_width = tifffile.TiffFile(mask_path).pages[0].shape
        max_width = max(max_width, img_width, mask_width)
        max_height = max(max_height, img_height, mask_height)
    return max_height, max_width

def copy_into_new_folder(file_paths, folder_path):
    max_h, max_w = get_max_shape(file_paths)
    v = len(str(len(file_paths)))
    for idx, path in enumerate(file_paths):
        img_path, mask_path = path    
        img = pad_to_shape(tifffile.imread(img_path), max_h, max_w)
        mask = pad_to_shape(tifffile.imread(mask_path), max_h, max_w)

        assert img.shape == mask.shape
        assert img.shape[1] == max_h and img.shape[2] == max_w

        tifffile.imwrite(os.path.join(folder_path, 'imgs', str(idx).zfill(v) + ".tif"), img)
        tifffile.imwrite(os.path.join(folder_path, 'masks', str(idx).zfill(v) + ".tif"), mask)
    


rusure = input("WARNING: ARE YOU SURE YOU WANT TO RESHUFFLE TRAIN TEST AND VALIDATION SPLITS? Y/N")
if rusure != "Y":
    exit()

imgs_paths, masks_paths = combine_01_02_get_paths(dataset_path)
# # study_empty_slice_counts(masks_paths)
train_paths, test_paths, val_paths = create_splits_on_file_paths(imgs_paths, masks_paths)
new_path = create_new_dir_struct(dataset_path)
copy_into_new_folder(train_paths, new_path + "/train")
copy_into_new_folder(test_paths, new_path + "/test")
copy_into_new_folder(val_paths, new_path + "/val")

print(len(val_paths), len(train_paths), len(test_paths))
print(len(os.listdir(new_path + "/train/imgs")))
print(len(os.listdir(new_path + "/test/imgs")))
print(len(os.listdir(new_path + "/val/imgs")))
print()
print(len(os.listdir(new_path + "/train/masks")))
print(len(os.listdir(new_path + "/test/masks")))
print(len(os.listdir(new_path + "/val/masks")))


