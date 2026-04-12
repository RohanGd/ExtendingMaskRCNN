import os
import numpy as np
import tifffile

def load_mask(path:str) -> np.ndarray:
    if path.endswith(".tif"):
        masks = tifffile.imread(path)
        masks = masks.squeeze(axis=1)
        # print(masks.shape)
        # print(np.unique(masks))
    
    else:
        target = np.load(path)
        # masks = target["masks"]
        masks = target["orignal_mask"]
    # print(masks.shape, np.unique_counts(masks))
    return masks

def drop_dim_by_coloring(mask: np.ndarray):
    num_cells, h, w = mask.shape
    new_mask = np.zeros((h, w), dtype=np.int16)

    for c in range(num_cells):
        m = mask[c]
        new_mask[m != 0] = c + 2

    return new_mask

def mkdir_2d_masks(dataset_path:str):
    dataset_path = dataset_path.removesuffix("/masks")
    os.makedirs(os.path.join(dataset_path, "2d_masks"), exist_ok=True)
    return os.path.join(dataset_path, "2d_masks")

def save_mask(mask, t:int, save_dir:str, save_name:str):
    _name = f"{save_name}{t:04d}.tif"
    save_path = os.path.join(save_dir, _name)
    tifffile.imwrite(save_path, mask)

def store_2d_dataset_masks(dataset, save_path):
    file_paths = sorted([os.path.join(dataset_path, f) for f in os.listdir(dataset_path)])

    # save_dir_2d_masks = mkdir_2d_masks(dataset_path=dataset_path)
    save_dir_2d_masks = save_path
    if os.path.exists(save_dir_2d_masks) == False:
        os.makedirs(save_dir_2d_masks)
    for t, file_path in enumerate(file_paths):
        mask = load_mask(path=file_path)
        # mask = drop_dim_by_coloring(mask)
        # print(mask.dtype)
        mask = mask.astype(np.uint16)
        save_mask(mask=mask, t=t, save_dir=save_dir_2d_masks, save_name="man_seg")



def mkdir_2d_preds(save_path):
    os.makedirs(save_path, exist_ok=True)
    return save_path

def store_2d_preds(renamed_preds_path, save_dir):
    file_paths = sorted([os.path.join(renamed_preds_path, f) for f in os.listdir(renamed_preds_path)])
    save_dir_2d_masks = mkdir_2d_preds(save_path=save_dir)

    for t, file_path in enumerate(file_paths):
        mask = load_mask(path=file_path)
        mask = drop_dim_by_coloring(mask)
        # print("mask", mask.dtype)
        mask = mask.astype(np.uint16)
        save_mask(mask=mask, t=t, save_dir=save_dir_2d_masks, save_name="mask")



# dataset_path = "datasets/12spheroids_Low/test/masks"
dataset_path = "datasets/Fluo-N3DH-SIM+/test/masks"
files = [
'channelFusion_20260328_214504',
# 'earlyFusion_GlobalGaussian_20260328_214504',
# 'earlyFusion_GlobalPerFPN_20260328_215310',
# 'earlyFusion_Pixel_20260328_215331',
# 'earlyFusion_Windowed_20260328_220034',
# 'lateFusion_conv3d_20260328_221338',
# 'lateFusion_mean_20260328_222135',
# 'lateFusion_onlyCenter_20260328_222937',
# 'lateFusion_SE_20260328_223711',
# "data/n1",
# "data/SIM+_n3/channelFusion_20260327_090820",
# "data/SIM+_n3/earlyFusion_GlobalGaussian_20260327_090820",
# "data/SIM+_n3/lateFusion_onlyCenter_20260327_091016",
# "data/12spheroids_Low_n5/channelFusion_20260330_163145",
# "data/12spheroids_Low_n5/earlyFusion_GlobalGaussian_20260330_174953",
# "data/12spheroids_Low_n5/lateFusion_onlyCenter_20260330_191518",
# "data/SIM+_n7/channelFusion_20260326_183827",
# "data/SIM+_n7/earlyFusion_GlobalGaussian_20260326_205246",
# "data/SIM+_n7/lateFusion_onlyCenter_20260327_141812"
]

for i, file in enumerate(files):
    # if i < 16:
    #     continue
    print(file)
    # exit(0)
    save_dir = f"test_dir/{str(i+1).zfill(2)}_GT/SEG"
    store_2d_dataset_masks(dataset_path, save_path=save_dir)
    
    save_dir = f"test_dir/{str(i+1).zfill(2)}_RES"
    # store_2d_preds(renamed_preds_path=f"/home/rohan/Dev/ExtendingMaskRCNN/data/SIM+_n7/{file}/renamed_preds", save_dir=save_dir)
    store_2d_preds(renamed_preds_path=f"/home/rohan/Dev/ExtendingMaskRCNN/data/SIM+_n5/{file}/renamed_preds", save_dir=save_dir)
    # store_2d_preds(renamed_preds_path=f"{file}/renamed_preds", save_dir=save_dir)

# store_2d_dataset_masks(dataset_path)
# store_2d_preds(renamed_preds_path)

