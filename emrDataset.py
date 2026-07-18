'''
Logic that returns n, H, W such that the n slices are consecutive from a volume.
Say n = 3, then we need to get 3 slices from say volume 0. However we need not load the enitre collume 0 as it is quite big and will cause slow dataloading which will affect training time and inference time.
Instead we save 2D slices and implement a logic that retrieves contiguous slices while also making sure that the slices are from the same volume.
Thus create an appropriate index.

The mask returned for n slices will be the center mask.
eg. n=3, idx = 5, and per volume size = 8 (0,1, ..7)
This returns volume 0 slices 4, 5, 6 and the mask for slice 5
idx = 6; volume 0 slices 5, 6, 7 and the mask for slice 6
idx = 7; volume 0 slices 6, 7, null and the mask for slice 7. Here idx(7) < v_size(8), but idx + (n-1)//2 = 8 >= v_size, then pad a null slice
idx = 0; volume 0 slice null, 0, 1 and the mask for slice 0. Here idx(0) > 0, but idx % v_size - (n-1)//2 = -1 , so pad a null slice before
idx = 9; volume 1 slice idx % v_size - (n-1)//2 so 0, 1, 2. And mask of slcie 1
'''

import os
from torch.utils.data import Dataset
import tifffile as tiff
import torch
from emrConfigManager import setup_logger, DATASETS_PATH
import warnings
import numpy
import json

warnings.simplefilter("ignore", category=FutureWarning)

class emrDataset(Dataset):
    def __init__(self, imgs_dir: str = None, masks_dir: str = None, num_slices: int = 3, dataset_name = "", logger=None, mode="train"):
        """
        Args:
            imgs_dir (str, optional): Relative path to the directory containing 3D TIFF image files.
            masks_dir (str, optional): Relative path to the directory containing 3D TIFF mask files.
            num_slices (int, optional): Number of slices to include in the 2.5D context window. Defaults to 3.
        """

        self.dataset_name = dataset_name
        self.mode = mode
        self.n = num_slices
        try:
            self.img_files = sorted(os.listdir(f"{DATASETS_PATH}/{self.dataset_name}/{self.mode}/imgs/"))
            self.mask_files = sorted(os.listdir(f"{DATASETS_PATH}/{self.dataset_name}/{self.mode}/masks/"))
            self.metadata = json.load(fp=open(f"{DATASETS_PATH}/{self.dataset_name}/{self.mode}/metadata.json", encoding="utf-8"))
        except FileNotFoundError:
            raise FileNotFoundError("Cannot find the required folder. Run dataSplitter.py to create dataset.")

        assert len(self.img_files) == len(self.mask_files), f"Mismatch between data and labels. imgs_dir - {imgs_dir} - {len(self.img_files), len(self.mask_files)}"
        if logger == None:
            logger = setup_logger("emrdataset.log")


        self.H, self.W = numpy.load(os.path.join(imgs_dir, self.img_files[0])).shape
        num_files = len(self.img_files)
        self.s = 6

        logger.info(f"Initialized dataset - {self.dataset_name} from  dim: ({self.H, self.W}), and num_files: {num_files} with {self.__len__()} slices. Find 2d slice files at datasets/{self.dataset_name}/{self.mode}/imgs/ or /masks")


    def __len__(self):
        '''
        If I have 10 volumes of size 5
        total 50 slices. SInce I am doing padding each one can be accessed.
        '''
        return len(self.img_files)

    def __get_volume_bounds__(self, idx):
        for key, values in self.metadata.items():
            if idx >= values[0] and idx <= values[-1]:
                return values[0], values[-1]
    
        return None, None


    def __getitem__(self, idx):
        """
        Retrieves the image and mask corresponding to the given slice idx.

        Args:
            idx (int): idx in the range [0, self.__len__() - 1].

        Returns:
            tuple: A tuple containing:
                - img (ndarray): Stack of n slices with the idx slice at the center.
                - mask (ndarray): Segmentation mask of the idx slice.
        """

        start_idx, end_idx = self.__get_volume_bounds__(idx)

        img_slices = []
        for i in range(-(self.n - 1)//2, (self.n - 1)//2 + 1):
            img_slice = None
            slice_idx = idx + i
            if start_idx > slice_idx or slice_idx > end_idx:
                # null slice or very small noise image.
                eps = 1e-8 
                img_slice = torch.ones(size=(self.H, self.W)) * eps
            else:
                img_slice = numpy.load(f"{DATASETS_PATH}/{self.dataset_name}/{self.mode}/imgs/{str(slice_idx).zfill(self.s)}.npy")
                img_slice = torch.from_numpy(img_slice)
            img_slices.append(img_slice)
        
        img_slices = torch.stack(img_slices)

        # mask / target
        slice_idx = idx
        target_npz = numpy.load(f"{DATASETS_PATH}/{self.dataset_name}/{self.mode}/masks/{str(slice_idx).zfill(self.s)}.npz")
        target = dict()
        for key in target_npz:
            target[key] = torch.from_numpy(target_npz[key])

        return img_slices, target


def emrCollate_fn(batch):
    '''
    Custom collate function for dataloader.
    Returns:
        data: torch.Size([B, n, H, W])
        targets: list of dicts of tensor. List size = B, each dict corresponds to the center slice
    '''
    data = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    data = torch.stack(data, dim=0)
    return data, targets
