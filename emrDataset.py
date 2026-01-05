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
from emrConfigManager import setup_logger
import warnings
import numpy

warnings.simplefilter("ignore", category=FutureWarning)

class emrDataset(Dataset):
    def __init__(self, imgs_dir: str = None, masks_dir: str = None, n: int = 3, load_from_cache=True, logger=None, mode="train"):
        """
        Initializes the dataset with image and mask directories and the 2.5D context window size.

        Args:
            imgs_dir (str, optional): Relative path to the directory containing 3D TIFF image files.
            masks_dir (str, optional): Relative path to the directory containing 3D TIFF mask files.
            n (int, optional): Number of slices to include in the 2.5D context window. Defaults to 3.
            load_from_cache: if True, doesnt resave individual 2d slices. Expects them to exist in datasets/dataset_name/imgs, masks

        Notes:
            Assumes that file names in both directories are sorted in order.
            Assumes every image is of a fixed shape - (D, H, W)
        """
        img_files, mask_files = sorted(os.listdir(imgs_dir)), sorted(os.listdir(masks_dir))
        assert len(img_files) == len(mask_files), f"Mismatch between data and labels. imgs_dir - {imgs_dir} - {len(img_files), len(mask_files)}"
        if logger == None:
            logger = setup_logger("emrdataset.log")

        self.dataset_name = os.path.normpath(imgs_dir).split(os.sep)[1]
        self.mode = mode
        self.n = n
        self.H, self.W = numpy.load(os.path.join(imgs_dir, img_files[0])).shape # total number of slices per volume /3d image, and H, W
        self.v_size = len([f for f in img_files if f.startswith("0000")])
        num_files = len(img_files)
        # self.v, self.s = len(str(num_files)) + 1, len(str(self.v_size)) + 1
        self.v, self.s = 4, 3

        try:
            self.img_files = sorted(os.listdir(f"datasets/{self.dataset_name}/{self.mode}/imgs/"))
            self.mask_files = sorted(os.listdir(f"datasets/{self.dataset_name}/{self.mode}/masks/"))
        except FileNotFoundError:
            raise FileNotFoundError("CAnnot find the required folder. Run dataSplitter.py to create dataset.")

        logger.info(f"Initialized dataset - {self.dataset_name} from  dim: ({self.v_size, self.H, self.W}), and num_files: {num_files} with {self.__len__()} slices. Find 2d slice files at datasets/{self.dataset_name}/{self.mode}/imgs/ or /masks")
        pass

    def __len__(self):
        '''
        If I have 10 volumes of size 5
        total 50 slices. SInce I am doing padding each one can be accessed.
        '''
        return len(self.img_files)


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

        # loop over n from  
        v_idx = idx // self.v_size

        img_slices = []
        for i in range(-(self.n - 1)//2, (self.n - 1)//2 + 1):
            slice_idx = idx % self.v_size + i 
            img_slice = None
            if slice_idx < 0 or slice_idx >= self.v_size:
                # null slice or very small noise image.
                eps = 1e-8 
                img_slice = torch.ones(size=(self.H, self.W)) * eps
            else:
                # get slice at v_idx at slice_idx
                img_slice = numpy.load(f"datasets/{self.dataset_name}/{self.mode}/imgs/{str(v_idx).zfill(self.v)}_{str(slice_idx).zfill(self.s)}.npy")
                img_slice = torch.from_numpy(img_slice)
            img_slices.append(img_slice)
        
        img_slices = torch.stack(img_slices)

        # mask / target
        slice_idx = idx % self.v_size
        target_npz = numpy.load(f"datasets/{self.dataset_name}/{self.mode}/masks/{str(v_idx).zfill(self.v)}_{str(slice_idx).zfill(self.s)}.npz")
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
