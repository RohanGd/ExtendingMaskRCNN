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
import logging
from torch.utils.data import Dataset
import tifffile as tiff
import torch

logger = logging.getLogger(__name__)
logger.setLevel(20)

class emrDataset(Dataset):
    def __init__(self, imgs_dir: str = None, masks_dir: str = None, n: int = 3, load_from_cache=False):
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
        assert len(img_files) == len(mask_files), "Mismatch between data and labels."

        self.dataset_name = os.path.normpath(imgs_dir).split(os.sep)[0]
        self.n = n
        self.v_size, self.H, self.W = tiff.TiffFile(os.path.join(imgs_dir, img_files[0])).series[0].shape # total number of slices per volume /3d image, and H, W
        num_files = len(img_files)
        self.v, self.s = len(str(num_files)) + 1, len(str(self.v_size)) + 1

        if not load_from_cache:
            for i in range(num_files):
                _img = tiff.imread(os.path.join(imgs_dir, img_files[i]))
                _mask = tiff.imread(os.path.join(masks_dir, mask_files[i]))
                assert _img.shape[0] == _mask.shape[0], f"Mismatch between number of slices of mask and image for {img_files[i]} and {mask_files[i]}"
                for slice_idx in range(_img.shape[0]):
                    self.save_as_2d_slice(slice_data=_img[slice_idx], volume_idx=i, slice_idx=slice_idx, type_="imgs")
                    self.save_as_2d_slice(slice_data=_mask[slice_idx], volume_idx=i, slice_idx=slice_idx, type_="masks") 
        else:
            pass       

        self.img_files = sorted(os.listdir(f"datasets/{self.dataset_name}/imgs/"))
        self.mask_files = sorted(os.listdir(f"datasets/{self.dataset_name}/masks/"))

        logger.info(f"Initialized dataset - {self.dataset_name} from  dim: ({self.v_size, self.H, self.W}), and num_files: {num_files} with {self.__len__()} slices. Find 2d slice files at datasets/{self.dataset_name}/imgs/ or /masks")
        pass

    def save_as_2d_slice(self, slice_data, volume_idx, slice_idx, type_):
        """
        type_: "imgs" or "masks"
        """
        save_dir = f"datasets/{self.dataset_name}/{type_}"
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        tiff.imwrite(f"{save_dir}/{str(volume_idx).zfill(self.v)}_{str(slice_idx).zfill(self.s)}.tif", data=slice_data)
        pass

    def get_target_from_mask(self, mask, image_id):
        """
        mask: torch.Tensor of shape [1, H, W]
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
                img_slice = tiff.imread(f"datasets/{self.dataset_name}/imgs/{str(v_idx).zfill(self.v)}_{str(slice_idx).zfill(self.s)}.tif")
                img_slice = torch.as_tensor(img_slice, dtype=torch.float32) # H,W
            img_slices.append(img_slice)
        
        img_slices = torch.stack(img_slices)

        # mask / target
        slice_idx = idx % self.v_size
        mask_slice = tiff.imread(f"datasets/{self.dataset_name}/imgs/{str(v_idx).zfill(self.v)}_{str(slice_idx).zfill(self.s)}.tif") # H, W
        mask_slice = torch.as_tensor(mask_slice, dtype=torch.float64) # H, W

        target = self.get_target_from_mask(mask_slice, idx)

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
