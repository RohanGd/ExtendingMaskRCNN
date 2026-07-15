import argparse
import random
from dataclasses import dataclass
from pathlib import Path
import json

import nrrd
import numpy as np
import tifffile as tiff
import torch
import torch.nn.functional as F
from tqdm import tqdm

from emrConfigManager import DATAPATH


DEFAULT_DATASET = "ATAS"
SPLIT_SEED = 42
SPLIT = (0.75, 0.15, 0.10)
BENCHMARK_ROOT = Path("data/Cell_Segmentation_Beyond_2D_Benchmark_Dataset")
DATASETS_ROOT = Path("datasets")


def resolve_path(path):
    path = Path(path)
    candidates = [
        path,
        Path(__file__).resolve().parent / path,
        Path(DATAPATH) / path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[1]


@dataclass(frozen=True)
class VolumePair:
    image_path: Path
    mask_path: Path


class DatasetSource:
    name = None
    output_name = None

    def get_pairs(self):
        raise NotImplementedError

    def read_volume(self, pair):
        image = tiff.imread(pair.image_path)
        mask = tiff.imread(pair.mask_path)
        return ensure_3d(image), ensure_3d(mask)


class BenchmarkTiffSource(DatasetSource):
    def __init__(self, name, root=BENCHMARK_ROOT):
        self.name = name
        self.output_name = name
        self.root = resolve_path(root)
        self.images_dir = self.root / "images" / name
        self.masks_dir = self.root / "masks" / name

    def get_pairs(self):
        if not self.images_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {self.images_dir}")
        if not self.masks_dir.exists():
            raise FileNotFoundError(f"Mask directory not found: {self.masks_dir}")

        image_paths = sorted(self.images_dir.glob("*.tif"))
        mask_by_name = {path.name: path for path in sorted(self.masks_dir.glob("*.tif"))}
        if not image_paths:
            raise FileNotFoundError(f"No .tif images found in {self.images_dir}")

        missing_masks = [path.name for path in image_paths if path.name not in mask_by_name]
        if missing_masks:
            raise FileNotFoundError(f"Missing masks for {self.name}: {missing_masks[:5]}")

        return [VolumePair(image_path, mask_by_name[image_path.name]) for image_path in image_paths]


class CellTrackingTiffSource(DatasetSource):
    def __init__(self, name, root=None):
        self.name = name
        self.output_name = name
        self.root = resolve_path(root or Path("data") / name)

    def get_pairs(self):
        pairs = []
        for folder_name in ["01", "02"]:
            image_dir = self.root / folder_name
            mask_dir_name = f"{folder_name}_GT/SEG" if "SIM+" in self.name else f"{folder_name}_ST/SEG"
            mask_dir = self.root / mask_dir_name
            image_paths = sorted(path for path in image_dir.iterdir() if path.is_file())
            mask_paths = sorted(path for path in mask_dir.iterdir() if path.is_file())
            if len(image_paths) != len(mask_paths):
                raise ValueError(f"Length mismatch between {image_dir} and {mask_dir}")
            pairs.extend(VolumePair(image_path, mask_path) for image_path, mask_path in zip(image_paths, mask_paths))
        return pairs


class SpheroidNrrdSource(DatasetSource):
    def __init__(self, anisotropy="High", root=Path("data/12spheroids")):
        self.name = "12spheroids"
        self.anisotropy = anisotropy
        self.output_name = f"12spheroids_{anisotropy}"
        self.root = resolve_path(root)

    def get_pairs(self):
        if self.anisotropy == "High":
            image_suffix = "_expanded_3.nrrd"
            mask_suffix = "_expanded_3_DT.nrrd"
        elif self.anisotropy == "Low":
            image_suffix = "_spheroid.nrrd"
            mask_suffix = "_GT.nrrd"
        else:
            raise ValueError('anisotropy must be either "High" or "Low"')

        image_paths = sorted((self.root / "spheroids").glob(f"*{image_suffix}"))
        mask_paths = sorted((self.root / "GT").glob(f"*{mask_suffix}"))
        if len(image_paths) != len(mask_paths):
            raise ValueError("Length mismatch between 12spheroids images and masks")
        return [VolumePair(image_path, mask_path) for image_path, mask_path in zip(image_paths, mask_paths)]

    def read_volume(self, pair):
        image, _ = nrrd.read(pair.image_path)
        mask, _ = nrrd.read(pair.mask_path)
        image = image.transpose(2, 0, 1)
        mask = mask.transpose(2, 0, 1)

        target_depth = 190 if self.anisotropy == "High" else 64
        padding = target_depth - image.shape[0]
        if padding > 0:
            padding_front = padding // 2
            padding_back = padding - padding_front
            image = np.pad(image, ((padding_front, padding_back), (0, 0), (0, 0)), mode="constant")
            mask = np.pad(mask, ((padding_front, padding_back), (0, 0), (0, 0)), mode="constant")

        return image, mask


def get_dataset_source(dataset_name, anisotropy):
    benchmark_sources = ["ATAS", "C_elegans_nuclei", "Mouse-Skull", "Mouse-Organoid"]
    if dataset_name in benchmark_sources:
        return BenchmarkTiffSource(dataset_name)
    if dataset_name == "Fluo-N3DH-SIM+":
        return CellTrackingTiffSource(dataset_name)
    if dataset_name == "12spheroids":
        return SpheroidNrrdSource(anisotropy=anisotropy)
    raise ValueError(f"Unknown dataset {dataset_name}. Choose from {available_datasets()}")


def available_datasets():
    return ["ATAS", "C_elegans_nuclei", "Mouse-Skull", "Mouse-Organoid", "Fluo-N3DH-SIM+", "12spheroids"]


def main():
    args = parse_args()
    source = get_dataset_source(args.dataset, args.anisotropy)
    save_dir = create_new_dir_struct(source.output_name, args.output_root)

    # response = input(
    #     "WARNING: ARE YOU SURE YOU WANT TO RESHUFFLE TRAIN TEST AND VALIDATION SPLITS? Y/N\n"
    #     f"For dataset: {source.output_name}\n"
    #     f"Seed: {args.seed}\n"
    #     "Enter Y/N:    "
    # )
    # if response != "Y":
    #     exit()

    pairs = source.get_pairs()
    train_paths, test_paths, val_paths = train_test_val_split_on_paths(pairs, split=args.split, seed=args.seed)
    save_name_digits = 6

    print("-" * 50, "\nCREATING VAL DATASET, number of volumes: ", len(val_paths))
    create_dataset(val_paths, save_dir, source, type_="val", s=save_name_digits)
    print("-" * 50, "\nCREATING TEST DATASET, number of volumes: ", len(test_paths))
    create_dataset(test_paths, save_dir, source, type_="test", s=save_name_digits)
    print("-" * 50, "\nCREATING TRAIN DATASET, number of volumes: ", len(train_paths))
    create_dataset(train_paths, save_dir, source, type_="train", s=save_name_digits)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate 2D Mask R-CNN slices from 3D cell segmentation datasets.")
    parser.add_argument("--dataset", choices=available_datasets(), default=DEFAULT_DATASET)
    parser.add_argument("--seed", type=int, default=SPLIT_SEED)
    parser.add_argument("--split", nargs=3, type=float, default=SPLIT, metavar=("TRAIN", "TEST", "VAL"))
    parser.add_argument("--anisotropy", choices=["High", "Low"], default="High")
    parser.add_argument("--output-root", type=Path, default=DATASETS_ROOT)
    return parser.parse_args()


def create_new_dir_struct(dataset_name, output_root):
    new_path = resolve_path(output_root) / dataset_name

    subdirs = ["train/imgs", "train/masks", "test/imgs", "test/masks", "val/imgs", "val/masks"]
    for subdir in subdirs:
        (new_path / subdir).mkdir(parents=True, exist_ok=True)
    return new_path


def train_test_val_split_on_paths(pairs, split=SPLIT, seed=None):
    """Put the largest volumes in train, then split the remainder reproducibly."""
    if not np.isclose(sum(split), 1.0):
        raise ValueError(f"split must sum to 1.0, got {split}")

    sorted_pairs = sorted(pairs, key=lambda x: x.image_path.stat().st_size, reverse=True)
    n = len(sorted_pairs)
    train_end = int(n * split[0])
    test_count = int(n * split[1])

    train_paths = sorted_pairs[:train_end]
    remaining_paths = sorted_pairs[train_end:]

    rng = random.Random(seed)
    rng.shuffle(remaining_paths)

    test_paths = remaining_paths[:test_count]
    val_paths = remaining_paths[test_count:]

    return train_paths, test_paths, val_paths


def create_dataset(file_paths, save_dir, source, type_, s):
    metadata = dict()
    next_file_id = 0
    for idx, path_pair in enumerate(tqdm(file_paths)):

        files_saved, next_file_id = make(path_pair, idx, next_file_id, save_dir, type_, source, s)
        metadata[idx] = files_saved
    
    metadata_file_path = Path(save_dir) / type_ / "metadata.json"
    print(metadata_file_path)
    json.dump(metadata, metadata_file_path.open("w", encoding="utf-8") )


def ensure_3d(volume):
    if volume.ndim == 2:
        return volume[np.newaxis, :, :]
    if volume.ndim == 3:
        return volume
    raise ValueError(f"Expected 2D or 3D volume, got shape {volume.shape}")


def resize_with_padding(img, target_size=512, is_mask=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_slices, height, width = img.shape

    img = torch.from_numpy(img).float().to(device)

    if height == target_size and width == target_size:
        return img.cpu()

    ratio = min(target_size / width, target_size / height)
    new_h = int(height * ratio)
    new_w = int(width * ratio)

    img_reshaped = img.unsqueeze(1).float()
    img_resized = F.interpolate(
        img_reshaped,
        size=(new_h, new_w),
        mode="nearest" if is_mask else "bilinear",
        align_corners=False if not is_mask else None,
    )

    pad_left = (target_size - new_w) // 2
    pad_right = target_size - new_w - pad_left
    pad_top = (target_size - new_h) // 2
    pad_bottom = target_size - new_h - pad_top

    img_padded = F.pad(
        img_resized,
        (pad_left, pad_right, pad_top, pad_bottom),
        mode="constant",
        value=0,
    )

    img_padded = img_padded.squeeze(1)
    if is_mask:
        img_padded = img_padded.int()

    del img, img_reshaped, img_resized
    return img_padded.cpu()


def get_target_from_mask(mask, image_id):
    """
    mask: torch.Tensor of shape [H, W]
    returns: target dict for Mask R-CNN
    """
    obj_ids = torch.unique(mask)
    obj_ids = obj_ids[obj_ids != 0]

    masks = (mask[None, :, :] == obj_ids[:, None, None]).to(torch.uint8)

    boxes = []
    for single_mask in masks:
        pos = torch.where(single_mask)
        if len(pos[0]) == 0:
            continue
        xmin, xmax = pos[1].min(), pos[1].max()
        ymin, ymax = pos[0].min(), pos[0].max()

        if xmax <= xmin or ymax <= ymin:
            continue
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
        "iscrowd": iscrowd,
        "orignal_mask": mask,
    }

    return target


def make(path_pair, volume_idx, next_file_id, save_dir, type_, source, s):
    image, mask = source.read_volume(path_pair)

    assert image.shape[0] == mask.shape[0], f"Mismatch between number of slices of mask and image for {path_pair}"
    image = resize_with_padding(image, is_mask=False)
    mask = resize_with_padding(mask, is_mask=True)
    assert image.shape[0] == mask.shape[0], f"Mismatch between number of slices of mask and image for {path_pair}"
    volume_depth = image.shape[0]

    files_saved = []
    for slice_idx in range(image.shape[0]):
        img_slice = image[slice_idx].cpu().numpy().copy()
        mask_slice = mask[slice_idx]

        save_slice_worker((img_slice, mask_slice, next_file_id, save_dir, type_, s))
        files_saved.append(next_file_id)
        next_file_id += 1
    return files_saved, next_file_id

def save_slice_worker(args):
    img_slice, mask_slice, file_id, save_dir, type_, s = args
    save_as_2d_slice(slice_data=img_slice, file_id=file_id, save_dir=save_dir, type_=type_)
    target = get_target_from_mask(mask=mask_slice, image_id=file_id)
    save_target(target, file_id=file_id, type_=type_, save_dir=save_dir)

def save_as_2d_slice(slice_data, file_id, save_dir, type_, s):
    filepath = Path(save_dir) / type_ / "imgs" / f"{str(file_id).zfill(s)}.npy"
    np.save(filepath, slice_data)


def save_target(target, file_id, type_, save_dir, s):
    filepath = Path(save_dir) / type_ / "masks" / f"{str(file_id).zfill(s)}.npz"

    target = {
        "boxes": target["boxes"].cpu().numpy(),
        "labels": target["labels"].cpu().numpy(),
        "masks": target["masks"].cpu().numpy(),
        "image_id": target["image_id"].cpu().numpy(),
        "area": target["area"].cpu().numpy(),
        "iscrowd": target["iscrowd"].cpu().numpy(),
        "orignal_mask": target["orignal_mask"].cpu().numpy(),
    }
    np.savez_compressed(filepath, **target)


if __name__ == "__main__":
    main()
