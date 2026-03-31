import tifffile
import napari
import sys
import numpy as np
from pathlib import Path


def load_data(img_path, mask_path=None):
    """Load image and optionally masks from various formats."""
    img_path = Path(img_path)
    
    # Load image
    if img_path.suffix == '.tif':
        img = tifffile.imread(img_path)
        if img.ndim == 4:
            img.squeeze(axis = 1)
    elif img_path.suffix == '.npy':
        img = np.load(img_path)
    else:
        raise ValueError(f"Unsupported image format: {img_path.suffix}")
    
    print(f"Loaded image: {img.shape}, {img.dtype}")
    
    # Load masks if provided
    masks = None
    if mask_path:
        mask_path = Path(mask_path)
        if mask_path.suffix == '.tif':
            masks = tifffile.imread(mask_path)
            if masks.ndim == 4:
                masks = masks.squeeze(axis=1)
        elif mask_path.suffix == '.npz':
            target_data = np.load(mask_path)
            masks = target_data['masks']  # Shape: (num_instances, H, W)
        else:
            raise ValueError(f"Unsupported mask format: {mask_path.suffix}")
        
        print(f"Loaded masks: {masks.shape}, {masks.dtype}")
    
    if mask_path == None:
        return img
    if img_path == None:
        return masks
    return img, masks



def create_label_image(masks, img):
    """Convert masks to label image matching image dimensions."""
    if masks is None or len(masks) == 0:
        return None
    
    # Case 1: masks is already a label volume (Z, H, W) - from .tif
    if masks.ndim == 3 and masks.shape == img.shape:
        return masks
    
    # Case 2: masks is 2D label image (H, W) - from single slice .tif
    if masks.ndim == 2 and img.ndim == 2:
        return masks
    
    # Case 3: masks is instance masks (num_instances, H, W) - from .npz
    if masks.ndim == 3:
        # Image is 2D slice - create 2D label image
        if img.ndim == 2:
            label_image = np.zeros(img.shape, dtype=np.int32)
            for i, mask in enumerate(masks):
                label_image[mask > 0] = i + 1
            return label_image
        
        # Image is 3D volume - ERROR: can't match instance masks to 3D volume
        else:
            print("WARNING: Instance masks provided but image is 3D volume.")
            print("Cannot determine which masks belong to which slice.")
            return None
    
    return None


def visualize(img, masks=None, contour_width=3):
    """Launch napari viewer with image and optional masks."""
    viewer = napari.Viewer()
    
    # Add image layer
    viewer.add_image(
        img,
        name="cells",
        colormap="gray",
    )
    
    # Add masks if available
    if masks is not None:
        label_image = create_label_image(masks, img)
        
        if label_image is not None:
            viewer.add_labels(label_image, name='Masks')
            if len(viewer.layers) > 1:
                viewer.layers[-1].contour = contour_width
    
    napari.run()


def visualize2(masks):
    viewer = napari.Viewer()
    
    colormaps = ("grey", "bop blue", "bop orange", "bop purple", "cyan", "green")
    for i, mask in enumerate(masks):
        print(mask.shape)
    # Add image layer
        viewer.add_image(
            mask,
            name=str(i),
            colormap=colormaps[i],
            opacity=0.5
        )
    
    napari.run()


if __name__ == "__main__":
    
    # # SIM+n3
    # paths = [
    # "/home/rohan/Dev/ExtendingMaskRCNN/datasets/Fluo-N3DH-SIM+/test/masks/0003_044.npz",
    # "/home/rohan/Dev/ExtendingMaskRCNN/data/SIM+_n3/lateFusion_onlyCenter_20260327_091016/renamed_preds/0003_044.tif",
    # "/home/rohan/Dev/ExtendingMaskRCNN/data/SIM+_n3/channelFusion_20260327_090820/renamed_preds/0003_044.tif",
    # ]

    # # SIM+n5
    # paths = [
    #     "/home/rohan/Dev/ExtendingMaskRCNN/datasets/Fluo-N3DH-SIM+/test/masks/0003_044.npz",
    #     "/home/rohan/Dev/ExtendingMaskRCNN/data/SIM+_n5/channelFusion_20260328_214504/renamed_preds/0003_044.tif",
    #     "/home/rohan/Dev/ExtendingMaskRCNN/data/SIM+_n5/lateFusion_onlyCenter_20260328_222937/renamed_preds/0003_044.tif",
    #     "/home/rohan/Dev/ExtendingMaskRCNN/data/SIM+_n5/earlyFusion_GlobalGaussian_20260328_214504/renamed_preds/0003_044.tif"
    # ]

    # # 12spheroids
    # paths = [
    #     "/home/rohan/Dev/ExtendingMaskRCNN/datasets/12spheroids_Low/test/masks/0000_035.npz",
    #     "/home/rohan/Dev/ExtendingMaskRCNN/Experiments/test/12spheroids_Low_n3/channelFusion_20260330_145609/renamed_preds/0000_035.tif",
    #     "/home/rohan/Dev/ExtendingMaskRCNN/Experiments/test/12spheroids_Low_n3/earlyFusion_GlobalGaussian_20260330_145748/renamed_preds/0000_035.tif",
    #     "/home/rohan/Dev/ExtendingMaskRCNN/Experiments/test/12spheroids_Low_n3/lateFusion_onlyCenter_20260330_145714/renamed_preds/0000_035.tif"
    # ]


    # masks = [load_data(path) for path in paths[1:]]
    # masks = [mask.any(axis=0) for mask in masks]

    # mask0 = paths[0]
    # mask0 = np.load(mask0)["masks"]
    # mask0 = mask0.any(axis=0)
    # masks.insert(0, mask0)

    # visualize2(masks)
    if len(sys.argv) == 1:
        # Default options - uncomment the pair you want to use
        
        # Option 1: Original .tif files
        # img_path = "data/Fluo-N3DH-SIM+/02/t050.tif"
        # mask_path = "data/Fluo-N3DH-SIM+/02_GT/SEG/man_seg050.tif"
        
        # Option 2: Processed .npy/.npz files (active)
        img_path = "/home/rohan/Dev/ExtendingMaskRCNN/datasets/Fluo-N3DH-SIM+/test/imgs/0008_018.npy"
        mask_path = "/home/rohan/Dev/ExtendingMaskRCNN/data/SIM+_n5/channelFusion_20260328_214504/renamed_preds/0008_018.tif"
        # mask_path = "/home/rohan/Dev/ExtendingMaskRCNN/datasets/Fluo-N3DH-SIM+/test/masks/0008_018.npz"
        img, masks = load_data(img_path, mask_path)
        visualize(img, masks)

    
    elif len(sys.argv) == 2:
        # Single argument: image only
        img, _ = load_data(sys.argv[1])
        visualize(img)
    
    elif len(sys.argv) == 3:
        # Two arguments: image and mask
        img, masks = load_data(sys.argv[1], sys.argv[2])
        visualize(img, masks)
    
    else:
        print("Usage:")
        print("  python script.py                          # Use default paths")
        print("  python script.py <img_path>               # View image only")
        print("  python script.py <img_path> <mask_path>   # View image + masks")
        sys.exit(1)