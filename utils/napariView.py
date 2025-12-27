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
        elif mask_path.suffix == '.npz':
            target_data = np.load(mask_path)
            masks = target_data['masks']  # Shape: (num_instances, H, W)
        else:
            raise ValueError(f"Unsupported mask format: {mask_path.suffix}")
        
        print(f"Loaded masks: {masks.shape}, {masks.dtype}")
    
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


if __name__ == "__main__":
    if len(sys.argv) == 1:
        # Default options - uncomment the pair you want to use
        
        # Option 1: Original .tif files
        # img_path = "data/Fluo-N3DH-SIM+/02/t050.tif"
        # mask_path = "data/Fluo-N3DH-SIM+/02_GT/SEG/man_seg050.tif"
        
        # Option 2: Processed .npy/.npz files (active)
        img_path = "datasets/Fluo-N3DH-SIM+/train/imgs/vol_0019/054.npy"
        mask_path = "datasets/Fluo-N3DH-SIM+/train/masks/vol_0019/054.npz"
        
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