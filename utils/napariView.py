import tifffile
import napari
import sys


if len(sys.argv) == 1:
        
    # Load the tif stacks
    img = tifffile.imread("Fluo-N3DH-SIM+/02/t050.tif")          # raw data, shape (59, 343, 649)
    img= tifffile.imread("Fluo-N3DH-SIM+_joined/val/imgs/002.tif")
    seg = tifffile.imread("Fluo-N3DH-SIM+/02_GT/SEG/man_seg050.tif")      # segmentation mask, same shape
    seg= tifffile.imread("Fluo-N3DH-SIM+_joined/val/masks/002.tif")


    # Launch napari viewer
    viewer = napari.Viewer()

    # Add raw image layer with contrast/gamma settings
    viewer.add_image(
        img,
        name="cells",
        colormap="gray",
        opacity=1.0,
        rendering="attenuated_mip",
        contrast_limits=(100, 150),  #almost removes noise (contrast clipping NOT stretching)
        gamma=1.2
    )

    # Add segmentation as a label layer
    # https://napari.org/stable/api/napari.layers.Labels.html
    # change countour = 3 thickness once opened. napari.viewer doesnt support contour parameter change napari.layer.Label next
    viewer.add_labels(
        seg,
        name="segmentation",
        opacity=1.0, # since contour will be set
    )

    viewer.layers[1].contour = 3 

    napari.run()

else:
    img = tifffile.imread(sys.argv[1])
    viewer = napari.Viewer()

    viewer.add_image(
        img,
        name="cells",
        colormap="gray",
        opacity=1.0,
        rendering="attenuated_mip",
    )

    napari.run()