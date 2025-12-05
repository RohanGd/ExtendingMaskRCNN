import tifffile as tiff
from pprint import pprint
import pandas as pd

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print(sys.path)

def study_empty_slice_counts(file_paths):
    good_slice_start_counts, good_slice_end_counts = list(), list()
    for img_path in file_paths:
        img = tiff.imread(img_path)
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
    pprint([{val: x.count(val) for val in set(x)} for x in (good_slice_start_counts, good_slice_end_counts)])


def study_size_distribution(file_paths):
    size_x, size_y, size_z = dict(), dict(), dict()
    for folder_path in file_paths:
        for path in os.listdir(folder_path):
            img_tif = tiff.TiffFile(os.path.join(folder_path, path))
            for page in img_tif.pages:
                x, y = page.shape
                size_x.update({x: size_x.get(x, 0)+1})
                size_y.update({y: size_y.get(y, 0)+1})
            size_z.update({len(img_tif.pages): size_z.get(len(img_tif.pages), 0) + 1})
    return size_x, size_y, size_z


dataset_path = "data/Fluo-N3DH-SIM+"

pprint(study_size_distribution((
    "data/Fluo-N3DH-SIM+/01", 
    "data/Fluo-N3DH-SIM+/02", 
    "data/Fluo-N3DH-SIM+/01_ERR_SEG", 
    "data/Fluo-N3DH-SIM+/02_ERR_SEG",
    "data/Fluo-N3DH-CHO/01", 
    "data/Fluo-N3DH-CHO/02", 
    "data/Fluo-N3DH-CHO/01_ERR_SEG", 
    "data/Fluo-N3DH-CHO/02_ERR_SEG",
    )))
