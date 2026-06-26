
As of 12:17 26.06.2026 I was storing data files under `/dataset/dataset_name/test|train|val/imgs|masks` in the form `volumeId_sliceId.tif`, for eg., `0000_001.tif`. While this worked so far for Fluo-N3DH-SIM+, Fluo-N3DH-CHO and 12Spheroids, this strategy wont work for the newer datasets that I am incorporating such as Mouse-Skull, Mouse-Orgenoid etc. This is because the number of slices per volume is not consistent in these datasets. This breaks the dataloading logic in the emrDataset class. 

New plan: 

- Store files as simply: `id.tif`
- Store a metadata file that keeps record for each `id.tif` which orignal volume it belongs to. This may simply be a `volume: [ids]` dict.
- Similarly store the corresponding masks.

emrDataset simply loads the center slice and corresponding neighbour slices. This logic should check if the neighbour actually exists and does not include slices from other volumes. Use the dataset metadata to handle this.

Then the model always simply predicts the 2d mask for the center slice. Then we can have a pipeline that recreates the 3d volume as a sstack of 2d slices using the metadata.


Challenge:
`train_test_val_split_on_paths` in `datasetGenerator.py` is going to shuffle the files randomly. For the datasets for which the number of slices per volume are not consistent, we should have ideally the volumes with more number of slices in the train set. 
So replace the shuffle function with something that sorts by file size instead. 
