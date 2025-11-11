'''
usage python testing_loop.py num_slices_per_batch num_epochs
set data dir inside file
'''
from emrDataset import emrDataset, emrCollate_fn
from emrmodel.extended_mask_rcnn import ExtendedMaskRCNN
from emrMetrics import emrMetrics
import os
import sys
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import matplotlib.pyplot as plt

test_imgs_dir = "Fluo-N3DH-CHO/02"
test_masks_dir = "Fluo-N3DH-CHO/02_ST/SEG"
saved_models_dir = "saved_models"
if len(sys.argv > 1):
    num_slices_per_batch = sys.argv[1]
num_slices_per_batch = 3
generator = torch.manual_seed(42)

test_dataset = emrDataset(imgs_dir=test_imgs_dir, masks_dir=test_masks_dir, n=num_slices_per_batch, load_from_cache=False)
dataset_name = test_dataset.dataset_name
test_dataloader = DataLoader(dataset=test_dataset, batch_size=2, shuffle=True, collate_fn=emrCollate_fn, generator=generator)

model = ExtendedMaskRCNN(n=num_slices_per_batch)
if len(sys.argv) > 2:
    model_num_epochs = sys.argv[2]
    model = torch.load(f"{saved_models_dir}/model_epochs_{model_num_epochs}_dataset_{dataset_name}.pt")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
model = model.to(device=device)
model.eval()


with torch.no_grad():

    metrics = emrMetrics()

    for images, targets in test_dataloader:
        images = images.to(device)
        targets = [{k:v.to(device) for k, v in t_dict.items()} for t_dict in targets]
        
        preds = model(images)
        metrics.update(preds, targets)

    print(metrics)