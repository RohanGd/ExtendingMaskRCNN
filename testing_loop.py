'''
usage python testing_loop.py num_slices_per_batch, num_epochs
set data dir inside file
'''
from emrDataset import emrDataset, emrCollate_fn
from emrmodel.extended_mask_rcnn import ExtendedMaskRCNN
from emrMetrics import emrMetrics
import sys
import torch
from torch.utils.data import DataLoader
from datetime import datetime

import configparser
import os
import logging
import warnings

logger = logging.getLogger(__name__)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
logger.setLevel(logging.INFO)
logger.addHandler(console_handler)

config = configparser.ConfigParser()
if len(sys.argv) > 1:
    config_file = os.path.join(os.path.dirname(__file__), 'config', sys.argv[1])
else:
    config_file = os.path.join(os.path.dirname(__file__), 'config', 'test_sim+epoch10.ini')

config.read(config_file)
test_imgs_dir = config.get('DATASET', 'imgs_dir')
test_masks_dir = config.get('DATASET', 'masks_dir')
saved_models_dir = config.get('DATASET', 'saved_models_dir')
model_num_epochs = config.getint('MODEL', 'start_epoch')
num_slices_per_batch = config.getint('MODEL', 'num_slices_per_batch')
batch_size = config.getint('TESTING', 'batch_size')
generator = torch.manual_seed(42)

test_dataset = emrDataset(imgs_dir=test_imgs_dir, masks_dir=test_masks_dir, n=num_slices_per_batch, load_from_cache=True)
dataset_name = test_dataset.dataset_name
test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True, collate_fn=emrCollate_fn, generator=generator)
logger.info(f"Number of items in test dataset: {len(test_dataloader)} with batch_size {batch_size}. Total slices: {len(test_dataset)}")

# load the model
model = ExtendedMaskRCNN(n=num_slices_per_batch)
if os.path.exists(f"{saved_models_dir}/model_epochs_{model_num_epochs}_dataset_{dataset_name}.pt"):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        model = torch.load(f"{saved_models_dir}/model_epochs_{model_num_epochs}_dataset_{dataset_name}.pt")
        logger.info(f"Loaded model from {saved_models_dir}/model_epochs_{model_num_epochs}_dataset_{dataset_name}.pt")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device=device)
model.eval()

# testing loop
start_time = datetime.now()
with torch.no_grad():

    metrics = emrMetrics()
    i = 0
    for images, targets in test_dataloader:
        images = images.to(device)
        targets = [{k:v.to(device) for k, v in t_dict.items()} for t_dict in targets]
        
        preds = model(images)
        metrics.update(preds, targets)

        i += 1
        logger.info(f"Progress {i} / {len(test_dataloader)}")

    logger.info(metrics)
    metrics.save(path=f"model_epochs_{model_num_epochs}_dataset_{dataset_name}.txt")
    logger.info(f"Saved results to results/model_epochs_{model_num_epochs}_dataset_{dataset_name}.txt")
end_time = datetime.now()
logger.info(f"TIME TAKEN: {end_time - start_time}")