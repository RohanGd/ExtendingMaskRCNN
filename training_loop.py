'''
Training loop
hyperparams: num_epochs, learning_rate, weight_decay, num_slices
python training_loop.py start_epochs, num_epochs, num_slices
'''
import warnings
from emrDataset import emrDataset, emrCollate_fn
from emrmodel.extended_mask_rcnn import ExtendedMaskRCNN
import os
import sys
import logging
import torch
from torch.utils.data import DataLoader
import datetime
import configparser

logger = logging.getLogger(__name__)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
logger.setLevel(logging.INFO)
logger.addHandler(console_handler)

## Load from config file
config = configparser.ConfigParser()
if len(sys.argv) > 1:
    config_file = os.path.join(os.path.dirname(__file__), 'config', sys.argv[1])
else:
    config_file = os.path.join(os.path.dirname(__file__), 'config', 'train_sim+epoch0+1.ini')
config.read(config_file)

print(config_file)
train_imgs_dir = config.get('DATASET', 'imgs_dir')
train_masks_dir = config.get('DATASET', 'masks_dir')
saved_models_dir = config.get('DATASET', 'saved_models_dir')
if not os.path.isdir(saved_models_dir):
    os.makedirs(saved_models_dir)

start_epochs = config.getint('MODEL', 'start_epoch')
num_slices_per_batch = config.getint('MODEL', 'num_slices_per_batch')

num_epochs = config.getint('TRAINING', 'num_epochs')
learning_rate = config.getfloat('TRAINING', 'learning_rate')
weight_decay = config.getfloat('TRAINING', 'weight_decay')
batch_size = config.getint('TRAINING', 'batch_size')

generator = torch.manual_seed(42)

## load dataset and dataloader
train_dataset = emrDataset(imgs_dir=train_imgs_dir, masks_dir=train_masks_dir, n=num_slices_per_batch, load_from_cache=True)
dataset_name = train_dataset.dataset_name
train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, collate_fn=emrCollate_fn, generator=generator)
logger.info(f"Number of items in train dataset: {len(train_dataloader)} with batch_size {batch_size}. Total slices: {len(train_dataset)}")

# load the model and optimizer
model = ExtendedMaskRCNN(n=num_slices_per_batch)
if start_epochs != 0:
    if os.path.exists(f"{saved_models_dir}/model_epochs_{start_epochs}_dataset_{dataset_name}.pt"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=FutureWarning)
            model = torch.load(f"{saved_models_dir}/model_epochs_{start_epochs}_dataset_{dataset_name}.pt")
            logger.info("Loaded model: " + f"{saved_models_dir}/model_epochs_{start_epochs}_dataset_{dataset_name}.pt")
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device=device)
model.train()

# training loop
for epoch in range(start_epochs, start_epochs + num_epochs):
    start_epoch_time = datetime.datetime.now()
    epoch_loss = 0
    iterations = 0
    for images, targets in train_dataloader:
        images = images.to(device)
        targets = [{k:v.to(device) for k, v in t_dict.items()} for t_dict in targets]
        loss_dict = model(images, targets) # {'loss_classifier': tensor, 'loss_box_reg': tensor, 'loss_rpn_box_reg': tensor}
        loss = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        iterations += 1
        if iterations % 100 == 0:
            logger.info(f"Loss at epoch {epoch} at iteration {iterations}: {loss:.4f}")
    
    end_epoch_time = datetime.datetime.now()
    logger.info(f"Epoch {epoch+1}/{start_epochs + num_epochs}, Average Loss: {epoch_loss/len(train_dataloader):.4f}, Time to loop: {end_epoch_time - start_epoch_time}")
    torch.save(model, f=f"{saved_models_dir}/model_epochs_{epoch}_dataset_{dataset_name}.pt")




