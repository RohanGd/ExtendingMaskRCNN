'''
Training loop
hyperparams: num_epochs, learning_rate, weight_decay, num_slices
python training_loop.py start_epochs, num_epochs, num_slices
'''
from emrDataset import emrDataset, emrCollate_fn
from emrmodel.extended_mask_rcnn import ExtendedMaskRCNN
import os
import sys
import logging
import torch
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
logger.setLevel(logging.INFO)
logger.addHandler(console_handler)


train_imgs_dir = "Fluo-N3DH-CHO/01"
# train_imgs_dir = "Fluo-N3DH-SIM+/01"
train_masks_dir = "Fluo-N3DH-CHO/01_ST/SEG"
# train_masks_dir = "Fluo-N3DH-SIM+/01_GT/SEG"
saved_models_dir = "saved_models"

if not os.path.isdir(saved_models_dir):
    os.makedirs(saved_models_dir)

start_epochs = 0 
num_epochs = 1
num_slices_per_batch = 3
if len(sys.argv) > 1:
    start_epochs, num_epochs, num_slices_per_batch = tuple(int(x) for x in sys.argv[1:4])

learning_rate = 1e-5
weight_decay = 1e-7

generator = torch.manual_seed(42)

train_dataset = emrDataset(imgs_dir=train_imgs_dir, masks_dir=train_masks_dir, n=num_slices_per_batch, load_from_cache=True)
dataset_name = train_dataset.dataset_name
train_dataloader = DataLoader(dataset=train_dataset, batch_size=4, shuffle=True, collate_fn=emrCollate_fn, generator=generator)

model = ExtendedMaskRCNN(n=num_slices_per_batch)
if start_epochs != 0:
    if os.path.exists(f"{saved_models_dir}/model_epochs_{start_epochs}_dataset_{dataset_name}.pt"):
        model = torch.load(f"{saved_models_dir}/model_epochs_{start_epochs}_dataset_{dataset_name}.pt")
        logger.info("Loaded model: " + f"{saved_models_dir}/model_epochs_{start_epochs}_dataset_{dataset_name}.pt")
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device=device)
model.train()

for epoch in range(start_epochs, start_epochs + num_epochs):
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
        if iterations % 10 == 0:
            print(f"Loss at epoch {epoch} at iteration {iterations}: {loss:.4f}")
    
    print(f"Epoch {epoch+1}/{start_epochs + num_epochs}, Average Loss: {epoch_loss/len(train_dataloader):.4f}")

torch.save(model, f=f"{saved_models_dir}/model_epochs_{start_epochs + num_epochs}_dataset_{dataset_name}.pt")




