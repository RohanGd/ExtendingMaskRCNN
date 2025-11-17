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

# test_imgs_dir = "Fluo-N3DH-CHO/02"
test_imgs_dir = "Fluo-N3DH-SIM+/01"
# test_masks_dir = "Fluo-N3DH-CHO/02_ST/SEG"
test_masks_dir = "Fluo-N3DH-SIM+/01_GT/SEG"
saved_models_dir = "saved_models"
if len(sys.argv) > 1:
    model_num_epochs, num_slices_per_batch = tuple(int(x) for x in sys.argv[1:3])
num_slices_per_batch = 3
generator = torch.manual_seed(42)

test_dataset = emrDataset(imgs_dir=test_imgs_dir, masks_dir=test_masks_dir, n=num_slices_per_batch, load_from_cache=False)
dataset_name = test_dataset.dataset_name
test_dataloader = DataLoader(dataset=test_dataset, batch_size=2, shuffle=True, collate_fn=emrCollate_fn, generator=generator)

model = ExtendedMaskRCNN(n=num_slices_per_batch)
if len(sys.argv) > 2:
    model = torch.load(f"{saved_models_dir}/model_epochs_{model_num_epochs}_dataset_{dataset_name}.pt")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device, len(test_dataloader))
model = model.to(device=device)
model.eval()


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
        print(f"Progress {i} / {len(test_dataloader)}")

    print(metrics)
    metrics.save(path=f"results_model_epochs_{model_num_epochs}_dataset_{dataset_name}.txt")
end_time = datetime.now()
print(f"TIME TAKEN: {end_time - start_time}")