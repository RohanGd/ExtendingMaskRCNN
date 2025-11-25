import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print(sys.path)

from emrMetrics import emrMetrics
import sys
import torch
from datetime import datetime
from emrConfigManager import emrConfigManager, create_experiment_folder, setup_logger
from emrDataloader import DataloaderBuilder
from emrModelBuilder import ModelBuilder
import os, tifffile
import matplotlib.pyplot as plt
import numpy as np




# load configs and setup logger
config_file = sys.argv[1] if len(sys.argv) > 1 else "config/test_sim+epoch10.ini"
cfg = emrConfigManager(config_file)
exp_dir, exp_name, log_file = create_experiment_folder(cfg)
logger = setup_logger(log_file)
logger.info(f"Experiment created at: {exp_dir}.\nUsing config file {config_file}\n")

# set up dataset and dataloader
loader_builder = DataloaderBuilder(cfg, logger)
test_dataset, test_dataloader = loader_builder.build(mode="test")

# set up model and optimizer
model_init = ModelBuilder(cfg, logger)
model = model_init.load_model(test_dataset.dataset_name)
optimizer = model_init.build_optimizer(model)
generator = torch.manual_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device=device)
model.eval()


def plot_all_masks_for_slice(masks,save_fname, scores=None):
# Example: random volume with shape (n_masks, h, w)
    n_masks, h, w = masks.shape
    if n_masks == 0: return

    if scores == None:
        scores = torch.ones((n_masks))

    # Compute grid size (rows, cols) for subplots
    cols = int(np.ceil(np.sqrt(n_masks)))
    rows = int(np.ceil(n_masks / cols)) + 1

    fig, axes = plt.subplots(rows, cols, figsize=(2*cols, 2*rows))

    # Flatten axes for easy iteration
    axes = axes.flatten()

    combined_mask = torch.zeros(size=(h, w), device=device, dtype=torch.bool)
    for i in range(n_masks):
        axes[i].imshow(masks[i].cpu(), cmap='gray')
        axes[i].set_title(f"Mask {i} score: {scores[i]:.2f}")
        axes[i].axis('off')
        combined_mask = combined_mask | (masks[i] > 0.5)
    
    axes[n_masks].imshow(combined_mask.cpu(), cmap = 'grey')
    axes[n_masks].set_title("All masks combined")
    # Hide any unused subplots
    for j in range(i+1, len(axes)):
        axes[j].axis('off')
    plt.tight_layout()
    plt.title(save_fname.split('/')[-1])
    plt.savefig(save_fname, dpi=300, bbox_inches="tight")
    plt.close(fig)

    # plt.show()

def save_targets_prediction_masks(targets, preds):
    assert len(targets) == len(preds)
    b = len(targets)
    for i in range(b):
        target, pred = targets[i], preds[i]
        target_masks, pred_masks = target["masks"], pred["masks"]
        pred_masks = pred_masks.squeeze(1)
        print(target_masks.shape, pred_masks.shape)
        idx = target["image_id"].item()
        if not os.path.exists(f"{exp_dir}/visualize_results/"):
            os.mkdir(f"{exp_dir}/visualize_results/")

        plot_all_masks_for_slice(target_masks, f"{exp_dir}/visualize_results/{idx}_gt.png")
        plot_all_masks_for_slice(pred_masks, f"{exp_dir}/visualize_results/{idx}_pred.png", pred["scores"])

print("box_score_thresh:", model.roi_heads.score_thresh)
print("box_nms_thresh:", model.roi_heads.nms_thresh)
print("rpn_nms_thresh:", model.rpn.nms_thresh)
print("detections_per_img:", model.roi_heads.detections_per_img)
# inspect what parameters the loaded model has
print("roi_heads attrs:", [a for a in dir(model.roi_heads) if "nms" in a or "score" in a or "thresh" in a])
print("rpn attrs:", [a for a in dir(model.rpn) if "nms" in a or "top_n" in a or "pre" in a or "post" in a])


# model.roi_heads.score_thresh = 0.2
# model.roi_heads.nms_thresh = 0.3
# model.rpn_nms_thresh = 0.5

# logger.info(f"Inference with: box_score_thresh: {model.roi_heads.score_thresh}, box_nms_thresH ")


num_epochs = model_init.start_epochs
print_rate = cfg.get_int("LOOP", "print_rate", 10)
# testing loop
start_time = datetime.now()
with torch.no_grad():
    metrics = emrMetrics()
    i = 0
    batch_start_time = datetime.now()
    for images, targets in test_dataloader:
        images = images.to(device)
        targets = [{k:v.to(device) for k, v in t_dict.items()} for t_dict in targets]
        
        preds = model(images)

        save_targets_prediction_masks(targets, preds)
        # metrics.update(preds, targets)

        i += 1
        if i % print_rate == 0:
            logger.info(f"Progress {i} / {len(test_dataloader)} - Time taken = {datetime.now() - batch_start_time}")
            batch_start_time = datetime.now()
            break


end_time = datetime.now()
logger.info(f"TIME TAKEN: {end_time - start_time}")