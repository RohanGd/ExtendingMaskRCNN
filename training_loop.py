'''
Training loop
python training_loop.py configfile
'''
import sys, os, subprocess
import torch
from datetime import datetime
from emrConfigManager import emrConfigManager, create_experiment_folder, setup_logger
from emrDataloader import DataloaderBuilder
from emrModelBuilder import ModelBuilder
from testing_SEG_helper_functions import save_preds, make_files_for_SEG

from multiprocessing import freeze_support

def main():
    # set seed
    torch.manual_seed(42)
    # load configs and setup logger
    try:
        config_file = sys.argv[1]
    except:
        raise Exception("NO CONFIG FILE SPECIFIED IN THE ARGS")
    cfg = emrConfigManager(config_file)
    exp_dir,exp_name, log_file = create_experiment_folder(cfg, mode="train")
    logger = setup_logger(log_file)
    logger.info(f"Experiment created at: {exp_dir}.\nUsing config file {config_file}\n")

    # set up dataset and dataloader
    loader_builder = DataloaderBuilder(cfg, logger)
    train_dataset, train_dataloader = loader_builder.build(mode="train")

    # set up model and optimizer
    model_init = ModelBuilder(cfg, logger)
    model = model_init.load_model(train_dataset.dataset_name)
    optimizer = model_init.build_optimizer(model)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device=device)
    model.train()

    # looping params
    start_epochs = model_init.start_epochs
    num_epochs = cfg.get_int("LOOP", "num_epochs", 1)
    print_rate = cfg.get_int("LOOP", "print_rate", 100)
    logger.info(model.early_mlp_fusion_module.static_logits.tolist())

    # training loop
    logger.info(f"Training model on dataset: {train_dataset.dataset_name} from {start_epochs} to {start_epochs + num_epochs} epochs.")
    moving_average_batch_time = 0
    for epoch in range(start_epochs, start_epochs + num_epochs):
        # training
        model.train()
        start_epoch_time = datetime.now()
        epoch_loss = 0
        iterations = 0
        for images, targets in train_dataloader:
            batch_start_time = datetime.now()
            images = images.to(device)
            targets = [{k:v.to(device) for k, v in t_dict.items()} for t_dict in targets]
            loss_dict = model(images, targets) # {'loss_classifier': tensor, 'loss_box_reg': tensor, 'loss_rpn_box_reg': tensor}
            loss = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            if iterations == 0:
                print(images.shape)
            iterations += 1
            if iterations % print_rate == 0:
                print_rate_time = datetime.now()
                average_batch_time = (print_rate_time - batch_start_time).total_seconds() / print_rate
                logger.info(f"Loss at epoch {epoch} at iteration {iterations}: {loss:.4f}, Avg batch time: {average_batch_time:.4f} seconds")
                logger.info(f"per_slice_bias: {model.early_mlp_fusion_module.static_logits.tolist()}")
                batch_start_time = datetime.now()
                moving_average_batch_time += average_batch_time
        
        end_epoch_time = datetime.now()
        logger.info(f"Epoch {epoch+1}/{start_epochs + num_epochs}, Average Loss: {epoch_loss/len(train_dataloader):.4f}, Time for Epoch: {end_epoch_time - start_epoch_time}, Moving Avg Batch Time: {moving_average_batch_time / (iterations // print_rate + 1):.4f} seconds")
        ckpt_path = f"{exp_dir}/({epoch+1}_of_{start_epochs+num_epochs}).pt"
        torch.save(model.state_dict(), f=ckpt_path)
        logger.info(f"Model Saved at location: {ckpt_path}")

        # validation
        model.eval()
        val_dataset, val_dataloader = loader_builder.build(mode='val')
        pred_masks_dir = os.path.join(exp_dir, "pred_masks")

        # erase contents of exp_dir/pred_masks but keep the empty folder
        if os.path.exists(pred_masks_dir):
            subprocess.run(["rm", "-rf", f"{pred_masks_dir}/*"])
        else:
            os.makedirs(pred_masks_dir)

        # then remove 01 rm -rf exp_dir/01
        subprocess.run(["rm", "-rf", f"{exp_dir}/01"])

        with torch.no_grad():
            for images, targets, in val_dataloader:
                images = images.to(device)
                targets = [{k:v.to(device) for k, v in t_dict.items()} for t_dict in targets]
                preds = model(images)
                save_preds(preds, pred_masks_dir)
        
        make_files_for_SEG(exp_dir=exp_dir, target_masks_dir=loader_builder.masks_dir["val"], pred_masks_dir=pred_masks_dir)

        SEG_result = subprocess.run(["./SEGMeasure", f"{os.path.abspath(exp_dir)}", "01","3"], stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True)
        logger.info(f"VALIDATION SEG SCORE: {SEG_result.stdout.strip()}")
        
if __name__ == "__main__":
    freeze_support()
    main()