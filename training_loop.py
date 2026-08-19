'''
Training loop
python training_loop.py configfile
'''
import sys, os, subprocess
import torch
from datetime import datetime
from emrConfigManager import emrConfigManager, create_experiment_folder, setup_logger, Fusion_Logger, REPO_ROOT
from emrDataloader import DataloaderBuilder
from emrModelBuilder import ModelBuilder
from SEG_helper_functions import save_preds, make_files_for_SEG
from metrics.metrics_volume import emrMetricsVolume
from torch.utils.tensorboard import SummaryWriter
from multiprocessing import freeze_support

def train_emr(config_file):
    # set seed
    torch.manual_seed(42)

    cfg = emrConfigManager(config_file)
    exp_dir,exp_name, log_file = create_experiment_folder(cfg, mode="train")
    logger = setup_logger(log_file, name="train")
    logger.info(f"Experiment created at: {exp_dir}.\nUsing config file {config_file}\n")
    Fusion_Logger.set(cfg, exp_dir)

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
    num_epochs = cfg.get_int("LOOP", "num_epochs", 1)
    print_rate = cfg.get_int("LOOP", "print_rate", 100)
    patience = cfg.get_int("LOOP", "early_stopping_patience", num_epochs)
    best_val_loss = float("inf")
    epochs_without_improvement = 0
    best_ckpt_path = None


    writer = SummaryWriter(log_dir=exp_dir)
    total_params = sum([p.numel() for p in model.parameters()])
    print(f"Total model params: {total_params}")
    # training loop
    global_iterations = 0
    start_epoch = model_init.start_epoch
    end_epoch = start_epoch + num_epochs
    logger.info(f"Training model on dataset: {train_dataset.dataset_name} for {num_epochs} epochs (epochs {start_epoch + 1} to {end_epoch}).")
    for epoch in range(start_epoch, end_epoch):
        # training
        model.train()
        start_epoch_time = datetime.now()
        epoch_loss = 0
        print_rate_loss = 0
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
            print_rate_loss += loss.item()

            if global_iterations == 0:
                print(images.shape)
            global_iterations += 1
            iterations += 1

            if iterations % print_rate == 0:
                print_rate_loss = print_rate_loss / print_rate
                logger.info(f"Loss at epoch {epoch} at iteration {iterations}: {print_rate_loss:.4f}")
                print_rate_loss = 0
                # logger.info(f"per_slice_bias: {model.early_mlp_fusion_module.static_logits.tolist()}")#

            for loss_name, loss_value in loss_dict.items():
                writer.add_scalar(f'Loss/{loss_name}', loss_value.item(), global_iterations)
            writer.add_scalar("Total Loss", loss, global_iterations)
        
        writer.add_scalar("Epoch Loss", epoch_loss/len(train_dataloader), epoch)
        end_epoch_time = datetime.now()
        logger.info(f"Epoch {epoch+1}/{end_epoch}, Average Loss: {epoch_loss/len(train_dataloader):.4f}, Time for Epoch: {end_epoch_time - start_epoch_time}")
        ckpt_path = f"{exp_dir}/epoch{epoch+1}.pt"
        torch.save({
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch + 1,
        }, f=ckpt_path)
        logger.info(f"Model Saved at location: {ckpt_path}")

        validation_enabled = cfg.get_bool("LOOP", "VALIDATION", False)
        if validation_enabled:
            val_loss = validation(model, loader_builder, exp_dir, device, epoch, logger, writer)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_without_improvement = 0

                # filename embeds the epoch, so the previous best (if any) is removed
                # rather than overwritten in place
                if best_ckpt_path is not None and os.path.exists(best_ckpt_path):
                    os.remove(best_ckpt_path)
                best_ckpt_path = f"{exp_dir}/best_model_epoch{epoch+1}.pt"
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch + 1,
                }, best_ckpt_path)
                logger.info(f"New best model saved: {best_ckpt_path}")

            else:
                epochs_without_improvement += 1
                logger.info(f"No improvement for {epochs_without_improvement} epochs")

            if epochs_without_improvement >= patience:
                logger.info("Early stopping triggered")
                break

    # prefer the best-val-loss checkpoint (if early stopping/validation was used and
    # found one) over the last epoch's -- this is the checkpoint callers like
    # run_train_test_pipeline.py should test with
    if validation_enabled and best_ckpt_path is not None:
        return best_ckpt_path
    return ckpt_path

def validation(model, loader_builder, exp_dir, device, epoch, logger, writer):
    # validation
    model.eval()
    val_dataset, val_dataloader = loader_builder.build(mode='val')
    pred_masks_dir = os.path.join(exp_dir, "pred_masks")

    # erase contents of exp_dir/pred_masks but keep the empty folder
    subprocess.run([f"rm -rf {pred_masks_dir}/*"], shell=True)

    # then remove 01 rm -rf exp_dir/01
    subprocess.run(f"rm -rf {exp_dir}/01", shell=True)

    with torch.no_grad():
        for images, targets, in val_dataloader:
            images = images.to(device)
            preds = model(images)
            save_preds(preds, pred_masks_dir)
    
    make_files_for_SEG(exp_dir=exp_dir, target_masks_dir=loader_builder.masks_dir["val"], pred_masks_dir=pred_masks_dir)

    val_seg = emrMetricsVolume(exp_dir).seg_score()
    logger.info(f"VALIDATION SEG SCORE: {val_seg['mean']}")
    writer.add_scalar("Val_SEG", val_seg["mean"], epoch)

    model.train()
    val_loss_total = 0.0

    for images, targets in val_dataloader:
        images = images.to(device)
        targets = [{k: v.to(device) for k, v in t_dict.items()} for t_dict in targets]

        with torch.no_grad(): 
            loss_dict = model(images, targets)  
            loss = sum(loss for loss in loss_dict.values())
        
        val_loss_total += loss.item()

    val_loss_avg = val_loss_total / len(val_dataloader)
    logger.info(f"Validation Loss at Epoch {epoch}: {val_loss_avg}")
    writer.add_scalar("Val_loss", val_loss_avg, epoch)

    return val_loss_avg

if __name__ == "__main__":
    freeze_support()
    # load configs and setup logger
    try:
        config_file = sys.argv[1]
    except:
        raise Exception("NO CONFIG FILE SPECIFIED IN THE ARGS")
    train_emr(config_file)