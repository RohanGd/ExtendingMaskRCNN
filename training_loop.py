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
from SEG_helper_functions import save_preds, make_files_for_SEG
from torch.utils.tensorboard import SummaryWriter
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
    num_epochs = cfg.get_int("LOOP", "num_epochs", 1)
    print_rate = cfg.get_int("LOOP", "print_rate", 100)

    writer = SummaryWriter(log_dir=exp_dir)
    total_params = sum([p.numel() for p in model.parameters()])
    print(f"Total model params: {total_params}")
    # training loop
    global_iterations = 0
    logger.info(f"Training model on dataset: {train_dataset.dataset_name} for {num_epochs} epochs.")
    for epoch in range(num_epochs):
        # training
        model.train()
        start_epoch_time = datetime.now()
        epoch_loss = 0
        print_rate_loss = 0
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
            if global_iterations % print_rate == 0:
                print_rate_loss = print_rate_loss / print_rate
                logger.info(f"Loss at epoch {epoch} at iteration {global_iterations%len(train_dataloader)}: {print_rate_loss:.4f}")
                print_rate_loss = 0
                # logger.info(f"per_slice_bias: {model.early_mlp_fusion_module.static_logits.tolist()}")
            for loss_name, loss_value in loss_dict.items():
                writer.add_scalar(f'Loss/{loss_name}', loss_value.item(), global_iterations)
            writer.add_scalar("Total Loss", loss, global_iterations)
        
        writer.add_scalar("Epoch Loss", epoch_loss/len(train_dataloader), epoch)
        
        end_epoch_time = datetime.now()
        logger.info(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {epoch_loss/len(train_dataloader):.4f}, Time for Epoch: {end_epoch_time - start_epoch_time}")
        ckpt_path = f"{exp_dir}/({epoch+1}_of_{num_epochs}).pt"
        torch.save(model.state_dict(), f=ckpt_path)
        logger.info(f"Model Saved at location: {ckpt_path}")

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

        SEG_result = subprocess.run(["./SEGMeasure", f"{os.path.abspath(exp_dir)}", "01","4"], stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True)
        logger.info(f"VALIDATION SEG SCORE: {SEG_result.stdout.strip()}")
        writer.add_scalar("Val_SEG", SEG_result, epoch)
        
if __name__ == "__main__":
    freeze_support()
    main()