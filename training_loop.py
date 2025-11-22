'''
Training loop
hyperparams: num_epochs, learning_rate, weight_decay, num_slices
python training_loop.py start_epochs, num_epochs, num_slices
'''
import sys
import torch
import datetime
from emrConfigManager import emrConfigManager, create_experiment_folder, setup_logger
from emrDataloader import DataloaderBuilder
from emrModelBuilder import ModelBuilder

# load configs and setup logger
config_file = sys.argv[1] if len(sys.argv) > 1 else "config/train_sim+epoch0+1.ini"
cfg = emrConfigManager(config_file)
exp_dir,exp_name, log_file = create_experiment_folder(cfg)
logger = setup_logger(log_file)
logger.info(f"Experiment created at: {exp_dir}.\nUsing config file {config_file}\n")

# set up dataset and dataloader
loader_builder = DataloaderBuilder(cfg, logger)
train_dataset, train_dataloader = loader_builder.build(mode="train")

# set up model and optimizer
model_init = ModelBuilder(cfg, logger)
model = model_init.load_model(train_dataset.dataset_name)
optimizer = model_init.build_optimizer(model)
generator = torch.manual_seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device=device)
model.train()

# looping params
start_epochs = model_init.start_epochs
num_epochs = cfg.get_int("LOOP", "num_epochs", 1)
save_path = cfg.get("LOOP", "save_path", str(datetime.datetime.now()))
print_rate = cfg.get_int("LOOP", "print_rate", 100)

# training loop
logger.info(f"Training model on dataset: {train_dataset.dataset_name} from {start_epochs} to {start_epochs + num_epochs} epochs.")
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
        if iterations % print_rate == 0:
            logger.info(f"Loss at epoch {epoch} at iteration {iterations}: {loss:.4f}")
    
    end_epoch_time = datetime.datetime.now()
    logger.info(f"Epoch {epoch+1}/{start_epochs + num_epochs}, Average Loss: {epoch_loss/len(train_dataloader):.4f}, Time for Epoch: {end_epoch_time - start_epoch_time}")
    ckpt_path = f"{save_path}({epoch+1}_outof_{start_epochs+num_epochs}).pt"
    torch.save(model, f=ckpt_path)
    logger.info(f"Model Saved at location: {ckpt_path}")