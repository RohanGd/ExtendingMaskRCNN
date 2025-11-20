'''
usage python testing_loop.py num_slices_per_batch, num_epochs
set data dir inside file
'''
from emrMetrics import emrMetrics
import sys
import torch
from datetime import datetime
from emrConfigManager import emrConfigManager, create_experiment_folder, setup_logger
from emrDataloader import DataloaderBuilder
from emrModelBuilder import ModelBuilder

# load configs and setup logger
config_file = sys.argv[1] if len(sys.argv) > 1 else "config/test_sim+epoch10.ini"
cfg = emrConfigManager(config_file)
exp_dir, log_file = create_experiment_folder(cfg)
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

num_epochs = model_init.start_epochs

# testing loop
start_time = datetime.now()
with torch.no_grad():
    metrics = emrMetrics()
    i = 0
    for images, targets in test_dataloader:
        batch_start_time = datetime.now()
        images = images.to(device)
        targets = [{k:v.to(device) for k, v in t_dict.items()} for t_dict in targets]
        
        preds = model(images)
        metrics.update(preds, targets)

        i += 1
        logger.info(f"Progress {i} / {len(test_dataloader)} - Time taken = {datetime.now() - batch_start_time}")

    logger.info(metrics)
    metrics.save(path=f"{exp_dir}/results_model_epochs_{num_epochs}_dataset_{test_dataset.dataset_name}.txt")
    logger.info(f"Saved results to results/model_epochs_{num_epochs}_dataset_{test_dataset.dataset_name}.txt")
end_time = datetime.now()
logger.info(f"TIME TAKEN: {end_time - start_time}")