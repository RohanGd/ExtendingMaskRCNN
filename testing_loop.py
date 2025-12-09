'''
usage python testing_loop.py configfilepath
set dataset_name inside file
'''
from emrMetrics import emrMetrics
import sys
import torch
from datetime import datetime
from emrConfigManager import emrConfigManager, create_experiment_folder, setup_logger
from emrDataloader import DataloaderBuilder
from emrModelBuilder import ModelBuilder
from multiprocessing import freeze_support

def main():
    torch.manual_seed(42) 
    # load configs and setup logger
    try:
        config_file = sys.argv[1]
    except:
        raise Exception("NO CONFIG FILE SPECIFIED IN THE ARGS")
    cfg = emrConfigManager(config_file)
    exp_dir, exp_name, log_file = create_experiment_folder(cfg, mode="test")
    logger = setup_logger(log_file)
    logger.info(f"Experiment created at: {exp_dir}.\nUsing config file {config_file}\n")

    # set up dataset and dataloader
    loader_builder = DataloaderBuilder(cfg, logger)
    test_dataset, test_dataloader = loader_builder.build(mode="test")

    # set up model and optimizer
    model_init = ModelBuilder(cfg, logger)
    model = model_init.load_model(test_dataset.dataset_name)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device=device)
    model.eval()

    print_rate = cfg.get_int("LOOP", "print_rate", 100)
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
            metrics.update(preds, targets)

            i += 1
            if i % print_rate == 0:
                logger.info(f"Progress {i} / {len(test_dataloader)} - Time taken = {datetime.now() - batch_start_time}")
                batch_start_time = datetime.now()


        logger.info(metrics)
        metrics_results_save_path=f"{exp_dir}/test_metrics.txt"
        metrics.save(path=metrics_results_save_path)
        logger.info(f"Saved results to {metrics_results_save_path}")
    end_time = datetime.now()
    logger.info(f"TIME TAKEN: {end_time - start_time}")

if __name__ == "__main__":
    freeze_support()
    main()