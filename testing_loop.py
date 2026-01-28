'''
usage python testing_loop.py configfilepath
set dataset_name inside file
'''
from emrMetrics import emrMetrics
import sys, os, subprocess
import torch, numpy, tifffile
from datetime import datetime
from emrConfigManager import emrConfigManager, create_experiment_folder, setup_logger, Fusion_Logger
from emrDataloader import DataloaderBuilder
from emrModelBuilder import ModelBuilder
from multiprocessing import freeze_support
from SEG_helper_functions import make_files_for_SEG, save_preds
from tqdm import tqdm

def main():
    torch.manual_seed(42) 
    # load configs and setup logger
    try:
        config_file = sys.argv[1]
    except:
        raise Exception("NO CONFIG FILE SPECIFIED IN THE ARGS")
    cfg = emrConfigManager(config_file)
    exp_dir, exp_name, log_file = create_experiment_folder(cfg, mode="test")
    pred_masks_dir = os.path.join(exp_dir, "pred_masks")

    logger = setup_logger(log_file)
    logger.info(f"Experiment created at: {exp_dir}.\nUsing config file {config_file}\n")
    Fusion_Logger.set(cfg, exp_dir)

    # set up dataset and dataloader
    loader_builder = DataloaderBuilder(cfg, logger)
    test_dataset, test_dataloader = loader_builder.build(mode="test")

    # set up model and optimizer
    model_init = ModelBuilder(cfg, logger)
    model = model_init.load_model(test_dataset.dataset_name)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device=device)
    model.eval()

    only_SEG_score = cfg.get_bool("LOOP", "only_SEG_score", 1)
    # testing loop
    start_time = datetime.now()
    metrics = emrMetrics()
    with torch.no_grad():
        for images, targets in tqdm(test_dataloader):
            images = images.to(device)
            targets = [{k:v.to(device) for k, v in t_dict.items()} for t_dict in targets]
            
            preds = model(images)
            # save the pred masks in exp_dir/pred_masks
            save_preds(preds, pred_masks_dir)

            if only_SEG_score == False: # this is slow, avoid if you only need SEG score
                metrics.update(preds, targets)

    make_files_for_SEG(exp_dir=exp_dir, target_masks_dir=loader_builder.masks_dir["test"], pred_masks_dir=pred_masks_dir)
    Fusion_Logger.save()
    logger.info(metrics)
    metrics_results_save_path=f"{exp_dir}/test_metrics.txt"
    metrics.save(path=metrics_results_save_path)
    logger.info(f"Saved results to {metrics_results_save_path}")
    end_time = datetime.now()
    logger.info(f"TIME TAKEN: {end_time - start_time}")

    SEG_result = subprocess.run(["./SEGMeasure", f"{os.path.abspath(exp_dir)}", "01","4"], stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True)
    logger.info(f"SEG score CLI Tool: {SEG_result.stdout.strip()}")

if __name__ == "__main__":
    freeze_support()
    main()
