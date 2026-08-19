'''
usage python testing_loop.py configfilepath
set dataset_name inside file
'''
from metrics.metrics_2d import emrMetrics2D
from metrics.metrics_volume import emrMetricsVolume
import sys, os, json
import torch, numpy, tifffile
from datetime import datetime
from emrConfigManager import emrConfigManager, create_experiment_folder, setup_logger, Fusion_Logger, REPO_ROOT
from emrDataloader import DataloaderBuilder
from emrModelBuilder import ModelBuilder
from multiprocessing import freeze_support
from SEG_helper_functions import make_files_for_SEG, save_preds
from tqdm import tqdm

def test_emr(config_file):
    torch.manual_seed(42)
    # load configs and setup logger
    cfg = emrConfigManager(config_file)
    exp_dir, exp_name, log_file = create_experiment_folder(cfg, mode="test")
    pred_masks_dir = os.path.join(exp_dir, "pred_masks")

    logger = setup_logger(log_file, name="test")
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

    overlap_thresholds = cfg.get_float_list("METRICS", "overlap_thresholds", [0.5, 0.75, 0.9])
    acc_thresholds = cfg.get_float_list("METRICS", "acc_thresholds", [0.3, 0.5])
    score_agg = cfg.get("METRICS", "volume_score_agg", "mean")
    run_2d = cfg.get_bool("METRICS", "run_2d_metrics", True)
    run_volume = cfg.get_bool("METRICS", "run_volume_metrics", True)

    # testing loop
    start_time = datetime.now()
    metrics_2d = emrMetrics2D(overlap_thresholds=overlap_thresholds)
    with torch.no_grad():
        for images, targets in tqdm(test_dataloader):
            images = images.to(device)
            targets = [{k:v.to(device) for k, v in t_dict.items()} for t_dict in targets]

            preds = model(images)
            # save the pred masks (+ per-instance scores) in exp_dir/pred_masks
            save_preds(preds, pred_masks_dir)

            if run_2d:
                metrics_2d.update(preds, targets)

    make_files_for_SEG(exp_dir=exp_dir, target_masks_dir=loader_builder.masks_dir["test"],
                        pred_masks_dir=pred_masks_dir, score_agg=score_agg)
    logger.info(Fusion_Logger.save())

    results = {}
    if run_2d:
        results["2d"] = metrics_2d.compute()
        logger.info(metrics_2d)
        metrics_2d.save(path=f"{exp_dir}/test_metrics_2d.txt")

    if run_volume:
        metrics_vol = emrMetricsVolume(exp_dir, overlap_thresholds=overlap_thresholds, acc_thresholds=acc_thresholds)
        results["volume"] = metrics_vol.compute()
        logger.info(metrics_vol)
        metrics_vol.save(path=f"{exp_dir}/test_metrics_volume.txt")

    with open(f"{exp_dir}/test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Saved results to {exp_dir}/test_metrics_2d.txt, test_metrics_volume.txt, test_metrics.json")
    end_time = datetime.now()
    logger.info(f"TIME TAKEN: {end_time - start_time}")

if __name__ == "__main__":
    freeze_support()
    try:
        config_file = sys.argv[1]
    except:
        raise Exception("NO CONFIG FILE SPECIFIED IN THE ARGS")
    test_emr(config_file)
