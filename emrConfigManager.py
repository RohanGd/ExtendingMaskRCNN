import os
import configparser
import time
import logging
from json import dumps as json_dumps
import pandas as pd
from torch import Tensor
from pathlib import Path

# Detect the Pegasus cluster by its mounts rather than by hostname, so any
# machine without these mounts (laptop, CI, ...) falls back to local paths.
IS_CLUSTER = os.path.isdir("/netscratch") and os.path.isdir("/ds")

REPO_ROOT = Path(__file__).resolve().parent

# /netscratch/gadgil: scratch space for temp files, experiment results, checkpoints. No backup.
# See https://pegasus.dfki.de/docs/guidelines/storage/
# Hardcoded rather than os.environ["USER"]: jobs on Pegasus run inside a SLURM/enroot
# container as root, so $USER resolves to "root" rather than the submitting user, and
# only /netscratch/gadgil is bind-mounted (see slurm/submit.sh), not /netscratch/root.
NETSCRATCH_PATH = Path("/netscratch/gadgil") if IS_CLUSTER else REPO_ROOT

# Raw downloaded/unzipped source data, staged before dataset generation.
DATA_PATH = NETSCRATCH_PATH / "data" if IS_CLUSTER else REPO_ROOT / "data"

# Generated 2D-slice datasets (output of datasetGenerator.py, input to emrDataset/emrDataloader).
# /ds is a read-only dataset share on the cluster (see https://pegasus.dfki.de/docs/guidelines/storage/:
# "dataset shares, read from here, don't write") -- jobs mount it :ro, so datasets we generate
# ourselves live in netscratch instead, alongside other per-user experiment output.
DATASETS_PATH = NETSCRATCH_PATH / "datasets" if IS_CLUSTER else REPO_ROOT / "datasets"


class emrConfigManager:
    def __init__(self, config_file):
        if not os.path.exists(config_file):
            raise FileNotFoundError(f"Config file not found: {config_file}")

        self.config = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
        self.config.read(config_file)
        self.path = config_file   
        self.as_dict = lambda: json_dumps({
            section: dict(self.config.items(section))
            for section in self.config.sections()
        }, indent=4)
     

    # ----- Generic getters -----
    def get(self, section, key, fallback=None):
        if self.config.has_option(section, key):
            return self.config.get(section, key)
        return fallback

    def get_int(self, section, key, fallback=None):
        if self.config.has_option(section, key):
            return self.config.getint(section, key)
        return fallback

    def get_float(self, section, key, fallback=None):
        if self.config.has_option(section, key):
            return self.config.getfloat(section, key)
        return fallback

    def get_bool(self, section, key, fallback=None):
        if self.config.has_option(section, key):
            return self.config.getboolean(section, key)
        return fallback

    # ----- Helpers for path creation -----
    def ensure_dir(self, section, key):
        path = self.get(section, key)
        if path and not os.path.isdir(path):
            os.makedirs(path)
        return path


def create_experiment_folder(cfg, mode):
    """Creates an experiment folder

    Args:
        cfg (emrConfigManager): emrConfigManager object for the configfile 
        mode (str): "train" | "test" | "val"

    Returns:
        exp_dir, name, log_file
    """
    root = NETSCRATCH_PATH / "Experiments"
    name = cfg.get("EXPERIMENT", "exp_name")

    t = time.strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join(root, mode, f"{name}_{t}")

    os.makedirs(exp_dir, exist_ok=True)

    # Save config for reproducibility
    cfg_copy = os.path.join(exp_dir, cfg.path.split('/')[-1])
    with open(cfg_copy, "w") as f:
        with open(cfg.path, "r") as orig:
            f.write(orig.read())

    log_file = os.path.join(exp_dir, f"{mode}.log")  

    os.makedirs(os.path.join(exp_dir, "pred_masks"), exist_ok=True)
    
    return exp_dir, name, log_file

def setup_logger(log_path, name="TRAIN"):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    fh = logging.FileHandler(log_path)
    sh = logging.StreamHandler()

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] [%(filename)s:%(lineno)d] -- %(message)s")
    fh.setFormatter(fmt)
    sh.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger

class fusion_weights_logger():
    def __init__(self):
        pass

    def set(self, cfg, exp_dir):
        self.exp_dir = exp_dir
        self.cfg = cfg
        self.columns = ["iter", "iter_batch", "logits_dynamic", "logits_static", "slice_weights", "weights_shape"]
        self.rows = list()
        self.iter = 0
    
    def log(self, logits_dynamic: Tensor, logits_static: Tensor, slice_weights: Tensor, weights_shape):
        B = weights_shape[0]
        for b in range(B):
            self.rows.append({
                'iter': self.iter,
                'iter_batch': b,
                'logits_dynamic': logits_dynamic[b].cpu().numpy(),
                'logits_static': logits_static.cpu().numpy(),
                'slice_weights': slice_weights[b].cpu().numpy(),
                'weights_shape': weights_shape
            })
            self.iter += 1
    
    def save(self):
        try:
            df = pd.DataFrame(self.rows, columns=self.columns)
            df.to_pickle(f"{self.exp_dir}/fusion_weights.pkl")
            return f"Fusion weights saved at {self.exp_dir}/fusion_weights.pkl"
        except Exception as e:
            return f"Exception during saving fusion weights: {type(e).__name__} - {e}"
    

Fusion_Logger = fusion_weights_logger()
