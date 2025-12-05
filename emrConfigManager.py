import os
import configparser
import time
import logging
from json import dumps as json_dumps


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
    root = "Experiments"
    name = cfg.get("EXPERIMENT", "exp_name")

    t = time.strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join(root, mode, f"{name}_{t}")

    os.makedirs(exp_dir, exist_ok=True)

    # Save config for reproducibility
    cfg_copy = os.path.join(exp_dir, cfg.path.split('/')[-1])
    with open(cfg_copy, "w") as f:
        with open(cfg.path, "r") as orig:
            f.write(orig.read())

    log_file = os.path.join(exp_dir, "train.log")        

    return exp_dir, name, log_file

def setup_logger(log_path):
    logger = logging.getLogger("TRAIN")
    logger.setLevel(logging.INFO)

    fh = logging.FileHandler(log_path)
    sh = logging.StreamHandler()

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] [%(filename)s:%(lineno)d] -- %(message)s")
    fh.setFormatter(fmt)
    sh.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger