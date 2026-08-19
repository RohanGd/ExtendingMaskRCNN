from training_loop import train_emr
from testing_loop import test_emr
import configparser
import os
import sys

try:
    config_file = sys.argv[1]
except IndexError:
    raise Exception("NO CONFIG FILE SPECIFIED IN THE ARGS")

model_ckpt = train_emr(config_file)
train_exp_dir = os.path.dirname(model_ckpt)

# Build a config for the test phase that points at the checkpoint training just
# produced, WITHOUT touching the original config_file on disk (which stays reusable/
# reproducible with its own originally-authored ckpt_path/backbone_ckpt_path). The
# resolved config is written into the training experiment's own output folder.
config = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
config.read(config_file)
config.set("LOOP", "ckpt_path", str(model_ckpt))
# The produced checkpoint is always a full, already-fused model -- never a bare
# backbone -- so it must be loaded via ckpt_path, not backbone_ckpt_path (ModelBuilder
# raises if both are set).
if config.has_option("LOOP", "backbone_ckpt_path"):
    config.set("LOOP", "backbone_ckpt_path", "")

test_config_file = os.path.join(train_exp_dir, "test_config.ini")
with open(test_config_file, "w") as f:
    config.write(f)

print(f"--- Training complete. Starting Test Phase with: {model_ckpt} ---")
print(f"--- Resolved test config written to: {test_config_file} ---")

test_emr(test_config_file)

print("Pipeline complete!")
