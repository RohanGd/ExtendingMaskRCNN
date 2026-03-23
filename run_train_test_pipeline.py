from training_loop import train_emr
from testing_loop import test_emr
import configparser
import sys 

try:
    config_file = sys.argv[1]
except IndexError:
    raise Exception("NO CONFIG FILE SPECIFIED IN THE ARGS")

model_ckpt = train_emr(config_file)

config = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
config.read(config_file)
config.set('LOOP', 'ckpt_path', str(model_ckpt))

with open(config_file, 'w') as f:
    config.write(f)

print(f"--- Config updated. Starting Test Phase with: {model_ckpt} ---")

test_emr(config_file)

print("Pipeline complete!")