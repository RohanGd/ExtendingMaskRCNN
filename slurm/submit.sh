#!/bin/bash
#SBATCH --job-name=emaskrcnn
#SBATCH --partition=batch
#SBATCH --gpus=1
#SBATCH --mem=30G
#SBATCH --ntasks=1
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=4
#SBATCH --output=/netscratch/gadgil/Emaskrcnn_%j.log

srun --container-image=/netscratch/gadgil/extending_maskrcnn_v1.sqsh \
     --container-mounts=/home/gadgil/ExtendingMaskRCNN:/home/gadgil/ExtendingMaskRCNN,/netscratch/gadgil:/netscratch/gadgil,/ds:/ds:ro \
     --container-workdir="/home/gadgil/ExtendingMaskRCNN" \
     python3 testing_loop.py config/early_mlp_experiments/slicese_only_center.ini