#!/bin/bash
#SBATCH --job-name=base_n1_MouseOrganoid
#SBATCH --partition=RTX3090
#SBATCH --gpus=1
#SBATCH --mem=30G
#SBATCH --ntasks=1
#SBATCH --time=11:00:00
#SBATCH --cpus-per-task=8
#SBATCH --output=/netscratch/gadgil/runlogs/%x_%j.log

srun \
  --container-image=/netscratch/gadgil/extending_maskrcnn_v1.sqsh \
  --container-mounts=/home/gadgil/ExtendingMaskRCNN:/home/gadgil/ExtendingMaskRCNN,/netscratch/gadgil:/netscratch/gadgil,/ds:/ds:ro \
  --container-workdir=/home/gadgil/ExtendingMaskRCNN \
  python3 training_loop.py config/base_n1_MouseOrganoid.ini