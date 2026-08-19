#!/bin/bash
# Regenerates 2D-slice datasets for all benchmark datasets via datasetGenerator.py.
# Dual-purpose: the #SBATCH lines below are no-ops for a plain `bash slurm/makedatasets.sh`
# (they're just comments to bash), so this same file also works as `sbatch slurm/makedatasets.sh`
# on the cluster.
#SBATCH --job-name=makedatasets
#SBATCH --partition=RTX3090
#SBATCH --gpus=1
#SBATCH --mem=30G
#SBATCH --ntasks=1
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --output=/netscratch/gadgil/runlogs/%x_%j.log

run() {
  if [ -n "$SLURM_JOB_ID" ]; then
    srun \
      --container-image=/netscratch/gadgil/extending_maskrcnn_v1.sqsh \
      --container-mounts=/home/gadgil/ExtendingMaskRCNN:/home/gadgil/ExtendingMaskRCNN,/netscratch/gadgil:/netscratch/gadgil,/ds:/ds:ro \
      --container-workdir=/home/gadgil/ExtendingMaskRCNN \
      python3 datasetGenerator.py "$@"
  else
    python3 datasetGenerator.py "$@"
  fi
}

run --dataset ATAS
run --dataset C_elegans_nuclei
run --dataset Mouse-Skull
run --dataset Mouse-Organoid
run --dataset Fluo-N3DH-SIM+
run --dataset 12spheroids --anisotropy High
run --dataset 12spheroids --anisotropy Low
