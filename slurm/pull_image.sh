#!/bin/bash
#SBATCH --job-name=emrcnn_pull_image
#SBATCH --ntasks=1
#SBATCH --time=00:30:00
#SBATCH --output=/netscratch/gadgil/runlogs/Emaskrcnn_pull_image_%j.log

# One-off: pull the project image from Docker Hub and import it as the .sqsh
# used by submit.sh. Run this whenever you push a new image version locally.
#
# Usage: sbatch slurm/pull_image.sh

IMAGE=rohangd/extending_maskrcnn_v1:latest
DEST=/netscratch/gadgil/extending_maskrcnn_v1.sqsh

enroot import -o "$DEST" "docker://${IMAGE}"

echo "Imported ${IMAGE} -> ${DEST}"
