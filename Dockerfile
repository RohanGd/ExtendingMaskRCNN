FROM python:3.11-slim

# Tells the NVIDIA container runtime hook (used by Pyxis/enroot on the cluster)
# to inject the host GPU driver/devices into the container. Without these, the
# hook silently skips GPU injection and torch.cuda.is_available() is False even
# though SLURM allocated a GPU (CUDA_VISIBLE_DEVICES is set but /dev/nvidia* is not).
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Runtime libs needed by opencv/PyQt5/napari/vispy at import time
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libxkbcommon0 \
    libxcb-cursor0 \
    libegl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cu126
