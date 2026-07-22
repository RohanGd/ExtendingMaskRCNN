FROM python:3.11-slim

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
