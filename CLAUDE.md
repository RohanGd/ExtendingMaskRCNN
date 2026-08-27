# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

A research codebase extending torchvision's Mask R-CNN to do **2.5D instance segmentation** of cells/nuclei in 3D microscopy volumes. Instead of segmenting a full 3D volume, the model takes `n` consecutive 2D slices (a window centered on a target slice) and predicts instance masks for just the center slice. All custom classes are prefixed `emr` (Extended Mask-RCNN) to avoid name clashes with torchvision's own classes.

The core research question is *how to fuse information across the `n` neighboring slices*, and the codebase implements several competing fusion strategies at two different points in the network (see Architecture below).

## Commands

Train a model:
```
python training_loop.py config/<file>.ini
```

Test a model (reads `ckpt_path` from the config):
```
python testing_loop.py config/<file>.ini
```

Train then test in one go (writes the resulting checkpoint path back into the config file's `ckpt_path`, then runs testing):
```
python run_train_test_pipeline.py config/<file>.ini
```

Generate a 2D-slice dataset from raw 3D volumes (run once per dataset before training):
```
python datasetGenerator.py --dataset {ATAS,C_elegans_nuclei,Mouse-Skull,Mouse-Organoid,Fluo-N3DH-SIM+,12spheroids} [--anisotropy High|Low] [--seed 42] [--split 0.8 0.2] [--val_from_test_fraction 0.2]
```
Reads raw data from `DATA_PATH`, writes slices/masks/`metadata.json` to `DATASETS_PATH/<dataset_name>/{train,test,val}/`.

Post-hoc metrics (manual, not wired into `testing_loop.py` beyond SEG score and `3d_semantic_iou`):
```
./SEGMeasure {experiment_folder_path} 01 4                 # official SEG score CLI tool
python utils/store2d_masks.py                               # hardcode exp_dir inside the file first
python utils/metrics2diou.py                                # 2D IoU / slicewise SEG, needs store2d_masks output folder
python utils/metrics_acc_X.py                                # Acc@X, needs store2d_masks output folder
```

There is no formal test suite / CI. `utils/unit_tests.py` is a set of ad-hoc manual scripts (some reference a `Fluo-N3DH-CHO` dataset and dataset-loading params that no longer exist) — treat it as illustrative, not runnable as-is.

Environment: Python 3.11, PyTorch/torchvision with CUDA 12.6 wheels (`requirements.txt`, `Dockerfile`). On the DFKI Pegasus SLURM cluster, jobs run inside a container (`slurm/submit.sh`, `slurm/run`) — see Paths below for how storage differs there.

`presentation/Tracking_experiments.xlsx` is the maintainer's manually-updated ledger of which checkpoint (path under `NETSCRATCH_PATH/Experiments/train/...`) corresponds to which dataset/experiment combination — check it before wiring a `ckpt_path`/`backbone_ckpt_path` into a config, since checkpoint directories are timestamped and not otherwise discoverable from the repo.

### Procedure: wiring up testing after a training run finishes on the cluster

The maintainer trains on the DFKI cluster, where Claude has no filesystem access, so checkpoint discovery is a manual handoff. After a batch of `slurm/{base,base_swin,channelFusion,earlyFusion,lateFusion}/*.sh` training jobs finishes:

1. Run `bash slurm/list_fusion_checkpoints.sh [Experiments/train dir]` on the cluster (or an equivalent `find`/tree listing for `base`/`channelFusion` runs — see that script for the pattern) and paste the output back.
2. Update `presentation/Tracking_experiments.xlsx`: for each dataset/experiment combination, use the **last-epoch checkpoint** (`epoch<N>.pt` where `N` is the highest epoch present in that run's directory), not `best_model_epoch<N>.pt` — this was an explicit maintainer preference. Preserve every existing cell exactly (read the sheet with explicit `ws.cell(row=r, column=c)` lookups, never positional tuples from `iter_rows()`, since row offsets shift whenever a new experiment-type row is inserted into a dataset block — a prior mistake here silently landed a value in the wrong dataset's row).
3. Create/update a matching config under `config/test/<family>/` for each entry: copy the training config, and replace whatever loads a checkpoint with a single `ckpt_path` pointing at the chosen checkpoint —
   - `base`/`base_swin`/`channelFusion` configs already use `ckpt_path` for training; just repoint it.
   - `earlyFusion`/`lateFusion` configs use `backbone_ckpt_path` + `freeze_backbone = True` for training (warm-starting the per-slice backbone from a `base_n1` checkpoint, since `base_n1`'s single-channel backbone is architecturally identical to the per-slice backbone these fusion variants use); for testing, delete both of those lines and add `ckpt_path` pointing at the fully-trained fusion checkpoint instead — `ckpt_path` and `backbone_ckpt_path` are mutually exclusive in `emrModelBuilder.py`.
4. Create matching `slurm/test/<family>/*.sh` sbatch jobs (mirror the existing training job template, job name prefixed `test_`, `--time=02:00:00` since testing has no backward pass) plus a `submit_all.sh`, and make sure `slurm/test/submit_all.sh` lists the new family.

## Architecture

### Data flow
1. **`datasetGenerator.py`** — converts raw 3D volumes (TIFF/NRRD, one `DatasetSource` subclass per raw format) into a flat directory of 2D slices. Volumes are resized/padded to a square (`resize_with_padding`), and every slice is saved as `imgs/<id>.npy` + `masks/<id>.npz` (boxes/labels/masks/area/iscrowd, produced by `get_target_from_mask`). A `metadata.json` maps volume index -> list of flat slice ids, because different datasets have inconsistent slice counts per volume — this replaced an older `volumeId_sliceId.tif` naming scheme that couldn't handle that (see `Docs/dataset_structure.md`).
   Data is split by **volume** only two ways, train and test (`train_test_split_on_paths`), puts the largest volumes in train (by file size) and shuffles the rest, reproducibly via `--seed`; it guarantees at least 1 volume in each of train/test. There is no dedicated val split of whole volumes — `create_val_from_test_slices` instead populates `val/` by copying a random sample of already-generated **test slices** (`--val_from_test_fraction`, default 0.2), each written as its own singleton "volume" in `val/metadata.json` so `emrDataset` zero-pads their neighbor-slice context rather than pulling in an unrelated slice. This `val/` is only meant for `training_loop.py`'s per-epoch `validation()` (val_loss for early stopping, optionally a 2D metric) — it is not a substitute for evaluating on the held-out `test/` set, since its slices are drawn from the same volume(s) as test.
2. **`emrDataset`** (`emrDataset.py`) — a `torch.utils.data.Dataset` over flat slice ids. For index `idx` it looks up the id's volume bounds in `metadata.json` (`__get_volume_bounds__`) and returns the `n` consecutive slices centered on `idx` (padding with a near-zero slice at volume edges) plus the target dict for the center slice only.
3. **`DataloaderBuilder`** (`emrDataloader.py`) — builds `emrDataset` + `DataLoader` from an `emrConfigManager`, using `emrCollate_fn` to stack images into `[B, n, H, W]` and keep targets as a list of per-sample dicts.
4. **`ExtendedMaskRCNN`** (`emrmodel/extended_mask_rcnn.py`) consumes `[B, n, H, W]` and predicts masks for the center slice only.

### Config-driven experiments
Everything (dataset, model, fusion strategy, training hyperparameters) is driven by an `.ini` file (see `config/*.ini`), parsed by `emrConfigManager`. `emrModelBuilder.ModelBuilder` reads the `[MODEL]` section into `ExtendedMaskRCNN` kwargs and optionally loads a checkpoint from `[LOOP] ckpt_path`. `training_loop.py`/`testing_loop.py`/`run_train_test_pipeline.py` are the entry points; `emrConfigManager.create_experiment_folder` copies the config into a timestamped experiment dir under `Experiments/{train,test}/` for reproducibility, and `Fusion_Logger` (module-level singleton) records fusion-weight diagnostics there.

### The two fusion points
This is the crux of the "extending" in the project name — two independent places where the `n`-slice stack can be fused into one, controlled by separate config knobs:

**A. Early fusion — inside the backbone**, config key `early_mlp_fusion` (`None|Global|GlobalPerFPN|Windowed|Pixel|PixelPerFPN`), implemented in `emrmodel/early_mlp_fusion.py`. When set, the backbone becomes `Stacked_Resnet50FPN_Backbone` (`emrmodel/stacked_fpn_backbone.py`), which runs *each slice through its own single-channel ResNet50-FPN* and returns, per FPN level, a *list* of per-slice feature maps (instead of one fused tensor). A `SliceSEFusion`-family module (squeeze-and-excite over the slice dimension, optionally per-FPN-level or per-pixel/window) then collapses that list back to one `[B,C,H,W]` map before the RPN. `early_mlp_bias` sets the fixed prior added to the learned logits (`gaussian`, `only_center`, `zero`) — a way of biasing fusion toward the center slice. When `early_mlp_fusion=None` and `roi_heads_fusion=None`, fusion instead happens trivially at the input: the backbone's `conv1` is widened to `in_channels=num_slices_per_batch` and all slices are stacked as channels ("channel fusion") through a single shared-weight ResNet50-FPN.

**B. Late fusion — inside the mask head**, config key `roi_heads_fusion`, implemented in `emrmodel/mask_features_fusion.py` and consumed in `emrmodel/roi_heads.py` (~line 822, `self.mask_features_fusion(mask_features)`). Here RoI-pooled mask features have shape `[N, C, S, H, W]` (`S` = slices) and are collapsed via `mean`, `only_center`, `conv3d` (a `Conv3d` collapsing the slice dim), or `SE` (per-slice MLP + softmax, analogous to early fusion but on RoI features instead of full feature maps).

`emrmodel/{generalized_rcnn,faster_rcnn,mask_rcnn,rpn,roi_heads}.py` are forked/patched copies of the corresponding torchvision `torchvision/models/detection/*.py` modules, modified to plumb the fusion hooks through; when debugging RPN/RoI internals it's often useful to diff against the installed torchvision source (see paths noted in `readme.md`) to see exactly what changed.

### Backbone choice
Config key `backbone` in `[MODEL]` (default unset -> ResNet50-FPN). Setting `backbone = Swin` swaps in a Swin-T + FPN backbone (`emrmodel/swin_fpn_backbone.py`), used the same way ResNet50 is in both fusion regimes above: `SwinFPNBackbone` (channel-fusion path, `in_channels=num_slices_per_batch` on the patch-embed conv) or `Stacked_SwinFPN_Backbone` (per-slice single-channel Swin, shared weights, used when `early_mlp_fusion`/`roi_heads_fusion` is set — mirrors `Stacked_Resnet50FPN_Backbone`). Swin's `features` Sequential exposes per-stage outputs at indices `1,3,5,7` (channels `[96,192,384,768]`, strides `[4,8,16,32]`), which line up with ResNet50's `layer1..layer4` strides, so it drops into the same `FeaturePyramidNetwork` / `MultiScaleRoIAlign(featmap_names=["0","1","2","3"])` setup unmodified. Swin stage outputs are channel-last (`B,H,W,C`) internally and get permuted to `B,C,H,W` before the FPN. Unlike the ResNet path, the Swin backbone is not partially frozen via `trainable_backbone_layers` — it's always fully fine-tuned. See `config/swin_channelFusion.ini` for an example.

### Paths (`emrConfigManager.py`)
Path constants switch automatically between a local checkout and the DFKI Pegasus cluster, detected via `IS_CLUSTER = /netscratch and /ds both exist`:
- `NETSCRATCH_PATH` — scratch space for experiment outputs/checkpoints (`/netscratch/$USER` on cluster, else repo root).
- `DS_PATH` / `DATASETS_PATH` — shared, already-generated train/test/val slice datasets (`/ds/3d/cellular` on cluster, else `./datasets`).
- `DATA_PATH` — raw downloaded source volumes staged for `datasetGenerator.py` (`/netscratch/$USER/data` on cluster, else `./data`).
Don't hardcode `datasets/` or `data/` paths in new code — import these constants instead so the code keeps working both locally and on the cluster.

### Metrics
`emrMetrics.py` accumulates per-batch prediction/target comparisons during testing (`3d_semantic_iou`, etc.) and is intentionally the *first* thing to look at in test logs. Everything else (Acc@X, 2D IoU, slicewise SEG) requires the separate manual `utils/store2d_masks.py` -> `utils/metrics2diou.py` / `utils/metrics_acc_X.py` pipeline described above, plus the official `SEGMeasure` CLI tool for the SEG score — none of this is automated end-to-end.

## Notes from the maintainer

- `readme.md` doubles as a running lab notebook (results tables, dated TODOs, architecture deep-dives, open questions) — worth grep'ing when investigating *why* something is the way it is, but treat anything not near the top as possibly stale/superseded.
- `Docs/dataset_structure.md` documents the rationale for the current flat-slice + `metadata.json` dataset layout (replacing an earlier `volumeId_sliceId.tif` scheme) — relevant if touching `datasetGenerator.py` or `emrDataset.py`.
- Fixed-window early fusion (`early_mlp_fusion=Windowed`) performed badly and was effectively replaced by pixel-level fusion (`Pixel`); its window size is only configurable by editing `SliceSEFusionFixedWindow.__init__` directly in `emrmodel/early_mlp_fusion.py`, not via the `.ini` file.
