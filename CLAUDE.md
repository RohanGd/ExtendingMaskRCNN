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

Train then test in one go (trains, then tests the checkpoint just produced):
```
python run_train_test_pipeline.py config/<file>.ini
```
It does **not** modify the config you pass in. It writes a resolved `test_config.ini` (with
`ckpt_path` set to the new checkpoint and `backbone_ckpt_path` cleared) into the training run's
own `Experiments/train/.../` folder and tests from that, so the original config stays reusable.
Which checkpoint gets tested depends on `VALIDATION` (`training_loop.py:133-135`): with
`VALIDATION = True` it returns `best_model_epoch<N>.pt`, otherwise the last epoch's
`epoch<N>.pt`. Since the tracking sheet records last-epoch checkpoints, configs driven by the
pipeline set `VALIDATION = False`.

Batch train+test on the cluster (one job per config, ATAS + C_elegans_nuclei across
`base`, `base_swin`, `channelFusion` n3/n5, `channelFusion_swin` n3/n5):
```
bash slurm/pipeline/submit_all.sh          # or one family: slurm/pipeline/<family>/submit_all.sh
```
These run `run_train_test_pipeline.py`, so unlike `slurm/test/*` they need no pre-wired
`ckpt_path` — nothing to backfill from the cluster afterwards except the tracking sheet.

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

## Working with the DFKI Pegasus cluster

### Access

`ssh pegasus1` (alias for `login1.pegasus.kl.dfki.de`) is key-based and works non-interactively, so
Claude can drive the cluster directly:
```
ssh -o BatchMode=yes pegasus1 '<command>'
```
Use `-o BatchMode=yes` so a missing key fails fast instead of hanging on a password prompt.

| what | where |
|---|---|
| cluster checkout | `/home/gadgil/ExtendingMaskRCNN` |
| experiment outputs | `/netscratch/gadgil/Experiments/{train,test}/<dataset>/<run>/` |
| job logs | `/netscratch/gadgil/runlogs/<jobname>_<jobid>.log` |
| container image | `/netscratch/gadgil/extending_maskrcnn_v1.sqsh` |
| datasets (read-only) | `/ds/3d/cellular` |

**The local checkout and the cluster checkout are separate, and git does not bridge them.**
`.gitignore` excludes `/config`, `/slurm` and `/presentation`, i.e. every config, every sbatch
script and the tracking workbook. Only Python sources sync via git. To change a config or an
sbatch script on the cluster, apply the *same targeted edit* over ssh (a `sed`/python line edit)
rather than `scp`-ing whole files — the cluster copy may carry edits the local one doesn't, and
copying would silently clobber them. Afterwards print the changed lines back and confirm both
sides agree.

### Submitting and monitoring

```
ssh pegasus1 'cd /home/gadgil/ExtendingMaskRCNN && bash slurm/pipeline/<family>/submit_all.sh'
ssh pegasus1 'squeue -u $USER -o "%.10i %.36j %.9P %.8T %.10M %.12l %.24R"'
ssh pegasus1 'sacct -j <id> --format=JobID,JobName%40,State,Elapsed,Timelimit,MaxRSS,ExitCode'
ssh pegasus1 'tail -40 /netscratch/gadgil/runlogs/<jobname>_<jobid>.log'
```

- **Editing an sbatch script does not change a job already in the queue.** Queued jobs keep the
  `--time`/`--partition` they were submitted with. To change them: `scancel` then resubmit (or
  `scontrol update JobId=... TimeLimit=... Partition=...` for a pending job, which avoids losing
  queue position but cannot change anything inside the script or config).
- Before a bulk `scancel`, assert every target job is still `PENDING` — cancelling a running job
  throws away hours of compute.
- **Do not run the top-level `submit_all.sh` when part of the batch has already completed**; it
  resubmits everything. Submit per-family instead.
- Partitions (verified 2026-09-16 via `sinfo -o "%.14P %.12l %.20G %.6D"` and node `AvailableFeatures`):

  | partition | max time | GPU | notes |
  |---|---|---|---|
  | `batch` (default) | 3 days | RTX6000, 24 GB, 10/node x 3 | same VRAM as RTX3090 |
  | `RTX3090` | 3 days | 24 GB, 8/node x 3 | what everything used historically |
  | `A100-40GB` / `A100-80GB` | 3 days | 40/80 GB | for anything that won't fit in 24 GB |
  | `H100`, `H200`, `B200` | 1 day | large | short jobs only |

- Sizing `--time`: measured on this repo, ATAS `base_n1` took **2.2 h** (17 epochs, 5971 slices)
  and C_elegans `base_n1` **9.2 h** (35 epochs, 26202 slices); stacked-backbone fusion
  (`earlyFusion`/`lateFusion`) costs roughly 2.5-3x per epoch over channel fusion, which runs the
  backbone once. Derive a limit from the nearest measured run rather than copying an old value —
  smaller requests get backfilled sooner, which is usually why jobs sit in `(Priority)`.

### Procedure: wiring up a separate test run (training_loop.py, not the pipeline)

`run_train_test_pipeline.py` tests its own checkpoint, so `slurm/pipeline/*` jobs need none of
this. It is still required after a plain `slurm/{base,base_swin,channelFusion,channelFusion_swin,earlyFusion,lateFusion}/*.sh`
training batch:

1. List the finished checkpoints (`slurm/list_fusion_checkpoints.sh`) and pick the **last-epoch**
   `epoch<N>.pt` per run.
2. Record them in `Sheet1` of the workbook.
3. Create/update `config/test/<family>/<run>.ini` — copy the training config and give it a single
   `ckpt_path`:
   - `base`/`base_swin`/`channelFusion`/`channelFusion_swin` already use `ckpt_path` for training;
     just repoint it.
   - `earlyFusion`/`lateFusion` train with `backbone_ckpt_path` + `freeze_backbone = True` (warm
     starting the per-slice backbone from a `base_n1` checkpoint, whose single-channel backbone is
     architecturally identical to the per-slice one). For testing, **delete both lines** and add
     `ckpt_path` pointing at the fully-trained fusion checkpoint — `ckpt_path` and
     `backbone_ckpt_path` are mutually exclusive in `emrModelBuilder.py`.
4. Create matching `slurm/test/<family>/*.sh` jobs (training template, job name prefixed `test_`,
   `--time=02:00:00` since testing has no backward pass) plus a `submit_all.sh`, and list the family
   in `slurm/test/submit_all.sh`.

Note that changing `anchor_sizes` changes the shape of `rpn.head.cls_logits`, so a test config must
carry the *same* anchor settings as the run that produced its checkpoint, or `load_state_dict` fails.

### Procedure: getting cluster results into `presentation/Tracking_experiments.xlsx`

The workbook lives only on the local machine. Three sheets, three different jobs:

| sheet | one row per | key | role |
|---|---|---|---|
| `Results` | (dataset, experiment) | cols A+B | the summary table, 17 columns |
| `Results_raw` | run | (dataset, `run_dir`) | full 64-column dump; **keeps history** |
| `Sheet1` | (dataset, experiment) | cols B+C | checkpoint ledger -> full path of the `.pt` |

`Results` column map (verify it against an intact row before writing):
`C` `2d.SEG_score` · `D` `2d.jaccard_index` · `E` `2d.dice_score` · `F` `2d.aji` · `G` `2d.ap@0_50`
· `H` `2d.pq@0_50` · `I` `2d.mean_difference_in_count` · `J` `volume.volume_SEG_score` ·
`K` `volume.volume_mean_semantic_3d_iou` · `L` `volume.volume_aji` · `M` `volume.volume_ap@0_50` ·
`N` `volume.volume_pq@0_50` · `O` `volume.volume_acc@0_50` · `P` `volume.volume_acc@0_30` ·
`Q` run directory. A run with no metrics yet is a *placeholder row*: the text
`no test_metrics.json — test not run / failed` in `C`, `C:P` merged, run dir still in `Q`.

1. **Collect metrics** — capture stdout over ssh, never from a terminal paste:
   ```
   ssh pegasus1 'cd /home/gadgil/ExtendingMaskRCNN && DATASETS="ATAS C_elegans_nuclei" bash slurm/collect_test_metrics.sh'
   ```
   Pasted terminal output *loses column boundaries*: runs of tabs collapse and adjacent values
   merge (`0.004047550.918197`, `000` for three zero columns). If you are stuck with a paste,
   split only tokens that are not a valid `f"{v:.6g}"`/int rendering, prefer the fewest pieces,
   and validate each row with the redundancies the metrics carry: `2d.sq@T == 2d.mean_iou@T` and
   `2d.rq@T == 2d.instance_f1@T` for every threshold, plus 61 metric values per row.
2. **Collect checkpoints for `Sheet1`** — `collect_test_metrics.sh` reports *test* directories, not
   training ones, so the ledger needs a separate listing:
   ```
   ssh pegasus1 'bash /home/gadgil/ExtendingMaskRCNN/slurm/list_fusion_checkpoints.sh'
   ```
   Record the **last-epoch** `epoch<N>.pt` (highest N), never `best_model_epoch<N>.pt` — an explicit
   maintainer preference. Configs driven by `run_train_test_pipeline.py` set `VALIDATION = False`
   precisely so the tested checkpoint is the last epoch.
3. **Write the workbook** following the rules below, then report before/after for changed rows.

### Editing the workbook with openpyxl — read this before touching it

`presentation/` is gitignored, so **there is no git history to recover from**. Copy the file
aside before editing.

- **`insert_rows` does not move merged cell ranges.** They stay at their original row numbers while
  the data shifts underneath them, so placeholder `C:P` merges end up sitting on metric rows. On
  the *next load*, openpyxl blanks every cell of a merged range except the top-left — silently
  destroying columns D..P of a real result row. This has already happened once and cost 10 rows.
  Fix: unmerge every data-row merge first, do the inserts, then re-merge exactly the placeholder
  rows at their final positions.
- **Verify by saving to a temp file and reloading it from disk**, comparing against a snapshot
  taken before the edit. An in-memory check after editing cannot see load-time blanking, and a
  reload-then-compare that re-reads both sides passes vacuously. Only replace the real file once
  the reloaded copy checks out.
- **Key rows by (dataset, experiment), never by row index** — offsets shift whenever a row is
  inserted into an earlier block.
- `run_dir` alone is **not unique** across datasets (Fluo and Mouse-Skull both have a
  `base_n1_20260823_145415`); match on (dataset, run_dir).
- If `Results` is damaged, recover from `Results_raw` — it has no merged cells and so is never hit
  by this failure.
- Keep superseded runs in `Results_raw` when rerunning an experiment (append the new row next to
  the old), so before/after comparisons survive; `Results` shows the current run only.

## Architecture

### Data flow
1. **`datasetGenerator.py`** — converts raw 3D volumes (TIFF/NRRD, one `DatasetSource` subclass per raw format) into a flat directory of 2D slices. Everything downstream assumes axis 0 is the stack axis, but not every source stores volumes that way: `DatasetSource.read_volume` moves `slice_axis` to axis 0, and `SLICE_AXIS_BY_DATASET` records the exceptions. **C_elegans_nuclei is `(Y, X, Z) = (140, 140, 1244)`** — the L1 larva lies along axis 2 — and slicing it on axis 0 (the original behaviour) produced mostly-empty 140x1244 planes that `resize_with_padding` squashed to 58x512, shrinking 13px nuclei to 5x4px boxes. If a new dataset scores near zero, check the slice axis first: labelled planes per axis is the diagnostic. Volumes are resized/padded to a square (`resize_with_padding`), and every slice is saved as `imgs/<id>.npy` + `masks/<id>.npz` (boxes/labels/masks/area/iscrowd, produced by `get_target_from_mask`). A `metadata.json` maps volume index -> list of flat slice ids, because different datasets have inconsistent slice counts per volume — this replaced an older `volumeId_sliceId.tif` naming scheme that couldn't handle that (see `Docs/dataset_structure.md`).
   Data is split by **volume** only two ways, train and test (`train_test_split_on_paths`), puts the largest volumes in train (by file size) and shuffles the rest, reproducibly via `--seed`; it guarantees at least 1 volume in each of train/test. There is no dedicated val split of whole volumes — `create_val_from_test_slices` instead populates `val/` by copying a random sample of already-generated **test slices** (`--val_from_test_fraction`, default 0.2), each written as its own singleton "volume" in `val/metadata.json` so `emrDataset` zero-pads their neighbor-slice context rather than pulling in an unrelated slice. This `val/` is only meant for `training_loop.py`'s per-epoch `validation()` (val_loss for early stopping, optionally a 2D metric) — it is not a substitute for evaluating on the held-out `test/` set, since its slices are drawn from the same volume(s) as test.
2. **`emrDataset`** (`emrDataset.py`) — a `torch.utils.data.Dataset` over flat slice ids. For index `idx` it looks up the id's volume bounds in `metadata.json` (`__get_volume_bounds__`) and returns the `n` consecutive slices centered on `idx` (padding with a near-zero slice at volume edges) plus the target dict for the center slice only.
3. **`DataloaderBuilder`** (`emrDataloader.py`) — builds `emrDataset` + `DataLoader` from an `emrConfigManager`, using `emrCollate_fn` to stack images into `[B, n, H, W]` and keep targets as a list of per-sample dicts.
4. **`ExtendedMaskRCNN`** (`emrmodel/extended_mask_rcnn.py`) consumes `[B, n, H, W]` and predicts masks for the center slice only.

### Config-driven experiments
Everything (dataset, model, fusion strategy, training hyperparameters) is driven by an `.ini` file (see `config/*.ini`), parsed by `emrConfigManager`. `emrModelBuilder.ModelBuilder` reads the `[MODEL]` section into `ExtendedMaskRCNN` kwargs and optionally loads a checkpoint from `[LOOP] ckpt_path`. Every key falls back to the torchvision default when absent, so adding one to a single dataset's config changes nothing elsewhere. Beyond the fusion knobs, the crowding/anchor keys matter on dense datasets:
- `box_detections_per_img`, `rpn_batch_size_per_image`, `box_batch_size_per_image`, `box_positive_fraction`, `rpn_post_nms_top_n_{train,test}` — the defaults (100 detections; `256*0.5` and `512*0.25` = 128 positive samples) sit *below* the object count on ATAS (median 93 instances/slice, max 364), so most cells never contribute a gradient.
- `anchor_sizes` / `anchor_aspect_ratios` — parsed by `build_anchor_generator`, one FPN level per `|`, sizes comma-separated within a level (`12,16,24 | 24,32,48 | ...`). Unset falls back to `_default_anchorgen()` in `emrmodel/faster_rcnn.py`. **Every level must carry the same number of sizes** — `RPNHead` uses one conv with a fixed channel count, and a ragged config fails at box decode. Note anchor sizes are bound to levels of fixed stride (4/8/16/32/64), so what governs RPN recall is the size-to-stride ratio, not absolute size: *lowering* the ladder makes matching worse (25.8% -> 7.5% of ATAS GT at >=0.7 IoU), while carrying 3 sizes per level raises it (34% -> 62% at 512px). Changing anchor counts changes `rpn.head.cls_logits` shape, so **old checkpoints will not load** into a config with new `anchor_sizes`.

`training_loop.py`/`testing_loop.py`/`run_train_test_pipeline.py` are the entry points; `emrConfigManager.create_experiment_folder` copies the config into a timestamped experiment dir under `Experiments/{train,test}/` for reproducibility, and `Fusion_Logger` (module-level singleton) records fusion-weight diagnostics there.

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
