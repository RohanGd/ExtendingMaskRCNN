"""Build presentation/Paper_tables.xlsx -- the paper-shaped view of the experiment results.

Reads presentation/Tracking_experiments.xlsx (the operational ledger) and datasets/, and writes
a second workbook holding the tables a paper on 2.5D slice fusion needs. Cells with no run are
left EMPTY; the RunQueue sheet lists every one of them as an actionable job.

    python utils/build_paper_tables.py [--source ...] [--out ...] [--no-dataset-stats]

Regenerate after each round of slurm/collect_test_metrics.sh. The output workbook is written
fresh every time -- never edited in place -- which avoids the openpyxl merged-range hazard
described in CLAUDE.md. It contains no merged cells.
"""
import argparse
import glob
import json
import os
import random
import re
import sys

import numpy as np
import openpyxl
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOURCE = os.path.join(REPO, "presentation", "Tracking_experiments.xlsx")
OUT = os.path.join(REPO, "presentation", "Paper_tables.xlsx")
DATASETS_DIR = os.path.join(REPO, "datasets")

DATASETS = ["Fluo-N3DH-SIM+", "Mouse-Organoid", "Mouse-Skull", "12spheroids_High",
            "ATAS", "C_elegans_nuclei"]
# Deep ablations are enumerated on one dataset only: the sole public benchmark of the six
# (Cell Tracking Challenge), the most test volumes (34, so per-volume metrics are least noisy),
# and the most existing variant coverage.
ABLATION_DS = "Fluo-N3DH-SIM+"

# Results_raw metric key -> (short column header, which block)
METRICS = [
    ("2d.SEG_score", "SEG", "2D"),
    ("2d.jaccard_index", "IoU", "2D"),
    ("2d.dice_score", "Dice", "2D"),
    ("2d.aji", "AJI", "2D"),
    ("2d.ap@0_50", "AP@.50", "2D"),
    ("2d.pq@0_50", "PQ@.50", "2D"),
    ("2d.mean_difference_in_count", "dCount", "2D"),
    ("volume.volume_SEG_score", "SEG", "3D"),
    ("volume.volume_mean_semantic_3d_iou", "SemIoU", "3D"),
    ("volume.volume_aji", "AJI", "3D"),
    ("volume.volume_ap@0_50", "AP@.50", "3D"),
    ("volume.volume_pq@0_50", "PQ@.50", "3D"),
    ("volume.volume_acc@0_50", "Acc@.50", "3D"),
    ("volume.volume_acc@0_30", "Acc@.30", "3D"),
]
HEADLINE = ["2d.SEG_score", "volume.volume_SEG_score"]

R50, SWIN = "ResNet50-FPN", "Swin-T"
NONE, CHAN, EARLY, LATE = "none", "input-channel", "backbone-early", "maskhead-late"
FREEZE_NONE, FREEZE_BB, FREEZE_BBR = "none", "backbone", "backbone+RPN+box"

DATASET_NOTES = {
    "Fluo-N3DH-SIM+": "Cell Tracking Challenge, simulated nuclei; ablation dataset",
    "Mouse-Organoid": "large cells, sparse; brightest signal",
    "Mouse-Skull": "smallest dataset (3 volumes); elongated cells",
    "12spheroids_High": "NRRD spheroids, High anisotropy (Low variant not generated)",
    "ATAS": "densest (p90 131 inst/slice); needs raised caps + 3 anchor sizes/level",
    "C_elegans_nuclei": "slice axis corrected to axis 2 (was axis 0); dataset grew 9x",
}


# --------------------------------------------------------------------------- source parsing
def parse_label(label):
    """Ledger experiment label -> (backbone, n, fusion_point, variant, bias, freeze)."""
    swin = "_swin" in label or label.endswith("swin")
    backbone = SWIN if swin else R50
    m = re.search(r"_n(\d)", label)
    n = int(m.group(1)) if m else 1
    if label.startswith("Base"):
        return backbone, n, NONE, "-", "-", FREEZE_NONE
    if label.startswith("Channel_fusion"):
        return backbone, n, CHAN, "-", "-", FREEZE_NONE
    if label.startswith("Early_fusion"):
        variant = label.split("_")[-1]
        return backbone, n, EARLY, variant, "zero", FREEZE_BB
    if label.startswith("Late_fusion"):
        freeze = FREEZE_NONE if "noFreeze" in label else (
            FREEZE_BBR if "frozenBackboneRPNBox" in label else FREEZE_BB)
        variant = "conv3d" if "conv3d" in label else ("only_center" if "onlyCenter" in label else "?")
        return backbone, n, LATE, variant, "-", freeze
    raise ValueError(f"unparsed experiment label: {label}")


def run_timestamp(run_dir):
    m = re.search(r"_(\d{8}_\d{6})$", run_dir or "")
    return m.group(1) if m else ""


def read_source(path):
    """Return (runs, placeholders, current).

    runs        every row of Results_raw (full history, including superseded runs)
    placeholders (dataset, label) -> run_dir, for rows with no metrics yet
    current     (dataset, label) -> run_dir that the Results summary sheet points at

    Results is authoritative on *which* run is current; Results_raw is the history. Looking a
    combo up in the history alone would resurrect a superseded pre-fix run as if it were current.
    """
    wb = openpyxl.load_workbook(path, data_only=True)
    raw, res = wb["Results_raw"], wb["Results"]
    keys = [raw.cell(row=1, column=c).value for c in range(1, raw.max_column + 1)]
    runs = []
    for r in range(2, raw.max_row + 1):
        ds = raw.cell(row=r, column=1).value
        if not ds:
            continue
        rec = {keys[c - 1]: raw.cell(row=r, column=c).value for c in range(1, len(keys) + 1)}
        runs.append(rec)

    # Read Results with explicit cell lookups; column C is the merge anchor of a placeholder row
    # so it always survives a load, and column Q sits outside the C:P merge.
    placeholders, current, cur = {}, {}, None
    for r in range(5, res.max_row + 1):
        a = res.cell(row=r, column=1).value
        if a:
            cur = a
        b = res.cell(row=r, column=2).value
        if not b:
            continue
        run_dir = res.cell(row=r, column=17).value
        current[(cur, b)] = run_dir
        if isinstance(res.cell(row=r, column=3).value, str):
            placeholders[(cur, b)] = run_dir
    return runs, placeholders, current


def index_runs(runs):
    """(dataset, run_dir) -> record, and the count of combos with more than one run."""
    by_run = {(r["dataset"], r["run_dir"]): r for r in runs}
    seen, extra = set(), 0
    for r in runs:
        key = (r["dataset"], r["experiment"])
        if key in seen:
            extra += 1
        seen.add(key)
    return by_run, extra


# --------------------------------------------------------------------------- intended matrix
def method(label, backbone, n, fusion, variant="-", bias="-", freeze=FREEZE_NONE, ledger=None):
    return dict(label=label, backbone=backbone, n=n, fusion=fusion, variant=variant,
                bias=bias, freeze=freeze, ledger=ledger)


def main_methods():
    """The 16 rows of T2_Main: every fusion point x backbone x window size."""
    M = []
    for bb, tag in ((R50, ""), (SWIN, "_swin")):
        M.append(method(f"Base n=1 ({bb})", bb, 1, NONE, ledger=f"Base_n1{tag}"))
    for bb, tag in ((R50, ""), (SWIN, "_swin")):
        for n in (3, 5):
            M.append(method(f"Channel n={n} ({bb})", bb, n, CHAN,
                            ledger=f"Channel_fusion{tag}_n{n}"))
    for bb, tag in ((R50, ""), (SWIN, "_swin")):
        for n in (3, 5):
            # Every early-fusion run used early_mlp_fusion = Global, early_mlp_bias = gaussian
            # (see config/earlyFusion/*.ini). The ledger key still says GlobalPerFPN; it is
            # only an ID, so it is kept as-is.
            M.append(method(f"Early Global n={n} ({bb})", bb, n, EARLY, "Global",
                            "gaussian", FREEZE_BB,
                            ledger=f"Early_fusion{tag}_n{n}_GlobalPerFPN"))
    for bb, tag in ((R50, ""), (SWIN, "_swin")):
        for n in (3, 5):
            M.append(method(f"Late conv3d n={n} ({bb})", bb, n, LATE, "conv3d", "-", FREEZE_BB,
                            ledger=f"Late_fusion{tag}_n{n}_conv3d"))
    for n in (3, 5):
        M.append(method(f"Late conv3d n={n} (+frozen RPN/box)", R50, n, LATE, "conv3d", "-",
                        FREEZE_BBR, ledger=f"Late_fusion_n{n}_frozenBackboneRPNBox_conv3d"))
    return M


def early_ablation():
    M = []
    for v in ("Global", "GlobalPerFPN", "Windowed", "Pixel", "PixelPerFPN"):
        for n in (3, 5):
            M.append(method(f"early={v}, n={n}, bias=zero", R50, n, EARLY, v, "zero", FREEZE_BB))
    # bias ablation around the variant actually run (Global); the gaussian cell is the
    # existing early-fusion run (ledger key Early_fusion_n{n}_GlobalPerFPN)
    for bias in ("only_center", "gaussian"):
        for n in (3, 5):
            M.append(method(f"early=Global, n={n}, bias={bias}", R50, n, EARLY, "Global", bias,
                            FREEZE_BB,
                            ledger=f"Early_fusion_n{n}_GlobalPerFPN" if bias == "gaussian" else None))
    return M


def late_ablation():
    M = []
    for v in ("mean", "only_center", "conv3d", "SE"):
        for n in (3, 5):
            led = f"Late_fusion_n{n}_conv3d" if v == "conv3d" else None
            M.append(method(f"late={v}, n={n}", R50, n, LATE, v, "-", FREEZE_BB, ledger=led))
    return M


def warmstart_ablation():
    M = []
    for freeze, tag in ((FREEZE_NONE, "noFreeze"), (FREEZE_BB, "frozenBackbone"),
                        (FREEZE_BBR, "frozenBackboneRPNBox")):
        for v in ("conv3d", "only_center"):
            for n in (3, 5):
                led = None
                if freeze == FREEZE_BB and v == "conv3d":
                    led = f"Late_fusion_n{n}_conv3d"
                elif freeze == FREEZE_BBR:
                    led = (f"Late_fusion_n{n}_frozenBackboneRPNBox_"
                           f"{'conv3d' if v == 'conv3d' else 'onlyCenter'}")
                elif freeze == FREEZE_NONE and v == "only_center":
                    led = f"Late_fusion_n{n}_onlyCenter_noFreeze"
                M.append(method(f"freeze={tag}, late={v}, n={n}", R50, n, LATE, v, "-", freeze,
                                ledger=led))
    return M


# --------------------------------------------------------------------------- status
# Test jobs that COMPLETED on the cluster but whose metrics were never pulled into the ledger
# (verified via sacct: job 3402474 test_base_n1_ATAS, 3h11m, 2026-09-16; job 3365452
# test_base_n1_swin_Fluo-N3DH-SIM+, 15m, 2026-09-07). These need no compute, only collection.
TEST_DONE_NOT_COLLECTED = {("ATAS", "Base_n1"), ("Fluo-N3DH-SIM+", "Base_n1_swin")}
# Submitted and not finished yet. Add (dataset, ledger label) while a job runs; remove an
# entry once its metrics are in the ledger.
IN_PROGRESS = set()


def resolve(ds, meth, by_run, placeholders, current):
    """-> (status, run_dir, metrics|None)."""
    led = meth.get("ledger")
    if led and (ds, led) in IN_PROGRESS:      # a rerun is on the cluster; any older row is moot
        return "running", (placeholders.get((ds, led)) or current.get((ds, led))), None
    if led and (ds, led) in current and (ds, led) not in placeholders:
        rec = by_run.get((ds, current[(ds, led)]))
        if rec is None:
            return "not-run", current[(ds, led)], None
        if rec.get("2d.SEG_score") == 0 and rec.get("2d.jaccard_index", 1) < 1e-3:
            return "failed", rec["run_dir"], None          # degenerate run, not a datapoint
        # C_elegans early/late fusion predate the slice-axis fix -> trained on broken data
        if ds == "C_elegans_nuclei" and meth["fusion"] in (EARLY, LATE) \
                and run_timestamp(rec["run_dir"]) < "20260907":
            return "superseded", rec["run_dir"], None
        return "done", rec["run_dir"], rec
    if led and (ds, led) in placeholders:
        run_dir = placeholders[(ds, led)]
        if (ds, led) in TEST_DONE_NOT_COLLECTED:
            return "pending-collect", run_dir, None
        # A placeholder carrying a run directory means training finished and a test config
        # exists, but the test job was never submitted. Without one, nothing has run at all.
        return ("test-not-submitted" if run_dir else "not-run"), run_dir, None
    return "not-configured", None, None


# --------------------------------------------------------------------------- dataset stats
def dataset_stats(name, n_sample=150):
    # The laptop's datasets/ can be stale (ATAS, Mouse-Organoid and Fluo-N3DH-SIM+ were,
    # Sep 2026): the splits that results are tested on live on the cluster. Point
    # EMR_DATASET_STATS at a JSON of this function's output measured there instead.
    stats_json = os.environ.get("EMR_DATASET_STATS")
    if stats_json:
        return json.load(open(stats_json, encoding="utf-8")).get(name)
    base = os.path.join(DATASETS_DIR, name)
    if not os.path.isdir(base):
        return None
    random.seed(0)
    out = {"dataset": name}
    for split in ("train", "test", "val"):
        out[f"slices_{split}"] = len(glob.glob(os.path.join(base, split, "imgs", "*.npy")))
    for split in ("train", "test"):
        meta = os.path.join(base, split, "metadata.json")
        out[f"vols_{split}"] = len(json.load(open(meta, encoding="utf-8"))) if os.path.isfile(meta) else None
    files = sorted(glob.glob(os.path.join(base, "test", "masks", "*.npz")))
    counts, widths = [], []
    for f in random.sample(files, min(n_sample, len(files))):
        z = np.load(f, allow_pickle=True)
        boxes = z["boxes"]
        counts.append(len(boxes))
        widths.extend(((b[2] - b[0]) + (b[3] - b[1])) / 2 for b in boxes)
    counts, widths = np.array(counts), np.array(widths)
    imgs = sorted(glob.glob(os.path.join(base, "train", "imgs", "*.npy")))
    sample = [np.load(f) for f in random.sample(imgs, min(30, len(imgs)))]
    out.update(inst_med=float(np.median(counts)), inst_p90=float(np.percentile(counts, 90)),
               inst_max=int(counts.max()), cell_px=float(np.median(widths)),
               empty_pct=float(100 * (counts == 0).mean()),
               intensity=float(np.mean([x.mean() for x in sample])))
    return out


# --------------------------------------------------------------------------- writing
HDR = Font(bold=True)
SUB = Font(bold=True, italic=True, size=9)
FILL = PatternFill("solid", fgColor="DDE5F0")
MISS = PatternFill("solid", fgColor="FFF2CC")
BAD = PatternFill("solid", fgColor="F8D7DA")
THIN = Border(bottom=Side(style="thin"))
NUM = "0.0000"


def autosize(ws, minw=8, maxw=52):
    for col in ws.columns:
        letter = None
        width = minw
        for c in col:
            if letter is None:
                letter = c.column_letter
            if c.value is not None:
                width = max(width, min(maxw, len(str(c.value)) + 2))
        if letter:
            ws.column_dimensions[letter].width = width


def write_readme(wb, generated_from):
    ws = wb.create_sheet("README")
    lines = [
        ("Paper tables -- 2.5D slice fusion for instance segmentation", HDR),
        ("", None),
        (f"Generated by utils/build_paper_tables.py from {generated_from}.", None),
        ("Regenerate after every round of slurm/collect_test_metrics.sh. Do not hand-edit:", None),
        ("edits are overwritten on the next run. Fix the source ledger instead.", None),
        ("", None),
        ("Sheets", HDR),
        ("T1_Datasets      dataset characteristics (measured from datasets/)", None),
        ("Data_AllRuns     source of truth: one row per (dataset, method), all 14 metrics", None),
        ("T2_Main          main comparison, all methods x all datasets", None),
        ("T3_WindowSize    effect of window size n", None),
        ("T4_Backbone      ResNet50-FPN vs Swin-T", None),
        (f"T5_EarlyFusion   early-fusion variants + bias init ({ABLATION_DS} only)", None),
        (f"T6_LateFusion    mask-head fusion variants ({ABLATION_DS} only)", None),
        (f"T7_Warmstart     backbone/RPN freezing regime ({ABLATION_DS} only)", None),
        ("RunQueue         every missing cell, as a job to run", None),
        ("", None),
        ("Status values", HDR),
        ("done             metrics collected and current", None),
        ("pending-collect  the test run finished; its metrics are not in the ledger yet", None),
        ("not-configured   no config exists; needs writing before it can run", None),
        ("failed           the run produced degenerate output (all-zero metrics)", None),
        ("superseded       result predates a data fix, so it is not comparable", None),
        ("", None),
        ("Conventions", HDR),
        ("An empty metric cell means the run has not produced a number. Nothing is imputed.", None),
        ("Where several runs exist for one (dataset, method), the newest by run-directory", None),
        ("timestamp is shown; older ones stay in the ledger's Results_raw sheet.", None),
        ("Metrics: 2D are per-slice, 3D are per-volume after Hungarian z-linking.", None),
        ("SEG is the Cell Tracking Challenge score; dCount is mean per-slice count error.", None),
        ("", None),
        (f"Ablation dataset: {ABLATION_DS} -- the only public benchmark of the six, the most", None),
        ("test volumes (34), and the most existing variant coverage.", None),
    ]
    for i, (text, font) in enumerate(lines, start=1):
        c = ws.cell(row=i, column=1, value=text)
        if font:
            c.font = font
    ws.column_dimensions["A"].width = 100
    return ws


def write_t1(wb, stats):
    ws = wb.create_sheet("T1_Datasets")
    ws.cell(row=1, column=1, value="Table 1: Dataset characteristics").font = HDR
    cols = ["Dataset", "Volumes tr/te", "Slices train", "Slices test", "Slices val",
            "Instances/slice (median)", "(p90)", "(max)", "Median cell diameter (px)",
            "Empty slices (%)", "Mean intensity", "Notes"]
    for j, h in enumerate(cols, start=1):
        c = ws.cell(row=3, column=j, value=h)
        c.font = HDR
        c.fill = FILL
        c.border = THIN
        c.alignment = Alignment(wrap_text=True, vertical="bottom")
    for i, ds in enumerate(DATASETS, start=4):
        s = stats.get(ds)
        ws.cell(row=i, column=1, value=ds)
        if not s:
            ws.cell(row=i, column=2, value="(not generated locally)")
        else:
            ws.cell(row=i, column=2, value=f"{s['vols_train']}/{s['vols_test']}")
            for j, k in enumerate(["slices_train", "slices_test", "slices_val", "inst_med",
                                   "inst_p90", "inst_max", "cell_px", "empty_pct",
                                   "intensity"], start=3):
                v = s[k]
                cell = ws.cell(row=i, column=j, value=v)
                if k in ("empty_pct", "intensity", "cell_px"):
                    cell.number_format = "0.0"
        ws.cell(row=i, column=12, value=DATASET_NOTES.get(ds, ""))
    autosize(ws)
    return ws


def write_data_all(wb, rows):
    ws = wb.create_sheet("Data_AllRuns")
    ws.cell(row=1, column=1, value="Source of truth -- every (dataset, method) cell").font = HDR
    head = ["Dataset", "Method", "Backbone", "n", "Fusion point", "Variant", "Bias", "Freeze",
            "Status", "Run directory"] + [f"{blk} {hdr}" for _, hdr, blk in METRICS]
    for j, h in enumerate(head, start=1):
        c = ws.cell(row=3, column=j, value=h)
        c.font = HDR
        c.fill = FILL
        c.border = THIN
        c.alignment = Alignment(wrap_text=True)
    for i, r in enumerate(rows, start=4):
        vals = [r["dataset"], r["label"], r["backbone"], r["n"], r["fusion"], r["variant"],
                r["bias"], r["freeze"], r["status"], r["run_dir"]]
        for j, v in enumerate(vals, start=1):
            ws.cell(row=i, column=j, value=v)
        st = ws.cell(row=i, column=9)
        if r["status"] == "failed":
            st.fill = BAD
        elif r["status"] != "done":
            st.fill = MISS
        for j, (key, _, _) in enumerate(METRICS, start=11):
            if r["metrics"]:
                cell = ws.cell(row=i, column=j, value=r["metrics"].get(key))
                cell.number_format = NUM
    ws.freeze_panes = "C4"
    autosize(ws)
    return ws


def write_matrix(wb, name, title, methods, datasets, index, note=None):
    """methods x datasets, two headline metrics per dataset."""
    ws = wb.create_sheet(name)
    ws.cell(row=1, column=1, value=title).font = HDR
    if note:
        ws.cell(row=2, column=1, value=note).font = SUB
    ws.cell(row=4, column=1, value="Method").font = HDR
    ws.cell(row=4, column=1).fill = FILL
    for d, ds in enumerate(datasets):
        base = 2 + d * len(HEADLINE)
        c = ws.cell(row=4, column=base, value=ds)
        c.font = HDR
        c.fill = FILL
        c.alignment = Alignment(horizontal="center")
        for k, key in enumerate(HEADLINE):
            lbl = "2D SEG" if key.startswith("2d") else "3D SEG"
            sc = ws.cell(row=5, column=base + k, value=lbl)
            sc.font = SUB
            sc.border = THIN
            sc.alignment = Alignment(horizontal="center")
    for i, meth in enumerate(methods, start=6):
        ws.cell(row=i, column=1, value=meth["label"])
        for d, ds in enumerate(datasets):
            r = index.get((ds, meth["label"]))
            base = 2 + d * len(HEADLINE)
            for k, key in enumerate(HEADLINE):
                cell = ws.cell(row=i, column=base + k)
                if r and r["metrics"]:
                    cell.value = r["metrics"].get(key)
                    cell.number_format = NUM
                else:
                    cell.fill = BAD if (r and r["status"] == "failed") else MISS
    ws.freeze_panes = "B6"
    autosize(ws)
    return ws


def write_runqueue(wb, queue):
    ws = wb.create_sheet("RunQueue")
    ws.cell(row=1, column=1, value="Every missing cell, as a job to run").font = HDR
    ws.cell(row=2, column=1,
            value="Priority A/B need no training. /netscratch must be back before anything runs."
            ).font = SUB
    head = ["Priority", "What it needs", "Dataset", "Method", "n", "Backbone", "Status",
            "Config", "How to run"]
    for j, h in enumerate(head, start=1):
        c = ws.cell(row=4, column=j, value=h)
        c.font = HDR
        c.fill = FILL
        c.border = THIN
    for i, q in enumerate(queue, start=5):
        for j, k in enumerate(["priority", "needs", "dataset", "label", "n", "backbone",
                               "status", "config", "how"], start=1):
            ws.cell(row=i, column=j, value=q.get(k))
    ws.freeze_panes = "A5"
    autosize(ws)
    return ws


# --------------------------------------------------------------------------- run queue
def build_queue(rows):
    """Classify every non-done cell into the priority groups from the plan."""
    out = []
    for r in rows:
        if r["status"] == "done":
            continue
        ds, meth = r["dataset"], r
        cfg, how, needs, prio = "", "", "", ""
        if r["status"] == "pending-collect":
            prio, needs = "A", "collect metrics only -- the test run already finished"
            how = "slurm/collect_test_metrics.sh once /netscratch is back"
        elif r["status"] == "test-not-submitted":
            prio, needs = "B", "test only -- trained, but the test job was never submitted"
            cfg = f"config/test/channelFusion_swin/channelFusion_swin_n{r['n']}_*.ini"
            how = "slurm/test/channelFusion_swin/submit_all.sh"
        elif r["status"] == "failed":
            prio, needs = "C", "rerun -- previous run produced all-zero metrics"
            how = f"slurm/pipeline/channelFusion/ (retrain), dataset {ds}"
        elif ds in ("ATAS", "C_elegans_nuclei") and r["fusion"] in (EARLY, LATE):
            prio = "D"
            needs = ("rerun on the corrected dataset" if r["status"] == "superseded"
                     else "resubmit -- job died in the /netscratch outage")
            cfg = f"config/pipeline/{'early' if r['fusion'] == EARLY else 'late'}Fusion/"
            how = "slurm/pipeline/{early,late}Fusion/submit_all.sh"
        elif r["backbone"] == SWIN:
            prio, needs = "E", "new config + train + test (backbone x fusion cell)"
            how = "add backbone = Swin to the matching fusion config, add pipeline job"
        else:
            prio, needs = "F", "new config + train + test (ablation variant)"
            how = "new config under config/pipeline/, add pipeline job"
        out.append(dict(priority=prio, needs=needs, dataset=ds, label=meth["label"],
                        n=meth["n"], backbone=meth["backbone"], status=r["status"],
                        config=cfg, how=how))
    out.sort(key=lambda q: (q["priority"], q["dataset"], q["label"]))
    return out


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default=SOURCE)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--no-dataset-stats", action="store_true",
                    help="skip T1 measurement (fast; leaves T1 blank)")
    args = ap.parse_args()

    if not os.path.isfile(args.source):
        sys.exit(f"source workbook not found: {args.source}")
    runs, placeholders, current = read_source(args.source)
    by_run, extra = index_runs(runs)
    print(f"source: {len(runs)} runs ({extra} superseded kept in history), "
          f"{len(current)} current combos, {len(placeholders)} placeholders")

    MAIN = main_methods()

    def expand(methods, datasets):
        rows = []
        for ds in datasets:
            for m in methods:
                status, run_dir, rec = resolve(ds, m, by_run, placeholders, current)
                rows.append({**m, "dataset": ds, "status": status,
                             "run_dir": run_dir, "metrics": rec})
        return rows

    main_rows = expand(MAIN, DATASETS)
    abl_rows = (expand(early_ablation(), [ABLATION_DS])
                + expand(late_ablation(), [ABLATION_DS])
                + expand(warmstart_ablation(), [ABLATION_DS]))

    # de-duplicate: an ablation row may repeat a main-table cell
    seen, all_rows = set(), []
    for r in main_rows + abl_rows:
        key = (r["dataset"], r["backbone"], r["n"], r["fusion"], r["variant"], r["bias"],
               r["freeze"])
        if key in seen:
            continue
        seen.add(key)
        all_rows.append(r)

    index = {(r["dataset"], r["label"]): r for r in main_rows + abl_rows}
    stats = {} if args.no_dataset_stats else {
        ds: dataset_stats(ds) for ds in DATASETS}

    wb = openpyxl.Workbook()
    wb.remove(wb.active)
    write_readme(wb, os.path.relpath(args.source, REPO))
    write_t1(wb, {k: v for k, v in stats.items() if v})
    write_data_all(wb, all_rows)
    write_matrix(wb, "T2_Main", "Table 2: Fusion strategy across all datasets", MAIN, DATASETS,
                 index, "Empty cell = not run. See RunQueue.")
    write_matrix(wb, "T3_WindowSize", "Table 3: Effect of window size n",
                 [m for m in MAIN if m["backbone"] == R50 and m["freeze"] != FREEZE_BBR],
                 DATASETS, index, "ResNet50-FPN only; n=1 is the no-fusion baseline.")
    write_matrix(wb, "T4_Backbone", "Table 4: Backbone -- ResNet50-FPN vs Swin-T",
                 [m for m in MAIN if m["freeze"] != FREEZE_BBR], DATASETS, index)
    write_matrix(wb, "T5_EarlyFusion",
                 "Table 5: Early (backbone) fusion variants and bias initialisation",
                 early_ablation(), [ABLATION_DS], index,
                 f"{ABLATION_DS} only. Windowed was superseded by Pixel (see CLAUDE.md).")
    write_matrix(wb, "T6_LateFusion", "Table 6: Late (mask-head) fusion variants",
                 late_ablation(), [ABLATION_DS], index, f"{ABLATION_DS} only.")
    write_matrix(wb, "T7_Warmstart", "Table 7: Backbone warm-start and freezing regime",
                 warmstart_ablation(), [ABLATION_DS], index,
                 f"{ABLATION_DS} only. Warm-started from a base_n1 checkpoint.")
    queue = build_queue(all_rows)
    write_runqueue(wb, queue)
    wb.save(args.out)

    done = sum(1 for r in all_rows if r["status"] == "done")
    print(f"wrote {args.out}")
    print(f"  {len(all_rows)} cells: {done} done, {len(all_rows) - done} outstanding")
    from collections import Counter
    for k, v in sorted(Counter(q["priority"] for q in queue).items()):
        print(f"  priority {k}: {v} jobs")


if __name__ == "__main__":
    main()
