"""Build presentation/Results_so_far.xlsx -- only tables that are already complete.

Companion to build_paper_tables.py. That workbook shows the full intended matrix with blanks
for everything still to run; this one shows ONLY what can be presented today: every table here
is 100% populated, with no empty cells to explain away.

    python utils/build_results_now_tables.py [--source ...] [--out ...] [--no-dataset-stats]

Each matrix table is pruned to its largest gap-free block: rows/columns with missing runs are
dropped and listed in the sheet's caption, so nothing is silently omitted. As runs land, the
tables grow on their own -- rerun after each collect_test_metrics.sh round.
"""
import argparse
import os
import sys

import openpyxl
from openpyxl.styles import Alignment, Font, PatternFill

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_paper_tables import (  # noqa: E402  (shared definitions, single source of truth)
    ABLATION_DS, DATASETS, DATASET_NOTES, EARLY, FREEZE_BB, FREEZE_BBR, FREEZE_NONE, HEADLINE,
    LATE, METRICS, R50, REPO, SOURCE, SWIN, autosize, dataset_stats, index_runs, late_ablation,
    main_methods, read_source, resolve, run_timestamp, warmstart_ablation, write_t1,
    FILL, HDR, NUM, SUB, THIN,
)

OUT = os.path.join(REPO, "presentation", "Results_so_far.xlsx")
STABLE = ["Fluo-N3DH-SIM+", "Mouse-Organoid", "Mouse-Skull", "12spheroids_High"]

# Metric sets for the matrix tables. "standard" is the default: segmentation quality (SEG),
# instance-aware overlap (AJI), detection (AP@.50) and panoptic (PQ@.50), at both 2D and 3D.
ALL_KEYS = [m[0] for m in METRICS]
MISSING_MARK = "\u2014"          # em dash: no usable result, never a zero
# Tables built with show_all keep every dataset and say WHY a cell is empty instead.
STATUS_MARK = {"running": "pending", "pending-collect": "pending", "test-not-submitted": "not tested",
               "not-run": "not run", "not-configured": "not run", "superseded": "superseded",
               "failed": "failed"}
PENDING_FILL = PatternFill("solid", fgColor="FFE699")   # amber: submitted, result on its way
MARK_FONT = Font(italic=True, color="808080")
MARK_LEGEND = ("Empty cells say why: 'pending' (amber) is running on the cluster now; 'not run' "
               "was never scheduled; 'superseded' predates a data fix; 'not tested' was trained "
               "but never tested. None of these is a zero.")
# Spelled out for a reader who has not seen the code; the ledger's short codes are cryptic.
LABEL = {
    "2d.SEG_score": "2D SEG", "2d.jaccard_index": "2D IoU", "2d.dice_score": "2D Dice",
    "2d.aji": "2D AJI", "2d.ap@0_50": "2D AP@0.50", "2d.pq@0_50": "2D PQ@0.50",
    "2d.mean_difference_in_count": "2D mean count error",
    "volume.volume_SEG_score": "3D SEG",
    "volume.volume_mean_semantic_3d_iou": "3D semantic IoU",
    "volume.volume_aji": "3D AJI", "volume.volume_ap@0_50": "3D AP@0.50",
    "volume.volume_pq@0_50": "3D PQ@0.50", "volume.volume_acc@0_50": "3D Acc@0.50",
    "volume.volume_acc@0_30": "3D Acc@0.30",
}
METRIC_SETS = {
    "headline": HEADLINE,
    "standard": ["2d.SEG_score", "2d.aji", "2d.ap@0_50", "2d.pq@0_50",
                 "volume.volume_SEG_score", "volume.volume_aji", "volume.volume_ap@0_50"],
    "all": ALL_KEYS,
}


def prune_to_complete(rows, cols, filled, require=None, constrain=None):
    """Drop rows/columns until no hole remains, removing the worst offender each pass.

    filled(row, col) -> bool.
    `constrain(rows) -> rows` repairs a candidate block, e.g. dropping a configuration that
    survived for only one backbone so a backbone comparison stays like-for-like.
    `require(rows, cols) -> bool` then rejects blocks that are complete but meaningless.
    Returns (rows, cols, dropped_rows, dropped_cols).
    """
    rows, cols = list(rows), list(cols)
    # Exact, not greedy: for every subset of columns, keep exactly the rows that are complete
    # over it. Columns are few (<=6 datasets), so 2**len(cols) enumeration is trivial, and greedy
    # pruning demonstrably picks worse blocks here (9x3=27 where 8x4=32 was available).
    best = ([], [], rows, cols)
    for mask in range(1, 1 << len(cols)):
        keep_c = [c for j, c in enumerate(cols) if mask >> j & 1]
        keep_r = [r for r in rows if all(filled(r, c) for c in keep_c)]
        if constrain:
            keep_r = constrain(keep_r)
        if not keep_r or (require and not require(keep_r, keep_c)):
            continue
        area = len(keep_r) * len(keep_c)
        best_area = len(best[0]) * len(best[1])
        # tie-break toward covering more datasets
        if (area, len(keep_c)) > (best_area, len(best[1])):
            best = (keep_r, keep_c,
                    [r for r in rows if r not in keep_r],
                    [c for c in cols if c not in keep_c])
    return best


def caption(ws, row, dropped_rows, dropped_cols, rowname=lambda r: r["label"]):
    """A readable note on scope, not a machine status line."""
    text = ("Limited to configurations with a completed run on every dataset shown, except "
            "where a dash marks a result that is not available.")
    if dropped_rows:
        text += " Not shown: " + "; ".join(rowname(r) for r in dropped_rows) + "."
    if dropped_cols:
        text += " Datasets not shown: " + ", ".join(dropped_cols) + "."
    c = ws.cell(row=row, column=1, value=text)
    c.font = SUB
    return c


def _block(ws, r0, label, rows, cols, get, metrics, rowname, merge_headers=False, mark=None):
    """Write one methods x (datasets x metrics) block starting at row r0. Returns next free row.

    merge_headers spans the dataset name across its metric columns. Safe here because this
    workbook is always written fresh -- the openpyxl hazard in CLAUDE.md is insert_rows on an
    existing file, which leaves merged ranges behind and blanks the cells they land on. Merges
    are confined to the header row; no merge ever covers a data cell.
    """
    ws.cell(row=r0, column=1, value=label).font = HDR
    hr = r0 + 1
    ws.cell(row=hr, column=1, value="Method").font = HDR
    ws.cell(row=hr, column=1).fill = FILL
    for d, col in enumerate(cols):
        base = 2 + d * len(metrics)
        if merge_headers and len(metrics) > 1:
            c = ws.cell(row=hr, column=base, value=col)
            c.font = HDR
            c.fill = FILL
            c.alignment = Alignment(horizontal="center")
            # fill the spanned cells so the banding stays continuous across the merge
            for k in range(1, len(metrics)):
                ws.cell(row=hr, column=base + k).fill = FILL
            ws.merge_cells(start_row=hr, start_column=base,
                           end_row=hr, end_column=base + len(metrics) - 1)
        else:
            for k in range(len(metrics)):
                # repeated rather than merged, so each column is self-describing when scrolled
                c = ws.cell(row=hr, column=base + k, value=col)
                c.font = HDR
                c.fill = FILL
                c.alignment = Alignment(horizontal="center")
        for k, key in enumerate(metrics):
            sc = ws.cell(row=hr + 1, column=base + k, value=LABEL[key])
            sc.font = SUB
            sc.border = THIN
            sc.alignment = Alignment(horizontal="center")
    r0d = hr + 2
    vals = {}                                   # (row index, dataset index, metric index) -> value
    for i, r in enumerate(rows):
        ws.cell(row=r0d + i, column=1, value=rowname(r))
        for d, col in enumerate(cols):
            rec = get(r, col)
            for k, key in enumerate(metrics):
                v = rec.get(key) if rec else None
                cell = ws.cell(row=r0d + i, column=2 + d * len(metrics) + k)
                if v is None:
                    text = mark(r, col) if mark else None
                    cell.value = text or MISSING_MARK
                    cell.alignment = Alignment(horizontal="center")
                    if text:
                        cell.font = MARK_FONT
                        if text == "pending":
                            cell.fill = PENDING_FILL
                else:
                    cell.value = v
                    cell.number_format = NUM
                vals[(i, d, k)] = v

    # Every metric carried in the matrix tables is higher-is-better, so plain maxima are right.
    for k in range(len(metrics)):
        for d in range(len(cols)):
            col_vals = [vals[(i, d, k)] for i in range(len(rows)) if vals[(i, d, k)] is not None]
            best_col = max(col_vals) if col_vals else None
            for i in range(len(rows)):
                v = vals[(i, d, k)]
                if v is None:
                    continue
                bold = v == best_col
                under = False
                if len(cols) > 1:               # comparing one method across datasets
                    row_vals = [vals[(i, dd, k)] for dd in range(len(cols))
                                if vals[(i, dd, k)] is not None]
                    under = bool(row_vals) and v == max(row_vals)
                if bold or under:
                    ws.cell(row=r0d + i, column=2 + d * len(metrics) + k).font = Font(
                        bold=bold, underline="single" if under else None)
    return r0d + len(rows)


def write_matrix(wb, name, title, rows, cols, get, dropped_r, dropped_c,
                 metrics=HEADLINE, rowname=lambda r: r["label"], note=None, footnote=None,
                 mark=None):
    """Sheet with two stacked views of the same block: the compact SEG-only summary, then the
    same rows with the fuller metric set below it."""
    ws = wb.create_sheet(name)
    ws.cell(row=1, column=1, value=title).font = HDR
    if mark:
        ws.cell(row=2, column=1, value="Every configuration and dataset is shown. Cells without a "
                "result are marked with the reason; see the note below the tables.").font = SUB
    else:
        caption(ws, 2, dropped_r, dropped_c, rowname)
    if note:
        ws.cell(row=3, column=1, value=note).font = SUB
    legend = "Bold marks the best result for each dataset."
    if len(cols) > 1:
        legend += (" Underlining marks each method's best dataset, read across the row "
                   "(datasets differ in difficulty, so this compares a method with itself).")
    ws.cell(row=4, column=1, value=legend).font = SUB
    nxt = _block(ws, 5, "SEG score", rows, cols, get, HEADLINE, rowname, mark=mark)
    last = nxt
    if list(metrics) != list(HEADLINE):
        last = _block(ws, nxt + 2, "All metrics",
                      rows, cols, get, metrics, rowname, merge_headers=True, mark=mark)
    if mark:
        ws.cell(row=last + 1, column=1, value=MARK_LEGEND).font = SUB
    if footnote:
        ws.cell(row=last + 2, column=1, value=footnote).font = SUB
    ws.freeze_panes = "B1"                      # column A only; two tables share the sheet
    autosize(ws)
    return ws


OVERVIEW = [
    ("2.5D slice fusion for cell instance segmentation", "h1"),
    ("Experimental results as of 5 October 2026", "sub"),
    ("", None),
    ("We segment cells in 3D microscopy volumes by giving Mask R-CNN a window of n consecutive", None),
    ("slices and predicting instance masks for the centre slice only. What these experiments", None),
    ("test is where in the network the n slices should be combined. Three options are compared", None),
    ("against a single-slice baseline:", None),
    ("", None),
    ("    input channels    the slices enter as extra channels of the first convolution", None),
    ("    backbone          each slice passes through its own backbone, and the feature", None),
    ("                      pyramids are combined before the region proposal network", None),
    ("    mask head         the slices are combined after RoI pooling, inside the mask head", None),
    ("", None),
    ("Datasets", "h2"),
    ("Six volumetric datasets, summarised in the Datasets sheet. Fluo-N3DH-SIM+ comes from the", None),
    ("Cell Tracking Challenge; the others are microscopy volumes spanning a wide range of cell", None),
    ("density and size, from 3 instances per slice to over 130.", None),
    ("", None),
    ("Metrics", "h2"),
    ("Results are reported at two levels. The 2D metrics are computed per slice. The 3D metrics", None),
    ("are computed per volume, after instances are linked across slices by Hungarian matching on", None),
    ("IoU. SEG is the Cell Tracking Challenge score, AJI the aggregated Jaccard index, and AP and", None),
    ("PQ the usual detection and panoptic measures at an IoU threshold of 0.50. For every metric", None),
    ("reported here, higher is better.", None),
    ("", None),
    ("Each results sheet gives the SEG score first, then the same rows again with the fuller set", None),
    ("of metrics below.", None),
    ("", None),
    ("Contents", "h2"),
    ("    T1  Datasets            size, density and cell size of each dataset", None),
    ("    T2  Fusion point        the main comparison: where the slices are combined, and", None),
    ("                            whether three or five slices work better", None),
    ("    T3  Backbone            ResNet50-FPN against Swin-T", None),
    ("    T4  Mask-head fusion    variants and freezing regimes on Fluo-N3DH-SIM+", None),
    ("    T5  Best per dataset    the strongest configuration on each dataset", None),
    ("    T6  All results         every completed run, with all fourteen metrics", None),
    ("", None),
    ("Coverage", "h2"),
    ("Each table is restricted to configurations that have been run on every dataset it covers,", None),
    ("with one exception: a dash marks a result that is not available, and a note beneath the", None),
    ("table says why. A dash is never a zero. The note under each title states what has been", None),
    ("left out entirely. Experiments still outstanding are tracked separately, not shown here.", None),
]


T1_DEFINITIONS = [
    ("Volumes tr/te", "Whole 3D volumes in the training and test splits. The split is by "
                      "volume, never by slice, so no volume contributes to both."),
    ("Slices train / test / val", "2D slices generated from those volumes. The validation "
                                  "slices are sampled from the test volumes and are used only "
                                  "to monitor training, never to report results."),
    ("Instances/slice (median, p90, max)", "Annotated cell instances on a single slice, over a "
                                           "sample of test slices. p90 is the 90th percentile, "
                                           "so a tenth of slices are denser than that."),
    ("Median cell diameter (px)", "Median of (width + height) / 2 of the annotated bounding "
                                  "boxes, in pixels, at the 512x512 working resolution the "
                                  "network sees."),
    ("Empty slices (%)", "Slices carrying no annotation at all, usually at the top and bottom "
                         "of a volume where the specimen ends."),
    ("Mean intensity", "Mean raw pixel value over a sample of training slices. The datasets "
                       "differ by more than an order of magnitude here, and no intensity "
                       "normalisation is applied."),
    ("Notes", "Anything specific to that dataset."),
]


def write_overview(wb):
    ws = wb.create_sheet("Overview")
    for i, (text, kind) in enumerate(OVERVIEW, start=1):
        c = ws.cell(row=i, column=1, value=text)
        if kind == "h1":
            c.font = Font(bold=True, size=14)
        elif kind == "h2":
            c.font = HDR
        elif kind == "sub":
            c.font = SUB
    ws.column_dimensions["A"].width = 98
    ws.sheet_view.showGridLines = False
    return ws


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default=SOURCE)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--no-dataset-stats", action="store_true")
    ap.add_argument("--metrics", choices=sorted(METRIC_SETS), default="standard",
                    help="metric set for the matrix tables (default: standard)")
    args = ap.parse_args()
    metrics = METRIC_SETS[args.metrics]

    runs, ph, current = read_source(args.source)
    by_run, _ = index_runs(runs)
    MAIN = main_methods()

    cache, status = {}, {}

    def rec_for(meth, ds):
        key = (ds, meth["label"])
        if key not in cache:
            st, _, rec = resolve(ds, meth, by_run, ph, current)
            cache[key] = rec if st == "done" else None
            status[key] = st
        return cache[key]

    def mark_for(meth, ds):
        rec_for(meth, ds)
        return STATUS_MARK.get(status[(ds, meth["label"])])

    def filled(meth, ds):
        return rec_for(meth, ds) is not None

    wb = openpyxl.Workbook()
    wb.remove(wb.active)
    write_overview(wb)
    stats = {} if args.no_dataset_stats else {d: dataset_stats(d) for d in DATASETS}
    t1 = write_t1(wb, {k: v for k, v in stats.items() if v})
    t1.title = "T1 Datasets"
    t1.cell(row=1, column=1, value="Dataset characteristics")
    r = 4 + len(DATASETS) + 2
    t1.cell(row=r, column=1, value="What the columns mean").font = HDR
    for j, (col, text) in enumerate(T1_DEFINITIONS, start=r + 1):
        t1.cell(row=j, column=1, value=col).font = SUB
        t1.cell(row=j, column=2, value=text).alignment = Alignment(vertical="top")

    by_label = {m["label"]: m for m in MAIN}

    def table(name, title, labels, cols, note=None, require=None, constrain=None,
              force_rows=(), force_cols=(), footnote=None, show_all=False):
        rows = [by_label[x] for x in labels]
        if show_all:                    # keep everything; empty cells carry a status marker
            r2, c2, dr, dc = rows, list(cols), [], []
        else:
            r2, c2, dr, dc = prune_to_complete(rows, cols, filled, require, constrain)
        # Re-admit rows/datasets the caller wants shown regardless, keeping declared order.
        if force_rows:
            keep = {r["label"] for r in r2} | set(force_rows)
            r2 = [by_label[x] for x in labels if x in keep]
            dr = [r for r in dr if r["label"] not in force_rows]
        if force_cols:
            keep = set(c2) | set(force_cols)
            c2 = [c for c in cols if c in keep]
            dc = [c for c in dc if c not in force_cols]
        write_matrix(wb, name, title, r2, c2,
                     lambda m, ds: rec_for(m, ds), dr, dc, metrics=metrics, note=note,
                     footnote=footnote, mark=mark_for if show_all else None)
        return sum(1 for m in r2 for d in c2 if filled(m, d))

    n2 = table("T2 Fusion point", "Where the slices are combined, and how many",
               ["Base n=1 (ResNet50-FPN)", "Channel n=3 (ResNet50-FPN)",
                "Channel n=5 (ResNet50-FPN)", "Early Global n=3 (ResNet50-FPN)",
                "Early Global n=5 (ResNet50-FPN)", "Late conv3d n=3 (ResNet50-FPN)",
                "Late conv3d n=5 (ResNet50-FPN)", "Late conv3d n=3 (+frozen RPN/box)",
                "Late conv3d n=5 (+frozen RPN/box)"],
               list(DATASETS),
               "ResNet50-FPN backbone throughout. Base n=1 sees a single slice and cannot fuse. "
               "Each fusion method appears at both n=3 and n=5, so the effect of the window size "
               "can be read within each pair of rows.",
               show_all=True)
    n4 = table("T3 Backbone", "Backbone: ResNet50-FPN against Swin-T",
               ["Base n=1 (ResNet50-FPN)", "Base n=1 (Swin-T)",
                "Channel n=3 (ResNet50-FPN)", "Channel n=3 (Swin-T)",
                "Channel n=5 (ResNet50-FPN)", "Channel n=5 (Swin-T)"],
               list(DATASETS),
               "Each configuration appears with both backbones, so every column is a "
               "like-for-like comparison.",
               # a backbone table is meaningless unless both backbones survive, and only
               # fully-paired configurations can be compared
               require=lambda rs, cs: len({r["backbone"] for r in rs}) == 2,
               constrain=lambda rs: [
                   r for r in rs
                   if sum(1 for o in rs if (o["fusion"], o["n"]) == (r["fusion"], r["n"])) == 2])

    # ---- T6: Fluo ablation, only rows with data --------------------------------------------
    abl = [m for m in warmstart_ablation() + late_ablation()]
    seen, rows6 = set(), []
    for m in abl:
        k = (m["variant"], m["n"], m["freeze"])
        if k in seen or not filled(m, ABLATION_DS):
            continue
        seen.add(k)
        rows6.append(m)
    write_matrix(wb, "T4 Mask-head fusion",
                 f"Mask-head fusion variants and freezing regime ({ABLATION_DS})",
                 rows6, [ABLATION_DS], lambda m, ds: rec_for(m, ds),
                 [m for m in abl if not filled(m, ABLATION_DS)], [], metrics=metrics,
                 note="All warm-started from a single-slice checkpoint; 'freeze' names the part "
                      "of the network held fixed during training.")

    # ---- T7: every metric for the best model per dataset ------------------------------------
    ws = wb.create_sheet("T5 Best per dataset")
    ws.cell(row=1, column=1, value="Best configuration on each dataset").font = HDR
    ws.cell(row=2, column=1,
            value="The configuration with the highest 2D SEG score among completed runs.").font = SUB
    head = ["Dataset", "Best configuration", "Run directory"] + [LABEL[k] for k in ALL_KEYS]
    for j, h in enumerate(head, start=1):
        c = ws.cell(row=4, column=j, value=h)
        c.font = HDR
        c.fill = FILL
        c.border = THIN
        c.alignment = Alignment(wrap_text=True)
    n7 = 0
    for i, ds in enumerate(DATASETS, start=5):
        cands = [(rec_for(m, ds), m) for m in MAIN if filled(m, ds)]
        if not cands:
            continue
        rec, m = max(cands, key=lambda t: t[0]["2d.SEG_score"])
        ws.cell(row=i, column=1, value=ds)
        ws.cell(row=i, column=2, value=m["label"])
        ws.cell(row=i, column=3, value=rec["run_dir"])
        for j, (key, _, _) in enumerate(METRICS, start=4):
            ws.cell(row=i, column=j, value=rec.get(key)).number_format = NUM
        n7 += 1
    autosize(ws)

    # ---- T8: every completed run, every metric ---------------------------------------------
    ws = wb.create_sheet("T6 All results")
    ws.cell(row=1, column=1, value="Every completed run").font = HDR
    ws.cell(row=2, column=1, value=(
        "One row per dataset and configuration, with all fourteen metrics. This is the full "
        "detail behind the preceding tables; nothing is omitted.")).font = SUB
    head = ["Dataset", "Configuration", "Backbone", "Slices n", "Fusion point",
            "Run directory"] + [LABEL[k] for k in ALL_KEYS]
    for j, h in enumerate(head, start=1):
        c = ws.cell(row=4, column=j, value=h)
        c.font = HDR
        c.fill = FILL
        c.border = THIN
        c.alignment = Alignment(wrap_text=True)
    i = 5
    n8 = 0
    seen8 = set()
    # MAIN plus the ablation-only variants, so no finished run is left out
    for ds in DATASETS:
        for m in MAIN + warmstart_ablation() + late_ablation():
            rec = rec_for(m, ds)
            if rec is None or (ds, rec["run_dir"]) in seen8:
                continue
            seen8.add((ds, rec["run_dir"]))
            fusion_words = {"none": "single slice", "input-channel": "input channels",
                            "backbone-early": "backbone", "maskhead-late": "mask head"}
            for j, v in enumerate([ds, m["label"], m["backbone"], m["n"],
                                   fusion_words.get(m["fusion"], m["fusion"]),
                                   rec["run_dir"]], start=1):
                ws.cell(row=i, column=j, value=v)
            for j, (key, _, _) in enumerate(METRICS, start=7):
                ws.cell(row=i, column=j, value=rec.get(key)).number_format = NUM
            i += 1
            n8 += 1
    ws.freeze_panes = "C5"
    autosize(ws)

    wb.save(args.out)
    print(f"wrote {args.out}  [metric set: {args.metrics} -> {len(metrics)} per dataset]")
    for nm, n in (("T2 Fusion point", n2), ("T3 Backbone", n4),
                  ("T4 Mask-head fusion", len(rows6)),
                  ("T5 Best per dataset", n7), ("T6 All results", n8)):
        print(f"  {nm:20s} {n} populated cells/rows")


if __name__ == "__main__":
    main()
