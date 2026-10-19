'''
Mask2Former local sanity check on real slices, using the config's own model, optimizer and LR recipe. A Mask R-CNN
config runs the same way, as a same-budget baseline.

Slice mode (default): train on --num_slices fixed training slices and evaluate on those same slices. A correct
model + loss must memorise them: the final-layer class loss should go to ~0 and every GT instance should be
claimed by exactly one query, with all other queries predicting "no object". If it can't, there is a bug or a
broken recipe -- run this before spending GPU days on a full schedule.

Volume mode (--train_volumes): train on every slice of those training volumes and evaluate on them and on a
held-out test volume, i.e. a miniature of the real train/test run.

python utils/m2f_overfit_check.py config/pipeline/mask2former/m2f_n1_ATAS.ini --dataset Fluo-N3DH-SIM+
python utils/m2f_overfit_check.py <cfg> --train_volumes 0 1 --test_volume 0 --iters 6000
'''
import argparse
import os
import sys
import time

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from emrConfigManager import emrConfigManager, setup_logger, DATASETS_PATH, REPO_ROOT
from emrDataset import emrDataset, emrCollate_fn
from emrModelBuilder import ModelBuilder

SMALL_AREA = 300


class LazySlices:
    """(image, target) pairs read from the dataset on access -- a couple of volumes of 512px masks don't fit in RAM."""

    def __init__(self, dataset, ids):
        self.dataset, self.ids = dataset, list(ids)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, k):
        return self.dataset[self.ids[k]]

    def __iter__(self):
        return (self.dataset[i] for i in self.ids)


def pick_slices(dataset, num_slices, empty_fraction, seed):
    """Deterministic sample of slice ids, mostly with instances plus a few empty ones (empty slices are ~37% of
    Fluo and must come out with zero predictions)."""
    masks_dir = f"{DATASETS_PATH}/{dataset.dataset_name}/{dataset.mode}/masks"
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(dataset))
    num_empty = int(round(num_slices * empty_fraction))
    full, empty = [], []
    for idx in order:
        n = np.load(f"{masks_dir}/{str(idx).zfill(dataset.s)}.npz")["masks"].shape[0]
        if n > 0 and len(full) < num_slices - num_empty:
            full.append(int(idx))
        elif n == 0 and len(empty) < num_empty:
            empty.append(int(idx))
        if len(full) + len(empty) == num_slices:
            break
    return sorted(full + empty)


def volume_slices(dataset, volumes):
    return [i for v in volumes for i in dataset.metadata[str(v)]]


def mask_iou(a, b):
    """a (N, P), b (M, P) bool -> (N, M) IoU."""
    a, b = a.float(), b.float()
    inter = a @ b.T
    return inter / (a.sum(1)[:, None] + b.sum(1)[None] - inter).clamp(min=1e-6)


@torch.no_grad()
def evaluate(model, samples, device, iou_thresh=0.5):
    model.eval()
    tp = num_pred = num_gt = dup = num_slices = 0
    matched_scores, unmatched_scores = [], []
    empty_slice_preds = 0
    gt_small = tp_small = 0  # GT under SMALL_AREA px, where a stride-4 mask grid has the least to work with
    for img, target in samples:
        num_slices += 1
        pred = model(img[None].to(device))[0]
        p = (pred["masks"][:, 0] > 0.5).flatten(1)
        g = target["masks"].to(device).bool().flatten(1)
        scores = pred["scores"].cpu().numpy()
        num_pred += p.shape[0]
        num_gt += g.shape[0]
        if g.shape[0] == 0:
            empty_slice_preds += p.shape[0]
        if p.shape[0] > 1:
            pp = mask_iou(p, p)
            pp.fill_diagonal_(0)
            dup += int((pp.max(1).values > iou_thresh).sum())
        matched = np.zeros(p.shape[0], dtype=bool)
        if p.shape[0] and g.shape[0]:
            iou = mask_iou(p, g).cpu().numpy()
            rows, cols = linear_sum_assignment(-iou)
            ok = iou[rows, cols] >= iou_thresh
            tp += int(ok.sum())
            matched[rows[ok]] = True
            small = (g.sum(1) < SMALL_AREA).cpu().numpy()
            tp_small += int(small[cols[ok]].sum())
        if g.shape[0]:
            gt_small += int((g.sum(1) < SMALL_AREA).sum())
        matched_scores += list(scores[matched])
        unmatched_scores += list(scores[~matched])
    model.train()
    return {
        "precision": tp / max(num_pred, 1),
        "recall": tp / max(num_gt, 1),
        "recall_small": tp_small / max(gt_small, 1),
        "recall_large": (tp - tp_small) / max(num_gt - gt_small, 1),
        "preds_per_slice": num_pred / max(num_slices, 1),
        "gt_per_slice": num_gt / max(num_slices, 1),
        "dup_frac": dup / max(num_pred, 1),
        "preds_on_empty_slices": empty_slice_preds,
        "score_matched": float(np.mean(matched_scores)) if matched_scores else float("nan"),
        "score_unmatched": float(np.mean(unmatched_scores)) if unmatched_scores else float("nan"),
    }


def fmt(m):
    return (f"P {m['precision']:.3f} R {m['recall']:.3f} (small {m['recall_small']:.2f} large {m['recall_large']:.2f}) | "
            f"preds/slice {m['preds_per_slice']:.1f} (GT {m['gt_per_slice']:.1f}) | dup {m['dup_frac']:.3f} | "
            f"on-empty {m['preds_on_empty_slices']} | "
            f"score matched {m['score_matched']:.3f} unmatched {m['score_unmatched']:.3f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", help="a Mask2Former .ini (its [MODEL]/[LOOP] keys are used as-is)")
    parser.add_argument("--dataset", default=None, help="override [DATASET] dataset_name")
    parser.add_argument("--num_slices", type=int, default=32, help="slice mode: how many training slices")
    parser.add_argument("--empty_fraction", type=float, default=0.25)
    parser.add_argument("--train_volumes", type=int, nargs="+", default=None, help="volume mode: train volume ids")
    parser.add_argument("--test_volume", type=int, default=0, help="volume mode: held-out test volume id")
    parser.add_argument("--eval_stride", type=int, default=2, help="volume mode: evaluate every k-th slice")
    parser.add_argument("--iters", type=int, default=3000)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--eval_every", type=int, default=500)
    parser.add_argument("--clip", type=float, default=None, help="override [LOOP] grad_clip_norm (0 = off)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ckpt", default=None, help="load this checkpoint first (with --iters 0: evaluate only)")
    parser.add_argument("--score_threshs", type=float, nargs="+", default=None,
                        help="Mask2Former: after training, re-evaluate every eval set at each of these score thresholds")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    cfg = emrConfigManager(args.config)
    # Mask R-CNN configs run too (same eval output format), as a same-budget baseline
    is_m2f = cfg.get("MODEL", "architecture", "MaskRCNN") == "Mask2Former"
    if args.dataset:
        cfg.config.set("DATASET", "dataset_name", args.dataset)
    cfg.config.set("LOOP", "ckpt_path", "")
    dataset_name = cfg.get("DATASET", "dataset_name")
    clip = args.clip if args.clip is not None else cfg.get_float("LOOP", "grad_clip_norm", 0.0)

    out_dir = REPO_ROOT / "Experiments" / "overfit_check"
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(str(out_dir / f"{dataset_name}_{time.strftime('%Y%m%d_%H%M%S')}.log"), name="overfit")
    logger.info(f"args: {vars(args)} | config: {args.config} | grad clip {clip}")

    num_slices_per_batch = cfg.get_int("MODEL", "num_slices_per_batch")

    def load(split):
        return emrDataset(imgs_dir=f"{DATASETS_PATH}/{dataset_name}/{split}/imgs",
                          masks_dir=f"{DATASETS_PATH}/{dataset_name}/{split}/masks",
                          num_slices=num_slices_per_batch, dataset_name=dataset_name, logger=logger, mode=split)

    train_dataset = load("train")
    if args.train_volumes is None:
        train_ids = pick_slices(train_dataset, args.num_slices, args.empty_fraction, args.seed)
        eval_sets = {"train": LazySlices(train_dataset, train_ids)}
    else:
        train_ids = volume_slices(train_dataset, args.train_volumes)
        test_dataset = load("test")
        test_ids = volume_slices(test_dataset, [args.test_volume])
        eval_sets = {"train": LazySlices(train_dataset, train_ids[::args.eval_stride]),
                     "test": LazySlices(test_dataset, test_ids[::args.eval_stride])}
    train = LazySlices(train_dataset, train_ids)
    logger.info(f"training on {len(train_ids)} slices ({dataset_name}/train), "
                + ", ".join(f"eval {k} on {len(v)}" for k, v in eval_sets.items()))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_init = ModelBuilder(cfg, logger)
    model = model_init.load_model(dataset_name).to(device)
    if args.ckpt:
        model.load_state_dict(torch.load(args.ckpt, weights_only=True)["model_state_dict"])
        logger.info(f"loaded {args.ckpt}")
    optimizer = model_init.build_optimizer(model)
    model.train()

    rng = np.random.default_rng(args.seed)
    running = {}
    start = time.time()
    for it in range(1, args.iters + 1):
        batch = [train[i] for i in rng.choice(len(train), args.batch_size, replace=False)]
        images, targets = emrCollate_fn(batch)
        images = images.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        loss = sum(loss_dict.values())
        optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip if clip > 0 else float("inf"))
        lr_factor = model_init.set_lr(optimizer, it - 1, args.iters)
        optimizer.step()

        # final decoder layer only, unweighted: this is the term that has to reach ~0 for duplicates to vanish
        terms = {"total": loss.item(), "grad_norm": grad_norm.item()}
        if is_m2f:
            terms["ce_final"] = model.last_loss_terms["loss_cross_entropy"].item() / model.config.class_weight
            terms["dice_final"] = model.last_loss_terms["loss_dice"].item() / model.config.dice_weight
        for k, v in terms.items():
            running[k] = running.get(k, 0.0) + v
        if it % 100 == 0:
            logger.info(f"iter {it}: " + " ".join(f"{k} {v / 100:.4f}" for k, v in running.items())
                        + f" lr_factor {lr_factor:.3f} | {time.time() - start:.0f}s")
            running = {}
        if it % args.eval_every == 0 or it == args.iters:
            for name, samples in eval_sets.items():
                logger.info(f"iter {it} eval {name}: {fmt(evaluate(model, samples, device))}")

    if args.iters > 0:
        ckpt_out = out_dir / f"{dataset_name}_{os.path.splitext(os.path.basename(args.config))[0]}_last.pt"
        torch.save({"model_state_dict": model.state_dict()}, ckpt_out)
        logger.info(f"saved {ckpt_out}")
    if args.score_threshs and is_m2f:
        for thresh in args.score_threshs:
            model.score_thresh = thresh
            for name, samples in eval_sets.items():
                logger.info(f"score_thresh {thresh} eval {name}: {fmt(evaluate(model, samples, device))}")
    if args.train_volumes is None and args.iters > 0:
        final = evaluate(model, eval_sets["train"], device)
        checks = {
            "precision >= 0.9": final["precision"] >= 0.9,
            "recall >= 0.9": final["recall"] >= 0.9,
            "dup_frac <= 0.05": final["dup_frac"] <= 0.05,
            "no preds on empty slices": final["preds_on_empty_slices"] == 0,
        }
        for name, ok in checks.items():
            logger.info(f"{'PASS' if ok else 'FAIL'}  {name}")
        logger.info("OVERFIT CHECK " + ("PASSED" if all(checks.values()) else "FAILED"))


if __name__ == "__main__":
    main()
