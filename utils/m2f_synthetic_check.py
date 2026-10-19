'''
Mask2Former synthetic benchmark: random non-overlapping discs on a flat background, trained on one set of images
and evaluated on both those images and a held-out set. Isolates the model, criterion, optimizer recipe and
postprocess from anything dataset-specific -- if a recipe can't score ~1.0 here, it won't work on cells.

python utils/m2f_synthetic_check.py --iters 3000 --clip 0.01 --poly
python utils/m2f_synthetic_check.py --bg 90 --contrast 10 --normalize    # Fluo-like: +10 on a ~90 background
'''
import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from emrmodel.mask2former import EMRMask2Former
from m2f_overfit_check import evaluate, fmt


def make_discs(num_images, size, bg, contrast, noise, empty_fraction, max_discs, rng):
    yy, xx = np.mgrid[:size, :size]
    samples = []
    for _ in range(num_images):
        img = np.full((size, size), bg, np.float32)
        masks = []
        n = 0 if rng.random() < empty_fraction else int(rng.integers(1, max_discs + 1))
        tries = 0
        while len(masks) < n and tries < 1000:
            tries += 1
            r = int(rng.integers(6, 25))
            cy, cx = rng.integers(r + 2, size - r - 2, size=2)
            m = (yy - cy) ** 2 + (xx - cx) ** 2 < r ** 2
            if any((m & o).any() for o in masks):
                continue
            masks.append(m)
            img[m] = bg + contrast
        if noise > 0:
            img += rng.normal(0, noise, img.shape).astype(np.float32)
        masks = np.stack(masks).astype(np.uint8) if masks else np.zeros((0, size, size), np.uint8)
        samples.append((torch.from_numpy(img)[None], {"masks": torch.from_numpy(masks),
                                                      "labels": torch.ones(len(masks), dtype=torch.int64)}))
    return samples


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # data
    p.add_argument("--num_train", type=int, default=64)
    p.add_argument("--num_test", type=int, default=32)
    p.add_argument("--size", type=int, default=256)
    p.add_argument("--bg", type=float, default=0.0)
    p.add_argument("--contrast", type=float, default=1.0)
    p.add_argument("--noise", type=float, default=0.0)
    p.add_argument("--empty_fraction", type=float, default=0.2)
    p.add_argument("--max_discs", type=int, default=8)
    p.add_argument("--normalize", action="store_true", help="image_mean/std from the training images")
    # model
    p.add_argument("--num_queries", type=int, default=20)
    p.add_argument("--no_object_weight", type=float, default=0.1)
    p.add_argument("--score_thresh", type=float, default=0.5)
    # optimisation
    p.add_argument("--iters", type=int, default=3000)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--backbone_lr_mult", type=float, default=0.1)
    p.add_argument("--weight_decay", type=float, default=0.05)
    p.add_argument("--clip", type=float, default=0.01, help="full-model grad-norm clip, 0 = off")
    p.add_argument("--warmup", type=int, default=0, help="linear LR warmup iterations")
    p.add_argument("--poly", action="store_true", help="poly(0.9) LR decay to 0 over --iters")
    p.add_argument("--eval_every", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    print(vars(args), flush=True)

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    gen = dict(size=args.size, bg=args.bg, contrast=args.contrast, noise=args.noise,
               empty_fraction=args.empty_fraction, max_discs=args.max_discs, rng=rng)
    train = make_discs(args.num_train, **gen)
    test = make_discs(args.num_test, **gen)

    norm = {}
    if args.normalize:
        pixels = torch.cat([img.flatten() for img, _ in train])
        norm = dict(image_mean=[pixels.mean().item()], image_std=[pixels.std().item()])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EMRMask2Former(num_slices_per_batch=1, num_queries=args.num_queries, box_score_thresh=args.score_thresh,
                           box_detections_per_img=args.num_queries, no_object_weight=args.no_object_weight,
                           **norm).to(device).train()
    backbone = [p_ for n, p_ in model.named_parameters() if "backbone" in n]
    others = [p_ for n, p_ in model.named_parameters() if "backbone" not in n]
    optimizer = torch.optim.AdamW([{"params": backbone, "lr": args.lr * args.backbone_lr_mult},
                                   {"params": others, "lr": args.lr}], lr=args.lr, weight_decay=args.weight_decay)

    def lr_factor(it):
        f = min(1.0, (it + 1) / args.warmup) if args.warmup else 1.0
        if args.poly:
            f *= (1 - it / args.iters) ** 0.9
        return f
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)

    start = time.time()
    for it in range(1, args.iters + 1):
        idx = rng.choice(len(train), args.batch_size, replace=False)
        images = torch.stack([train[i][0] for i in idx]).to(device)
        targets = [{k: v.to(device) for k, v in train[i][1].items()} for i in idx]
        loss = sum(model(images, targets).values())
        optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip if args.clip > 0 else float("inf"))
        optimizer.step()
        scheduler.step()
        if it % args.eval_every == 0 or it == args.iters:
            terms = model.last_loss_terms
            print(f"iter {it} loss {loss.item():.2f} gn {grad_norm.item():.1f} "
                  f"ce_final {terms['loss_cross_entropy'].item() / model.config.class_weight:.3f} "
                  f"dice_final {terms['loss_dice'].item() / model.config.dice_weight:.3f} | {time.time() - start:.0f}s",
                  flush=True)
            print(f"  train: {fmt(evaluate(model, train, device))}", flush=True)
            print(f"  test:  {fmt(evaluate(model, test, device))}", flush=True)
            # same weights, panoptic-style overlap resolution in the postprocess
            model.overlap_thresh = 0.8
            print(f"  test+overlap0.8:  {fmt(evaluate(model, test, device))}", flush=True)
            model.overlap_thresh = 0.0


if __name__ == "__main__":
    main()
