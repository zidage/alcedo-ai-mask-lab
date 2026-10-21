#!/usr/bin/env python3
"""Train the people and sky head on the frozen SAM 2.1 tiny image_embed (step 1).

Input:  --train-dir (one or more split directories with targets/ and image_embed.npy; the
        training set is their concatenation) and --val-dir (one or more splits; the first one
        selects best.pt).
Output: --out-dir/last.pt (each epoch; --resume continues from it), --out-dir/best.pt (highest
        head_metrics.selection_score on the first validation split), --out-dir/metrics.jsonl
        (one line per epoch: mean loss terms, `val` of the first validation split, `val_extra`
        of the other splits by directory name), --out-dir/run.json.

Optimizer: AdamW, linear warmup for --warmup-epochs, then cosine decay to zero. The objective is
head_losses.total_loss. Float32 on --device (cuda on Colab). The head only reads the cache, so the time of an epoch
is mostly the disk read of the image_embed cache (2 MB per image).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import head_data as hd  # noqa: E402
import head_losses as hl  # noqa: E402
import head_metrics as hmet  # noqa: E402
import head_model as hm  # noqa: E402


def to_device(batch: dict, device) -> dict:
    return {k: v.to(device, non_blocking=True) if hasattr(v, "to") else v
            for k, v in batch.items()}


def optional(value: float | None) -> str:
    """A measurement for the log line; '-' when it is not measured on the split."""
    return "-" if value is None or value != value else f"{value:.3f}"


def evaluate(model, loader, device) -> dict:
    import torch

    model.eval()
    evaluation = hmet.Evaluation()
    with torch.no_grad():
        for batch in loader:
            batch = to_device(batch, device)
            evaluation.add(model(batch["image_embed"]), batch)
    return evaluation.result()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train-dir", required=True, nargs="+")
    parser.add_argument("--val-dir", required=True, nargs="+")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--warmup-epochs", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--width", type=int, default=96)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--block", type=int, default=256,
                        help="consecutive images per shuffle block (disk read locality)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", default="cuda",
                        help="cuda (Colab) or cpu (a check of the script on a few images)")
    for name, value in hl.DEFAULT_WEIGHTS.items():
        parser.add_argument(f"--w-{name.replace('_', '-')}", type=float, default=value)
    args = parser.parse_args()

    import torch

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("no CUDA device: select a GPU runtime")
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    weights = {name: getattr(args, f"w_{name}") for name in hl.DEFAULT_WEIGHTS}
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_parts = [hd.EmbedDataset(d) for d in args.train_dir]
    train_set = torch.utils.data.ConcatDataset(train_parts)
    val_sets = {Path(d).name: hd.EmbedDataset(d) for d in args.val_dir}
    if len(val_sets) != len(args.val_dir):
        raise ValueError(f"--val-dir names must differ: {args.val_dir}")
    primary_val = Path(args.val_dir[0]).name
    sampler = hd.BlockShuffleSampler(len(train_set), args.block, args.seed)
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch_size, sampler=sampler, num_workers=args.workers,
        pin_memory=device.type == "cuda", drop_last=True, persistent_workers=args.workers > 0)
    val_loaders = {name: torch.utils.data.DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
        pin_memory=device.type == "cuda") for name, val_set in val_sets.items()}

    model = hm.PeopleSkyHead(width=args.width).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps_per_epoch = len(train_loader)
    total_steps = args.epochs * steps_per_epoch
    warmup_steps = max(1, int(args.warmup_epochs * steps_per_epoch))

    def learning_rate_factor(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, learning_rate_factor)
    start_epoch, best_score = 0, -1.0
    if args.resume:
        state = torch.load(out_dir / "last.pt", map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start_epoch, best_score = state["epoch"] + 1, state["best_score"]
        print(f"[resume] from epoch {state['epoch']}, best score {best_score:.4f}")

    hd.write_json(out_dir / "run.json", {
        "step": "README section 5, step 1: people and sky head training",
        "arguments": vars(args),
        "loss_weights": weights,
        "model": model.config,
        "parameters": hm.parameter_count(model),
        "train_images": len(train_set),
        "train_images_per_split": {d: len(part) for d, part in zip(args.train_dir, train_parts)},
        "val_images": len(val_sets[primary_val]),
        "val_images_per_split": {name: len(v) for name, v in val_sets.items()},
        "selection_split": primary_val,
        "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "torch": torch.__version__,
    })
    print(f"[model] {hm.parameter_count(model)} parameters, {steps_per_epoch} steps per epoch")

    for epoch in range(start_epoch, args.epochs):
        model.train()
        sampler.set_epoch(epoch)
        sums = {name: 0.0 for name in hl.DEFAULT_WEIGHTS}
        total_sum, started = 0.0, time.perf_counter()
        for step, batch in enumerate(train_loader):
            batch = to_device(batch, device)
            total, terms = hl.total_loss(model(batch["image_embed"]), batch, weights)
            optimizer.zero_grad(set_to_none=True)
            total.backward()
            optimizer.step()
            scheduler.step()
            total_sum += float(total.detach())
            for name, value in terms.items():
                sums[name] += value
            if (step + 1) % 100 == 0:
                print(f"[epoch {epoch}] step {step + 1} / {steps_per_epoch}, "
                      f"loss {total_sum / (step + 1):.4f}")
        train_seconds = time.perf_counter() - started
        results = {name: evaluate(model, loader, device) for name, loader in val_loaders.items()}
        result = results[primary_val]
        score = hmet.selection_score(result)
        line = {"epoch": epoch, "train_loss": total_sum / steps_per_epoch,
                "train_terms": {k: v / steps_per_epoch for k, v in sums.items()},
                "learning_rate": scheduler.get_last_lr()[0],
                "train_seconds": round(train_seconds, 1), "val": result,
                "val_extra": {k: v for k, v in results.items() if k != primary_val},
                "selection_score": score}
        with open(out_dir / "metrics.jsonl", "a", encoding="utf-8", newline="\n") as file:
            file.write(json.dumps(line) + "\n")
        if score > best_score:
            best_score = score
            torch.save({"model": model.state_dict(), "config": model.config, "epoch": epoch,
                        "val": result}, out_dir / "best.pt")
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(), "config": model.config,
                    "epoch": epoch, "best_score": best_score}, out_dir / "last.pt")
        print(f"[epoch {epoch}] loss {line['train_loss']:.4f}, {train_seconds:.0f} s | "
              f"sky IoU {result['sky_cell_iou']:.3f}, person AP50 {result['person_ap50']:.3f}, "
              f"top-1 {result['person_top1_hit']:.3f}, presence AUC sky "
              f"{result['presence_auc_sky']:.3f} person {result['presence_auc_person']:.3f}"
              f" | score {score:.4f} (best {best_score:.4f})")
        for name, extra in line["val_extra"].items():
            print(f"  [{name}] sky point AUC {optional(extra['sky_point_auc'])}, false sky "
                  f"{optional(extra['sky_point_false_sky_share'])}, person AP50 "
                  f"{optional(extra['person_ap50'])}, presence AUC sky "
                  f"{optional(extra['presence_auc_sky'])} person "
                  f"{optional(extra['presence_auc_person'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
