#!/usr/bin/env python3
"""Validation measurements of a trained people and sky head checkpoint, without training (step 1).

Input:  --checkpoint (best.pt or last.pt of train_head.py), --val-dir (one or more splits with
        targets/ and image_embed.npy).
Output: --out (JSON: the checkpoint epoch, the head_metrics.Evaluation result of each split by
        directory name, the device).

Use: the measurements that head_metrics adds after a training run (for example the People list
threshold `person_list_best_f1_nms` and `person_box_fit_nms`) on an existing checkpoint.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import head_data as hd  # noqa: E402
import head_metrics as hmet  # noqa: E402
import head_model as hm  # noqa: E402


def optional(value: float | None) -> str:
    """A measurement for the log line; '-' when it is not measured on the split."""
    return "-" if value is None or value != value else f"{value:.3f}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--val-dir", required=True, nargs="+")
    parser.add_argument("--out", required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", default="cuda",
                        help="cuda (Colab) or cpu (a check of the script on a few images)")
    args = parser.parse_args()

    import torch

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("no CUDA device: select a GPU runtime")
    device = torch.device(args.device)
    state = torch.load(args.checkpoint, map_location=device, weights_only=True)
    config = state["config"]
    model = hm.PeopleSkyHead(config["in_channels"], config["width"], tuple(config["dilations"]),
                             config["groups"]).to(device)
    model.load_state_dict(state["model"])
    model.eval()
    results, images = {}, {}
    for val_dir in args.val_dir:
        name = Path(val_dir).name
        if name in results:
            raise ValueError(f"--val-dir names must differ: {args.val_dir}")
        dataset = hd.EmbedDataset(val_dir)
        loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                                             num_workers=args.workers,
                                             pin_memory=device.type == "cuda")
        evaluation = hmet.Evaluation()
        with torch.no_grad():
            for batch in loader:
                batch = {k: v.to(device, non_blocking=True) if hasattr(v, "to") else v
                         for k, v in batch.items()}
                evaluation.add(model(batch["image_embed"]), batch)
        results[name], images[name] = evaluation.result(), len(dataset)
    hd.write_json(Path(args.out), {
        "step": "README section 5, step 1: validation measurements of a checkpoint",
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": state["epoch"],
        "val_images": images,
        "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "torch": torch.__version__,
        "validation": results,
    })
    for name, result in results.items():
        listing = result["person_list_best_f1_nms"]
        print(f"[{name}] epoch {state['epoch']}: sky cell IoU {result['sky_cell_iou']:.3f}, sky "
              f"point AUC {optional(result['sky_point_auc'])}, false sky "
              f"{optional(result['sky_point_false_sky_share'])}; presence AUC sky "
              f"{optional(result['presence_auc_sky'])} person "
              f"{optional(result['presence_auc_person'])}; AP50 {optional(result['person_ap50'])},"
              f" after the suppression {optional(result['person_ap50_nms'])}; list threshold "
              f"{optional(listing['threshold'])} (F1 {listing['f1']:.3f})")
        for size, fit in result["person_box_fit_nms"].items():
            print(f"  {size}: {fit['persons']} persons, mean best IoU "
                  f"{optional(fit['mean_best_iou'])}, IoU >= 0.75 "
                  f"{optional(fit['share_best_iou_ge_0_75'])}, median log height ratio "
                  f"{optional(fit['median_log_height_ratio'])}, width "
                  f"{optional(fit['median_log_width_ratio'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
