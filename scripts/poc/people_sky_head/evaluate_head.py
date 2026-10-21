#!/usr/bin/env python3
"""Validation measurements of a trained people and sky head checkpoint, without training (step 1).

Input:  --checkpoint (best.pt or last.pt of train_head.py), --val-dir (a split with targets/ and
        image_embed.npy).
Output: --out (JSON: the checkpoint epoch, the head_metrics.Evaluation result, the device).

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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--val-dir", required=True)
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
    dataset = hd.EmbedDataset(args.val_dir)
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                                         num_workers=args.workers,
                                         pin_memory=device.type == "cuda")
    evaluation = hmet.Evaluation()
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device, non_blocking=True) if hasattr(v, "to") else v
                     for k, v in batch.items()}
            evaluation.add(model(batch["image_embed"]), batch)
    result = evaluation.result()
    hd.write_json(Path(args.out), {
        "step": "README section 5, step 1: validation measurements of a checkpoint",
        "checkpoint": Path(args.checkpoint).name,
        "checkpoint_epoch": state["epoch"],
        "val_images": len(dataset),
        "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "torch": torch.__version__,
        "validation": result,
    })
    listing = result["person_list_best_f1_nms"]
    print(f"[done] epoch {state['epoch']}: AP50 {result['person_ap50']:.3f}, after the "
          f"suppression {result['person_ap50_nms']:.3f}; list threshold "
          f"{listing['threshold']:.3f} (F1 {listing['f1']:.3f})")
    for name, fit in result["person_box_fit_nms"].items():
        print(f"  {name}: {fit['persons']} persons, mean best IoU {fit['mean_best_iou']:.3f}, "
              f"IoU >= 0.75 {fit['share_best_iou_ge_0_75']:.3f}, median log height ratio "
              f"{fit['median_log_height_ratio']:+.3f}, width {fit['median_log_width_ratio']:+.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
