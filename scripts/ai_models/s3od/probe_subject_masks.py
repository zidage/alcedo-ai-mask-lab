#!/usr/bin/env python3
"""Quality metrics of Subject alphas on the quality photo set, S3OD, IS-Net, universal matting,
and BiRefNet (issue #6).

Input:  one or more --alpha LABEL=KIND:DIR (KIND s3od: the unpadded alpha of the long edge 1024 px
        of compare_onnx_outputs.py or measure_directml_cost.py; KIND birefnet: the 1024 x 1024
        alpha of the E2 compare_onnx_outputs.py or of measure_directml_cost.py; KIND isnet: the
        1024 x 1024 alpha of the IS-Net compare_onnx_outputs.py or of measure_directml_cost.py;
        KIND universal_matting: the alpha at the photo size of the universal matting
        compare_onnx_outputs.py or of measure_directml_cost.py), the photos of --prompts in --photos, the E1 SAM 2.1 box masks (--subject-masks).
Output: --out (JSON): per label and photo the metrics below, and the mean over the photos.

All alphas are compared on one grid per photo: the photo aspect with the long edge 1024 px (the
`resized_size` of s3od_graphs.pad_info, the S3OD model grid). A BiRefNet or IS-Net alpha is
resized from 1024 x 1024, a universal matting alpha from the photo size, to that grid (PyTorch
bilinear, `antialias=True`). Metrics on the
alpha > 0.5 mask, with the definitions of the E2 probe_mattes.py:

- foreground: share of the grid pixels;
- soft: share of pixels with 0.05 < alpha < 0.95;
- edge width: soft pixels divided by the boundary pixels of the mask, in grid pixels;
- parts: connected parts with at least 1 % of the foreground area; hole and island fractions;
- subject recall: share of the SAM 2.1 box mask (E1 reference mask of the first box prompt of the
  photo) inside the foreground;
- outside box: share of the foreground outside the box of that prompt;
- IoU with each --iou-reference (`iou_with_<label>`: the BiRefNet lite mask, and the PyTorch
  float32 S3OD mask to show a wrong DirectML output), and for each --region the soft share and
  the mean alpha in that rectangle.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

import s3od_graphs as graphs  # noqa: E402

HERE = Path(__file__).resolve().parent
KINDS = ("s3od", "birefnet", "isnet", "universal_matting")


def load_probe_mattes():
    """`holes_and_islands`, `boundary_pixels`, `parse_region`, and the soft range of E2."""
    path = HERE.parent / "birefnet" / "probe_mattes.py"
    spec = importlib.util.spec_from_file_location("probe_mattes", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_alpha(text: str) -> tuple[str, str, Path]:
    label, rest = text.split("=", 1)
    kind, directory = rest.split(":", 1)
    if kind not in KINDS:
        raise ValueError(f"{text!r}: kind must be one of {KINDS}")
    return label, kind, Path(directory)


def alpha_on_grid(path: Path, kind: str, grid: tuple[int, int]) -> np.ndarray:
    alpha = np.load(path).astype(np.float32)
    if kind == "s3od":
        if alpha.shape != grid:
            raise ValueError(f"{path}: shape {alpha.shape}, expected {grid}")
        return alpha
    return graphs.masks_to_photo(alpha[None], *grid)[0].clip(0.0, 1.0)


def iou(a: np.ndarray, b: np.ndarray) -> float:
    union = np.logical_or(a, b).sum()
    return 1.0 if union == 0 else float(np.logical_and(a, b).sum() / union)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--alpha", action="append", required=True, type=parse_alpha)
    parser.add_argument("--iou-reference", action="append", default=[],
                        help="label of a reference alpha for the IoU (repeatable)")
    parser.add_argument("--photos", required=True)
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--subject-masks", required=True,
                        help="E1 reference masks (<stem>__<box name>__single0.png)")
    parser.add_argument("--region", action="append", default=[])
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    probe = load_probe_mattes()
    regions = [probe.parse_region(text) for text in args.region]
    prompts = json.loads(Path(args.prompts).read_text(encoding="utf-8"))
    results: dict[str, list[dict]] = {label: [] for label, _, _ in args.alpha}
    for entry in prompts["images"]:
        stem = Path(entry["file"]).stem
        with Image.open(Path(args.photos) / entry["file"]) as photo:
            width, height = photo.size
        grid = graphs.pad_info(height, width)["resized_size"]
        box_prompt = entry["boxes"][0]
        subject = np.asarray(Image.open(
            Path(args.subject_masks) / f"{stem}__{box_prompt['name']}__single0.png"
        ).convert("L").resize(grid[::-1], Image.Resampling.NEAREST)) > 127
        x0, y0, x1, y1 = box_prompt["box"]
        inside_box = np.zeros(grid, dtype=bool)
        inside_box[round(y0 * grid[0]):round(y1 * grid[0]), round(x0 * grid[1]):round(x1 * grid[1])] \
            = True
        alphas = {label: alpha_on_grid(directory / f"{stem}.npy", kind, grid)
                  for label, kind, directory in args.alpha}
        references = {label: alphas[label] > 0.5 for label in args.iou_reference}
        for label, alpha in alphas.items():
            mask = alpha > 0.5
            soft = (alpha > probe.SOFT_LOW) & (alpha < probe.SOFT_HIGH)
            holes, islands, components = probe.holes_and_islands(mask)
            boundary = probe.boundary_pixels(mask)
            record = {
                "image": entry["file"], "grid": list(grid),
                "foreground": float(mask.mean()),
                "soft": float(soft.mean()),
                "edge_width_px": float(soft.sum() / boundary) if boundary else 0.0,
                "parts": components,
                "hole_fraction": holes,
                "island_fraction": islands,
                "subject": box_prompt["name"],
                "subject_recall": float((mask & subject).sum() / max(subject.sum(), 1)),
                "outside_box": float((mask & ~inside_box).sum() / max(mask.sum(), 1)),
                "regions": {},
            }
            for reference_label, reference in references.items():
                record[f"iou_with_{reference_label}"] = iou(mask, reference)
            for file, name, (rx0, ry0, rx1, ry1) in regions:
                if file != entry["file"]:
                    continue
                crop = (slice(round(ry0 * grid[0]), round(ry1 * grid[0])),
                        slice(round(rx0 * grid[1]), round(rx1 * grid[1])))
                record["regions"][name] = {"soft": float(soft[crop].mean()),
                                           "mean_alpha": float(alpha[crop].mean())}
            results[label].append(record)
            print(f"[probe] {label} {stem}: fg {record['foreground']:.3f} "
                  f"edge {record['edge_width_px']:.1f} px parts {components} "
                  f"recall {record['subject_recall']:.3f} outside {record['outside_box']:.3f}"
                  + "".join(f" IoU {key} {record[f'iou_with_{key}']:.3f}" for key in references),
                  flush=True)

    keys = ("foreground", "soft", "edge_width_px", "parts", "hole_fraction", "island_fraction",
            "subject_recall", "outside_box") + tuple(f"iou_with_{label}"
                                                     for label in args.iou_reference)
    means = {label: {key: float(np.mean([r[key] for r in records]))
                     for key in keys if key in records[0]}
             for label, records in results.items()}
    report = {"grid": "photo aspect, long edge 1024 px", "soft_range": [probe.SOFT_LOW,
                                                                       probe.SOFT_HIGH],
              "iou_references": args.iou_reference,
              "alphas": {label: {"kind": kind, "dir": directory.as_posix()}
                         for label, kind, directory in args.alpha},
              "means": means, "results": results}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
