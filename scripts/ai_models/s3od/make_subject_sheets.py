#!/usr/bin/env python3
"""Comparison sheets of Subject alphas for the check by eye (issue #6). Local only.

Input:  one or more --alpha LABEL=KIND:DIR (the format of probe_subject_masks.py), the photos of
        --prompts in --photos, optional --crop FILE:NAME:X0,Y0,X1,Y1 (normalized photo
        coordinates).
Output: in --out-dir (a directory in .gitignore, for example docs/exploration/images/), per photo
        `<stem>.jpg`: one panel per alpha, the photo at a long edge of --panel px with the alpha
        > 0.5 area tinted red and the alpha 0.05 to 0.95 band tinted yellow; per crop
        `<stem>_<name>.jpg`: the crop at the photo resolution with the alpha resized to the photo
        size (bilinear), one panel per alpha.

The sheets show the quality photos. They must not be committed or shared (AGENTS.md).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

import s3od_graphs as graphs  # noqa: E402
from probe_subject_masks import alpha_on_grid, load_probe_mattes, parse_alpha  # noqa: E402

LABEL_HEIGHT = 28


def tint(image: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """Red tint on alpha > 0.5, yellow tint on the soft band 0.05 < alpha < 0.95."""
    out = image.astype(np.float32)
    inside = alpha > 0.5
    soft = (alpha > 0.05) & (alpha < 0.95)
    out[inside] = out[inside] * 0.55 + np.array([255, 0, 0]) * 0.45
    out[soft] = out[soft] * 0.4 + np.array([255, 230, 0]) * 0.6
    return out.clip(0, 255).astype(np.uint8)


def sheet(panels: list[tuple[str, np.ndarray]]) -> Image.Image:
    height, width = panels[0][1].shape[:2]
    canvas = Image.new("RGB", (width * len(panels), height + LABEL_HEIGHT), "white")
    draw = ImageDraw.Draw(canvas)
    for index, (label, panel) in enumerate(panels):
        canvas.paste(Image.fromarray(panel), (index * width, LABEL_HEIGHT))
        draw.text((index * width + 6, 6), label, fill="black")
    return canvas


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--alpha", action="append", required=True, type=parse_alpha)
    parser.add_argument("--photos", required=True)
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--crop", action="append", default=[])
    parser.add_argument("--panel", type=int, default=900)
    parser.add_argument("--quality", type=int, default=80)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    probe = load_probe_mattes()
    crops = [probe.parse_region(text) for text in args.crop]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prompts = json.loads(Path(args.prompts).read_text(encoding="utf-8"))
    for entry in prompts["images"]:
        stem = Path(entry["file"]).stem
        photo = Image.open(Path(args.photos) / entry["file"]).convert("RGB")
        width, height = photo.size
        grid = graphs.pad_info(height, width)["resized_size"]
        alphas = {label: alpha_on_grid(directory / f"{stem}.npy", kind, grid)
                  for label, kind, directory in args.alpha}
        scale = args.panel / max(width, height)
        size = (round(width * scale), round(height * scale))
        small = np.asarray(photo.resize(size, Image.Resampling.LANCZOS))
        panels = [(label, tint(small, graphs.masks_to_photo(alpha[None], size[1], size[0])[0]))
                  for label, alpha in alphas.items()]
        sheet(panels).save(out_dir / f"{stem}.jpg", quality=args.quality)
        for file, name, (x0, y0, x1, y1) in crops:
            if file != entry["file"]:
                continue
            box = (round(x0 * width), round(y0 * height), round(x1 * width), round(y1 * height))
            region = np.asarray(photo.crop(box))
            crop_panels = []
            for label, alpha in alphas.items():
                full = graphs.masks_to_photo(alpha[None], height, width)[0]
                crop_panels.append((label, tint(region, full[box[1]:box[3], box[0]:box[2]])))
            sheet(crop_panels).save(out_dir / f"{stem}_{name}.jpg", quality=args.quality)
        print(f"[sheets] {stem}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
