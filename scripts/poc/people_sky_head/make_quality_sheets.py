#!/usr/bin/env python3
"""Overlay sheets of the end-to-end run on the quality photos (step 1, local only).

Input:  --photos, --results (evaluate_quality_photos.py quality_photos.json), --logits (its
        --logit-dir), --min-score (the People list entries drawn).
Output: --out-dir/<photo>.jpg, two panels per photo at --width px each:
          left   sky: the union of the region masks (blue) and the region points (white)
          right  people: the list boxes with score >= --min-score (rank and score; yellow), the
                 top-1 mask (green) and, on photos with a referenced person, the mask of the
                 list entry with the highest IoU (magenta) where it is not the top-1 entry

The sheets show the quality photos. They must not be committed or shared (`*.jpg` is in
.gitignore).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

GRID = 64
CELL_MODEL_PIXELS = 16


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--photos", required=True)
    parser.add_argument("--results", required=True)
    parser.add_argument("--logits", required=True)
    parser.add_argument("--min-score", type=float, default=0.2)
    parser.add_argument("--width", type=int, default=900)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    import cv2
    from PIL import Image, ImageDraw

    results = json.loads(Path(args.results).read_text(encoding="utf-8"))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logits = Path(args.logits)

    for stem, record in results["photos"].items():
        photo = Image.open(Path(args.photos) / f"{stem}.jpg").convert("RGB")
        height = round(photo.height * args.width / photo.width)
        base = np.asarray(photo.resize((args.width, height), Image.BILINEAR), np.float32)

        def mask_of(name: str) -> np.ndarray:
            data = np.load(logits / f"{stem}__{name}.npy").astype(np.float32)
            return cv2.resize(data, (args.width, height), interpolation=cv2.INTER_LINEAR) > 0

        def tint(image: np.ndarray, mask: np.ndarray, color: tuple[int, int, int]) -> None:
            image[mask] = 0.45 * image[mask] + 0.55 * np.array(color, np.float32)

        def model_to_panel(x: float, y: float) -> tuple[float, float]:
            return x / 1024 * args.width, y / 1024 * height

        sky_panel = base.copy()
        for index in range(len(record["sky"]["regions"])):
            tint(sky_panel, mask_of(f"sky_region{index}"), (40, 110, 255))
        people_panel = base.copy()
        entries = record["person"]["list"]
        if entries and "iou" in entries[0]:
            tint(people_panel, mask_of("person_rank1"), (40, 220, 80))
            best_rank = record["person"]["best_rank"]
            if best_rank and best_rank != 1:
                tint(people_panel, mask_of(f"person_rank{best_rank}"), (230, 40, 230))

        sky_image = Image.fromarray(sky_panel.clip(0, 255).astype(np.uint8))
        draw = ImageDraw.Draw(sky_image)
        for region in record["sky"]["regions"]:
            for x, y in region["points"]:
                px, py = model_to_panel(x, y)
                draw.ellipse([px - 5, py - 5, px + 5, py + 5], outline=(255, 255, 255), width=2)
        sky = record["sky"]
        draw.text((8, 8), f"sky {sky['status']}: presence {sky['presence']:.3f}, "
                          f"found {sky['found']}, IoU {sky.get('iou', float('nan')):.3f}",
                  fill=(255, 255, 0))

        people_image = Image.fromarray(people_panel.clip(0, 255).astype(np.uint8))
        draw = ImageDraw.Draw(people_image)
        for entry in entries:
            if entry["score"] < args.min_score:
                continue
            x0, y0, x1, y1 = (v * CELL_MODEL_PIXELS for v in entry["box_cells"])
            (px0, py0), (px1, py1) = model_to_panel(x0, y0), model_to_panel(x1, y1)
            draw.rectangle([px0, py0, px1, py1], outline=(255, 220, 0), width=2)
            draw.text((px0 + 3, py0 + 2), f"{entry['rank']}:{entry['score']:.2f}",
                      fill=(255, 220, 0))
        person = record["person"]
        draw.text((8, 8), f"person {person['status']}: presence {person['presence']:.3f}, "
                          f"found {person['found']}, top-1 IoU "
                          f"{person.get('top1_iou', float('nan')):.3f}, best "
                          f"{person.get('best_iou', float('nan')):.3f} at rank "
                          f"{person.get('best_rank')}", fill=(255, 255, 0))

        sheet = Image.new("RGB", (2 * args.width + 10, height), (0, 0, 0))
        sheet.paste(sky_image, (0, 0))
        sheet.paste(people_image, (args.width + 10, 0))
        sheet.save(out_dir / f"{stem}.jpg", quality=88)
        print(f"[sheet] {out_dir / f'{stem}.jpg'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
