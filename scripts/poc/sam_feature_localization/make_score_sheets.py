#!/usr/bin/env python3
"""Score map overlays per photo and class (issue #1, step 7). The sheets show the photos: they stay
local and are not committed (`*.jpg` is in .gitignore).

Input:  --photos, --scores (score_maps.npz, metrics.json, end_to_end.json), --labels
        (labels.npz), --logits (decode_top_cells.py --logit-dir).
Output: --out-dir/<stem>.jpg, one sheet per photo: one row per class, one column per feature
        source, and a last column with the SAM 2.1 tiny masks of the image_embed prompts.

Score panels: the photo darkened, with the score in red. The color scale is the same for all
photos of one feature source and class: 0 is transparent, the highest score of the source and
class over all photos is full red. White outline: the class cells of the label grid. Green
squares: the 3 highest-scoring cells. Rectangles (person): the region boxes of
decode_top_cells.py, cyan for the rule above_zero and yellow for half_max. Mask panel: magenta is
the mask of the 3 points, the cyan outline is the mask of the above_zero region box.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import feature_localization as fl  # noqa: E402

TILE = 340
TITLE = 18
BOX_COLORS = {"region_box_above_zero": (0, 230, 255), "region_box_half_max": (255, 220, 0)}


def cell_rectangle(row: int, column: int, width: int, height: int):
    return (column * width / fl.GRID, row * height / fl.GRID,
            (column + 1) * width / fl.GRID, (row + 1) * height / fl.GRID)


def outline(mask: np.ndarray) -> np.ndarray:
    from scipy import ndimage

    return mask & ~ndimage.binary_erosion(mask)


def to_pixels(grid: np.ndarray, width: int, height: int) -> np.ndarray:
    rows = np.minimum((np.arange(height) * fl.GRID / height).astype(int), fl.GRID - 1)
    columns = np.minimum((np.arange(width) * fl.GRID / width).astype(int), fl.GRID - 1)
    return grid[np.ix_(rows, columns)]


def score_panel(photo, score, scale, positive, top, boxes):
    from PIL import Image, ImageDraw

    base = np.asarray(photo).astype(np.float32) * 0.45
    height, width = base.shape[:2]
    alpha = to_pixels(np.clip(score / scale, 0, 1), width, height)[..., None] * 0.85
    image = base * (1 - alpha) + np.array([255, 40, 30], np.float32) * alpha
    edge = outline(to_pixels(positive, width, height))
    image[edge] = (255, 255, 255)
    panel = Image.fromarray(image.clip(0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(panel)
    for row, column in top:
        draw.rectangle(cell_rectangle(row, column, width, height), outline=(60, 255, 60), width=2)
    for box, color in boxes:
        draw.rectangle([box[0] / fl.IMAGE_SIZE * width, box[1] / fl.IMAGE_SIZE * height,
                        box[2] / fl.IMAGE_SIZE * width, box[3] / fl.IMAGE_SIZE * height],
                       outline=color, width=2)
    return panel


def mask_panel(photo, points_logits, box_logits):
    import cv2
    from PIL import Image

    base = np.asarray(photo).astype(np.float32) * 0.6
    height, width = base.shape[:2]
    if points_logits is not None:
        mask = cv2.resize(points_logits.astype(np.float32), (width, height)) > 0
        base[mask] = base[mask] * 0.4 + np.array([255, 0, 220]) * 0.6
    if box_logits is not None:
        mask = cv2.resize(box_logits.astype(np.float32), (width, height)) > 0
        base[outline(mask)] = (0, 230, 255)
    return Image.fromarray(base.clip(0, 255).astype(np.uint8))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--photos", required=True)
    parser.add_argument("--scores", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--logits", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--quality", type=int, default=82)
    args = parser.parse_args()

    from PIL import Image, ImageDraw

    scores_dir = Path(args.scores)
    score_maps = np.load(scores_dir / "score_maps.npz")
    metrics = fl.read_json(scores_dir / "metrics.json")["sources"]
    end_to_end = fl.read_json(scores_dir / "end_to_end.json")["results"]
    labels = np.load(Path(args.labels) / "labels.npz")
    boxes = {(r["source"], r["photo"], r["class"], r["prompt"]): r["coords"][0] + r["coords"][1]
             for r in end_to_end if r["prompt"] in BOX_COLORS}
    ious = {(r["source"], r["photo"], r["class"], r["prompt"]): r["iou"] for r in end_to_end}
    stems = sorted(metrics["image_embed"]["person"]["photos"])
    scales = {(s, c): max(float(score_maps[f"{s}__{stem}__{c}"].max()) for stem in stems)
              for s in fl.FEATURE_SOURCES for c in fl.CLASSES}
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for stem in stems:
        photo = Image.open(Path(args.photos) / f"{stem}.jpg").convert("RGB")
        photo.thumbnail((TILE, TILE))
        width, height = photo.size
        columns = len(fl.FEATURE_SOURCES) + 1
        sheet = Image.new("RGB", (columns * (width + 6), len(fl.CLASSES) * (height + TITLE + 6)),
                          (20, 20, 20))
        draw = ImageDraw.Draw(sheet)
        for row, name in enumerate(fl.CLASSES):
            y = row * (height + TITLE + 6)
            status = metrics["image_embed"][name]["photos"][stem]["status"]
            for column, source in enumerate(fl.FEATURE_SOURCES):
                score = score_maps[f"{source}__{stem}__{name}"]
                drawn = [(boxes[(source, stem, name, prompt)], color)
                         for prompt, color in BOX_COLORS.items()
                         if (source, stem, name, prompt) in boxes]
                panel = score_panel(photo, score, scales[(source, name)],
                                    labels[f"{stem}__{name}_positive"],
                                    fl.top_cells(score, 3), drawn)
                x = column * (width + 6)
                sheet.paste(panel, (x, y + TITLE))
                m = metrics[source][name]["photos"][stem]
                text = f"{name} | {source} | max {m['max_score']:.3f}"
                if status == "referenced":
                    text += f" | AUC {m['auc']:.3f}" if m["auc"] is not None else ""
                draw.text((x + 2, y + 3), text, fill=(230, 230, 230))
            x = len(fl.FEATURE_SOURCES) * (width + 6)
            if status == "referenced":
                logit_file = Path(args.logits) / f"image_embed__{stem}__{name}__"
                points = np.load(f"{logit_file}top3_points.npy")
                box_path = Path(f"{logit_file}region_box_above_zero.npy")
                box_logits = np.load(box_path) if box_path.exists() else None
                sheet.paste(mask_panel(photo, points, box_logits), (x, y + TITLE))
                text = f"image_embed masks | points IoU " \
                       f"{ious[('image_embed', stem, name, 'top3_points')]:.2f}"
                if box_logits is not None:
                    key = ("image_embed", stem, name, "region_box_above_zero")
                    text += f" | box IoU {ious[key]:.2f}"
            else:
                text = f"{name}: {status}"
            draw.text((x + 2, y + 3), text, fill=(230, 230, 230))
        sheet.save(out_dir / f"{stem}.jpg", quality=args.quality)
        print(f"[sheet] {out_dir / f'{stem}.jpg'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
