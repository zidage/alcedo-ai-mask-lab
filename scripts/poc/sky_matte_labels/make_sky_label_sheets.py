#!/usr/bin/env python3
"""Check sheets of the sky matte labels (local only, the photos are not committed).

Input:  --manifest, --photo-dir, --panoptic-zip, --panoptic-json, --label-dir (the --out-dir of
        make_sky_matte_labels.py), optional --image-ids.
Output: --sheet-dir/<group>_<image id>.jpg and --sheet-dir/index.html (the sheets by group, with
        the statistics of label_stats.json).

Step: sky matte labels (README decision U17). Each sheet has two rows:

  row 1  photo | coarse COCO sky (blue tint) | trimap (white sky, gray unknown) |
         alpha_label (gray), red where label_weight is 0 (ignored by the training loss)
  row 2  zoom of the 256 x 256 photo window with the most fractional alpha_label:
         photo | coarse sky | alpha_label as a blue tint | the same, red where label_weight is 0
"""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sky_label_data as sd  # noqa: E402

TILE = 480
ZOOM_WINDOW = 256
TINT = np.array([40.0, 110.0, 255.0])
TINT_STRENGTH = 0.6


def tint(photo: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    weight = (TINT_STRENGTH * alpha)[..., None]
    return (photo * (1.0 - weight) + TINT * weight).astype(np.uint8)


def mark_ignored(image: np.ndarray, ignored: np.ndarray) -> np.ndarray:
    marked = image.copy()
    marked[ignored] = (255, 0, 0)
    return marked


def gray(values: np.ndarray) -> np.ndarray:
    channel = np.round(np.clip(values, 0.0, 1.0) * 255).astype(np.uint8)
    return np.repeat(channel[..., None], 3, axis=2)


def fit(image: np.ndarray, size: int) -> Image.Image:
    picture = Image.fromarray(image)
    picture.thumbnail((size, size), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (size, size), (24, 24, 24))
    canvas.paste(picture, ((size - picture.width) // 2, (size - picture.height) // 2))
    return canvas


def zoom_window(alpha: np.ndarray) -> tuple[int, int]:
    """Top-left corner of the window with the most pixels with a fractional alpha."""
    fractional = ((alpha > 0.05) & (alpha < 0.95)).astype(np.float64)
    size = min(ZOOM_WINDOW, *alpha.shape)
    counts = ndimage.uniform_filter(fractional, size=size, mode="constant")
    half = size // 2
    inner = counts[half:alpha.shape[0] - half + 1, half:alpha.shape[1] - half + 1]
    y, x = np.unravel_index(np.argmax(inner), inner.shape)
    return int(y), int(x)


def write_index(sheet_dir: Path, items: list[dict], stats: dict[int, dict]) -> None:
    """index.html: one section per group, each sheet with its label statistics."""
    keys = ("unknown_share", "label_coarse_iou", "label_fractional_share",
            "unstable_share_of_unknown")
    parts = ["<!doctype html><meta charset='utf-8'><title>Sky matte labels</title>",
             "<style>body{background:#181818;color:#ddd;font:14px sans-serif}"
             "img{width:100%;max-width:1920px}</style>"]
    for group in ("tree", "building", "other"):
        members = [item for item in items if item["group"] == group]
        parts.append(f"<h2>{group} ({len(members)})</h2>")
        for item in members:
            entry = stats.get(item["id"], {})
            values = " ".join(f"{key} {entry[key]}" for key in keys if key in entry)
            name = f"{group}_{item['id']:012d}.jpg"
            parts.append(f"<p>{item['id']} license {item['license']} {values}</p>"
                         f"<img src='{name}' loading='lazy'>")
    (sheet_dir / "index.html").write_text("\n".join(parts) + "\n", encoding="utf-8",
                                          newline="\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--photo-dir", type=Path, required=True)
    parser.add_argument("--panoptic-zip", type=Path, required=True)
    parser.add_argument("--panoptic-json", type=Path, required=True)
    parser.add_argument("--label-dir", type=Path, required=True)
    parser.add_argument("--sheet-dir", type=Path, required=True)
    parser.add_argument("--image-ids", type=str, default="")
    args = parser.parse_args()

    items = json.loads(args.manifest.read_text(encoding="utf-8"))["images"]
    if args.image_ids:
        wanted = {int(part) for part in args.image_ids.split(",")}
        items = [item for item in items if item["id"] in wanted]
    items = [item for item in items
             if (sd.label_dir(args.label_dir, item["id"]) / "alpha_label.png").exists()]
    data = json.loads(args.panoptic_json.read_text(encoding="utf-8"))
    sky_ids = {c["id"] for c in data["categories"] if c["name"] == sd.SKY_CATEGORY}
    segments = {a["image_id"]: a["segments_info"] for a in data["annotations"]}
    del data

    args.sheet_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.panoptic_zip) as archive:
        for item in items:
            photo = np.asarray(Image.open(sd.photo_path(args.photo_dir, item["id"]))
                               .convert("RGB"), dtype=np.float64)
            height, width = photo.shape[:2]
            segment_ids = sd.read_panoptic_png(archive, item["panoptic_file_name"])
            coarse = sd.resize_mask(sd.category_mask(segment_ids, segments[item["id"]], sky_ids),
                                    width, height).astype(np.float64)
            directory = sd.label_dir(args.label_dir, item["id"])
            trimap = np.asarray(Image.open(directory / "trimap.png"), dtype=np.float64) / 255
            alpha = np.asarray(Image.open(directory / "alpha_label.png"),
                               dtype=np.float64) / 255
            ignored = np.asarray(Image.open(directory / "label_weight.png")) < 128

            row1 = [photo.astype(np.uint8), tint(photo, coarse), gray(trimap),
                    mark_ignored(gray(alpha), ignored)]
            y, x = zoom_window(alpha)
            size = min(ZOOM_WINDOW, height, width)
            window = (slice(y, y + size), slice(x, x + size))
            crop = photo[window]
            tinted = tint(crop, alpha[window])
            row2 = [crop.astype(np.uint8), tint(crop, coarse[window]), tinted,
                    mark_ignored(tinted, ignored[window])]

            sheet = Image.new("RGB", (4 * TILE, 2 * TILE), (24, 24, 24))
            for column, tile in enumerate(row1):
                sheet.paste(fit(tile, TILE), (column * TILE, 0))
            for column, tile in enumerate(row2):
                picture = Image.fromarray(tile).resize((TILE, TILE), Image.Resampling.NEAREST)
                sheet.paste(picture, (column * TILE, TILE))
            draw = ImageDraw.Draw(sheet)
            labels = ["photo", "coarse COCO sky", "trimap", "alpha_label, red ignored",
                      f"zoom x={x} y={y}", "coarse", "alpha_label", "red ignored"]
            for index, label in enumerate(labels):
                draw.text((index % 4 * TILE + 6, index // 4 * TILE + 4), label,
                          fill=(255, 255, 0))
            sheet.save(args.sheet_dir / f"{item['group']}_{item['id']:012d}.jpg", quality=90)
            print(item["id"], flush=True)
    stats_path = args.label_dir / "label_stats.json"
    stats = ({entry["id"]: entry for entry in
              json.loads(stats_path.read_text(encoding="utf-8"))["images"]}
             if stats_path.exists() else {})
    write_index(args.sheet_dir, items, stats)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
