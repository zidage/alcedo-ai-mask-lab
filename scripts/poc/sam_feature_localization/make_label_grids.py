#!/usr/bin/env python3
"""Label grid per photo on the 64 x 64 encoder grid (issue #1, step 2).

Input:  --reference-masks (E1 reference masks at the 4096 px input, E4 input set),
        --reference-scores (E1 base plus compare.json, selects the best multimask sky mask), and
        class_presence.json of this directory.
Output: --out-dir/labels.npz and --out-dir/labels.json; with --sheet, a contact sheet of the
        label grids over the photos (local only: it shows the photos).

Labels per cell (a cell is 1/64 of the photo width by 1/64 of the photo height):

  person (1)    : at least --pure-fraction (0.9) of the cell pixels are in the person reference
  sky (2)       : at least --pure-fraction of the cell pixels are in the sky reference
  other (0)     : at most 1 - --pure-fraction of the cell pixels are in a person or sky
                  reference, and no pixel is in an ignore box of class_presence.json
  excluded (-1) : the remaining cells (mixed cells, and cells of unreferenced people or sky)

The label grid gives the cells of the prototypes. The measurements of each class use two sets
per class, so that a cell that is uncertain for one class can still count for the other:

  <class>_positive : the cells with the label of the class
  <class>_negative : at most 1 - --pure-fraction of the cell pixels are in the class reference,
                     and no pixel is in an ignore box of that class (the cells of the other class
                     are negatives)

The person reference is the union of the E1 single-mask references of the photo, minus the
excluded references (the board of 2088064441). The sky reference is the E1 multimask output with
the highest predicted IoU of the sky point prompt. labels.npz also holds, per photo and class, the
cell fraction and the cells whose center pixel is inside the reference (used by the
"center" precision of score_prototypes.py).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import feature_localization as fl  # noqa: E402

LABEL_COLORS = {fl.LABEL_PERSON: (230, 60, 40), fl.LABEL_SKY: (250, 210, 0)}


def label_sheet(photos_dir: Path, grids: dict, out: Path, tile: int = 360) -> None:
    from PIL import Image, ImageDraw

    stems = sorted(grids)
    columns = 5
    rows = (len(stems) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * tile, rows * (tile + 20)), (24, 24, 24))
    draw_sheet = ImageDraw.Draw(sheet)
    for index, stem in enumerate(stems):
        photo = Image.open(photos_dir / f"{stem}.jpg").convert("RGB")
        photo.thumbnail((tile, tile))
        width, height = photo.size
        base = np.asarray(photo).astype(np.float32)
        label = grids[stem]
        overlay = np.zeros_like(base)
        alpha = np.zeros(base.shape[:2], np.float32)
        ys = np.minimum((np.arange(height) * fl.GRID / height).astype(int), fl.GRID - 1)
        xs = np.minimum((np.arange(width) * fl.GRID / width).astype(int), fl.GRID - 1)
        per_pixel = label[np.ix_(ys, xs)]
        for value, color in LABEL_COLORS.items():
            overlay[per_pixel == value] = color
            alpha[per_pixel == value] = 0.55
        excluded = per_pixel == fl.LABEL_EXCLUDED
        overlay[excluded] = (0, 0, 0)
        alpha[excluded] = 0.6
        mixed = base * (1 - alpha[..., None]) + overlay * alpha[..., None]
        image = Image.fromarray(mixed.clip(0, 255).astype(np.uint8))
        x0 = (index % columns) * tile
        y0 = (index // columns) * (tile + 20)
        sheet.paste(image, (x0 + (tile - width) // 2, y0 + 20 + (tile - height) // 2))
        draw_sheet.text((x0 + 4, y0 + 4), stem, fill=(230, 230, 230))
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out, quality=85)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reference-masks", required=True)
    parser.add_argument("--reference-scores", required=True, help="E1 base plus compare.json")
    parser.add_argument("--photos", required=True, help="E4 input set photos (for the sizes)")
    parser.add_argument("--presence", default=str(fl.HERE / "class_presence.json"))
    parser.add_argument("--pure-fraction", type=float, default=0.9)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--sheet", default="", help="contact sheet path (local only)")
    args = parser.parse_args()

    from PIL import Image

    presence = fl.read_json(Path(args.presence))["photos"]
    compare_results = fl.read_json(Path(args.reference_scores))["results"]
    arrays = {}
    summary = {}
    grids = {}
    for stem, classes in sorted(presence.items()):
        with Image.open(Path(args.photos) / f"{stem}.jpg") as photo:
            width, height = photo.size
        fractions = {}
        photo_summary = {"width": width, "height": height}
        ignore_boxes = []
        for name in fl.CLASSES:
            entry = classes[name]
            ignore_boxes += entry.get("ignore_boxes", [])
            class_ignored = fl.cell_fraction(
                fl.box_mask(entry.get("ignore_boxes", []), width, height)) > 0
            item = {"status": entry["status"]}
            if entry["status"] == "referenced":
                mask, files = fl.class_reference_mask(stem, entry, Path(args.reference_masks),
                                                      compare_results)
                if mask.shape != (height, width):
                    raise ValueError(f"{stem} {name}: reference {mask.shape} != photo "
                                     f"{(height, width)}")
                fractions[name] = fl.cell_fraction(mask)
                arrays[f"{stem}__{name}_center"] = fl.cell_center_inside(mask)
                item["reference_files"] = files
                item["reference_area_fraction"] = float(mask.mean())
            else:
                fractions[name] = np.zeros((fl.GRID, fl.GRID), np.float32)
                arrays[f"{stem}__{name}_center"] = np.zeros((fl.GRID, fl.GRID), bool)
            arrays[f"{stem}__{name}_fraction"] = fractions[name]
            positive = fractions[name] >= args.pure_fraction
            negative = (fractions[name] <= 1 - args.pure_fraction) & ~class_ignored
            arrays[f"{stem}__{name}_positive"] = positive
            arrays[f"{stem}__{name}_negative"] = negative
            item["positive_cells"] = int(positive.sum())
            item["negative_cells"] = int(negative.sum())
            item["center_cells"] = int(arrays[f"{stem}__{name}_center"].sum())
            photo_summary[name] = item
        ignored = fl.cell_fraction(fl.box_mask(ignore_boxes, width, height)) > 0
        label = np.full((fl.GRID, fl.GRID), fl.LABEL_EXCLUDED, np.int8)
        labeled = fractions["person"] + fractions["sky"]
        label[(labeled <= 1 - args.pure_fraction) & ~ignored] = fl.LABEL_OTHER
        label[fractions["person"] >= args.pure_fraction] = fl.LABEL_PERSON
        label[fractions["sky"] >= args.pure_fraction] = fl.LABEL_SKY
        arrays[f"{stem}__label"] = label
        arrays[f"{stem}__ignored"] = ignored
        grids[stem] = label
        photo_summary["cells"] = {
            "person": int((label == fl.LABEL_PERSON).sum()),
            "sky": int((label == fl.LABEL_SKY).sum()),
            "other": int((label == fl.LABEL_OTHER).sum()),
            "excluded": int((label == fl.LABEL_EXCLUDED).sum()),
            "ignored": int(ignored.sum()),
        }
        summary[stem] = photo_summary
        print(f"[labels] {stem}: {photo_summary['cells']}, "
              + ", ".join(f"{n} +{photo_summary[n]['positive_cells']}"
                          f"/-{photo_summary[n]['negative_cells']}"
                          f"/center {photo_summary[n]['center_cells']}" for n in fl.CLASSES))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / "labels.npz", **arrays)
    fl.write_json(out_dir / "labels.json", {
        "step": "issue #1, step 2: reference cells",
        "pure_fraction": args.pure_fraction,
        "grid": fl.GRID,
        "label_values": {"excluded": fl.LABEL_EXCLUDED, "other": fl.LABEL_OTHER,
                         "person": fl.LABEL_PERSON, "sky": fl.LABEL_SKY},
        "presence_file": Path(args.presence).name,
        "presence_sha256": fl.file_sha256(Path(args.presence)),
        "photos": summary,
    })
    if args.sheet:
        label_sheet(Path(args.photos), grids, Path(args.sheet))
        print(f"[sheet] {args.sheet}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
