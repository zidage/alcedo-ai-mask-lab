#!/usr/bin/env python3
"""Per-cell targets of the people and sky head from Open Images V7 (step 1, second data source).

Input:  --images-csv (image list with OriginalURL, License, Rotation, Author of the split),
        --points-csv (point labels), --boxes-csv (box annotations), --image-labels-csv
        (human-verified image-level labels), --split (`train` or `validation`, the folder of the
        image URL), --image-dir (the photos are downloaded and stored here when missing).
Output: --out-dir/manifest.json and --out-dir/targets/*.npy (head_data.TARGET_ARRAYS).

Selection. Only photos with a license in --licenses (default CC BY 2.0, decision U2) and the
rotation 0 (the image list has no rotation for 12.8 percent of the validation photos and a
rotation for 1.2 percent; for these the orientation of the labels and the pixels is not known).
Two pools, each shuffled with --shuffle-seed:
  sky pool     photos with a Sky or Cloud point label (yes or no): the first --sky-images
  person pool  photos with a box of a person class: the first --person-images that are not
               already selected
The selected photos are shuffled again, so the image order of the split is random.

Photos. Each photo is downloaded from --image-url and stored as downloaded. The default URL is
the copy of the Common Visual Data Foundation (CVDF): the photo with the long edge reduced to
1024 px and no EXIF data. Some Flickr originals are removed (HTTP 404), so the originals are not
used. The OriginalMD5 of the image list is the MD5 of the Flickr original; it does not apply to
the CVDF copy, so the check is that the photo decodes. The manifest records the stored size.

Targets on the 64 x 64 stretched grid (cell = 1/64 of the width by 1/64 of the height):
  sky_coverage, sky_valid  only the cells of Sky and Cloud point labels: a Sky or Cloud "yes"
                 point counts 1, a Sky "no" point counts 0 (a Cloud "no" point is not used: the
                 point can be on clear sky); the share of a cell is the mean of its points.
                 "unsure" points are not used. sky_dense = False (no Dice term)
  person         the boxes of Person, Man, Woman, Boy, Girl. A box with IsGroupOf or IsDepiction
                 is an ignored area (crowd_boxes; heatmap_valid False on its cells). Boxes of
                 different person classes with IoU >= PERSON_DUPLICATE_IOU are one person (the
                 larger box is kept). Open Images annotates every instance of each positive label
                 of a box-annotated photo, so the person targets are complete when the photo is in
                 the box annotations and has a person box or a verified negative Person label.
                 Otherwise the person losses ignore the photo (heatmap_valid False, presence
                 ignored)
  presence       sky: 1 with a Sky or Cloud "yes" point or a verified positive Sky label, 0 with
                 a verified negative Sky label and no "yes" point, else ignored. person: 1 with a
                 person box (not group-of, not depiction) of at least one cell area, 0 when the
                 person targets are complete and the photo has no person box, else ignored

Open Images V7 annotations are CC BY 4.0 (Google LLC). The image list names the license of each
photo; Open Images asks the user to verify the license of each photo. The manifest records the
license, the author, and the source page of each selected photo.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import sys
import time
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import head_data as hd  # noqa: E402

SKY = "/m/01bqvp"
CLOUD = "/m/0csby"
PERSON = "/m/01g317"
PERSON_CLASSES = {PERSON: "Person", "/m/04yx4": "Man", "/m/03bt1vf": "Woman",
                  "/m/01bl7v": "Boy", "/m/05r655": "Girl"}
CC_BY_2_0 = "https://creativecommons.org/licenses/by/2.0/"
DEFAULT_IMAGE_URL = "https://open-images-dataset.s3.amazonaws.com/{split}/{image_id}.jpg"
PERSON_DUPLICATE_IOU = 0.7
DOWNLOAD_ATTEMPTS = 3


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for block in iter(lambda: file.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def header_index(path: Path, names: list[str]) -> tuple[int, dict[str, int]]:
    """Column index of each name in the header of a CSV file; fails when a column is missing."""
    with open(path, encoding="utf-8", newline="") as file:
        header = next(csv.reader(file))
    missing = [n for n in names if n not in header]
    if missing:
        raise ValueError(f"{path}: no columns {missing} in {header}")
    return len(header), {n: header.index(n) for n in names}


def rows_with(path: Path, needles: tuple[str, ...]):
    """Parsed rows of the lines that contain one of `needles` (a substring test before the CSV
    parse keeps the pass over the large files fast)."""
    with open(path, encoding="utf-8", newline="") as file:
        next(file)
        for line in file:
            if any(n in line for n in needles):
                yield next(csv.reader([line]))


def read_points(path: Path) -> dict[str, list[tuple[float, float, float]]]:
    """Sky targets per image: (x, y, value) with value 1 (Sky or Cloud yes) or 0 (Sky no)."""
    _, column = header_index(path, ["ImageId", "X", "Y", "Label", "EstimatedYesNo"])
    points: dict[str, list[tuple[float, float, float]]] = defaultdict(list)
    for row in rows_with(path, (SKY, CLOUD)):
        label, answer = row[column["Label"]], row[column["EstimatedYesNo"]]
        if label not in (SKY, CLOUD) or answer not in ("yes", "no"):
            continue
        if label == CLOUD and answer == "no":
            continue
        points[row[column["ImageId"]]].append(
            (float(row[column["X"]]), float(row[column["Y"]]), 1.0 if answer == "yes" else 0.0))
    return points


def read_image_labels(path: Path) -> dict[str, dict[str, float]]:
    """Verified Sky and person-class labels per image: {label: confidence (1 or 0)}."""
    _, column = header_index(path, ["ImageID", "LabelName", "Confidence"])
    labels: dict[str, dict[str, float]] = defaultdict(dict)
    wanted = (SKY,) + tuple(PERSON_CLASSES)
    for row in rows_with(path, wanted):
        if row[column["LabelName"]] in wanted:
            labels[row[column["ImageID"]]][row[column["LabelName"]]] = float(
                row[column["Confidence"]])
    return labels


def read_boxes(path: Path) -> tuple[set[str], dict[str, list[dict]]]:
    """All image ids of the box annotations, and the person-class boxes per image."""
    _, column = header_index(path, ["ImageID", "LabelName", "XMin", "XMax", "YMin", "YMax",
                                    "IsGroupOf", "IsDepiction"])
    annotated: set[str] = set()
    boxes: dict[str, list[dict]] = defaultdict(list)
    with open(path, encoding="utf-8", newline="") as file:
        next(file)
        for line in file:
            image_id = line.split(",", 1)[0]
            annotated.add(image_id)
            if not any(code in line for code in PERSON_CLASSES):
                continue
            row = next(csv.reader([line]))
            if row[column["LabelName"]] not in PERSON_CLASSES:
                continue
            boxes[image_id].append({
                "box": [float(row[column[k]]) for k in ("XMin", "YMin", "XMax", "YMax")],
                "group": row[column["IsGroupOf"]] == "1",
                "depiction": row[column["IsDepiction"]] == "1"})
    return annotated, boxes


def read_image_info(path: Path, wanted: set[str]) -> dict[str, dict]:
    """Image list rows of the wanted image ids."""
    width, column = header_index(path, ["ImageID", "OriginalURL", "OriginalLandingURL", "License",
                                        "AuthorProfileURL", "Author", "Title", "Rotation"])
    info = {}
    with open(path, encoding="utf-8", newline="") as file:
        next(file)
        for line in file:
            if line.split(",", 1)[0] not in wanted:
                continue
            row = next(csv.reader([line]))
            if len(row) != width:
                raise ValueError(f"{path}: {len(row)} columns in the row of {row[0]}")
            info[row[column["ImageID"]]] = {k: row[i] for k, i in column.items()}
    return info


def box_iou(a: list[float], b: list[float]) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def cells_of_boxes(boxes: np.ndarray) -> np.ndarray:
    """[64, 64] bool: the cells that a box of [K, 4] (cells) overlaps."""
    covered = np.zeros((hd.GRID, hd.GRID), bool)
    for x0, y0, x1, y1 in boxes:
        c0, c1 = int(np.clip(np.floor(x0), 0, hd.GRID)), int(np.clip(np.ceil(x1), 0, hd.GRID))
        r0, r1 = int(np.clip(np.floor(y0), 0, hd.GRID)), int(np.clip(np.ceil(y1), 0, hd.GRID))
        covered[r0:r1, c0:c1] = True
    return covered


def image_targets(image_id: str, points, labels: dict[str, float], annotated: bool,
                  person_boxes: list[dict]) -> dict:
    """All target arrays of one image (see the module docstring)."""
    sky_sum = np.zeros((hd.GRID, hd.GRID), np.float64)
    sky_count = np.zeros((hd.GRID, hd.GRID), np.int32)
    for x, y, value in points:
        column, row = min(int(x * hd.GRID), hd.GRID - 1), min(int(y * hd.GRID), hd.GRID - 1)
        sky_sum[row, column] += value
        sky_count[row, column] += 1
    sky_valid = sky_count > 0
    sky_coverage = np.zeros((hd.GRID, hd.GRID), np.uint8)
    sky_coverage[sky_valid] = np.round(sky_sum[sky_valid] / sky_count[sky_valid] * 255)
    sky_yes = any(v == 1.0 for _, _, v in points)
    sky_label = labels.get(SKY)
    conflict = bool(sky_label == 0.0 and sky_yes)
    sky_presence = (1 if (sky_yes or sky_label == 1.0) and not conflict
                    else 0 if sky_label == 0.0 and not sky_yes else hd.PRESENCE_IGNORE)

    scale = np.array([hd.GRID] * 4, np.float32)
    people = sorted([b for b in person_boxes if not b["group"] and not b["depiction"]],
                    key=lambda b: -(b["box"][2] - b["box"][0]) * (b["box"][3] - b["box"][1]))
    kept: list[list[float]] = []
    for entry in people:
        if all(box_iou(entry["box"], k) < PERSON_DUPLICATE_IOU for k in kept):
            kept.append(entry["box"])
    ignored = [b["box"] for b in person_boxes if b["group"] or b["depiction"]]
    person_cells = np.array(kept, np.float32).reshape(-1, 4) * scale
    ignore_cells = np.array(ignored, np.float32).reshape(-1, 4) * scale
    complete = annotated and (bool(person_boxes) or labels.get(PERSON) == 0.0)

    boxes = np.zeros((hd.MAX_PERSONS, 4), np.float32)
    crowd_slots = np.zeros((hd.MAX_CROWDS, 4), np.float32)
    if complete:
        targets = hd.person_targets(person_cells)
        heatmap_valid = ~cells_of_boxes(ignore_cells) | (targets["center_heatmap"] >= 1.0)
        areas = (person_cells[:, 2] - person_cells[:, 0]) * (person_cells[:, 3] - person_cells[:, 1])
        person_presence = (1 if (areas >= 1.0).any()
                           else 0 if not person_boxes else hd.PRESENCE_IGNORE)
        boxes[:min(len(person_cells), hd.MAX_PERSONS)] = person_cells[:hd.MAX_PERSONS]
        crowd_slots[:min(len(ignore_cells), hd.MAX_CROWDS)] = ignore_cells[:hd.MAX_CROWDS]
    else:
        targets = hd.person_targets(np.zeros((0, 4), np.float32))
        heatmap_valid = np.zeros((hd.GRID, hd.GRID), bool)
        person_presence = hd.PRESENCE_IGNORE
    return {
        "sky_coverage": sky_coverage,
        "sky_valid": sky_valid,
        "center_heatmap": targets["center_heatmap"].astype(np.float16),
        "heatmap_valid": heatmap_valid,
        "person_cell": targets["person_cell"],
        "person_offset": targets["person_offset"],
        "person_log_size": targets["person_log_size"],
        "person_boxes": boxes,
        "crowd_boxes": crowd_slots,
        "presence": np.array([sky_presence, person_presence], np.int8),
        "sky_dense": np.array(False),
        "_sky_points": len(points),
        "_persons": len(person_cells) if complete else 0,
        "_duplicates": len(people) - len(kept),
        "_ignored_boxes": len(ignored) if complete else 0,
        "_person_complete": complete,
        "_sky_conflict": conflict,
    }


def download_image(task: tuple[str, Path]) -> tuple[str, dict | str]:
    """Download one photo (when missing) and check that it decodes.

    Returns (image id, {"stored_size": [w, h]}) or (image id, error text)."""
    from PIL import Image

    url, path = task
    image_id = path.stem
    error = ""
    for _ in range(DOWNLOAD_ATTEMPTS):
        try:
            if not path.exists():
                temporary = path.with_suffix(".part")
                with urllib.request.urlopen(url, timeout=60) as response:
                    temporary.write_bytes(response.read())
                temporary.rename(path)
            with Image.open(path) as image:
                image.verify()
            with Image.open(path) as image:
                image.convert("RGB")
                return image_id, {"stored_size": list(image.size)}
        except Exception as caught:  # The same URL is tried again; the last error is reported.
            error = f"{url}: {type(caught).__name__}: {caught}"
            path.unlink(missing_ok=True)
    return image_id, error


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--images-csv", required=True)
    parser.add_argument("--points-csv", required=True)
    parser.add_argument("--boxes-csv", required=True)
    parser.add_argument("--image-labels-csv", required=True)
    parser.add_argument("--split", required=True, choices=("train", "validation", "test"))
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--licenses", default=CC_BY_2_0,
                        help="comma-separated license URLs of the image list")
    parser.add_argument("--sky-images", type=int, required=True)
    parser.add_argument("--person-images", type=int, required=True)
    parser.add_argument("--shuffle-seed", type=int, default=0)
    parser.add_argument("--image-url", default=DEFAULT_IMAGE_URL,
                        help="URL pattern with {split} and {image_id}")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--download-workers", type=int, default=32)
    args = parser.parse_args()

    started = time.perf_counter()
    paths = {name: Path(getattr(args, name.replace("-", "_")))
             for name in ("images-csv", "points-csv", "boxes-csv", "image-labels-csv")}
    allowed = sorted({u.strip() for u in args.licenses.split(",") if u.strip()})

    points = read_points(paths["points-csv"])
    print(f"[points] {len(points)} images with Sky or Cloud point labels "
          f"({time.perf_counter() - started:.0f} s)")
    labels = read_image_labels(paths["image-labels-csv"])
    print(f"[labels] {len(labels)} images with verified Sky or person labels "
          f"({time.perf_counter() - started:.0f} s)")
    annotated, person_boxes = read_boxes(paths["boxes-csv"])
    print(f"[boxes] {len(annotated)} box-annotated images, {len(person_boxes)} with person boxes "
          f"({time.perf_counter() - started:.0f} s)")
    info = read_image_info(paths["images-csv"], set(points) | set(person_boxes))
    print(f"[images] {len(info)} image list rows ({time.perf_counter() - started:.0f} s)")

    rejected = {"license": 0, "rotation": 0, "not_in_image_list": 0}

    def usable(image_id: str) -> bool:
        entry = info.get(image_id)
        if entry is None:
            rejected["not_in_image_list"] += 1
            return False
        if entry["License"] not in allowed:
            rejected["license"] += 1
            return False
        if entry["Rotation"] not in ("0", "0.0"):
            rejected["rotation"] += 1
            return False
        return True

    rng = np.random.default_rng(args.shuffle_seed)
    sky_pool = [i for i in sorted(points) if usable(i)]
    person_pool = [i for i in sorted(person_boxes) if usable(i)]
    sky_selected = [sky_pool[i] for i in rng.permutation(len(sky_pool))][:args.sky_images]
    chosen = set(sky_selected)
    person_order = [person_pool[i] for i in rng.permutation(len(person_pool))]
    person_selected = [i for i in person_order if i not in chosen][:args.person_images]
    selected = sky_selected + person_selected
    selected = [selected[i] for i in rng.permutation(len(selected))]
    print(f"[select] sky pool {len(sky_pool)}, person pool {len(person_pool)}; selected "
          f"{len(sky_selected)} + {len(person_selected)} = {len(selected)}; rejected {rejected}")

    image_dir = Path(args.image_dir)
    image_dir.mkdir(parents=True, exist_ok=True)
    tasks = [(args.image_url.format(split=args.split, image_id=i), image_dir / f"{i}.jpg")
             for i in selected]
    sizes, failures = {}, []
    with ThreadPoolExecutor(args.download_workers) as executor:
        for done, (image_id, result) in enumerate(executor.map(download_image, tasks), start=1):
            if isinstance(result, str):
                failures.append(result)
            else:
                sizes[image_id] = result
            if done % 2000 == 0 or done == len(tasks):
                print(f"[download] {done} / {len(tasks)} checked, {len(failures)} failed "
                      f"({time.perf_counter() - started:.0f} s)")
    if failures:
        raise RuntimeError(f"{len(failures)} photos did not download or decode:\n"
                           + "\n".join(failures[:20]))

    arrays = {name: np.zeros((len(selected),) + shape, dtype)
              for name, (dtype, shape) in hd.TARGET_ARRAYS.items()}
    counts = defaultdict(int)
    for index, image_id in enumerate(selected):
        result = image_targets(image_id, points.get(image_id, []), labels.get(image_id, {}),
                               image_id in annotated, person_boxes.get(image_id, []))
        for name in hd.TARGET_ARRAYS:
            arrays[name][index] = result[name]
        counts["sky_points"] += result["_sky_points"]
        counts["images_with_sky_points"] += int(result["_sky_points"] > 0)
        counts["persons"] += result["_persons"]
        counts["person_duplicates_removed"] += result["_duplicates"]
        counts["ignored_person_boxes"] += result["_ignored_boxes"]
        counts["images_with_complete_person_targets"] += int(result["_person_complete"])
        counts["sky_presence_conflicts"] += int(result["_sky_conflict"])
        counts["persons_above_max"] += max(0, result["_persons"] - hd.MAX_PERSONS)

    out_dir = Path(args.out_dir)
    hd.target_dir(out_dir).mkdir(parents=True, exist_ok=True)
    for name, array in arrays.items():
        np.save(hd.target_dir(out_dir) / f"{name}.npy", array)
    presence = arrays["presence"]
    presence_counts = {
        cls: {"present": int((presence[:, k] == 1).sum()),
              "absent": int((presence[:, k] == 0).sum()),
              "ignored": int((presence[:, k] == hd.PRESENCE_IGNORE).sum())}
        for k, cls in enumerate(hd.PRESENCE_CLASSES)}
    sky_cells = arrays["sky_coverage"][arrays["sky_valid"]]
    license_counts = defaultdict(int)
    for image_id in selected:
        license_counts[info[image_id]["License"]] += 1
    hd.write_json(out_dir / "manifest.json", {
        "step": "README section 5, step 1: targets of the people and sky head (Open Images V7)",
        "sources": {name: {"path": path.name, "sha256": file_sha256(path)}
                    for name, path in paths.items()},
        "annotation_license": "Open Images V7 annotations, CC BY 4.0 (Google LLC)",
        "split": args.split,
        "allowed_licenses": allowed,
        "rotation_rule": "only the photos with the rotation 0 in the image list",
        "photo_source": args.image_url,
        "photo_storage": "as downloaded (the CVDF copy: long edge 1024 px, no EXIF data); the "
                         "check is that the photo decodes",
        "classes": {"sky": {SKY: "Sky", CLOUD: "Cloud"}, "person": PERSON_CLASSES},
        "person_duplicate_iou": PERSON_DUPLICATE_IOU,
        "shuffle_seed": args.shuffle_seed,
        "quotas": {"sky_images": args.sky_images, "person_images": args.person_images},
        "pools": {"sky": len(sky_pool), "person": len(person_pool)},
        "rejected_candidates": rejected,
        "counts": {"images": len(selected), "sky_selected": len(sky_selected),
                   "person_selected": len(person_selected), **counts,
                   "sky_point_cells": {"sky": int((sky_cells >= 128).sum()),
                                       "not_sky": int((sky_cells < 128).sum())},
                   "presence": presence_counts,
                   "images_per_license": dict(license_counts)},
        "images": [{"id": i, "file_name": f"{i}.jpg", "license": info[i]["License"],
                    "author": info[i]["Author"], "author_profile_url": info[i]["AuthorProfileURL"],
                    "landing_url": info[i]["OriginalLandingURL"],
                    "original_url": info[i]["OriginalURL"], "title": info[i]["Title"],
                    **sizes[i]} for i in selected],
        "seconds": round(time.perf_counter() - started, 1),
    })
    print(f"[done] {len(selected)} images, {counts['persons']} persons, {counts['sky_points']} "
          f"sky points, presence {presence_counts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
