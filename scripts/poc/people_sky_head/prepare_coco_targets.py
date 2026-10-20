#!/usr/bin/env python3
"""Per-cell targets of the people and sky head from COCO panoptic annotations (step 1).

Input:  --panoptic-json and --panoptic-png-dir (COCO 2017 panoptic annotations of one split),
        --image-info-json (a COCO 2017 annotation file of the same split with the `images` and
        `licenses` lists, for example captions_train2017.json), --licenses (the Flickr license
        ids that the split may use), --image-dir (the photos are downloaded here when missing).
Output: --out-dir/manifest.json and --out-dir/targets/*.npy (head_data.TARGET_ARRAYS).

The selected images are shuffled with --shuffle-seed, and the first --max-images are kept, so the
image order of the split is random. For each image on the 64 x 64 stretched grid:

  sky_coverage    share of the cell pixels in `sky-other-merged` segments
  sky_valid       the cell has at most 50 percent unlabeled pixels
  center_heatmap  CenterNet Gaussian at the box center of each non-crowd person
  heatmap_valid   False on cells covered by a crowd person segment (except the peaks)
  person_*        box target of each non-crowd person (center offset, log size, box)
  crowd_boxes     boxes of the crowd person segments (the evaluation ignores detections there)
  presence        sky: 1 if the sky area >= 1 percent, 0 if no sky pixel, else ignored;
                  person: 1 if a non-crowd person has at least one cell area, 0 if the image has
                  no person segment, else ignored

COCO 2017 annotations are CC BY 4.0. The photos are Flickr photos with the license that the
`images` entry names; README decision U2 allows only licenses that permit commercial use.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import head_data as hd  # noqa: E402

PERSON_CATEGORY = "person"
SKY_CATEGORY = "sky-other-merged"
DOWNLOAD_ATTEMPTS = 3


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for block in iter(lambda: file.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_licenses(text: str, known: dict[int, str]) -> list[int]:
    if text == "all":
        return sorted(known)
    ids = sorted({int(part) for part in text.split(",")})
    unknown = [i for i in ids if i not in known]
    if unknown:
        raise ValueError(f"license ids {unknown} are not in the image info file {known}")
    return ids


def download_image(task: tuple[str, Path, int, int]) -> str | None:
    """Download one photo if missing; check that it decodes with the annotated size.

    Returns None on success, otherwise the error text of the last attempt."""
    from PIL import Image

    url, path, width, height = task
    error = None
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
                if image.size != (width, height):
                    raise ValueError(f"size {image.size}, annotation {width}x{height}")
            return None
        except Exception as caught:  # The same URL is tried again; the last error is reported.
            error = f"{type(caught).__name__}: {caught}"
            path.unlink(missing_ok=True)
    return f"{url}: {error}"


def area_share(mask: np.ndarray) -> np.ndarray:
    """[H, W] bool -> [64, 64] float32 share of each stretched cell (pixel-area average)."""
    import cv2

    return cv2.resize(mask.astype(np.float32), (hd.GRID, hd.GRID), interpolation=cv2.INTER_AREA)


def boxes_in_cells(segments: list[dict], width: int, height: int) -> np.ndarray:
    """COCO [x, y, w, h] pixel boxes -> [P, 4] x0, y0, x1, y1 in cells, largest area first."""
    if not segments:
        return np.zeros((0, 4), np.float32)
    boxes = np.array([s["bbox"] for s in segments], np.float32)
    scale = np.array([hd.GRID / width, hd.GRID / height] * 2, np.float32)
    cells = np.concatenate([boxes[:, :2], boxes[:, :2] + boxes[:, 2:]], axis=1) * scale
    order = np.argsort(-(boxes[:, 2] * boxes[:, 3]), kind="stable")
    return cells[order]


def image_targets(task: dict) -> dict:
    """All target arrays of one image (see the module docstring)."""
    from PIL import Image

    width, height = task["width"], task["height"]
    with Image.open(task["png_path"]) as png:
        rgb = np.asarray(png.convert("RGB"), dtype=np.uint32)
    if rgb.shape[:2] != (height, width):
        raise ValueError(f"{task['png_path']}: size {rgb.shape[1]}x{rgb.shape[0]}, "
                         f"annotation {width}x{height}")
    segment_ids = rgb[..., 0] + 256 * rgb[..., 1] + 65536 * rgb[..., 2]

    segments = task["segments_info"]
    sky_ids = [s["id"] for s in segments if s["category_id"] == task["sky_id"]]
    persons = [s for s in segments if s["category_id"] == task["person_id"] and not s["iscrowd"]]
    crowds = [s for s in segments if s["category_id"] == task["person_id"] and s["iscrowd"]]

    sky_coverage = area_share(np.isin(segment_ids, sky_ids))
    unlabeled = area_share(segment_ids == 0)
    person_boxes = boxes_in_cells(persons, width, height)
    crowd_boxes = boxes_in_cells(crowds, width, height)
    targets = hd.person_targets(person_boxes)
    heatmap_valid = np.ones((hd.GRID, hd.GRID), bool)
    if crowds:
        crowd_coverage = area_share(np.isin(segment_ids, [s["id"] for s in crowds]))
        heatmap_valid = (crowd_coverage == 0) | (targets["center_heatmap"] >= 1.0)

    image_area = float(width * height)
    sky_area = sum(s["area"] for s in segments if s["category_id"] == task["sky_id"]) / image_area
    sky_presence = (1 if sky_area >= hd.SKY_PRESENT_MIN_AREA
                    else 0 if sky_area == 0 else hd.PRESENCE_IGNORE)
    visible = any(s["area"] / image_area >= hd.PERSON_PRESENT_MIN_AREA for s in persons)
    person_presence = (1 if visible
                       else 0 if not persons and not crowds else hd.PRESENCE_IGNORE)

    boxes = np.zeros((hd.MAX_PERSONS, 4), np.float32)
    boxes[:min(len(person_boxes), hd.MAX_PERSONS)] = person_boxes[:hd.MAX_PERSONS]
    crowd_slots = np.zeros((hd.MAX_CROWDS, 4), np.float32)
    crowd_slots[:min(len(crowd_boxes), hd.MAX_CROWDS)] = crowd_boxes[:hd.MAX_CROWDS]
    return {
        "sky_coverage": np.round(sky_coverage * 255).astype(np.uint8),
        "sky_valid": unlabeled <= hd.SKY_MAX_UNLABELED,
        "center_heatmap": targets["center_heatmap"].astype(np.float16),
        "heatmap_valid": heatmap_valid,
        "person_cell": targets["person_cell"],
        "person_offset": targets["person_offset"],
        "person_log_size": targets["person_log_size"],
        "person_boxes": boxes,
        "crowd_boxes": crowd_slots,
        "presence": np.array([sky_presence, person_presence], np.int8),
        "_persons": len(person_boxes),
        "_crowds": len(crowd_boxes),
        "_box_targets": int((targets["person_cell"] >= 0).sum()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--panoptic-json", required=True)
    parser.add_argument("--panoptic-png-dir", required=True)
    parser.add_argument("--image-info-json", required=True)
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--licenses", required=True,
                        help="comma-separated COCO license ids, or 'all'")
    parser.add_argument("--max-images", type=int, default=0, help="0 = all selected images")
    parser.add_argument("--shuffle-seed", type=int, default=0)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--download-workers", type=int, default=32)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()

    started = time.perf_counter()
    panoptic_json, info_json = Path(args.panoptic_json), Path(args.image_info_json)
    panoptic = hd.read_json(panoptic_json)
    info = hd.read_json(info_json)
    licenses = {entry["id"]: entry["name"] for entry in info["licenses"]}
    allowed = parse_licenses(args.licenses, licenses)
    categories = {c["name"]: c["id"] for c in panoptic["categories"]}
    for name in (PERSON_CATEGORY, SKY_CATEGORY):
        if name not in categories:
            raise ValueError(f"{panoptic_json}: no category '{name}'")
    images = {entry["id"]: entry for entry in info["images"]}
    annotations = {entry["image_id"]: entry for entry in panoptic["annotations"]}
    missing = [i for i in annotations if i not in images]
    if missing:
        raise ValueError(f"{len(missing)} panoptic images are not in {info_json}, "
                         f"for example {missing[:5]}")

    per_license_all: dict[int, int] = {}
    candidates = []
    for image_id in sorted(annotations):
        license_id = images[image_id]["license"]
        per_license_all[license_id] = per_license_all.get(license_id, 0) + 1
        if license_id in allowed:
            candidates.append(image_id)
    print("[licenses] panoptic images per license:")
    for license_id in sorted(per_license_all):
        mark = "use" if license_id in allowed else "skip"
        print(f"  {license_id} {licenses[license_id]}: {per_license_all[license_id]} ({mark})")
    rng = np.random.default_rng(args.shuffle_seed)
    selected = [candidates[i] for i in rng.permutation(len(candidates))]
    if args.max_images:
        selected = selected[:args.max_images]
    print(f"[select] {len(selected)} of {len(candidates)} images with an allowed license")

    image_dir = Path(args.image_dir)
    image_dir.mkdir(parents=True, exist_ok=True)
    downloads = [(images[i]["coco_url"], image_dir / images[i]["file_name"],
                  images[i]["width"], images[i]["height"]) for i in selected]
    failures = []
    with ThreadPoolExecutor(args.download_workers) as executor:
        for done, error in enumerate(executor.map(download_image, downloads), start=1):
            if error:
                failures.append(error)
            if done % 2000 == 0 or done == len(downloads):
                print(f"[download] {done} / {len(downloads)} checked, {len(failures)} failed")
    if failures:
        raise RuntimeError(f"{len(failures)} photos did not download or decode:\n"
                           + "\n".join(failures[:20]))

    png_dir = Path(args.panoptic_png_dir)
    tasks = [{"png_path": str(png_dir / annotations[i]["file_name"]),
              "width": images[i]["width"], "height": images[i]["height"],
              "segments_info": annotations[i]["segments_info"],
              "person_id": categories[PERSON_CATEGORY], "sky_id": categories[SKY_CATEGORY]}
             for i in selected]
    arrays = {name: np.zeros((len(selected),) + shape, dtype)
              for name, (dtype, shape) in hd.TARGET_ARRAYS.items()}
    persons = crowds = box_targets = truncated = 0
    with Pool(args.workers) as pool:
        for index, result in enumerate(pool.imap(image_targets, tasks, chunksize=32)):
            for name in hd.TARGET_ARRAYS:
                arrays[name][index] = result[name]
            persons += result["_persons"]
            crowds += result["_crowds"]
            box_targets += result["_box_targets"]
            truncated += max(0, result["_persons"] - hd.MAX_PERSONS)
            if (index + 1) % 5000 == 0:
                print(f"[targets] {index + 1} / {len(tasks)}")

    out_dir = Path(args.out_dir)
    hd.target_dir(out_dir).mkdir(parents=True, exist_ok=True)
    for name, array in arrays.items():
        np.save(hd.target_dir(out_dir) / f"{name}.npy", array)
    presence = arrays["presence"]
    presence_counts = {
        cls: {"present": int((presence[:, k] == 1).sum()), "absent": int((presence[:, k] == 0).sum()),
              "ignored": int((presence[:, k] == hd.PRESENCE_IGNORE).sum())}
        for k, cls in enumerate(hd.PRESENCE_CLASSES)}
    hd.write_json(out_dir / "manifest.json", {
        "step": "README section 5, step 1: targets of the people and sky head",
        "sources": {
            "panoptic_json": {"path": panoptic_json.name, "sha256": file_sha256(panoptic_json)},
            "image_info_json": {"path": info_json.name, "sha256": file_sha256(info_json)},
            "annotation_license": "COCO 2017 annotations, CC BY 4.0",
        },
        "categories": {PERSON_CATEGORY: categories[PERSON_CATEGORY],
                       SKY_CATEGORY: categories[SKY_CATEGORY]},
        "licenses": {str(k): v for k, v in licenses.items()},
        "allowed_licenses": allowed,
        "panoptic_images_per_license": {str(k): v for k, v in sorted(per_license_all.items())},
        "shuffle_seed": args.shuffle_seed,
        "max_images": args.max_images,
        "counts": {"images": len(selected), "non_crowd_persons": persons,
                   "persons_above_max": truncated,
                   "box_targets": box_targets, "crowd_segments": crowds,
                   "presence": presence_counts},
        "images": [{"id": i, "file_name": images[i]["file_name"], "license": images[i]["license"],
                    "flickr_url": images[i].get("flickr_url", "")} for i in selected],
        "seconds": round(time.perf_counter() - started, 1),
    })
    print(f"[done] {len(selected)} images, {persons} persons ({box_targets} box targets), "
          f"presence {presence_counts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
