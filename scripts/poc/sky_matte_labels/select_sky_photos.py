#!/usr/bin/env python3
"""Select COCO 2017 train photos with sky and download them at 1024 px from Flickr.

Input:  --panoptic-json (panoptic_train2017.json; it holds the license of each photo),
        --count-tree, --count-building, --count-other, --shuffle-seed.
Output: --out-dir/manifest.json and --out-dir/photos/<image id>.jpg.

Step: sky matte labels (README decision U17). Only the photos with a license that permits
commercial use are selected (sky_label_data.ALLOWED_LICENSES, the rule of P1). A photo is a
candidate when its coarse sky covers 5 to 85 percent of the photo and at most 10 percent of the
photo is unlabeled. The candidates are put in three groups by the segments of the photo:
"tree" (a tree segment), "building" (a building segment, no tree), and "other". Each group is
shuffled with --shuffle-seed and filled in order.

The COCO copy of a photo has a long edge of 640 px. The script downloads the Flickr size "_b"
(long edge 1024 px) of the same photo. A download is accepted only when Flickr returns a JPEG
(not the "photo unavailable" image), the long edge is at least 800 px, and the aspect ratio
differs from the COCO size by at most 1.5 percent (the same photo without a crop). A rejected
photo is recorded with its reason and the next candidate of the group is used.
"""

from __future__ import annotations

import argparse
import io
import json
import random
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sky_label_data as sd  # noqa: E402

MIN_SKY_SHARE = 0.05
MAX_SKY_SHARE = 0.85
MAX_UNLABELED_SHARE = 0.10
MIN_LONG_EDGE = 800
MAX_ASPECT_DIFFERENCE = 0.015
DOWNLOAD_THREADS = 4


def flickr_large_url(flickr_url: str) -> str:
    """COCO flickr_url (size "_z", 640 px) -> the same photo at size "_b" (1024 px)."""
    if not flickr_url.endswith("_z.jpg"):
        raise ValueError(f"unexpected flickr_url: {flickr_url}")
    return flickr_url[: -len("_z.jpg")].replace("http://", "https://") + "_b.jpg"


def download_photo(task: dict) -> dict:
    """Download one photo and check it; returns the task with 'error' or the photo size."""
    url, path = task["download_url"], Path(task["path"])
    try:
        with urllib.request.urlopen(url, timeout=60) as response:
            final_url = response.geturl()
            data = response.read()
        if "photo_unavailable" in final_url:
            return {**task, "error": "photo unavailable"}
        image = Image.open(io.BytesIO(data))
        if image.format != "JPEG":
            return {**task, "error": f"format {image.format}"}
        width, height = image.size
        if max(width, height) < MIN_LONG_EDGE:
            return {**task, "error": f"long edge {max(width, height)} px"}
        coco_aspect = task["coco_width"] / task["coco_height"]
        difference = abs(width / height - coco_aspect) / coco_aspect
        if difference > MAX_ASPECT_DIFFERENCE:
            return {**task, "error": f"aspect ratio differs by {difference:.4f}"}
        path.write_bytes(data)
        return {**task, "width": width, "height": height}
    except Exception as caught:  # The error is recorded and the next candidate is used.
        return {**task, "error": f"{type(caught).__name__}: {caught}"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--panoptic-json", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--count-tree", type=int, default=100)
    parser.add_argument("--count-building", type=int, default=60)
    parser.add_argument("--count-other", type=int, default=40)
    parser.add_argument("--shuffle-seed", type=int, default=0)
    args = parser.parse_args()
    started = time.perf_counter()

    data = json.loads(args.panoptic_json.read_text(encoding="utf-8"))
    categories = {c["name"]: c["id"] for c in data["categories"]}
    sky_id = categories[sd.SKY_CATEGORY]
    tree_ids = {categories[name] for name in sd.TREE_CATEGORIES}
    building_ids = {categories[name] for name in sd.BUILDING_CATEGORIES}
    images = {image["id"]: image for image in data["images"]}

    groups: dict[str, list[dict]] = {"tree": [], "building": [], "other": []}
    for annotation in data["annotations"]:
        image = images[annotation["image_id"]]
        if image["license"] not in sd.ALLOWED_LICENSES:
            continue
        area = image["width"] * image["height"]
        segments = annotation["segments_info"]
        sky_share = sum(s["area"] for s in segments if s["category_id"] == sky_id) / area
        unlabeled_share = 1.0 - sum(s["area"] for s in segments) / area
        if not MIN_SKY_SHARE <= sky_share <= MAX_SKY_SHARE:
            continue
        if unlabeled_share > MAX_UNLABELED_SHARE:
            continue
        present = {s["category_id"] for s in segments}
        group = "tree" if present & tree_ids else "building" if present & building_ids else "other"
        groups[group].append({
            "id": image["id"], "file_name": image["file_name"],
            "panoptic_file_name": annotation["file_name"], "license": image["license"],
            "flickr_url": image["flickr_url"], "coco_width": image["width"],
            "coco_height": image["height"], "group": group,
            "coarse_sky_share": round(sky_share, 4),
            "unlabeled_share": round(unlabeled_share, 4)})
    candidate_counts = {name: len(items) for name, items in groups.items()}

    photo_dir = args.out_dir / "photos"
    photo_dir.mkdir(parents=True, exist_ok=True)
    wanted = {"tree": args.count_tree, "building": args.count_building,
              "other": args.count_other}
    selected: list[dict] = []
    rejected: list[dict] = []
    with ThreadPoolExecutor(DOWNLOAD_THREADS) as pool:
        for name, items in groups.items():
            random.Random(f"{args.shuffle_seed}-{name}").shuffle(items)
            accepted: list[dict] = []
            position = 0
            while len(accepted) < wanted[name] and position < len(items):
                batch = items[position:position + wanted[name] - len(accepted)]
                position += len(batch)
                tasks = [{**item, "download_url": flickr_large_url(item["flickr_url"]),
                          "path": str(sd.photo_path(photo_dir, item["id"]))} for item in batch]
                for result in pool.map(download_photo, tasks):
                    result.pop("path")
                    (rejected if "error" in result else accepted).append(result)
            if len(accepted) < wanted[name]:
                raise RuntimeError(f"group {name}: {len(accepted)} of {wanted[name]} photos")
            selected.extend(accepted)

    license_names = {entry["id"]: entry["name"] for entry in data["licenses"]}
    manifest = {
        "step": "sky matte labels (README decision U17): photo selection",
        "sources": {"panoptic_json": args.panoptic_json.name,
                    "annotation_license": "COCO 2017 annotations, CC BY 4.0",
                    "photos": "Flickr size _b of the COCO flickr_url, license per photo"},
        "allowed_licenses": {str(i): license_names[i] for i in sd.ALLOWED_LICENSES},
        "rules": {"min_sky_share": MIN_SKY_SHARE, "max_sky_share": MAX_SKY_SHARE,
                  "max_unlabeled_share": MAX_UNLABELED_SHARE,
                  "min_long_edge": MIN_LONG_EDGE,
                  "max_aspect_difference": MAX_ASPECT_DIFFERENCE},
        "shuffle_seed": args.shuffle_seed,
        "candidate_counts": candidate_counts,
        "selected_counts": {name: sum(1 for s in selected if s["group"] == name)
                            for name in groups},
        "selected_per_license": {str(i): sum(1 for s in selected if s["license"] == i)
                                 for i in sd.ALLOWED_LICENSES},
        "rejected_count": len(rejected),
        "images": selected,
        "rejected": rejected,
        "seconds": round(time.perf_counter() - started, 1),
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n",
                                                encoding="utf-8", newline="\n")
    print(json.dumps({k: manifest[k] for k in ("candidate_counts", "selected_counts",
                                               "selected_per_license", "rejected_count",
                                               "seconds")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
