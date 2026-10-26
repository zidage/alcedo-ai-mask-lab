#!/usr/bin/env python3
"""Sky matte labels: coarse COCO panoptic sky -> trimap -> closed-form matting alpha.

Input:  --manifest (select_sky_photos.py), --photo-dir, --panoptic-zip (panoptic_train2017.zip),
        --panoptic-json (panoptic_train2017.json), optional --limit and --image-ids.
Output: --out-dir/labels/<image id>/{trimap.png, alpha_eps7.png, alpha_eps5.png,
        alpha_label.png, label_weight.png} (uint8, local evidence) and
        --out-dir/label_stats.json (one entry per photo).

Step: sky matte labels (README decision U17). The labels are computed offline; no model runs
and nothing here is used at run time. Closed-form matting (Levin, Lischinski, Weiss 2008;
pymatting.estimate_alpha_cf, no learned weights) computes the sky alpha in the unknown area of
the trimap (sky_label_data.sky_trimap) two times, with two values of the regularization epsilon
of the matting Laplacian:

  alpha_eps7    epsilon 1e-7 (the PyMatting default): keeps thin structures, but shows the
                8 x 8 JPEG blocks of the Flickr photo in flat grey sky
  alpha_eps5    epsilon 1e-5: no JPEG blocks, but makes some dark opaque parts semi-transparent
  alpha_label   mean of the two
  label_weight  0 (the training loss ignores the pixel) in the unknown area where
                |alpha_eps7 - alpha_eps5| > STABLE_DIFFERENCE, or where alpha_label is in
                (PLATEAU_LOW, PLATEAU_HIGH) and stays so after an erosion of PLATEAU_RADIUS px
                (a semi-transparent plateau: true mixed pixels follow an edge and are only a few
                pixels wide; a plateau comes from similar colors, for example a dim sign in
                overcast sky); else 1

Statistics per photo (shares are of the photo area unless stated):

  coarse_sky_share          coarse COCO sky, with the unlabeled components that the sky
                            encloses (sky_label_data.add_enclosed_void_to_sky)
  void_to_sky_share         unlabeled area added to the coarse sky by that rule
  unknown_share             unknown area of the trimap
  label_sky_share           mean of alpha_label
  label_coarse_iou          IoU of (alpha_label >= 0.5) and the coarse sky
  label_fractional_share    share of the unknown area with 0.05 < alpha_label < 0.95
  label_sky_in_tree_share   share of the tree pixels with alpha_label >= 0.5
  unstable_share_of_unknown share of the unknown area with label_weight 0
  eps_mean_difference       mean |alpha_eps7 - alpha_eps5| in the unknown area
"""

from __future__ import annotations

import os

# The resource rules of AGENTS.md: at most 4 CPU threads. Set before NumPy and Numba load.
for variable in ("NUMBA_NUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                 "MKL_NUM_THREADS"):
    os.environ.setdefault(variable, "4")

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import zipfile  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402
from pymatting import estimate_alpha_cf  # noqa: E402
from scipy import ndimage  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sky_label_data as sd  # noqa: E402

EPSILON_FINE = 1e-7
EPSILON_SMOOTH = 1e-5
STABLE_DIFFERENCE = 0.15
PLATEAU_LOW = 0.15
PLATEAU_HIGH = 0.85
PLATEAU_RADIUS = 3
FRACTIONAL_LOW = 0.05
FRACTIONAL_HIGH = 0.95


def save_unit_image(path: Path, values: np.ndarray) -> None:
    Image.fromarray(np.round(np.clip(values, 0.0, 1.0) * 255).astype(np.uint8)).save(path)


def closed_form_alpha(image: np.ndarray, trimap: np.ndarray, epsilon: float) -> np.ndarray:
    alpha = estimate_alpha_cf(image, trimap, laplacian_kwargs={"epsilon": epsilon})
    return np.clip(alpha, 0.0, 1.0)


def label_photo(item: dict, photo_dir: Path, archive: zipfile.ZipFile, segments_info: list[dict],
                category_ids: dict[str, set[int]], out_dir: Path) -> dict:
    photo = Image.open(sd.photo_path(photo_dir, item["id"])).convert("RGB")
    width, height = photo.size
    image = np.asarray(photo, dtype=np.float64) / 255.0
    segment_ids = sd.read_panoptic_png(archive, item["panoptic_file_name"])
    annotated_sky = sd.category_mask(segment_ids, segments_info, category_ids["sky"])
    annotated_void = segment_ids == 0
    merged_sky = sd.add_enclosed_void_to_sky(annotated_sky, annotated_void)
    void_to_sky_share = float((merged_sky & ~annotated_sky).mean())
    coarse_sky = sd.resize_mask(merged_sky, width, height)
    tree = sd.resize_mask(sd.category_mask(segment_ids, segments_info, category_ids["tree"]),
                          width, height)
    void = sd.resize_mask(annotated_void & ~merged_sky, width, height)
    sky_colored_material = any(segment["category_id"] in category_ids["sky_colored"]
                               for segment in segments_info)
    trimap = sd.sky_trimap(image, coarse_sky, tree, void, sky_colored_material)
    unknown = trimap == sd.TRIMAP_UNKNOWN

    started = time.perf_counter()
    alpha_eps7 = closed_form_alpha(image, trimap, EPSILON_FINE)
    alpha_eps5 = closed_form_alpha(image, trimap, EPSILON_SMOOTH)
    seconds = time.perf_counter() - started
    alpha_label = 0.5 * (alpha_eps7 + alpha_eps5)
    difference = np.abs(alpha_eps7 - alpha_eps5)
    fractional = (alpha_label > PLATEAU_LOW) & (alpha_label < PLATEAU_HIGH)
    plateau = ndimage.binary_dilation(
        ndimage.binary_erosion(fractional, structure=sd.disk(PLATEAU_RADIUS)),
        structure=sd.disk(PLATEAU_RADIUS)) & fractional
    label_weight = ~unknown | ((difference <= STABLE_DIFFERENCE) & ~plateau)

    directory = sd.label_dir(out_dir, item["id"])
    directory.mkdir(parents=True, exist_ok=True)
    save_unit_image(directory / "trimap.png", trimap)
    save_unit_image(directory / "alpha_eps7.png", alpha_eps7)
    save_unit_image(directory / "alpha_eps5.png", alpha_eps5)
    save_unit_image(directory / "alpha_label.png", alpha_label)
    save_unit_image(directory / "label_weight.png", label_weight.astype(np.float64))

    label_sky = alpha_label >= 0.5
    union = np.count_nonzero(label_sky | coarse_sky)
    unknown_count = max(1, np.count_nonzero(unknown))
    in_unknown = alpha_label[unknown]
    return {
        "id": item["id"], "group": item["group"], "license": item["license"],
        "width": width, "height": height, "sky_colored_material": sky_colored_material,
        "coarse_sky_share": round(float(coarse_sky.mean()), 4),
        "void_to_sky_share": round(void_to_sky_share, 4),
        "unknown_share": round(float(unknown.mean()), 4),
        "label_sky_share": round(float(alpha_label.mean()), 4),
        "label_coarse_iou": round(np.count_nonzero(label_sky & coarse_sky) / max(1, union), 4),
        "label_fractional_share": round(np.count_nonzero(
            (in_unknown > FRACTIONAL_LOW) & (in_unknown < FRACTIONAL_HIGH)) / unknown_count, 4),
        "label_sky_in_tree_share": (round(float(label_sky[tree].mean()), 4)
                                    if tree.any() else None),
        "unstable_share_of_unknown": round(
            np.count_nonzero(unknown & ~label_weight) / unknown_count, 4),
        "eps_mean_difference": round(float(difference[unknown].mean())
                                     if unknown.any() else 0.0, 4),
        "seconds": round(seconds, 2),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--photo-dir", type=Path, required=True)
    parser.add_argument("--panoptic-zip", type=Path, required=True)
    parser.add_argument("--panoptic-json", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0, help="first N photos only (0 = all)")
    parser.add_argument("--image-ids", type=str, default="", help="comma-separated ids only")
    args = parser.parse_args()
    started = time.perf_counter()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    items = manifest["images"]
    if args.image_ids:
        wanted = {int(part) for part in args.image_ids.split(",")}
        items = [item for item in items if item["id"] in wanted]
    if args.limit:
        items = items[: args.limit]

    data = json.loads(args.panoptic_json.read_text(encoding="utf-8"))
    names = {c["name"]: c["id"] for c in data["categories"]}
    category_ids = {"sky": {names[sd.SKY_CATEGORY]},
                    "tree": {names[name] for name in sd.TREE_CATEGORIES},
                    "sky_colored": {names[name] for name in sd.SKY_COLORED_CATEGORIES}}
    segments = {a["image_id"]: a["segments_info"] for a in data["annotations"]}
    del data

    stats_path = args.out_dir / "label_stats.json"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    stats: list[dict] = []
    with zipfile.ZipFile(args.panoptic_zip) as archive:
        for index, item in enumerate(items, start=1):
            entry = label_photo(item, args.photo_dir, archive, segments[item["id"]],
                                category_ids, args.out_dir)
            stats.append(entry)
            print(f"{index}/{len(items)} {item['id']} {entry['seconds']} s "
                  f"unknown {entry['unknown_share']} "
                  f"unstable {entry['unstable_share_of_unknown']}", flush=True)
            stats_path.write_text(json.dumps({
                "step": "sky matte labels (README decision U17): labels",
                "algorithm": "pymatting.estimate_alpha_cf (closed-form matting)",
                "epsilons": [EPSILON_FINE, EPSILON_SMOOTH],
                "stable_difference": STABLE_DIFFERENCE,
                "plateau": {"low": PLATEAU_LOW, "high": PLATEAU_HIGH,
                            "radius": PLATEAU_RADIUS},
                "trimap": {"boundary_band_fraction": sd.BOUNDARY_BAND_FRACTION,
                           "void_depth_fraction": sd.VOID_DEPTH_FRACTION,
                           "sky_color_far": sd.SKY_COLOR_FAR,
                           "sky_color_near": sd.SKY_COLOR_NEAR,
                           "sky_color_neighbors": sd.SKY_COLOR_NEIGHBORS,
                           "sky_flat_window": sd.SKY_FLAT_WINDOW,
                           "sky_flat_std": sd.SKY_FLAT_STD,
                           "dark_sky_luminance": sd.DARK_SKY_LUMINANCE,
                           "void_to_sky_max_share": sd.VOID_TO_SKY_MAX_SHARE,
                           "sky_outlier_ring_fraction": sd.SKY_OUTLIER_RING_FRACTION,
                           "sky_color_samples": sd.SKY_COLOR_SAMPLES},
                "photos": len(stats),
                "seconds": round(time.perf_counter() - started, 1),
                "images": stats}, indent=1) + "\n", encoding="utf-8", newline="\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
