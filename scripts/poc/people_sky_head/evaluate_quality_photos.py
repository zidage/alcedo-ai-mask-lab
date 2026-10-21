#!/usr/bin/env python3
"""End to end on the quality photos: people_sky_head.onnx prompts -> SAM 2.1 tiny decoder (step 1).

Input:  --head-onnx and --head-json (export_head_onnx.py output; the presence thresholds are the
        best-F1 thresholds of the COCO validation split in the json), --photos, --onnx-dir
        (SAM 2.1 tiny `point_encoder.onnx` and `point_decoder.onnx`), --reference-masks,
        --reference-scores (E1 compare.json), --presence (the P0 class_presence.json).
Output: --out-dir/quality_photos.json and, with --logit-dir, the selected 256 x 256 logits of each
        prompt (float16, for make_quality_sheets.py).

Per photo, the encoder runs once (ONNX Runtime CPU, --threads intra-op threads, CPU memory arena
off), then the head reads `image_embed`:

  sky     found when presence[sky] >= the sky threshold. The prompts are one set per 8-connected
          region of cells with sky probability >= 0.5 and at least --sky-region-cells cells: the
          centers of the 3 most probable cells of the region as positive points. The sky mask is
          the union of the region masks.
  people  found when presence[person] >= the person threshold. The People list is
          head_model.decode_people with nms_iou = --nms-iou (score >= 0.05, at most 20 boxes);
          each box, clipped to the photo, is one SAM 2.1 box prompt (a box that extends past
          the photo edge selects the wrong area). All boxes are decoded on the photos with a
          referenced person, so the list at any score threshold can be read from the output.

Mask selection as in E1 and P0: a box or several points give the single-mask selection
(`select_single_mask`). IoU at the photo size (bilinear upsampling of the logits, threshold 0)
against the class reference of P0 (E1 base plus masks). On photos with a referenced person, each
list entry has its IoU with the referenced person; the photo has one reference, so a box on
another person has an IoU near 0.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "sam_feature_localization"))
import feature_localization as fl  # noqa: E402
import head_model as hm  # noqa: E402

SKY_POINTS_PER_REGION = 3


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def sky_regions(sky_probability: np.ndarray, min_cells: int) -> list[np.ndarray]:
    """Boolean [64, 64] masks of the 8-connected regions of cells with probability >= 0.5 that
    have at least `min_cells` cells, largest first."""
    from scipy import ndimage

    selected = sky_probability >= 0.5
    regions, count = ndimage.label(selected, structure=np.ones((3, 3), int))
    masks = [regions == i for i in range(1, count + 1)]
    masks = [m for m in masks if int(m.sum()) >= min_cells]
    return sorted(masks, key=lambda m: -int(m.sum()))


def region_points(sky_probability: np.ndarray, region: np.ndarray) -> list[list[float]]:
    """Model-pixel centers of the SKY_POINTS_PER_REGION most probable cells of a region."""
    score = np.where(region, sky_probability, -1.0)
    return [fl.cell_center_model(r, c) for r, c in fl.top_cells(score, SKY_POINTS_PER_REGION)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--head-onnx", required=True)
    parser.add_argument("--head-json", required=True)
    parser.add_argument("--photos", required=True)
    parser.add_argument("--onnx-dir", required=True)
    parser.add_argument("--reference-masks", required=True)
    parser.add_argument("--reference-scores", required=True)
    parser.add_argument("--presence", default=str(fl.HERE / "class_presence.json"))
    parser.add_argument("--sky-region-cells", type=int, default=SKY_POINTS_PER_REGION)
    parser.add_argument("--nms-iou", type=float, default=0.5)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--logit-dir", default="")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()

    fl.limit_threads(args.threads)
    fl.add_sam2_scripts_to_path()
    import cv2
    import torch
    from PIL import Image

    import sam2_graphs as graphs
    from compare_onnx_outputs import LOGIT_LIMIT, iou, select_single_mask

    head_info = fl.read_json(Path(args.head_json))
    validation = head_info["validation"]
    thresholds = {"sky": validation["presence_best_f1_sky"]["threshold"],
                  "person": validation["presence_best_f1_person"]["threshold"]}
    head_sha256 = fl.file_sha256(Path(args.head_onnx))
    if head_sha256 != head_info["onnx"]["sha256"]:
        raise ValueError(f"{args.head_onnx}: SHA-256 {head_sha256}, "
                         f"{args.head_json} names {head_info['onnx']['sha256']}")
    head = fl.create_cpu_session(Path(args.head_onnx), args.threads)
    encoder = fl.create_cpu_session(Path(args.onnx_dir) / "point_encoder.onnx", args.threads)
    decoder = fl.create_cpu_session(Path(args.onnx_dir) / "point_decoder.onnx", args.threads)
    presence = fl.read_json(Path(args.presence))["photos"]
    compare_results = fl.read_json(Path(args.reference_scores))["results"]
    logit_dir = Path(args.logit_dir) if args.logit_dir else None
    if logit_dir:
        logit_dir.mkdir(parents=True, exist_ok=True)

    timings = {"encoder_ms": [], "head_ms": [], "decoder_ms": []}

    def decode(features, coords, labels) -> np.ndarray:
        size = graphs.LOW_RES_MASK_SIZE
        started = time.perf_counter()
        logits, scores, _ = decoder.run(None, {
            "image_embed": features[0], "high_res_feats_0": features[1],
            "high_res_feats_1": features[2],
            "point_coords": np.array([coords], np.float32),
            "point_labels": np.array([labels], np.float32),
            "mask_input": np.zeros((1, 1, size, size), np.float32),
            "has_mask_input": np.zeros(1, np.float32)})
        timings["decoder_ms"].append((time.perf_counter() - started) * 1000)
        selected, _ = select_single_mask(logits[0], scores[0])
        return np.clip(selected, -LOGIT_LIMIT, LOGIT_LIMIT)

    photos = {}
    for stem in sorted(presence):
        image = Image.open(Path(args.photos) / f"{stem}.jpg").convert("RGB")
        width, height = image.size
        started = time.perf_counter()
        features = encoder.run(None, {"pixel_values": graphs.preprocess_image(image).numpy()})
        timings["encoder_ms"].append((time.perf_counter() - started) * 1000)
        started = time.perf_counter()
        head_outputs = dict(zip(hm.OUTPUT_NAMES, head.run(None, {"image_embed": features[0]})))
        timings["head_ms"].append((time.perf_counter() - started) * 1000)

        def to_mask(logits: np.ndarray, name: str) -> np.ndarray:
            if logit_dir:
                np.save(logit_dir / f"{stem}__{name}.npy", logits.astype(np.float16))
            return cv2.resize(logits, (width, height), interpolation=cv2.INTER_LINEAR) > 0

        def reference(name: str) -> np.ndarray | None:
            entry = presence[stem][name]
            if entry["status"] != "referenced":
                return None
            mask, _ = fl.class_reference_mask(stem, entry, Path(args.reference_masks),
                                              compare_results)
            return mask

        presence_scores = sigmoid(head_outputs["presence_logits"][0])
        record = {"width": width, "height": height}

        # Sky.
        sky_probability = sigmoid(head_outputs["sky_logits"][0, 0])
        sky_found = bool(presence_scores[0] >= thresholds["sky"])
        regions = sky_regions(sky_probability, args.sky_region_cells) if sky_found else []
        sky = {"status": presence[stem]["sky"]["status"], "presence": float(presence_scores[0]),
               "found": sky_found, "cells_ge_0_5": int((sky_probability >= 0.5).sum()),
               "regions": []}
        sky_reference = reference("sky")
        union = np.zeros((height, width), bool)
        for index, region in enumerate(regions):
            points = region_points(sky_probability, region)
            mask = to_mask(decode(features, points, [1.0] * len(points)), f"sky_region{index}")
            union |= mask
            sky["regions"].append({"cells": int(region.sum()), "points": points,
                                   "mask_area_fraction": float(mask.mean())})
        sky["mask_area_fraction"] = float(union.mean())
        if sky_reference is not None:
            sky["iou"] = iou(union, sky_reference) if regions else 0.0
            # The COCO measurement sky_cell_iou on this photo: probability >= 0.5 against a
            # reference share >= 0.5, over all cells.
            sky["cell_iou"] = iou(sky_probability >= 0.5, fl.cell_fraction(sky_reference) >= 0.5)
        record["sky"] = sky

        # People.
        torch_outputs = {k: torch.from_numpy(v) for k, v in head_outputs.items()}
        people_list = hm.decode_people(torch_outputs, nms_iou=args.nms_iou)[0]
        all_maxima = hm.decode_people(torch_outputs)[0]
        person_reference = reference("person")
        people = {"status": presence[stem]["person"]["status"],
                  "presence": float(presence_scores[1]),
                  "found": bool(presence_scores[1] >= thresholds["person"]),
                  "local_maxima": len(all_maxima["boxes"]), "list": []}
        for rank, (box, score) in enumerate(zip(people_list["boxes"].tolist(),
                                                people_list["scores"].tolist()), start=1):
            model_box = [min(max(v, 0.0), float(fl.GRID)) * fl.CELL_MODEL_PIXELS for v in box]
            entry = {"rank": rank, "score": score, "box_cells": box}
            if person_reference is not None:
                mask = to_mask(decode(features, [model_box[:2], model_box[2:]], [2.0, 3.0]),
                               f"person_rank{rank}")
                entry["iou"] = iou(mask, person_reference)
                entry["mask_area_fraction"] = float(mask.mean())
            people["list"].append(entry)
        if person_reference is not None:
            ious = [e["iou"] for e in people["list"]]
            people["top1_iou"] = ious[0] if ious else 0.0
            best = int(np.argmax(ious)) if ious else -1
            people["best_iou"] = ious[best] if ious else 0.0
            people["best_rank"] = best + 1 if ious else None
            people["best_score"] = people["list"][best]["score"] if ious else None
        record["person"] = people
        photos[stem] = record
        print(f"[photo] {stem}: sky {sky['presence']:.3f} found {sky_found} "
              f"regions {len(regions)} IoU {sky.get('iou', float('nan')):.3f} | person "
              f"{people['presence']:.3f} list {len(people['list'])} "
              f"top-1 IoU {people.get('top1_iou', float('nan')):.3f} best "
              f"{people.get('best_iou', float('nan')):.3f} at rank {people.get('best_rank')}")

    fl.write_json(Path(args.out_dir) / "quality_photos.json", {
        "step": "README section 5, step 1: head prompts on the quality photos, end to end",
        "head": {"path": Path(args.head_onnx).name, "sha256": head_sha256,
                 "checkpoint_epoch": head_info["checkpoint_epoch"]},
        "sam2_onnx": {name: fl.file_sha256(Path(args.onnx_dir) / name)
                      for name in ("point_encoder.onnx", "point_decoder.onnx")},
        "presence_thresholds": thresholds,
        "presence_threshold_source": "best-F1 thresholds of the COCO validation split "
                                     "(export json, validation.presence_best_f1_*)",
        "sky_region_cells": args.sky_region_cells,
        "sky_points_per_region": SKY_POINTS_PER_REGION,
        "nms_iou": args.nms_iou,
        "threads": args.threads,
        "timings_median_ms": {k: float(np.median(v)) for k, v in timings.items()},
        "decoder_runs": len(timings["decoder_ms"]),
        "peak_working_set_bytes": fl.peak_working_set_bytes(),
        "machine": fl.machine_description(),
        "photos": photos,
    })
    print(f"[done] median encoder {np.median(timings['encoder_ms']):.0f} ms, head "
          f"{np.median(timings['head_ms']):.2f} ms, decoder {np.median(timings['decoder_ms']):.1f}"
          f" ms ({len(timings['decoder_ms'])} runs), peak working set "
          f"{fl.peak_working_set_bytes() / 2**20:.0f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
