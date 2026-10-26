#!/usr/bin/env python3
"""Write the reference masks of the accepted model set for the MAM check.

Input:  the photos and the hand-drawn boxes of `scripts/poc/box_s3od/boxes.json` (Wikimedia
        Commons photos, not the quality photos; `--photos`, default
        `build/tmp/box_s3od_poc/photos`) and the MVP graphs (`mask_pipeline.GraphPaths.default`)
        with the accepted S3OD graph (`--subject-onnx`).
Output: under `--out` (default `build/tmp/mam_poc/references/`):
        - `references.json`: for each photo with a Sky box, the head message and, for each sky
          region, the normalized (x, y) points that the MVP gives to the SAM 2.1 decoder (P1
          section 3.3: the 3 most probable cells of each region); for each box, the S3OD IoU
          scores and the selected output.
        - `<stem>_b<index>_s3od.png`: S3OD on the crop of the box (`check_box_s3od.py` rules):
          the selected alpha for an object box, 1 - the selected alpha for a Sky box, at the photo
          size, 8-bit.
        - `<stem>_b<index>_sam21.png`: one SAM 2.1 tiny box prompt with the same box (0 or 255).
        - `<stem>_skyp1.png`: the MVP one-click Sky (0 or 255).

Step: MAM proof of concept. `check_mam.py` gives the sky points to the SAM ViT-B prompt encoder of
MAM (the same prompt as the MVP Sky) and shows these masks beside the MAM alpha.

The SAM 2.1 graphs read the photo resized to 1024 x 1024 (no letterbox), so a model pixel divided
by 1024 is the normalized photo coordinate.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "box_s3od"))
import check_box_s3od as box  # noqa: E402

mp = box.mp
DEFAULT_WORK = mp.REPOSITORY / "build" / "tmp"
MODEL_INPUT_SIZE = 1024.0


def save_alpha(path: Path, alpha: np.ndarray) -> None:
    from PIL import Image

    Image.fromarray(np.round(np.clip(alpha, 0.0, 1.0) * 255.0).astype(np.uint8)).save(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--photos", type=Path, default=DEFAULT_WORK / "box_s3od_poc" / "photos")
    parser.add_argument("--boxes", type=Path, default=HERE.parent / "box_s3od" / "boxes.json")
    parser.add_argument("--subject-onnx", type=Path, default=box.DEFAULT_SUBJECT)
    parser.add_argument("--out", type=Path, default=DEFAULT_WORK / "mam_poc" / "references")
    mp.add_provider_arguments(parser, default="directml")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()

    paths = mp.GraphPaths.default()
    paths.subject_onnx = args.subject_onnx
    pipeline = mp.MaskPipeline(paths, provider=args.provider, threads=args.threads,
                               **mp.pipeline_provider_options(args))
    if pipeline.subject_kind != "s3od":
        raise ValueError(f"{args.subject_onnx}: needs an S3OD graph, not {pipeline.subject_kind}")
    args.out.mkdir(parents=True, exist_ok=True)
    entries = json.loads(args.boxes.read_text(encoding="utf-8"))["photos"]
    photos = {}
    for entry in entries:
        path = args.photos / entry["file"]
        state = pipeline.open_photo(path)
        width, height = state.photo.width, state.photo.height
        rgb8 = np.asarray(state.photo.image)
        stem = path.stem
        record = {"size": [width, height], "boxes": []}
        if any(b["label"] == "sky" for b in entry["boxes"]):
            result = pipeline.select_sky(state)
            record["sky"] = {
                "message": result.message, "found": result.found,
                "regions": [[[x / MODEL_INPUT_SIZE, y / MODEL_INPUT_SIZE]
                             for x, y in region["points"]] for region in result.regions]}
            sky_p1, _ = box.p1_sky_mask(pipeline, state)
            save_alpha(args.out / f"{stem}_skyp1.png", sky_p1)
            print(f"[sky] {stem}: {result.message}", flush=True)
        for index, item in enumerate(entry["boxes"]):
            label, rect_box = item["label"], item["box"]
            is_sky = label == "sky"
            rect = box.crop_rect(rect_box, box.SKY_MARGIN if is_sky else box.OBJECT_MARGIN,
                                 width, height)
            crop = mp.person_crop_image(rgb8, rect, pipeline.s3od)
            masks, scores, _ = box.s3od_outputs(pipeline, crop, path.name, label)
            selected = int(np.argmax(scores))
            alpha = box.paste(masks[selected], rect, width, height)
            if is_sky:
                x0, y0, x1, y1 = rect
                sky = np.zeros_like(alpha)
                sky[y0:y1, x0:x1] = 1.0 - alpha[y0:y1, x0:x1]
                alpha = sky
            save_alpha(args.out / f"{stem}_b{index}_s3od.png", alpha)
            save_alpha(args.out / f"{stem}_b{index}_sam21.png",
                       box.sam_box_mask(pipeline, state, rect_box, label))
            record["boxes"].append({"label": label, "rect": list(rect),
                                    "s3od_iou_scores": [round(float(s), 4) for s in scores],
                                    "s3od_selected": selected})
            print(f"[box] {stem} {label}: S3OD output {selected}", flush=True)
        photos[entry["file"]] = record
    mp.save_json(args.out / "references.json", {
        "step": "MAM PoC: reference masks of the accepted model set",
        "provider": args.provider, "subject_onnx": str(args.subject_onnx), "photos": photos})
    print(f"[done] {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
