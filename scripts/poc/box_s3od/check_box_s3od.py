#!/usr/bin/env python3
"""Proof of concept: S3OD on the crop of a box, for Sky (1 - alpha) and for objects (trees,
buildings, cars, dogs).

Input:  the photos and the hand-drawn boxes of `boxes.json` (Wikimedia Commons photos, not the
        quality photos; `--photos`, default `build/tmp/box_s3od_poc/photos`), the MVP graphs
        (`mask_pipeline.GraphPaths.default`) with the S3OD graph `--subject-onnx`.
Output: under --out (default `build/tmp/box_s3od_poc/check/`): one HTML page per photo,
        `index.html`, and `summary.json` (crop rectangles, S3OD IoU scores, the selected output,
        the mask areas, the run times).

Question: when a box gives the region (a Sky box from the top of the photo to the horizon, or a
box around a tree or an object), does S3OD on the crop of the box give a better mask than one
SAM 2.1 box prompt?

Rules:

- Crop rectangle: the box widened by `margin` of the box width on the left and on the right and
  of the box height on the top and on the bottom, clipped to the photo (the People rule E3 of
  `mask_pipeline.person_crop_rect`). Objects: margin 0.1 (the People value). Sky: margin 0, because
  the Sky box ends at the horizon and a margin below the box adds ground to the crop.
- Crop image: `mask_pipeline.person_crop_image` (long edge at most 1024 kept, else the S3OD
  resize).
- S3OD: letterbox to 1024, the three output masks and the three IoU scores; the selected mask is
  the mask with the largest IoU score (the Subject rule).
- Object mask: the selected alpha inside the crop rectangle, 0 outside.
- Sky mask ("background of the crop"): 1 - the selected alpha inside the crop rectangle, 0
  outside.
- References on the same photo: one SAM 2.1 box prompt with the same box (the E1 single mask
  selection, threshold 0); for Sky also the one-click Sky of the MVP (P1 run 2 head, 3 points per
  region) and the photo Background (1 - S3OD alpha of the photo).

The check is visual: the photos have no reference masks. The summary records the areas of the
masks (share of the photo pixels with a value >= 0.5) so that large differences are visible.
"""

from __future__ import annotations

import argparse
import html
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "ai_mask_mvp"))
import mask_pipeline as mp  # noqa: E402

REPOSITORY = mp.REPOSITORY
DEFAULT_WORK = REPOSITORY / "build" / "tmp" / "box_s3od_poc"
DEFAULT_SUBJECT = (mp.DEFAULT_EXPLORE / "s3od" / "onnx_mha_chunk8_final" / "float16"
                   / "subject_masks.onnx")
OBJECT_MARGIN = 0.1
SKY_MARGIN = 0.0
PANEL_MAX_EDGE = 420
OVERVIEW_MAX_EDGE = 900
OVERLAY_OPACITY = 0.55
COLORS = {"s3od": (0, 200, 255), "sam": (255, 160, 0), "sky_p1": (255, 40, 200),
          "background": (255, 40, 80)}


def overlay(rgb8: np.ndarray, alpha: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:
    """`rgb8` with `color` blended by `alpha` (in [0, 1]) times OVERLAY_OPACITY."""
    weight = (np.clip(alpha, 0.0, 1.0) * OVERLAY_OPACITY)[..., None]
    return np.round(rgb8 * (1.0 - weight) + np.array(color) * weight).astype(np.uint8)


def resized(rgb8: np.ndarray, max_edge: int) -> np.ndarray:
    from PIL import Image

    image = Image.fromarray(rgb8)
    image.thumbnail((max_edge, max_edge), Image.Resampling.LANCZOS)
    return np.asarray(image)


def crop_rect(box: list[float], margin: float, width: int,
              height: int) -> tuple[int, int, int, int]:
    """The pixel rectangle (x0, y0, x1, y1), end exclusive, of the normalized `box` widened by
    `margin` of its size on each side, clipped to the photo (mask_pipeline.person_crop_rect with
    another margin)."""
    x0, y0, x1, y1 = box
    margin_x, margin_y = margin * (x1 - x0), margin * (y1 - y0)
    left, right = mp.pixel_span(max(x0 - margin_x, 0.0), min(x1 + margin_x, 1.0), width)
    top, bottom = mp.pixel_span(max(y0 - margin_y, 0.0), min(y1 + margin_y, 1.0), height)
    return left, top, right, bottom


def s3od_outputs(pipeline: mp.MaskPipeline, crop: np.ndarray, photo: str,
                 label: str) -> tuple[np.ndarray, np.ndarray, float]:
    """S3OD on the crop image: the three masks unpadded (crop grid), the three IoU scores, and
    the run time."""
    s3od = pipeline.s3od
    pixels, info = s3od.letterbox(np.ascontiguousarray(crop), pipeline.subject_size)
    (masks, scores), milliseconds = pipeline._run("subject_matte", {"pixel_values": pixels},
                                                  "s3od_box", photo=photo, box=label)
    return (np.ascontiguousarray(s3od.unpad(masks[0], info), np.float32), scores[0].copy(),
            milliseconds)


def paste(alpha: np.ndarray, rect: tuple[int, int, int, int], width: int,
          height: int) -> np.ndarray:
    """The crop-grid alpha at the photo size inside `rect`, 0 outside (section 2.5 sampling)."""
    full = np.zeros((height, width), np.float32)
    values, (rows, columns) = mp.alpha_on_image(alpha, mp.mask_box_of_rect(rect, width, height),
                                                width, height)
    full[rows, columns] = values
    return full


def sam_box_mask(pipeline: mp.MaskPipeline, state: mp.PhotoState, box: list[float],
                 label: str) -> np.ndarray:
    """One SAM 2.1 box prompt (corners with labels 2 and 3), the E1 single mask, at the photo
    size, as float32 0 or 1."""
    model_box = [v * 1024.0 for v in box]
    run = pipeline.decode(state, [model_box[:2], model_box[2:]], [2.0, 3.0], None,
                          f"box {label}")
    return mp.logits_to_mask(pipeline.single_mask(run), state.photo.width,
                             state.photo.height).astype(np.float32)


def p1_sky_mask(pipeline: mp.MaskPipeline, state: mp.PhotoState) -> tuple[np.ndarray, str]:
    """The one-click Sky of the MVP at the photo size, and its message."""
    result = pipeline.select_sky(state)
    mask = np.zeros((state.photo.height, state.photo.width), np.float32)
    for region in result.regions:
        mask = np.maximum(mask, mp.logits_to_mask(region["logits"], state.photo.width,
                                                  state.photo.height))
    return mask, result.message


def area(alpha: np.ndarray) -> float:
    return round(float((alpha >= mp.ALPHA_INSIDE).mean()), 4)


def panel(rgb8: np.ndarray, alpha: np.ndarray, color: tuple[int, int, int], rect,
          box_px, out: Path, name: str, max_edge: int = PANEL_MAX_EDGE) -> str:
    """The crop `rect` of `rgb8` with `alpha` (photo size) over it and the box outline."""
    from PIL import Image, ImageDraw

    x0, y0, x1, y1 = rect
    image = Image.fromarray(resized(overlay(rgb8[y0:y1, x0:x1], alpha[y0:y1, x0:x1], color),
                                    max_edge))
    draw = ImageDraw.Draw(image)
    sx, sy = image.width / (x1 - x0), image.height / (y1 - y0)
    bx0, by0, bx1, by1 = box_px
    draw.rectangle([(bx0 - x0) * sx, (by0 - y0) * sy, (bx1 - x0) * sx - 1, (by1 - y0) * sy - 1],
                   outline=(255, 255, 0), width=2)
    image.save(out / name, quality=88)
    return name


def measure_photo(pipeline: mp.MaskPipeline, path: Path, entry: dict, out: Path) -> dict:
    state = pipeline.open_photo(path)
    photo = state.photo
    width, height = photo.width, photo.height
    rgb8 = np.asarray(photo.image)
    stem = path.stem
    s3od = pipeline.s3od
    record = {"photo": path.name, "license": entry["license"], "page": entry["page"],
              "size": [width, height], "boxes": []}
    has_sky = any(b["label"] == "sky" for b in entry["boxes"])
    if has_sky:
        background = 1.0 - pipeline.subject_alpha_at_photo(state)
        sky_p1, sky_message = p1_sky_mask(pipeline, state)
        record["sky_p1"] = {"message": sky_message, "area": area(sky_p1)}
        record["background_area"] = area(background)
    for index, item in enumerate(entry["boxes"]):
        label, box = item["label"], item["box"]
        is_sky = label == "sky"
        margin = SKY_MARGIN if is_sky else OBJECT_MARGIN
        rect = crop_rect(box, margin, width, height)
        box_px = crop_rect(box, 0.0, width, height)
        crop = mp.person_crop_image(rgb8, rect, s3od)
        masks, scores, milliseconds = s3od_outputs(pipeline, crop, path.name, label)
        selected = int(np.argmax(scores))
        outputs = [paste(m, rect, width, height) for m in masks]
        if is_sky:
            result = np.zeros((height, width), np.float32)
            x0, y0, x1, y1 = rect
            result[y0:y1, x0:x1] = 1.0 - outputs[selected][y0:y1, x0:x1]
        else:
            result = outputs[selected]
        sam = sam_box_mask(pipeline, state, box, label)
        prefix = f"{stem}_b{index}"
        panels = []
        for k, alpha in enumerate(outputs):
            mark = " (selected)" if k == selected else ""
            panels.append((f"S3OD output {k}, IoU {scores[k]:.3f}{mark}",
                           panel(rgb8, alpha, COLORS["s3od"], rect, box_px, out,
                                 f"{prefix}_s3od{k}.jpg")))
        if is_sky:
            panels.append(("Sky = 1 - selected S3OD alpha (in the box)",
                           panel(rgb8, result, COLORS["sky_p1"], rect, box_px, out,
                                 f"{prefix}_sky.jpg")))
        panels.append(("SAM 2.1 tiny, same box prompt",
                       panel(rgb8, sam, COLORS["sam"], rect, box_px, out, f"{prefix}_sam.jpg")))
        if is_sky:
            panels.append((f"MVP one-click Sky (P1 head + SAM points): {sky_message}",
                           panel(rgb8, sky_p1, COLORS["sky_p1"], rect, box_px, out,
                                 f"{prefix}_skyp1.jpg")))
            panels.append(("Photo Background = 1 - S3OD alpha of the photo",
                           panel(rgb8, background, COLORS["background"], rect, box_px, out,
                                 f"{prefix}_bg.jpg")))
        record["boxes"].append({
            "label": label, "box": box, "margin": margin, "rect": list(rect),
            "crop_size": list(crop.shape[1::-1]), "iou_scores": [round(float(s), 4)
                                                                 for s in scores],
            "selected": selected, "s3od_ms": round(milliseconds, 1),
            "area_result": area(result), "area_sam": area(sam),
            "panels": panels})
        print(f"[box] {stem} {label}: IoU {np.round(scores, 3).tolist()} selected {selected}, "
              f"{milliseconds:.0f} ms", flush=True)
    return record


def write_page(record: dict, out: Path) -> str:
    stem = Path(record["photo"]).stem
    sections = []
    for item in record["boxes"]:
        cells = "".join(f"<figure><img src='{name}'><figcaption>{html.escape(caption)}"
                        f"</figcaption></figure>" for caption, name in item["panels"])
        sections.append(
            f"<h2>{html.escape(item['label'])} box {item['box']}</h2>"
            f"<p>crop rectangle {item['rect']} (margin {item['margin']}), crop image "
            f"{item['crop_size'][0]} x {item['crop_size'][1]}, S3OD {item['s3od_ms']} ms; "
            f"mask area: S3OD result {item['area_result']}, SAM box {item['area_sam']}. "
            f"Yellow line: the box.</p><div class='row'>{cells}</div>")
    page = ("<!doctype html><html><head><meta charset='utf-8'>"
            f"<title>{html.escape(stem)}</title><style>body{{font-family:sans-serif;"
            "background:#222;color:#ddd}.row{display:flex;flex-wrap:wrap;gap:8px}"
            "figure{margin:0;max-width:430px}figcaption{font-size:12px}</style></head><body>"
            f"<h1>{html.escape(stem)}</h1><p><a href='index.html'>index</a> | "
            f"<a href='{html.escape(record['page'])}'>source</a> ({html.escape(record['license'])})"
            f", {record['size'][0]} x {record['size'][1]}</p>" + "".join(sections)
            + "</body></html>")
    name = f"{stem}.html"
    (out / name).write_text(page, encoding="utf-8", newline="\n")
    return name


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--photos", type=Path, default=DEFAULT_WORK / "photos")
    parser.add_argument("--boxes", type=Path, default=HERE / "boxes.json")
    parser.add_argument("--subject-onnx", type=Path, default=DEFAULT_SUBJECT)
    parser.add_argument("--out", type=Path, default=DEFAULT_WORK / "check")
    mp.add_provider_arguments(parser, default="directml")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()

    paths = mp.GraphPaths.default()
    paths.subject_onnx = args.subject_onnx
    started = time.perf_counter()
    pipeline = mp.MaskPipeline(paths, provider=args.provider, threads=args.threads,
                               **mp.pipeline_provider_options(args))
    if pipeline.subject_kind != "s3od":
        raise ValueError(f"{args.subject_onnx}: the check needs an S3OD graph, not "
                         f"{pipeline.subject_kind}")
    args.out.mkdir(parents=True, exist_ok=True)
    entries = json.loads(args.boxes.read_text(encoding="utf-8"))["photos"]
    records = [measure_photo(pipeline, args.photos / entry["file"], entry, args.out)
               for entry in entries]
    links = [f"<li><a href='{write_page(r, args.out)}'>{html.escape(r['photo'])}</a>: "
             + ", ".join(html.escape(b["label"]) for b in r["boxes"]) + "</li>" for r in records]
    (args.out / "index.html").write_text(
        "<!doctype html><html><head><meta charset='utf-8'><title>S3OD on box crops</title>"
        "</head><body><h1>S3OD on box crops</h1><ul>" + "".join(links) + "</ul></body></html>",
        encoding="utf-8", newline="\n")
    for record in records:
        for item in record["boxes"]:
            item["panels"] = [name for _, name in item["panels"]]
    mp.save_json(args.out / "summary.json", {
        "step": "PoC: S3OD on box crops (Sky and objects)", "provider": args.provider,
        "pipeline": {k: v for k, v in pipeline.log().items() if k != "events"},
        "object_margin": OBJECT_MARGIN, "sky_margin": SKY_MARGIN, "photos": records,
        "seconds": round(time.perf_counter() - started, 1)})
    print(f"[done] {args.out / 'index.html'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
