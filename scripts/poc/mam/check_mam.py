#!/usr/bin/env python3
"""Proof of concept: Matting Anything (MAM, SAM ViT-B + M2M) for Sky and for object boxes, beside
S3OD on the box crop, the SAM 2.1 tiny box prompt, and the MVP one-click Sky.

Input:  the photos and the hand-drawn boxes of `scripts/poc/box_s3od/boxes.json` (Wikimedia
        Commons photos, not the quality photos; `--photos`, default
        `build/tmp/box_s3od_poc/photos`), the reference masks and sky points of
        `export_mvp_references.py` (`--references`), the MAM source tree (`--mam-source`) and the
        checkpoint `mam_sam_vitb.pth` (`--checkpoint`).
Output: under `--out` (default `build/tmp/mam_poc/check/`): one HTML page per photo,
        `index.html`, the MAM alphas (`alphas/*.png`, 8-bit), and `summary.json` (SAM IoU scores,
        mask areas, agreement with the references, share of soft alpha, run times).

Question: does MAM give a usable Sky mask with fine edges (trees, twigs, roofs against the sky),
and are its object masks on a box better than S3OD on the box crop?

Variants (PyTorch CPU, `mam_model.py`):

- Object box, `mam_crop`: MAM on the crop of the box (`check_box_s3od.crop_rect` with margin
  0.1, the S3OD box crop), box prompt = the box in crop pixels. MAM resizes the crop to a long edge
  of 1024.
- Object box, `mam_crop_mask`: as `mam_crop` with the output fusion `mask_one_side` (the demo
  "mask" mode: the inside of the SAM mask stays 1, M2M changes only the band outside it).
- Object box, `mam_photo`: MAM on the photo with the box prompt (the MAM demo use).
- Sky, `mam_sky_points_<target>_<guidance>`: MAM on the photo with the MVP sky points (one prompt
  of 3 points per sky region), the union (maximum) of the region skies. This is the one-click Sky
  of the MVP with MAM in place of SAM 2.1. `SKY_MODES` lists the target and the output fusion
  (`mam_model.py`): `selection_alpha` (the demo default: the sky is the alpha), `selection_mask_
  one_side` (the demo "mask" mode), and `complement_alpha` (M2M mattes the non-sky part, the sky
  is 1 - alpha).
- Sky, `mam_sky_box`: MAM on the photo with the Sky box as the box prompt (demo default).

The check is visual (the photos have no reference masks). The numbers are indicators: area (share
of the photo pixels with alpha >= 0.5), IoU of the binarized mask (>= 0.5) against S3OD on the
box crop and against the SAM 2.1 box prompt, and the soft share (share of the box pixels with
0.02 < alpha < 0.98). The zoom panels show the same window of 384 x 384 photo pixels for each
method at 1:1 (the window inside the box with the most soft MAM and S3OD pixels).
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
sys.path.insert(0, str(HERE.parent / "box_s3od"))
sys.path.insert(0, str(HERE))
import check_box_s3od as box  # noqa: E402
import mam_model  # noqa: E402

mp = box.mp
DEFAULT_WORK = mp.REPOSITORY / "build" / "tmp" / "mam_poc"
SOFT_LOW, SOFT_HIGH = 0.02, 0.98
ZOOM = 384
SKY_MODES = (("selection", "alpha"), ("selection", "mask_one_side"), ("complement", "alpha"))
COLORS = {"mam": (60, 255, 60), "sam_vitb": (255, 160, 0), "s3od": (0, 200, 255),
          "sam21": (255, 160, 0), "sky": (255, 40, 200)}


def read_alpha(path: Path) -> np.ndarray:
    from PIL import Image

    return np.asarray(Image.open(path), np.float32) / 255.0


def save_alpha(path: Path, alpha: np.ndarray) -> None:
    from PIL import Image

    Image.fromarray(np.round(np.clip(alpha, 0.0, 1.0) * 255.0).astype(np.uint8)).save(path)


def iou(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a >= mp.ALPHA_INSIDE, b >= mp.ALPHA_INSIDE
    union = int((a | b).sum())
    return round(float((a & b).sum() / union), 4) if union else 1.0


def soft_share(alpha: np.ndarray, rect: tuple[int, int, int, int]) -> float:
    x0, y0, x1, y1 = rect
    part = alpha[y0:y1, x0:x1]
    return round(float(((part > SOFT_LOW) & (part < SOFT_HIGH)).mean()), 4)


def zoom_window(soft_maps: list[np.ndarray], rect: tuple[int, int, int, int]) -> tuple[int, int]:
    """Top-left corner of the ZOOM x ZOOM window inside `rect` (clipped to the photo) with the
    most soft pixels in the sum of `soft_maps`."""
    import cv2

    x0, y0, x1, y1 = rect
    score = sum(((m > SOFT_LOW) & (m < SOFT_HIGH)).astype(np.float32) for m in soft_maps)
    height, width = score.shape
    density = cv2.blur(score, (ZOOM, ZOOM), borderType=cv2.BORDER_CONSTANT)
    half = ZOOM // 2

    def centers(start: int, end: int, size: int) -> tuple[int, int]:
        """The window centers on one axis; the clipped center of the span when the span is
        shorter than the window."""
        low, high = max(start + half, half), min(end - half, size - half)
        if low > high:
            low = high = min(max((start + end) // 2, half), size - half)
        return low, high

    cy0, cy1 = centers(y0, y1, height)
    cx0, cx1 = centers(x0, x1, width)
    part = density[cy0:cy1 + 1, cx0:cx1 + 1]
    row, column = np.unravel_index(int(np.argmax(part)), part.shape)
    return int(cx0 + column - half), int(cy0 + row - half)


def zoom_panel(rgb8: np.ndarray, alpha: np.ndarray, corner: tuple[int, int], color, out: Path,
               name: str) -> str:
    """The zoom window at 1:1: the photo with the alpha overlay on the left, the alpha as gray on
    the right."""
    from PIL import Image

    x, y = corner
    photo = rgb8[y:y + ZOOM, x:x + ZOOM]
    part = alpha[y:y + ZOOM, x:x + ZOOM]
    gray = np.repeat(np.round(np.clip(part, 0, 1) * 255).astype(np.uint8)[..., None], 3, axis=2)
    Image.fromarray(np.concatenate([box.overlay(photo, part, color), gray], axis=1)).save(
        out / name, quality=92)
    return name


def run_crop(model: mam_model.MamModel, rgb8: np.ndarray, rect, box_px,
             guidance_modes: tuple[str, ...]):
    """MAM on the crop `rect` with the box `box_px` (photo pixels) as the prompt, once per output
    fusion of `guidance_modes`; per mode the alpha pasted at the photo size (0 outside the crop)
    and the result; the SAM mask pasted at the photo size; the encoder time."""
    x0, y0, x1, y1 = rect
    encoded = model.encode(np.ascontiguousarray(rgb8[y0:y1, x0:x1]))
    bx0, by0, bx1, by1 = box_px
    height, width = rgb8.shape[:2]
    outputs = {}
    for guidance in guidance_modes:
        result = model.predict(encoded, box=[bx0 - x0, by0 - y0, bx1 - x0, by1 - y0],
                               guidance=guidance)
        alpha = np.zeros((height, width), np.float32)
        alpha[y0:y1, x0:x1] = result.alpha
        outputs[guidance] = (alpha, result)
    sam = np.zeros((height, width), np.float32)
    sam[y0:y1, x0:x1] = result.sam_mask
    return outputs, sam, encoded.milliseconds


def measure_photo(model: mam_model.MamModel, path: Path, entry: dict, references: dict,
                  reference_dir: Path, out: Path) -> dict:
    photo = mp.load_photo(path)
    rgb8 = np.asarray(photo.image)
    width, height = photo.width, photo.height
    stem = path.stem
    reference = references[entry["file"]]
    if reference["size"] != [width, height]:
        raise ValueError(f"{path}: size {width} x {height}, references {reference['size']}")
    encoded = model.encode(rgb8)
    record = {"photo": path.name, "license": entry["license"], "page": entry["page"],
              "size": [width, height], "encoder_ms_photo": round(encoded.milliseconds, 1),
              "boxes": []}
    print(f"[photo] {stem}: encoder {encoded.milliseconds:.0f} ms", flush=True)
    for index, item in enumerate(entry["boxes"]):
        label, norm_box = item["label"], item["box"]
        is_sky = label == "sky"
        prefix = f"{stem}_b{index}"
        box_px = box.crop_rect(norm_box, 0.0, width, height)
        s3od = read_alpha(reference_dir / f"{prefix}_s3od.png")
        sam21 = read_alpha(reference_dir / f"{prefix}_sam21.png")
        variants, panels, numbers = {}, [], {}
        if is_sky:
            rect = box.crop_rect(norm_box, box.SKY_MARGIN, width, height)
            regions = reference["sky"]["regions"]
            sky_p1 = read_alpha(reference_dir / f"{stem}_skyp1.png")
            shown, zooms = [], []
            for target, guidance in SKY_MODES:
                key = f"mam_sky_points_{target}_{guidance}"
                sky = np.zeros((height, width), np.float32)
                sam = np.zeros((height, width), np.float32)
                scores, milliseconds = [], 0.0
                for points in regions:
                    result = model.predict(encoded, points=[[x * width, y * height]
                                                            for x, y in points],
                                           guidance=guidance, target=target)
                    region_sky = result.alpha if target == "selection" else 1.0 - result.alpha
                    sky = np.maximum(sky, region_sky)
                    sam = np.maximum(sam, result.sam_mask if target == "selection"
                                     else 1.0 - result.sam_mask)
                    scores.append([round(v, 4) for v in result.iou_scores])
                    milliseconds += result.milliseconds
                variants[key] = sky
                numbers[key] = {"sam_iou_scores": scores, "ms": round(milliseconds, 1)}
                caption = f"MAM sky points, {target}, {guidance}"
                shown.append((caption, sky, COLORS["mam"]))
                zooms.append((caption, sky, COLORS["mam"]))
            shown.insert(1, (f"SAM ViT-B sky mask ({len(regions)} region(s))", sam,
                             COLORS["sam_vitb"]))
            result = model.predict(encoded, box=list(box_px))
            variants["mam_sky_box"] = result.alpha
            numbers["mam_sky_box"] = {"sam_iou_scores": [round(v, 4) for v in result.iou_scores],
                                      "ms": round(result.milliseconds, 1)}
            main = variants["mam_sky_points_complement_alpha"]
            shown += [("MAM, Sky box prompt (selection, alpha)", result.alpha, COLORS["mam"]),
                      (f"MVP one-click Sky (SAM 2.1): {reference['sky']['message']}", sky_p1,
                       COLORS["sky"]),
                      ("1 - S3OD alpha of the Sky box crop", s3od, COLORS["s3od"])]
            zooms += [("MVP one-click Sky", sky_p1, COLORS["sky"]),
                      ("1 - S3OD alpha (Sky box crop)", s3od, COLORS["s3od"])]
            compare = {"mvp_sky": sky_p1, "s3od": s3od, "sam21_box": sam21}
        else:
            rect = box.crop_rect(norm_box, box.OBJECT_MARGIN, width, height)
            outputs, sam, encoder_ms = run_crop(model, rgb8, rect, box_px,
                                                ("alpha", "mask_one_side"))
            for key, guidance in (("mam_crop", "alpha"), ("mam_crop_mask", "mask_one_side")):
                alpha, result = outputs[guidance]
                variants[key] = alpha
                numbers[key] = {"sam_iou_scores": [round(v, 4) for v in result.iou_scores],
                                "ms": round(result.milliseconds, 1),
                                "encoder_ms": round(encoder_ms, 1)}
            alpha = variants["mam_crop"]
            photo_result = model.predict(encoded, box=list(box_px))
            variants["mam_photo"] = photo_result.alpha
            numbers["mam_photo"] = {
                "sam_iou_scores": [round(s, 4) for s in photo_result.iou_scores],
                "ms": round(photo_result.milliseconds, 1)}
            main = alpha
            shown = [("MAM on the box crop", alpha, COLORS["mam"]),
                     ("SAM ViT-B mask that guides M2M (crop)", sam, COLORS["sam_vitb"]),
                     ("MAM on the box crop, mask mode", variants["mam_crop_mask"],
                      COLORS["mam"]),
                     ("MAM on the photo, box prompt", photo_result.alpha, COLORS["mam"]),
                     ("S3OD on the box crop (selected)", s3od, COLORS["s3od"]),
                     ("SAM 2.1 tiny, box prompt", sam21, COLORS["sam21"])]
            zooms = [("MAM on the box crop", alpha, COLORS["mam"]),
                     ("MAM on the box crop, mask mode", variants["mam_crop_mask"],
                      COLORS["mam"]),
                     ("MAM on the photo", photo_result.alpha, COLORS["mam"]),
                     ("S3OD on the box crop", s3od, COLORS["s3od"])]
            compare = {"s3od": s3od, "sam21_box": sam21}
        for key, alpha in variants.items():
            save_alpha(out / "alphas" / f"{prefix}_{key}.png", alpha)
            numbers[key].update({"area": box.area(alpha), "soft_share": soft_share(alpha, rect)})
            numbers[key].update({f"iou_{name}": iou(alpha, other)
                                 for name, other in compare.items()})
        reference_numbers = {name: {"area": box.area(other), "soft_share": soft_share(other, rect)}
                             for name, other in compare.items()}
        for k, (caption, alpha, color) in enumerate(shown):
            panels.append((caption, box.panel(rgb8, alpha, color, rect, box_px, out,
                                              f"{prefix}_p{k}.jpg")))
        corner = zoom_window([main, s3od], rect)
        zoom_panels = [(caption, zoom_panel(rgb8, alpha, corner, color, out,
                                            f"{prefix}_z{k}.jpg"))
                       for k, (caption, alpha, color) in enumerate(zooms)]
        record["boxes"].append({"label": label, "box": norm_box, "rect": list(rect),
                                "zoom_corner": list(corner), "variants": numbers,
                                "references": reference_numbers, "panels": panels,
                                "zoom_panels": zoom_panels})
        summary = ", ".join(f"{k} area {v['area']} IoU S3OD {v['iou_s3od']}"
                            for k, v in numbers.items())
        print(f"[box] {stem} {label}: {summary}", flush=True)
    return record


def write_page(record: dict, out: Path) -> str:
    stem = Path(record["photo"]).stem
    sections = []
    for item in record["boxes"]:
        cells = "".join(f"<figure><img src='{name}'><figcaption>{html.escape(caption)}"
                        f"</figcaption></figure>" for caption, name in item["panels"])
        zooms = "".join(f"<figure class='zoom'><img src='{name}'><figcaption>"
                        f"{html.escape(caption)} (left: overlay, right: alpha)</figcaption>"
                        f"</figure>" for caption, name in item["zoom_panels"])
        numbers = "; ".join(
            f"{html.escape(k)}: area {v['area']}, soft {v['soft_share']}, "
            + ", ".join(f"IoU {n[4:]} {v[n]}" for n in v if n.startswith("iou_"))
            for k, v in item["variants"].items())
        references = "; ".join(f"{html.escape(k)}: area {v['area']}, soft {v['soft_share']}"
                               for k, v in item["references"].items())
        sections.append(
            f"<h2>{html.escape(item['label'])} box {item['box']}</h2>"
            f"<p>{numbers}<br>references: {references}<br>Yellow line: the box. Zoom window "
            f"{ZOOM} x {ZOOM} px at {item['zoom_corner']}.</p><div class='row'>{cells}</div>"
            f"<div class='row'>{zooms}</div>")
    page = ("<!doctype html><html><head><meta charset='utf-8'>"
            f"<title>{html.escape(stem)}</title><style>body{{font-family:sans-serif;"
            "background:#222;color:#ddd}.row{display:flex;flex-wrap:wrap;gap:8px;"
            "margin-bottom:8px}figure{margin:0;max-width:430px}figure.zoom{max-width:none}"
            "figcaption{font-size:12px}</style></head><body>"
            f"<h1>{html.escape(stem)}</h1><p><a href='index.html'>index</a> | "
            f"<a href='{html.escape(record['page'])}'>source</a> ({html.escape(record['license'])})"
            f", {record['size'][0]} x {record['size'][1]}</p>" + "".join(sections)
            + "</body></html>")
    name = f"{stem}.html"
    (out / name).write_text(page, encoding="utf-8", newline="\n")
    return name


def sam_weights_difference(model: mam_model.MamModel, sam_checkpoint: Path) -> float:
    """Largest absolute difference between the SAM ViT-B weights in the MAM checkpoint and the
    official `sam_vit_b_01ec64.pth` (MAM freezes SAM, so 0 is expected)."""
    import torch

    official = torch.load(sam_checkpoint, map_location="cpu", weights_only=True)
    if set(official) != set(model.seg_state):
        raise ValueError(f"{sam_checkpoint}: other SAM keys than the MAM checkpoint")
    return max(float((official[k].float() - v).abs().max()) for k, v in model.seg_state.items())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--photos", type=Path,
                        default=mp.REPOSITORY / "build" / "tmp" / "box_s3od_poc" / "photos")
    parser.add_argument("--boxes", type=Path, default=HERE.parent / "box_s3od" / "boxes.json")
    parser.add_argument("--references", type=Path, default=DEFAULT_WORK / "references")
    parser.add_argument("--mam-source", type=Path,
                        default=DEFAULT_WORK / "src" / "Matting-Anything")
    parser.add_argument("--checkpoint", type=Path,
                        default=DEFAULT_WORK / "weights" / "mam_sam_vitb.pth")
    parser.add_argument("--sam-checkpoint", type=Path,
                        default=DEFAULT_WORK / "weights" / "sam_vit_b_01ec64.pth")
    parser.add_argument("--out", type=Path, default=DEFAULT_WORK / "check")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()

    started = time.perf_counter()
    model = mam_model.MamModel(args.mam_source, args.checkpoint, threads=args.threads)
    difference = sam_weights_difference(model, args.sam_checkpoint)
    print(f"[model] SAM ViT-B weights in the MAM checkpoint vs official: max |diff| {difference}",
          flush=True)
    (args.out / "alphas").mkdir(parents=True, exist_ok=True)
    references = json.loads((args.references / "references.json").read_text(
        encoding="utf-8"))["photos"]
    entries = json.loads(args.boxes.read_text(encoding="utf-8"))["photos"]
    records = [measure_photo(model, args.photos / e["file"], e, references, args.references,
                             args.out) for e in entries]
    links = [f"<li><a href='{write_page(r, args.out)}'>{html.escape(r['photo'])}</a>: "
             + ", ".join(html.escape(b["label"]) for b in r["boxes"]) + "</li>" for r in records]
    (args.out / "index.html").write_text(
        "<!doctype html><html><head><meta charset='utf-8'><title>MAM check</title></head><body>"
        "<h1>Matting Anything: Sky and object boxes</h1><ul>" + "".join(links)
        + "</ul></body></html>", encoding="utf-8", newline="\n")
    for record in records:
        for item in record["boxes"]:
            item["panels"] = [name for _, name in item["panels"]]
            item["zoom_panels"] = [name for _, name in item["zoom_panels"]]
    import torch

    mp.save_json(args.out / "summary.json", {
        "step": "PoC: Matting Anything (MAM ViT-B) for Sky and object boxes",
        "runtime": f"PyTorch {torch.__version__} CPU, {args.threads} threads",
        "checkpoint": args.checkpoint.name, "sam_weights_max_abs_difference": difference,
        "object_margin": box.OBJECT_MARGIN, "sky_margin": box.SKY_MARGIN, "zoom": ZOOM,
        "soft_range": [SOFT_LOW, SOFT_HIGH], "photos": records,
        "seconds": round(time.perf_counter() - started, 1)})
    print(f"[done] {args.out / 'index.html'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
