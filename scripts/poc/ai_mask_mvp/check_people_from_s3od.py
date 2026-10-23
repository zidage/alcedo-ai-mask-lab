#!/usr/bin/env python3
"""Check pages of People from S3OD (AlcedoStudio SM3.D, section 7.1 steps 3 and 4).

Input:  the quality photos (`build/tmp/explore/e4/input_4096/photos`), the SAM 2.1 tiny graphs,
        the people and sky head, and the S3OD graph `subject_masks.onnx` (the model directory of
        AlcedoStudio SM3.C).
Output: under --out (default `build/tmp/explore/sm3d/`): one HTML page per photo with a non-empty
        People list and `index.html`; `summary.json` with the per-person values, the agreement
        counts of each `PERSON_IN_SUBJECT_FRACTION` candidate, and the S3OD times.

For each photo with people, the page shows:

  - the photo with the Subject alpha (S3OD on the photo) and the person boxes: green when the
    person is included by decision E8 (`subject_fraction >= PERSON_IN_SUBJECT_FRACTION`), white
    when not;
  - for each listed person, the crop rectangle of decision E3 with three overlays: the Subject
    alpha, the S3OD alpha of the person crop (decision E4, section 2.4), and the SM3.C mask (one
    SAM 2.1 box prompt) for comparison;
  - a table: the box score, the crop rectangle, `subject_fraction`, the lab rule share, the S3OD
    times.

The lab rule of decision E8 for a person: at least half of the pixels whose person crop alpha is
>= 0.5 have a Subject alpha >= 0.5 (both at the photo size, section 2.5 sampling). The selected
value is the value in steps of 0.05 that agrees with the lab rule for the most listed people of
all photos; on equal counts the smaller value. A person whose crop alpha has no value >= 0.5
(decision E9) has no lab rule value and is not counted.
"""

from __future__ import annotations

import argparse
import html
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import mask_pipeline as mp  # noqa: E402

OVERVIEW_MAX_EDGE = 1600
PERSON_PANEL_MAX_EDGE = 360
CANDIDATES = [round(0.05 * step, 2) for step in range(1, 20)]
COLORS = {"subject": (255, 40, 80), "person": (0, 200, 255), "sam": (255, 160, 0)}
OVERLAY_OPACITY = 0.5


def overlay(rgb8: np.ndarray, alpha: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:
    """`rgb8` with `color` blended by `alpha` (in [0, 1]) times OVERLAY_OPACITY."""
    weight = (np.clip(alpha, 0.0, 1.0) * OVERLAY_OPACITY)[..., None]
    return np.round(rgb8 * (1.0 - weight) + np.array(color) * weight).astype(np.uint8)


def resized(rgb8: np.ndarray, max_edge: int) -> np.ndarray:
    from PIL import Image

    image = Image.fromarray(rgb8)
    image.thumbnail((max_edge, max_edge), Image.Resampling.LANCZOS)
    return np.asarray(image)


def save_jpeg(rgb8: np.ndarray, path: Path) -> None:
    from PIL import Image

    Image.fromarray(rgb8).save(path, quality=90)


def measure_photo(pipeline: mp.MaskPipeline, path: Path) -> dict | None:
    """The People result of one photo, each person crop alpha, the SM3.C box mask, and the lab
    rule share. None when the People list is empty."""
    state = pipeline.open_photo(path)
    result = pipeline.select_people(state)
    photo = state.photo
    record = {"photo": path.stem, "size": [photo.width, photo.height],
              "presence": result.presence, "message": result.message,
              "crop_ms": round(state.people.crop_ms, 2),
              "crop_bytes": [int(p.image.nbytes) for p in state.people.people]}
    if not result.found:
        print(f"[people] {path.stem}: {result.message}", flush=True)
        return {**record, "found": False}
    s3od_events = [e for e in pipeline.events if e["stage"] == "s3od"
                   and e.get("photo") == photo.path.name]
    record["subject_s3od_ms"] = s3od_events[-1]["ms"]
    record["subject_score"] = state.subject_score
    width, height = photo.width, photo.height
    subject_at_photo, _ = mp.alpha_on_image(result.subject_alpha, [0.0, 0.0, 1.0, 1.0], width,
                                            height)
    subject_inside = subject_at_photo >= mp.ALPHA_INSIDE
    rgb8 = np.asarray(photo.image)
    people = []
    for person, fraction in zip(result.people, result.subject_fractions):
        pipeline.person_alpha(state, person.index)
        mask_box = person.mask_box(width, height)
        pasted, (rows, columns) = mp.alpha_on_image(person.alpha, mask_box, width, height)
        person_inside = pasted >= mp.ALPHA_INSIDE
        inside_count = int(person_inside.sum())
        share = (float((person_inside & subject_inside[rows, columns]).sum()) / inside_count
                 if inside_count else None)
        sam = mp.logits_to_mask(pipeline.sam_box_logits(state, person), width, height)
        x0, y0, x1, y1 = person.rect
        people.append({
            "index": person.index, "score": person.score, "box": person.box,
            "rect": list(person.rect), "crop_size": list(person.image.shape[1::-1]),
            "alpha_size": list(person.alpha.shape[::-1]), "alpha_score": person.alpha_score,
            "found": person.found, "s3od_ms": round(person.s3od_ms, 2),
            "subject_fraction": fraction, "lab_share": share,
            "panels": {"rgb8": rgb8[y0:y1, x0:x1].copy(),
                       "subject": subject_at_photo[y0:y1, x0:x1].copy(),
                       "person": pasted, "sam": sam[y0:y1, x0:x1].astype(np.float32)}})
    record.update(found=True, people=people,
                  overview=resized(overlay(rgb8, subject_at_photo, COLORS["subject"]),
                                   OVERVIEW_MAX_EDGE))
    print(f"[people] {path.stem}: {len(people)} people, Subject S3OD "
          f"{record['subject_s3od_ms']:.1f} ms, person S3OD "
          f"{', '.join(f'{p['s3od_ms']:.0f}' for p in people)} ms", flush=True)
    return record


def agreement(records: list[dict]) -> dict:
    """The number of listed people whose E8 decision equals the lab rule, per candidate."""
    people = [p for r in records if r["found"] for p in r["people"] if p["lab_share"] is not None]
    counts = {value: sum((p["subject_fraction"] >= value) == (p["lab_share"] >= 0.5)
                         for p in people) for value in CANDIDATES}
    best = max(counts.values())
    selected = min(value for value, count in counts.items() if count == best)
    return {"people": len(people), "counts": counts, "selected": selected, "agreeing": best}


def write_page(record: dict, threshold: float, out: Path) -> str:
    from PIL import Image, ImageDraw

    stem = record["photo"]
    width, height = record["size"]
    overview = Image.fromarray(record["overview"])
    scale = overview.width / width
    draw = ImageDraw.Draw(overview)
    for person in record["people"]:
        included = person["subject_fraction"] >= threshold
        x0, y0, x1, y1 = person["box"]
        color = (0, 230, 0) if included else (255, 255, 255)
        draw.rectangle([x0 * overview.width, y0 * overview.height,
                        x1 * overview.width, y1 * overview.height], outline=color, width=3)
        draw.text((x0 * overview.width + 4, y0 * overview.height + 2), str(person["index"]),
                  fill=color)
    overview.save(out / f"{stem}_overview.jpg", quality=90)
    rows = []
    for person in record["people"]:
        panels = person["panels"]
        names = []
        for kind in ("subject", "person", "sam"):
            image = Image.fromarray(resized(overlay(panels["rgb8"], panels[kind], COLORS[kind]),
                                            PERSON_PANEL_MAX_EDGE))
            draw = ImageDraw.Draw(image)
            rx0, ry0, rx1, ry1 = person["rect"]
            bx0, by0, bx1, by1 = person["box"]
            sx, sy = image.width / (rx1 - rx0), image.height / (ry1 - ry0)
            draw.rectangle([(bx0 * width - rx0) * sx, (by0 * height - ry0) * sy,
                            (bx1 * width - rx0) * sx - 1, (by1 * height - ry0) * sy - 1],
                           outline=(255, 255, 255), width=1)
            name = f"{stem}_person{person['index']}_{kind}.jpg"
            image.save(out / name, quality=90)
            names.append(name)
        share = "no alpha >= 0.5" if person["lab_share"] is None else f"{person['lab_share']:.3f}"
        lab_included = None if person["lab_share"] is None else person["lab_share"] >= 0.5
        e8_included = person["subject_fraction"] >= threshold
        rows.append(
            f"<tr><td>{person['index']}</td><td>{person['score']:.3f}</td>"
            f"<td>{person['rect']}<br>crop {person['crop_size'][0]} x {person['crop_size'][1]}, "
            f"alpha {person['alpha_size'][0]} x {person['alpha_size'][1]}</td>"
            f"<td>{person['subject_fraction']:.3f} ({'included' if e8_included else 'not'})</td>"
            f"<td>{share}"
            f"{'' if lab_included is None else (' (included)' if lab_included else ' (not)')}"
            f"{'' if lab_included is None or lab_included == e8_included else ' DIFFERENT'}</td>"
            f"<td>{person['alpha_score']:.3f}</td><td>{person['s3od_ms']:.1f}</td>"
            + "".join(f'<td><img src="{name}"></td>' for name in names) + "</tr>")
    page = (f"<!doctype html><html><head><meta charset='utf-8'><title>{stem}</title>"
            "<style>body{font-family:sans-serif;background:#222;color:#ddd}"
            "td,th{border:1px solid #555;padding:4px;vertical-align:top}"
            "table{border-collapse:collapse}img{max-width:100%}</style></head><body>"
            f"<h1>{html.escape(stem)}</h1><p><a href='index.html'>index</a></p>"
            f"<p>{width} x {height}; {html.escape(record['message'])} Subject S3OD "
            f"{record['subject_s3od_ms']:.1f} ms (score {record['subject_score']:.3f}); crops "
            f"{record['crop_ms']:.1f} ms, {sum(record['crop_bytes']) / 1e6:.2f} MB. Red: Subject "
            f"alpha. Boxes: green included (subject_fraction >= {threshold}), white not included."
            f"</p><img src='{stem}_overview.jpg'>"
            "<h2>People</h2><p>Each row: the crop rectangle (box + 10 % per side) with the Subject "
            "alpha (red), the S3OD alpha of the person crop (blue), and the SM3.C mask of one SAM "
            "2.1 box prompt (orange). The white line is the person box.</p>"
            "<table><tr><th>#</th><th>score</th><th>crop rectangle (px)</th>"
            "<th>subject_fraction</th><th>lab rule share</th><th>crop S3OD score</th>"
            "<th>crop S3OD ms</th><th>Subject alpha</th><th>person crop alpha</th>"
            "<th>SM3.C SAM 2.1 box</th></tr>" + "".join(rows) + "</table></body></html>")
    name = f"{stem}.html"
    (out / name).write_text(page, encoding="utf-8")
    return name


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--photos", type=Path,
                        default=mp.DEFAULT_EXPLORE / "e4" / "input_4096" / "photos")
    parser.add_argument("--model-dir", type=Path, required=True,
                        help="directory with point_encoder.onnx, point_decoder.onnx, "
                             "people_sky_head.onnx and .json, and subject_masks.onnx")
    parser.add_argument("--out", type=Path, default=mp.DEFAULT_EXPLORE / "sm3d")
    mp.add_provider_arguments(parser, default="directml")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()

    paths = mp.GraphPaths(sam2_dir=args.model_dir,
                          head_onnx=args.model_dir / "people_sky_head.onnx",
                          head_json=args.model_dir / "people_sky_head.json",
                          subject_onnx=args.model_dir / "subject_masks.onnx")
    started = time.perf_counter()
    pipeline = mp.MaskPipeline(paths, provider=args.provider, threads=args.threads,
                               **mp.pipeline_provider_options(args))
    args.out.mkdir(parents=True, exist_ok=True)
    records = [record for path in sorted(args.photos.glob("*.jpg"))
               if (record := measure_photo(pipeline, path)) is not None]
    summary = agreement(records)
    threshold = summary["selected"]
    print(f"[E8] {summary['agreeing']} of {summary['people']} people agree at {threshold}; "
          f"counts {summary['counts']}", flush=True)
    links = []
    for record in records:
        if record["found"]:
            name = write_page(record, threshold, args.out)
            links.append(f"<li><a href='{name}'>{record['photo']}</a>: "
                         f"{len(record['people'])} people</li>")
        else:
            links.append(f"<li>{record['photo']}: {html.escape(record['message'])}</li>")
    (args.out / "index.html").write_text(
        "<!doctype html><html><head><meta charset='utf-8'><title>SM3.D People from S3OD</title>"
        "</head><body><h1>People from S3OD</h1>"
        f"<p>PERSON_IN_SUBJECT_FRACTION {threshold}: {summary['agreeing']} of {summary['people']} "
        f"people agree with the lab rule. Counts per value: {summary['counts']}</p><ul>"
        + "".join(links) + "</ul></body></html>", encoding="utf-8")
    for record in records:
        record.pop("overview", None)
        for person in record.get("people", []):
            person.pop("panels")
    mp.save_json(args.out / "summary.json", {
        "step": "AlcedoStudio SM3.D: People from S3OD", "provider": args.provider,
        "pipeline": {k: v for k, v in pipeline.log().items() if k != "events"},
        "person_crop_margin": mp.PERSON_CROP_MARGIN, "agreement": summary, "photos": records,
        "seconds": round(time.perf_counter() - started, 1)})
    print(f"[done] {args.out / 'index.html'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
