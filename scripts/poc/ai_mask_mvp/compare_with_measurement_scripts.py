#!/usr/bin/env python3
"""Compare the AI Mask MVP pipeline with the measurement scripts on the quality photos (issue #5).

Input:  --photos (the 14 quality photos at the application input size, E4 `input_4096/photos`),
        --quality-results and --quality-logits (`evaluate_quality_photos.py` with run 2: the
        committed quality_photos.json and the saved 256 x 256 logits), --strokes-report (the E3
        `segmentation_runtime_report strokes` report of SAM 2.1 tiny), --subject-alpha (the
        `alpha/` directory of the E2 `compare_onnx_outputs.py` run of the same BiRefNet graph on
        the same photos, ONNX Runtime CPU).
Output: --out (JSON): one entry per check and photo, and the result of each check.

The MVP runs with --provider (cpu for these checks). Checks and tolerances:

  SkyPeoplePromptsMatchQualityScript
      Per photo: the sky and people presence decisions are equal (MVP `present`, script
      `found`: presence >= the run 2 threshold); the sky region points are equal
      (cell centers, absolute difference <= 1e-6 model px); every People box of decode_people
      (all ranks) is within 1e-4 cell of the script box; the sky union mask and each mask of the
      People list (where the script saved the logits) have IoU >= 0.999 with the script mask
      (logits upsampled bilinearly to the photo size, threshold 0, in both).
  NotFoundGivesMessage
      Where the script gives presence below the threshold, the MVP result is "not found", has a
      message, and has no mask (no region, no person). The cases where the presence is above the
      threshold but the People list is empty (no box score >= the list threshold) are recorded as
      `present_without_list`; they also give a message and no mask.
  StrokeDecoderInputsMatchProbeScript
      For the E3 sequences that use the MVP rule (all points with the previous mask as
      mask_input, first click with the best predicted IoU): click_then_clicks,
      click_then_strokes, stroke_then_strokes, full_stroke_then_strokes. Each step is given to
      PromptSelection.add_points. The decoder inputs are compared with the inputs that
      compare_stroke_outputs.py (`interactions`, `OnnxDecoder`) gives on its own CPU sessions:
      point_coords within 1e-4 model px, point_labels and has_mask_input equal, mask_input within
      1e-4 logit. Also: stroke_points(start, end, N) gives the recorded stroke points within 1e-6
      (normalized coordinates).
  SubjectAlphaMatchesE2Output
      The BiRefNet alpha at 1024 x 1024 has a maximum absolute difference <= 1e-3 against the E2
      ONNX Runtime CPU alpha of the same graph (stored as float16 by the E2 script).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import mask_pipeline as mp  # noqa: E402

fl = mp.fl
REQUIRED_MASK_IOU = 0.999
BOX_TOLERANCE_CELLS = 1e-4
POINT_TOLERANCE_MODEL_PX = 1e-6
STROKE_POINT_TOLERANCE = 1e-6
DECODER_COORD_TOLERANCE = 1e-4
MASK_INPUT_TOLERANCE = 1e-4
ALPHA_TOLERANCE = 1e-3
STROKE_SEQUENCES = ("click_then_clicks", "click_then_strokes", "stroke_then_strokes",
                    "full_stroke_then_strokes")


def iou(a: np.ndarray, b: np.ndarray) -> float:
    union = np.logical_or(a, b).sum()
    return 1.0 if union == 0 else float(np.logical_and(a, b).sum() / union)


def script_mask(logit_dir: Path, name: str, width: int, height: int) -> np.ndarray | None:
    path = logit_dir / f"{name}.npy"
    if not path.is_file():
        return None
    return mp.logits_to_mask(np.load(path).astype(np.float32), width, height)


def compare_sky_people(pipeline, state, expected: dict, logit_dir: Path) -> dict:
    stem = state.photo.path.stem
    width, height = state.photo.width, state.photo.height
    sky = pipeline.select_sky(state)
    people = pipeline.select_people(state)
    record = {"photo": stem, "size": [width, height]}

    # Sky.
    expected_sky = expected["sky"]
    sky_record = {"presence": sky.presence, "script_presence": expected_sky["presence"],
                  "present": sky.present, "found": sky.found,
                  "script_found": expected_sky["found"],
                  "regions": len(sky.regions), "script_regions": len(expected_sky["regions"]),
                  "message": sky.message}
    point_difference = 0.0
    points_equal = len(sky.regions) == len(expected_sky["regions"])
    if points_equal:
        for region, script_region in zip(sky.regions, expected_sky["regions"]):
            a, b = np.array(region["points"]), np.array(script_region["points"])
            if a.shape != b.shape:
                points_equal = False
                break
            point_difference = max(point_difference, float(np.abs(a - b).max()))
    sky_record["max_point_difference_model_px"] = point_difference
    sky_record["points_equal"] = points_equal and point_difference <= POINT_TOLERANCE_MODEL_PX
    if sky.regions:
        script_union = np.zeros((height, width), bool)
        missing = False
        for index in range(len(expected_sky["regions"])):
            mask = script_mask(logit_dir, f"{stem}__sky_region{index}", width, height)
            if mask is None:
                missing = True
                break
            script_union |= mask
        sky_record["mask_iou"] = None if missing else iou(sky.mask(width, height), script_union)
    sky_record["passed"] = (sky.present == expected_sky["found"] and sky_record["points_equal"]
                            and (not sky.regions or (sky_record.get("mask_iou") or 0.0)
                                 >= REQUIRED_MASK_IOU))
    record["sky"] = sky_record

    # People.
    expected_people = expected["person"]
    script_list = expected_people["list"]
    box_difference = 0.0
    boxes_equal = len(people.all_boxes) == len(script_list)
    if boxes_equal:
        for entry, script_entry in zip(people.all_boxes, script_list):
            box_difference = max(box_difference, float(np.abs(
                np.array(entry["box_cells"]) - np.array(script_entry["box_cells"])).max()))
    people_record = {"presence": people.presence, "script_presence": expected_people["presence"],
                     "present": people.present, "found": people.found,
                     "script_found": expected_people["found"],
                     "boxes": len(people.all_boxes), "script_boxes": len(script_list),
                     "max_box_difference_cells": box_difference,
                     "boxes_equal": boxes_equal and box_difference <= BOX_TOLERANCE_CELLS,
                     "listed": len(people.people), "message": people.message, "masks": []}
    for person in people.people:
        mask = script_mask(logit_dir, f"{stem}__person_rank{person['rank']}", width, height)
        if mask is None:
            people_record["masks"].append({"rank": person["rank"], "iou": None,
                                           "note": "the script decoded no box on this photo "
                                                   "(no referenced person)"})
            continue
        people_record["masks"].append({
            "rank": person["rank"], "score": person["score"],
            "iou": iou(mp.logits_to_mask(person["logits"], width, height), mask)})
    compared = [m["iou"] for m in people_record["masks"] if m["iou"] is not None]
    people_record["masks_compared"] = len(compared)
    people_record["minimum_mask_iou"] = min(compared) if compared else None
    people_record["passed"] = (people.present == expected_people["found"]
                               and people_record["boxes_equal"]
                               and all(v >= REQUIRED_MASK_IOU for v in compared))
    record["person"] = people_record

    # Not found.
    not_found = []
    for name, result, mask_items in (("sky", sky, sky.regions), ("person", people, people.people)):
        script_found = expected[name]["found"]
        if not script_found:
            not_found.append({"class": name, "found": result.found, "message": result.message,
                              "mask_items": len(mask_items),
                              "passed": (not result.found) and bool(result.message)
                              and not mask_items})
    record["not_found"] = not_found
    record["present_without_list"] = [
        {"class": name, "presence": result.presence, "message": result.message,
         "mask_items": len(items), "passed": bool(result.message) and not items}
        for name, result, items in (("sky", sky, sky.regions), ("person", people, people.people))
        if result.present and not result.found]
    print(f"[sky-people] {stem}: sky present {sky.present} found {sky.found} points "
          f"{'equal' if sky_record['points_equal'] else 'DIFFERENT'} IoU "
          f"{sky_record.get('mask_iou')} | people present {people.present} found "
          f"{people.found} ({len(people.people)} listed) boxes "
          f"{'equal' if people_record['boxes_equal'] else 'DIFFERENT'} "
          f"(max {box_difference:.2e} cell), {len(compared)} masks, min IoU "
          f"{people_record['minimum_mask_iou']}", flush=True)
    return record


class ReferenceStrokes:
    """The decoder inputs of compare_stroke_outputs.py on its own CPU sessions."""

    def __init__(self, sam2_dir: Path, threads: int) -> None:
        import compare_stroke_outputs as cso
        import sam2_graphs as graphs

        self.cso = cso
        self.graphs = graphs
        self.encoder = fl.create_cpu_session(sam2_dir / "point_encoder.onnx", threads)
        self.decoder = fl.create_cpu_session(sam2_dir / "point_decoder.onnx", threads)

    def encode(self, image) -> list[np.ndarray]:
        return self.encoder.run(None, {"pixel_values": self.graphs.preprocess_image(image).numpy()})

    def replay(self, features: list[np.ndarray], sequence: dict) -> list[dict]:
        decode = self.cso.OnnxDecoder(self.decoder, features)
        size = self.graphs.LOW_RES_MASK_SIZE
        current = None
        steps = []
        for step, points, labels, use_mask in self.cso.interactions(sequence):
            previous = current if use_mask else None
            steps.append({
                "point_coords": np.array([[[x * self.graphs.IMAGE_SIZE, y * self.graphs.IMAGE_SIZE]
                                           for x, y in points]], np.float32),
                "point_labels": np.array([[float(v) for v in labels]], np.float32),
                "mask_input": (previous[None, None].astype(np.float32) if previous is not None
                               else np.zeros((1, 1, size, size), np.float32)),
                "has_mask_input": np.array([1.0 if previous is not None else 0.0], np.float32),
            })
            current = self.cso.combine(sequence["combination"], step["label"], current,
                                       decode(points, labels, previous, len(points) == 1))
            steps[-1]["selected"] = current
        return steps


def compare_strokes(pipeline, state, image_entry: dict, reference: ReferenceStrokes) -> list[dict]:
    photo = state.photo
    features = reference.encode(photo.image)
    records = []
    for prompt in image_entry["prompts"]:
        for name in STROKE_SEQUENCES:
            sequence = prompt.get("sequences", {}).get(name)
            if not sequence or "steps" not in sequence:
                continue
            if sequence["combination"] != "points_and_previous_mask":
                raise ValueError(f"{name}: combination {sequence['combination']}")
            expected = reference.replay(features, sequence)
            selection = pipeline.new_selection(state)
            for step, reference_step in zip(sequence["steps"], expected):
                points = [list(p) for p in step["points"]]
                selection.add_points(points, step["label"] == 0, step["interaction"])
                actual = selection.last_run.inputs
                sampled_difference = None
                if step["interaction"] == "stroke":
                    stroke = step["stroke"]
                    sampled = mp.stroke_points([stroke["start"], stroke["end"]], len(points),
                                               photo.width, photo.height)
                    sampled_difference = float(np.abs(np.array(sampled) - np.array(points)).max())
                coords_difference = (float(np.abs(actual["point_coords"]
                                                  - reference_step["point_coords"]).max())
                                     if actual["point_coords"].shape
                                     == reference_step["point_coords"].shape else None)
                result = {
                    "photo": photo.path.stem, "prompt": prompt["name"], "sequence": name,
                    "step": step["step"], "interaction": step["interaction"],
                    "label": step["label"], "decoder_points": int(actual["point_coords"].shape[1]),
                    "max_coord_difference_model_px": coords_difference,
                    "labels_equal": bool(np.array_equal(actual["point_labels"],
                                                        reference_step["point_labels"])),
                    "has_mask_input": float(actual["has_mask_input"][0]),
                    "has_mask_input_equal": bool(np.array_equal(
                        actual["has_mask_input"], reference_step["has_mask_input"])),
                    "max_mask_input_difference": float(np.abs(
                        actual["mask_input"] - reference_step["mask_input"]).max()),
                    "selection_iou_256": iou(selection.logits > 0, reference_step["selected"] > 0),
                    "max_stroke_point_difference": sampled_difference,
                }
                result["passed"] = (coords_difference is not None
                                    and coords_difference <= DECODER_COORD_TOLERANCE
                                    and result["labels_equal"] and result["has_mask_input_equal"]
                                    and result["max_mask_input_difference"]
                                    <= MASK_INPUT_TOLERANCE
                                    and (sampled_difference is None
                                         or sampled_difference <= STROKE_POINT_TOLERANCE))
                records.append(result)
    failed = sum(not r["passed"] for r in records)
    print(f"[strokes] {photo.path.stem}: {len(records)} decoder calls, {failed} failed", flush=True)
    return records


def compare_subject(pipeline, state, alpha_dir: Path) -> dict:
    stem = state.photo.path.stem
    alpha = pipeline.select_subject(state)
    reference = np.load(alpha_dir / f"{stem}.npy").astype(np.float32)
    difference = np.abs(alpha - reference)
    width, height = state.photo.width, state.photo.height
    record = {"photo": stem, "max_abs_alpha_difference": float(difference.max()),
              "mean_abs_alpha_difference": float(difference.mean()),
              "photo_iou_alpha_gt_0_5": iou(
                  pipeline.birefnet.alpha_to_photo(alpha, width, height) > 0.5,
                  pipeline.birefnet.alpha_to_photo(reference, width, height) > 0.5)}
    record["passed"] = record["max_abs_alpha_difference"] <= ALPHA_TOLERANCE
    print(f"[subject] {stem}: max |dalpha| {record['max_abs_alpha_difference']:.2e}", flush=True)
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    defaults = mp.GraphPaths.default()
    parser.add_argument("--photos", required=True)
    parser.add_argument("--quality-results", required=True)
    parser.add_argument("--quality-logits", required=True)
    parser.add_argument("--strokes-report", required=True)
    parser.add_argument("--subject-alpha", required=True)
    parser.add_argument("--sam2-dir", default=str(defaults.sam2_dir))
    parser.add_argument("--head-onnx", default=str(defaults.head_onnx))
    parser.add_argument("--head-json", default=str(defaults.head_json))
    parser.add_argument("--subject-onnx", default=str(defaults.subject_onnx))
    parser.add_argument("--provider", choices=sorted(mp.PROVIDERS), default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    paths = mp.GraphPaths(Path(args.sam2_dir), Path(args.head_onnx), Path(args.head_json),
                          Path(args.subject_onnx))
    started = time.perf_counter()
    pipeline = mp.MaskPipeline(paths, provider=args.provider, threads=args.threads)
    quality = fl.read_json(Path(args.quality_results))
    strokes_report = fl.read_json(Path(args.strokes_report))
    stroke_images = {Path(entry["file"]).stem: entry for entry in strokes_report["images"]}
    reference_strokes = ReferenceStrokes(paths.sam2_dir, args.threads)

    sky_people, strokes, subject = [], [], []
    for stem in sorted(quality["photos"]):
        state = pipeline.open_photo(Path(args.photos) / f"{stem}.jpg")
        sky_people.append(compare_sky_people(pipeline, state, quality["photos"][stem],
                                             Path(args.quality_logits)))
        if stem in stroke_images:
            strokes += compare_strokes(pipeline, state, stroke_images[stem], reference_strokes)
        subject.append(compare_subject(pipeline, state, Path(args.subject_alpha)))

    not_found = [item for record in sky_people for item in record["not_found"]]
    mask_ious = ([r["sky"]["mask_iou"] for r in sky_people if r["sky"].get("mask_iou") is not None]
                 + [m["iou"] for r in sky_people for m in r["person"]["masks"]
                    if m["iou"] is not None])
    checks = {
        "SkyPeoplePromptsMatchQualityScript": {
            "rule": "found equal; sky points within 1e-6 model px; all People boxes within 1e-4 "
                    "cell; mask IoU >= 0.999 against the saved script logits",
            "photos": len(sky_people),
            "passed_photos": sum(r["sky"]["passed"] and r["person"]["passed"] for r in sky_people),
            "masks_compared": len(mask_ious),
            "minimum_mask_iou": min(mask_ious) if mask_ious else None,
            "max_box_difference_cells": max(r["person"]["max_box_difference_cells"]
                                            for r in sky_people),
            "max_presence_difference": max(
                max(abs(r["sky"]["presence"] - r["sky"]["script_presence"]),
                    abs(r["person"]["presence"] - r["person"]["script_presence"]))
                for r in sky_people),
        },
        "NotFoundGivesMessage": {
            "rule": "script presence below the threshold -> MVP not found, message, no mask",
            "cases": len(not_found), "passed_cases": sum(i["passed"] for i in not_found),
            "present_without_list": [{"photo": r["photo"], **i} for r in sky_people
                                     for i in r["present_without_list"]],
        },
        "StrokeDecoderInputsMatchProbeScript": {
            "rule": "point_coords within 1e-4 model px, labels and has_mask_input equal, "
                    "mask_input within 1e-4; stroke_points within 1e-6",
            "sequences": list(STROKE_SEQUENCES), "decoder_calls": len(strokes),
            "passed_calls": sum(r["passed"] for r in strokes),
            "with_mask_input": sum(r["has_mask_input"] > 0 for r in strokes),
            "with_subtract": sum(r["label"] == 0 for r in strokes),
            "max_mask_input_difference": max((r["max_mask_input_difference"] for r in strokes),
                                             default=None),
            "minimum_selection_iou_256": min((r["selection_iou_256"] for r in strokes),
                                             default=None),
        },
        "SubjectAlphaMatchesE2Output": {
            "rule": "max |alpha difference| <= 1e-3 at 1024 x 1024",
            "photos": len(subject), "passed_photos": sum(r["passed"] for r in subject),
            "max_abs_alpha_difference": max(r["max_abs_alpha_difference"] for r in subject),
        },
    }
    checks["SkyPeoplePromptsMatchQualityScript"]["passed"] = (
        checks["SkyPeoplePromptsMatchQualityScript"]["passed_photos"] == len(sky_people))
    checks["NotFoundGivesMessage"]["passed"] = (
        checks["NotFoundGivesMessage"]["passed_cases"] == len(not_found) > 0
        and all(i["passed"] for i in checks["NotFoundGivesMessage"]["present_without_list"]))
    checks["StrokeDecoderInputsMatchProbeScript"]["passed"] = (
        checks["StrokeDecoderInputsMatchProbeScript"]["passed_calls"] == len(strokes) > 0)
    checks["SubjectAlphaMatchesE2Output"]["passed"] = (
        checks["SubjectAlphaMatchesE2Output"]["passed_photos"] == len(subject))
    passed = all(c["passed"] for c in checks.values())
    fl.write_json(Path(args.out), {
        "step": "issue #5: AI Mask MVP compared with the measurement scripts",
        "provider": args.provider, "threads": args.threads, "passed": passed,
        "inputs": {"photos": args.photos, "quality_results": args.quality_results,
                   "quality_logits": args.quality_logits, "strokes_report": args.strokes_report,
                   "subject_alpha": args.subject_alpha},
        "pipeline": {k: v for k, v in pipeline.log().items() if k != "events"},
        "checks": checks, "seconds": round(time.perf_counter() - started, 1),
        "peak_working_set_bytes": fl.peak_working_set_bytes(),
        "machine": fl.machine_description(),
        "sky_people": sky_people, "strokes": strokes, "subject": subject,
    })
    for name, check in checks.items():
        print(f"[check] {name}: {'PASS' if check['passed'] else 'FAIL'}")
    print(f"[done] {args.out}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
