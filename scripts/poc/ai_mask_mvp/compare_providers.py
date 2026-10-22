#!/usr/bin/env python3
"""Compare the AI Mask MVP outputs of two ONNX Runtime providers on the quality photos (issue #8:
Core ML against the CPU provider on the same machine and the same graphs).

Subcommands:
  run      --provider (cpu, coreml, coreml_native, directml), --coreml-model-format,
           --coreml-compute-units, --photos (the quality photos at the application input size),
           --strokes-report (the E3 strokes report), --subject-onnx (the Subject graph; the
           accepted S3OD graph), --subject-coreml (coreml_native: the compiled Subject model;
           default mask_pipeline.GraphPaths.coreml_models), --out-dir.
           All graphs load in one mask_pipeline.MaskPipeline. Per photo: the encoder outputs, the
           head outputs, Sky (region points, presence, the selected logits per region), People
           (presence, all decode_people boxes, the People list, the selected logits per listed
           person), the E3 stroke sequences click_then_clicks, click_then_strokes,
           stroke_then_strokes, and full_stroke_then_strokes given to PromptSelection.add_points
           (the selected logits after each step; each provider uses its own previous mask as
           mask_input, as in the UI), and the Subject alpha. Output: <out-dir>/<stem>.npz (the
           arrays), <out-dir>/<stem>.json (the decisions and prompts), <out-dir>/run.json (the
           pipeline record and the stage times).
  compare  --reference (the `run` directory of the CPU), --candidate (the `run` directory of the
           other provider), --photos, --out (JSON).

Checks of `compare` (the acceptance criteria of issue #8):
  PresenceDecisionsEqual  per photo: sky present and found, people present and found are equal
  PeopleListEqual         per photo: the listed ranks are equal (the number and the order); the
                          largest box difference (cells) and score difference are recorded
  SkyRegionsEqual         per photo: the number of sky regions is equal; the largest point
                          difference (model px) is recorded
  MaskIoU                 the sky union, each listed person mask, and the selection after each
                          stroke step have IoU >= 0.999 with the reference (logits upsampled
                          bilinearly to the photo size, threshold 0: mask_pipeline.logits_to_mask)
  SubjectAlphaIoU         the Subject alpha > 0.5 has IoU >= 0.998 with the reference on the model
                          grid (S3OD: the unpadded grid, long edge 1024 px); the largest absolute
                          alpha difference is recorded
  OutputsFinite           no NaN or infinite value in any candidate output
Recorded without a tolerance: per encoder output the largest absolute difference, the relative
error ||candidate - reference|| / ||reference||, and the cosine similarity; per head output the
largest absolute difference.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import mask_pipeline as mp  # noqa: E402

fl = mp.fl
STROKE_SEQUENCES = ("click_then_clicks", "click_then_strokes", "stroke_then_strokes",
                    "full_stroke_then_strokes")
FEATURE_NAMES = ("image_embed", "high_res_feats_0", "high_res_feats_1")
REQUIRED_MASK_IOU = 0.999
REQUIRED_SUBJECT_IOU = 0.998


def iou(a: np.ndarray, b: np.ndarray) -> float:
    union = np.logical_or(a, b).sum()
    return 1.0 if union == 0 else float(np.logical_and(a, b).sum() / union)


def run(args) -> int:
    paths = mp.GraphPaths.default(Path(args.explore))
    paths.subject_onnx = Path(args.subject_onnx)
    if args.subject_coreml:
        paths.coreml_models["subject_matte"] = Path(args.subject_coreml)
    started = time.perf_counter()
    pipeline = mp.MaskPipeline(paths, provider=args.provider, threads=args.threads,
                               **mp.pipeline_provider_options(args))
    strokes = fl.read_json(Path(args.strokes_report))
    stroke_images = {Path(entry["file"]).stem: entry for entry in strokes["images"]}
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for path in sorted(Path(args.photos).glob("*.jpg")):
        stem = path.stem
        state = pipeline.open_photo(path)
        arrays = dict(zip(FEATURE_NAMES, state.features))
        arrays.update({f"head__{name}": value for name, value in state.head_outputs.items()})
        sky = pipeline.select_sky(state)
        people = pipeline.select_people(state)
        record = {"photo": stem, "size": [state.photo.width, state.photo.height],
                  "sky": {"present": sky.present, "found": sky.found, "presence": sky.presence,
                          "regions": [{"cells": r["cells"], "points": r["points"]}
                                      for r in sky.regions]},
                  "people": {"present": people.present, "found": people.found,
                             "presence": people.presence, "all_boxes": people.all_boxes,
                             "listed_ranks": [p["rank"] for p in people.people]},
                  "strokes": []}
        for index, region in enumerate(sky.regions):
            arrays[f"sky_region{index}"] = region["logits"]
        for person in people.people:
            arrays[f"person_rank{person['rank']}"] = person["logits"]
        for prompt in stroke_images.get(stem, {}).get("prompts", []):
            for name in STROKE_SEQUENCES:
                sequence = prompt.get("sequences", {}).get(name)
                if not sequence or "steps" not in sequence:
                    continue
                selection = pipeline.new_selection(state)
                for step in sequence["steps"]:
                    selection.add_points([list(p) for p in step["points"]], step["label"] == 0,
                                         step["interaction"])
                    key = f"stroke__{prompt['name']}__{name}__{step['step']}"
                    arrays[key] = selection.logits
                    record["strokes"].append({"key": key, "interaction": step["interaction"],
                                              "label": step["label"]})
        arrays["subject_alpha"] = pipeline.select_subject(state)
        np.savez(out_dir / f"{stem}.npz", **arrays)
        fl.write_json(out_dir / f"{stem}.json", record)
        print(f"[run] {stem}: sky {sky.found} ({len(sky.regions)}), people {people.found} "
              f"({len(people.people)}), {len(record['strokes'])} stroke steps", flush=True)
    fl.write_json(out_dir / "run.json", {
        "step": "issue #8: MVP outputs for the provider comparison", "provider": args.provider,
        "pipeline": pipeline.log(), "seconds": round(time.perf_counter() - started, 1),
        "machine": fl.machine_description()})
    return 0


def feature_errors(candidate: np.ndarray, reference: np.ndarray) -> dict:
    a, b = candidate.astype(np.float64).ravel(), reference.astype(np.float64).ravel()
    difference = a - b
    return {"max_abs_difference": float(np.abs(difference).max()),
            "relative_error": float(np.linalg.norm(difference) / max(np.linalg.norm(b), 1e-30)),
            "cosine": float(a @ b / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-30))}


def compare(args) -> int:
    reference_dir, candidate_dir = Path(args.reference), Path(args.candidate)
    reference_run = fl.read_json(reference_dir / "run.json")
    candidate_run = fl.read_json(candidate_dir / "run.json")
    photos = []
    for path in sorted(Path(args.photos).glob("*.jpg")):
        stem = path.stem
        ref_json, cand_json = (fl.read_json(d / f"{stem}.json")
                               for d in (reference_dir, candidate_dir))
        ref, cand = (np.load(d / f"{stem}.npz") for d in (reference_dir, candidate_dir))
        width, height = ref_json["size"]
        non_finite = {key: int(np.count_nonzero(~np.isfinite(cand[key]))) for key in cand.files}
        record = {"photo": stem,
                  "encoder": {name: feature_errors(cand[name], ref[name])
                              for name in FEATURE_NAMES},
                  "head_max_abs_difference": {
                      key[len("head__"):]: float(np.abs(cand[key] - ref[key]).max())
                      for key in ref.files if key.startswith("head__")},
                  "non_finite_values": sum(non_finite.values()),
                  "non_finite_outputs": [k for k, v in non_finite.items() if v]}
        decisions = {f"{name}_{field}": (cand_json[name][field], ref_json[name][field])
                     for name in ("sky", "people") for field in ("present", "found")}
        record["decisions"] = {k: {"candidate": c, "reference": r} for k, (c, r) in
                               decisions.items()}
        record["decisions_equal"] = all(c == r for c, r in decisions.values())
        record["presence_difference"] = {
            name: abs(cand_json[name]["presence"] - ref_json[name]["presence"])
            for name in ("sky", "people")}

        ref_ranks, cand_ranks = (j["people"]["listed_ranks"] for j in (ref_json, cand_json))
        ref_boxes, cand_boxes = (j["people"]["all_boxes"] for j in (ref_json, cand_json))
        compared_boxes = list(zip(cand_boxes, ref_boxes))
        record["people_list"] = {
            "candidate_ranks": cand_ranks, "reference_ranks": ref_ranks,
            "equal": cand_ranks == ref_ranks,
            "boxes": [len(cand_boxes), len(ref_boxes)],
            "max_box_difference_cells": max((float(np.abs(np.array(c["box_cells"])
                                                          - np.array(r["box_cells"])).max())
                                             for c, r in compared_boxes), default=0.0),
            "max_score_difference": max((abs(c["score"] - r["score"])
                                         for c, r in compared_boxes), default=0.0)}
        ref_regions, cand_regions = (j["sky"]["regions"] for j in (ref_json, cand_json))
        point_difference = None
        if len(ref_regions) == len(cand_regions):
            differences = [float(np.abs(np.array(c["points"]) - np.array(r["points"])).max())
                           for c, r in zip(cand_regions, ref_regions)
                           if np.shape(c["points"]) == np.shape(r["points"])]
            point_difference = max(differences, default=0.0)
        record["sky_regions"] = {"candidate": len(cand_regions), "reference": len(ref_regions),
                                 "equal": len(cand_regions) == len(ref_regions),
                                 "max_point_difference_model_px": point_difference}

        masks = []

        def add_mask(kind: str, name: str, candidate: np.ndarray | None,
                     reference: np.ndarray | None) -> None:
            if candidate is None or reference is None:
                masks.append({"kind": kind, "name": name, "iou": None,
                              "note": "missing in one provider"})
                return
            masks.append({"kind": kind, "name": name, "iou": iou(candidate, reference)})

        def union(arrays, json_record) -> np.ndarray:
            total = np.zeros((height, width), bool)
            for index in range(len(json_record["sky"]["regions"])):
                total |= mp.logits_to_mask(arrays[f"sky_region{index}"], width, height)
            return total

        if ref_regions or cand_regions:
            add_mask("sky", "union", union(cand, cand_json), union(ref, ref_json))
        for rank in sorted(set(ref_ranks) | set(cand_ranks)):
            key = f"person_rank{rank}"
            add_mask("person", key,
                     mp.logits_to_mask(cand[key], width, height) if key in cand.files else None,
                     mp.logits_to_mask(ref[key], width, height) if key in ref.files else None)
        for step in ref_json["strokes"]:
            key = step["key"]
            add_mask("stroke", key,
                     mp.logits_to_mask(cand[key], width, height) if key in cand.files else None,
                     mp.logits_to_mask(ref[key], width, height))
        record["masks"] = masks
        record["minimum_mask_iou"] = min((m["iou"] for m in masks if m["iou"] is not None),
                                         default=None)
        record["masks_passed"] = all(m["iou"] is not None and m["iou"] >= REQUIRED_MASK_IOU
                                     for m in masks)

        ref_alpha = ref["subject_alpha"].astype(np.float32)
        cand_alpha = cand["subject_alpha"].astype(np.float32)
        record["subject"] = {
            "grid": list(ref_alpha.shape),
            "iou_alpha_gt_0_5": (iou(cand_alpha > 0.5, ref_alpha > 0.5)
                                 if cand_alpha.shape == ref_alpha.shape else None),
            "max_abs_alpha_difference": (float(np.abs(cand_alpha - ref_alpha).max())
                                         if cand_alpha.shape == ref_alpha.shape else None)}
        record["subject"]["passed"] = ((record["subject"]["iou_alpha_gt_0_5"] or 0.0)
                                       >= REQUIRED_SUBJECT_IOU)
        photos.append(record)
        print(f"[compare] {stem}: decisions {'equal' if record['decisions_equal'] else 'DIFFERENT'}"
              f", people list {'equal' if record['people_list']['equal'] else 'DIFFERENT'}, "
              f"{len(masks)} masks min IoU {record['minimum_mask_iou']}, subject IoU "
              f"{record['subject']['iou_alpha_gt_0_5']}, non-finite "
              f"{record['non_finite_values']}", flush=True)

    mask_ious = [m["iou"] for p in photos for m in p["masks"] if m["iou"] is not None]
    checks = {
        "PresenceDecisionsEqual": {"rule": "sky and people present and found equal",
                                   "passed_photos": sum(p["decisions_equal"] for p in photos)},
        "PeopleListEqual": {"rule": "listed ranks equal (number and order)",
                            "passed_photos": sum(p["people_list"]["equal"] for p in photos)},
        "SkyRegionsEqual": {"rule": "number of sky regions equal",
                            "passed_photos": sum(p["sky_regions"]["equal"] for p in photos)},
        "MaskIoU": {"rule": f"IoU >= {REQUIRED_MASK_IOU} at the photo size",
                    "masks": len(mask_ious),
                    "masks_missing": sum(m["iou"] is None for p in photos for m in p["masks"]),
                    "passed_masks": sum(v >= REQUIRED_MASK_IOU for v in mask_ious),
                    "minimum_iou": min(mask_ious, default=None),
                    "passed_photos": sum(p["masks_passed"] for p in photos)},
        "SubjectAlphaIoU": {"rule": f"IoU of alpha > 0.5 >= {REQUIRED_SUBJECT_IOU} on the "
                                    f"model grid",
                            "passed_photos": sum(p["subject"]["passed"] for p in photos),
                            "minimum_iou": min((p["subject"]["iou_alpha_gt_0_5"] or 0.0)
                                               for p in photos)},
        "OutputsFinite": {"rule": "no NaN or infinite candidate output",
                          "passed_photos": sum(p["non_finite_values"] == 0 for p in photos)},
    }
    for check in checks.values():
        check["photos"] = len(photos)
        check["passed"] = check["passed_photos"] == len(photos)
    encoder = {name: {"max_abs_difference": max(p["encoder"][name]["max_abs_difference"]
                                                for p in photos),
                      "max_relative_error": max(p["encoder"][name]["relative_error"]
                                                for p in photos),
                      "minimum_cosine": min(p["encoder"][name]["cosine"] for p in photos)}
               for name in FEATURE_NAMES}
    passed = all(c["passed"] for c in checks.values())
    fl.write_json(Path(args.out), {
        "step": "issue #8: MVP outputs of two providers compared",
        "reference": {"dir": mp.repository_path(reference_dir),
                      "provider": reference_run["provider"],
                      "provider_settings": reference_run["pipeline"].get("provider_settings"),
                      "graphs": reference_run["pipeline"]["graphs"]},
        "candidate": {"dir": mp.repository_path(candidate_dir),
                      "provider": candidate_run["provider"],
                      "provider_settings": candidate_run["pipeline"].get("provider_settings"),
                      "graphs": candidate_run["pipeline"]["graphs"]},
        "onnxruntime": candidate_run["pipeline"]["onnxruntime"], "passed": passed,
        "checks": checks, "encoder_summary": encoder, "photos": photos,
        "machine": candidate_run["machine"]})
    for name, check in checks.items():
        print(f"[check] {name}: {'PASS' if check['passed'] else 'FAIL'}")
    return 0 if passed else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("run", "compare"))
    parser.add_argument("--photos", required=True)
    parser.add_argument("--explore", default=str(mp.DEFAULT_EXPLORE))
    mp.add_provider_arguments(parser)
    parser.add_argument("--subject-coreml", default="")
    parser.add_argument("--strokes-report")
    parser.add_argument("--subject-onnx")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--out-dir")
    parser.add_argument("--reference")
    parser.add_argument("--candidate")
    parser.add_argument("--out")
    args = parser.parse_args()
    if args.command == "run":
        if not (args.strokes_report and args.subject_onnx and args.out_dir):
            parser.error("run needs --strokes-report, --subject-onnx, and --out-dir")
        return run(args)
    if not (args.reference and args.candidate and args.out):
        parser.error("compare needs --reference, --candidate, and --out")
    return compare(args)


if __name__ == "__main__":
    raise SystemExit(main())
