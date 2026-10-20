#!/usr/bin/env python3
"""End to end: the highest-scoring cells as SAM 2.1 tiny prompts (issue #1, step 6).

Input:  --photos, --onnx-dir (SAM 2.1 tiny `point_encoder.onnx` and `point_decoder.onnx`),
        --scores (score_prototypes.py output directory), --reference-masks, --reference-scores,
        and --prompts (quality prompts.json).
Output: --out-dir/end_to_end.json and, with --logit-dir, the selected 256 x 256 logits of each
        prompt (float16, for make_score_sheets.py).

For each feature source, and each photo where the class is referenced, the prompts are:

  top3_points    : the centers of the 3 highest-scoring cells as positive points
  region_box_<rule> : person only. The bounding box of the largest 8-connected region of cells
                   above a threshold. Two threshold rules, both reported:
                     above_zero : score > 0 (the cell is closer to the person prototype than to
                                  the other prototype)
                     half_max   : score >= --box-fraction (0.5) times the highest score of the
                                  photo (relative: the scores are not calibrated across photos)
  region_box_<rule>_and_top3_points : person only, the box and the 3 points together
  e1_prompt      : the E1 prompt of the reference (box or sky point), decoded with tiny. It is the
                   value that the same decoder reaches with the prompt of a user

Mask selection as in E1: one point gives the multimask output with the highest predicted IoU;
a box or several points give the single-mask selection (`select_single_mask`). The IoU is computed
at the photo size (bilinear upsampling of the logits, threshold 0) against the class reference of
make_label_grids.py (E1 base plus masks). The encoder runs once per photo on ONNX Runtime CPU
(--threads intra-op threads, CPU memory arena off).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import feature_localization as fl  # noqa: E402


def largest_region_box(selected: np.ndarray) -> tuple[list[float], int]:
    """Model-pixel box of the largest 8-connected region of the selected cells."""
    from scipy import ndimage

    regions, count = ndimage.label(selected, structure=np.ones((3, 3), int))
    sizes = ndimage.sum(selected, regions, index=np.arange(1, count + 1))
    largest = 1 + int(np.argmax(sizes))
    rows, columns = np.nonzero(regions == largest)
    size = fl.CELL_MODEL_PIXELS
    box = [float(columns.min() * size), float(rows.min() * size),
           float((columns.max() + 1) * size), float((rows.max() + 1) * size)]
    return box, int(sizes[largest - 1])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--photos", required=True)
    parser.add_argument("--onnx-dir", required=True)
    parser.add_argument("--scores", required=True)
    parser.add_argument("--reference-masks", required=True)
    parser.add_argument("--reference-scores", required=True)
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--presence", default=str(fl.HERE / "class_presence.json"))
    parser.add_argument("--box-fraction", type=float, default=0.5)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--logit-dir", default="")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()

    fl.limit_threads(args.threads)
    fl.add_sam2_scripts_to_path()
    import cv2
    from PIL import Image

    import sam2_graphs as graphs
    from compare_onnx_outputs import LOGIT_LIMIT, iou, select_single_mask

    encoder = fl.create_cpu_session(Path(args.onnx_dir) / "point_encoder.onnx", args.threads)
    decoder = fl.create_cpu_session(Path(args.onnx_dir) / "point_decoder.onnx", args.threads)
    presence = fl.read_json(Path(args.presence))["photos"]
    compare_results = fl.read_json(Path(args.reference_scores))["results"]
    prompt_sets = {Path(e["file"]).stem: graphs.prompt_sets(e)
                   for e in fl.read_json(Path(args.prompts))["images"]}
    score_maps = np.load(Path(args.scores) / "score_maps.npz")
    logit_dir = Path(args.logit_dir) if args.logit_dir else None
    if logit_dir:
        logit_dir.mkdir(parents=True, exist_ok=True)

    def decode(features, coords, labels):
        size = graphs.LOW_RES_MASK_SIZE
        started = time.perf_counter()
        logits, scores, _ = decoder.run(None, {
            "image_embed": features[0], "high_res_feats_0": features[1],
            "high_res_feats_1": features[2],
            "point_coords": np.array([coords], np.float32),
            "point_labels": np.array([labels], np.float32),
            "mask_input": np.zeros((1, 1, size, size), np.float32),
            "has_mask_input": np.zeros(1, np.float32)})
        elapsed = (time.perf_counter() - started) * 1000
        if len(labels) == 1:
            best = 1 + int(np.argmax(scores[0][1:]))
            selected, score = logits[0][best], float(scores[0][best])
        else:
            selected, score = select_single_mask(logits[0], scores[0])
        return np.clip(selected, -LOGIT_LIMIT, LOGIT_LIMIT), score, elapsed

    results = []
    decoder_ms = []
    for stem in sorted(presence):
        referenced = [n for n in fl.CLASSES if presence[stem][n]["status"] == "referenced"]
        if not referenced:
            continue
        image = Image.open(Path(args.photos) / f"{stem}.jpg").convert("RGB")
        width, height = image.size
        features = encoder.run(None, {"pixel_values": graphs.preprocess_image(image).numpy()})
        for name in referenced:
            entry = presence[stem][name]
            reference, _ = fl.class_reference_mask(stem, entry, Path(args.reference_masks),
                                                   compare_results)

            def evaluate(prompt: str, source: str, coords, labels, extra=None):
                logits, score, elapsed = decode(features, coords, labels)
                decoder_ms.append(elapsed)
                mask = cv2.resize(logits, (width, height), interpolation=cv2.INTER_LINEAR) > 0
                if logit_dir:
                    np.save(logit_dir / f"{source}__{stem}__{name}__{prompt}.npy",
                            logits.astype(np.float16))
                item = {"photo": stem, "class": name, "source": source, "prompt": prompt,
                        "coords": coords, "labels": labels, "predicted_iou": score,
                        "iou": iou(mask, reference),
                        "mask_area_fraction": float(mask.mean())}
                item.update(extra or {})
                results.append(item)
                print(f"[decode] {stem} {name} {source} {prompt}: IoU {item['iou']:.3f}")

            e1_name = next(i["prompt"] for i in entry["reference"] if not i.get("exclude"))
            e1 = next(p for p in prompt_sets[stem] if p["name"] == e1_name)
            coords, labels = graphs.model_prompt(e1)
            evaluate("e1_prompt", "e1", coords, [float(v) for v in labels],
                     {"e1_prompt_name": e1_name})

            for source in fl.FEATURE_SOURCES:
                score = score_maps[f"{source}__{stem}__{name}"]
                points = [fl.cell_center_model(r, c) for r, c in fl.top_cells(score, 3)]
                evaluate("top3_points", source, points, [1.0] * 3)
                if name != "person":
                    continue
                rules = {"above_zero": score > 0,
                         "half_max": score >= args.box_fraction * score.max()}
                for rule, selected in rules.items():
                    box, cells = largest_region_box(selected)
                    region = {"region_rule": rule, "region_cells": cells}
                    evaluate(f"region_box_{rule}", source, [box[:2], box[2:]], [2.0, 3.0],
                             region)
                    evaluate(f"region_box_{rule}_and_top3_points", source,
                             [box[:2], box[2:]] + points, [2.0, 3.0, 1.0, 1.0, 1.0], region)

    fl.write_json(Path(args.out_dir) / "end_to_end.json", {
        "step": "issue #1, step 6: end to end",
        "box_fraction": args.box_fraction,
        "decoder_cpu_ms_median": float(np.median(decoder_ms)),
        "peak_working_set_bytes": fl.peak_working_set_bytes(),
        "results": results,
    })
    print(f"[done] {len(results)} prompts, median decoder {np.median(decoder_ms):.1f} ms, "
          f"peak working set {fl.peak_working_set_bytes() / 2**20:.0f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
