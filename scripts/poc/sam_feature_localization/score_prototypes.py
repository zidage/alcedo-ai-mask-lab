#!/usr/bin/env python3
"""Prototype similarity without training, and its measurements (issue #1, steps 4 and 5).

Input:  --features (extract_cell_features.py output) and --labels (make_label_grids.py output).
Output: --out-dir/score_maps.npz  score map per feature source, photo, and class [64, 64]
        --out-dir/metrics.json    all measurements
        --out-dir/summary.md      the tables of the report

Prototypes (leave one photo out). For each photo, the person, sky, and other prototypes are the
means of the unit-length cell vectors with that label on the other 13 photos (a mean over cells,
not over photos). The score of a cell for a class is

    score = cosine(cell, prototype(class)) - cosine(cell, prototype(other))

Measurements per feature source, class, and photo:

  area under ROC : score map against the class cells (positive) and the cells with at most 10
                   percent of the class (negative, outside the ignore boxes of the class); only on
                   photos where the class is referenced
  peak precision : share of the 3 and the 10 highest-scoring cells of the whole photo that are
                   class cells ("label": at least 90 percent of the cell is the class) or whose
                   center pixel is inside the reference ("center")
  peak scores    : the highest and the third-highest cell score of the photo

Separation per feature source and class: margin = the lowest third-highest score on the photos
where the class is referenced, minus the highest score on the photos without the class. A
positive margin means that one threshold (the midpoint) finds the class on every referenced photo
by its three best cells and on no photo without the class. Photos with unreferenced or uncertain
presence are reported against that threshold but do not set it.

Acceptance criteria of the issue (proposal), checked per feature source and class:

  peak : precision at 3 is 1 on every referenced photo. The "label" rule applies; a photo with
         fewer than 3 class cells (the small boarder of 2088064441) uses the "center" rule
  separation : margin > 0
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import feature_localization as fl  # noqa: E402

PEAK_COUNTS = (3, 10)


def prototypes(vectors: dict, labels: dict, held_out: str) -> dict:
    """Mean unit cell vector per label over all photos except `held_out`."""
    result = {}
    for name, value in (("person", fl.LABEL_PERSON), ("sky", fl.LABEL_SKY),
                        ("other", fl.LABEL_OTHER)):
        cells = [vectors[stem][labels[stem].ravel() == value]
                 for stem in vectors if stem != held_out]
        stacked = np.concatenate(cells, axis=0)
        if len(stacked) == 0:
            raise ValueError(f"no {name} cells outside {held_out}")
        result[name] = (stacked.mean(axis=0), len(stacked))
    return result


def cosine(cells: np.ndarray, prototype: np.ndarray) -> np.ndarray:
    return cells @ prototype / np.linalg.norm(prototype)


def peak_precision(score: np.ndarray, inside: np.ndarray, count: int) -> float:
    cells = fl.top_cells(score, count)
    return float(np.mean([inside[r, c] for r, c in cells]))


def photo_measurements(score: np.ndarray, status: str, data, stem: str, name: str) -> dict:
    ordered = np.sort(score.ravel())[::-1]
    result = {"status": status, "max_score": float(ordered[0]),
              "third_score": float(ordered[2]),
              "top_cells": [list(rc) for rc in fl.top_cells(score, 10)]}
    ignored = data[f"{stem}__ignored"]
    result["top3_in_ignore_boxes"] = int(sum(ignored[r, c] for r, c in fl.top_cells(score, 3)))
    if status != "referenced":
        return result
    positive = data[f"{stem}__{name}_positive"]
    negative = data[f"{stem}__{name}_negative"]
    center = data[f"{stem}__{name}_center"]
    evaluated = positive | negative
    result["positive_cells"] = int(positive.sum())
    result["negative_cells"] = int(negative.sum())
    result["auc"] = fl.area_under_roc(score[evaluated], positive[evaluated])
    for count in PEAK_COUNTS:
        result[f"precision_at_{count}_label"] = peak_precision(score, positive, count)
        result[f"precision_at_{count}_center"] = peak_precision(score, center, count)
    rule = "label" if positive.sum() >= 3 else "center"
    result["acceptance_rule"] = rule
    result["acceptance_precision_at_3"] = result[f"precision_at_3_{rule}"]
    return result


def separation(per_photo: dict) -> dict:
    present = {s: m["third_score"] for s, m in per_photo.items() if m["status"] == "referenced"}
    absent = {s: m["max_score"] for s, m in per_photo.items() if m["status"] == "absent"}
    lowest_present = min(present, key=present.get)
    highest_absent = max(absent, key=absent.get)
    margin = present[lowest_present] - absent[highest_absent]
    threshold = (present[lowest_present] + absent[highest_absent]) / 2
    found = {s: {"status": m["status"], "third_score": m["third_score"],
                 "found_at_threshold": bool(m["third_score"] >= threshold)}
             for s, m in per_photo.items()
             if m["status"] in ("present_unreferenced", "uncertain")}
    return {"lowest_present_third_score": present[lowest_present],
            "lowest_present_photo": lowest_present,
            "highest_absent_max_score": absent[highest_absent],
            "highest_absent_photo": highest_absent,
            "margin": margin, "threshold": threshold if margin > 0 else None,
            "present_photos": len(present), "absent_photos": len(absent),
            "unreferenced_or_uncertain": found}


def fmt(value, digits=3) -> str:
    if value is None:
        return "-"
    return f"{value:.{digits}f}"


def summary_tables(metrics: dict) -> str:
    lines = ["# Proof of concept: SAM 2.1 tiny features for people and sky", "",
             "Generated by `score_prototypes.py`. Scores are "
             "`cosine(class prototype) - cosine(other prototype)`, leave one photo out.", ""]
    sources = list(metrics["sources"])
    lines += ["## Acceptance criteria per feature source", "",
              "| Feature source | Class | Precision at 3 = 1 on every referenced photo | "
              "Separation margin | Usable |", "| --- | --- | --- | ---: | --- |"]
    for source in sources:
        for name in fl.CLASSES:
            item = metrics["sources"][source][name]
            acceptance = item["acceptance"]
            lines.append(f"| {source} | {name} | {acceptance['peak_passed_photos']} / "
                         f"{acceptance['peak_photos']} | {fmt(item['separation']['margin'])} | "
                         f"{'yes' if acceptance['usable'] else 'no'} |")
    for name in fl.CLASSES:
        lines += ["", f"## {name}: referenced photos", "",
                  "AUC / precision at 3 (label, center) / precision at 10 (label, center) / "
                  "third-highest score.", ""]
        header = "| Photo | Class cells | " + " | ".join(sources) + " |"
        lines += [header, "| --- | ---: | " + " | ".join("---" for _ in sources) + " |"]
        stems = [s for s, m in metrics["sources"][sources[0]][name]["photos"].items()
                 if m["status"] == "referenced"]
        for stem in stems:
            cells = metrics["sources"][sources[0]][name]["photos"][stem]["positive_cells"]
            row = [f"`{stem}`", str(cells)]
            for source in sources:
                m = metrics["sources"][source][name]["photos"][stem]
                row.append(f"{fmt(m['auc'])} / {fmt(m['precision_at_3_label'], 2)}, "
                           f"{fmt(m['precision_at_3_center'], 2)} / "
                           f"{fmt(m['precision_at_10_label'], 1)}, "
                           f"{fmt(m['precision_at_10_center'], 1)} / {fmt(m['third_score'])}")
            lines.append("| " + " | ".join(row) + " |")
        lines += ["", f"## {name}: highest score per photo", "",
                  "Maximum / third-highest score. The status comes from `class_presence.json`.",
                  "", "| Photo | Status | " + " | ".join(sources) + " |",
                  "| --- | --- | " + " | ".join("---" for _ in sources) + " |"]
        for stem, m0 in metrics["sources"][sources[0]][name]["photos"].items():
            row = [f"`{stem}`", m0["status"]]
            for source in sources:
                m = metrics["sources"][source][name]["photos"][stem]
                row.append(f"{fmt(m['max_score'])} / {fmt(m['third_score'])}")
            lines.append("| " + " | ".join(row) + " |")
        lines += ["", f"## {name}: separation", "",
                  "| Feature source | Lowest third-highest score, referenced | "
                  "Highest score, absent | Margin | Threshold | Unreferenced or uncertain "
                  "photos found at the threshold |",
                  "| --- | --- | --- | ---: | ---: | --- |"]
        for source in sources:
            sep = metrics["sources"][source][name]["separation"]
            found = sep["unreferenced_or_uncertain"]
            found_text = (", ".join(f"`{s}` {'yes' if f['found_at_threshold'] else 'no'}"
                                    for s, f in found.items())
                          if sep["threshold"] is not None else "no threshold")
            lines.append(f"| {source} | {fmt(sep['lowest_present_third_score'])} "
                         f"(`{sep['lowest_present_photo']}`) | "
                         f"{fmt(sep['highest_absent_max_score'])} "
                         f"(`{sep['highest_absent_photo']}`) | {fmt(sep['margin'])} | "
                         f"{fmt(sep['threshold'])} | {found_text} |")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--features", required=True)
    parser.add_argument("--labels", required=True, help="directory with labels.npz/json")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()

    fl.limit_threads(args.threads)
    labels_dir = Path(args.labels)
    label_info = fl.read_json(labels_dir / "labels.json")["photos"]
    data = dict(np.load(labels_dir / "labels.npz"))
    stems = sorted(label_info)
    labels = {stem: data[f"{stem}__label"] for stem in stems}
    score_maps = {}
    metrics = {"step": "issue #1, steps 4 and 5", "sources": {}}
    for source in fl.FEATURE_SOURCES:
        vectors = {stem: fl.cell_vectors(Path(args.features) / f"{stem}.npz", source)
                   for stem in stems}
        per_class = {name: {} for name in fl.CLASSES}
        prototype_cells = {}
        scoring_ms = []
        for stem in stems:
            protos = prototypes(vectors, labels, stem)
            prototype_cells[stem] = {n: c for n, (_, c) in protos.items()}
            started = time.perf_counter()
            other = cosine(vectors[stem], protos["other"][0])
            scores = {name: (cosine(vectors[stem], protos[name][0]) - other).reshape(
                fl.GRID, fl.GRID) for name in fl.CLASSES}
            scoring_ms.append((time.perf_counter() - started) * 1000)
            for name in fl.CLASSES:
                score_maps[f"{source}__{stem}__{name}"] = scores[name].astype(np.float32)
                per_class[name][stem] = photo_measurements(
                    scores[name], label_info[stem][name]["status"], data, stem, name)
        source_metrics = {"title": fl.FEATURE_SOURCE_TITLES[source],
                          "channels": int(vectors[stems[0]].shape[1]),
                          "prototype_cells": prototype_cells,
                          "scoring_ms_median": float(np.median(scoring_ms))}
        for name in fl.CLASSES:
            photos = per_class[name]
            referenced = {s: m for s, m in photos.items() if m["status"] == "referenced"}
            peak_passed = [s for s, m in referenced.items()
                           if m["acceptance_precision_at_3"] == 1.0]
            sep = separation(photos)
            aucs = [m["auc"] for m in referenced.values() if m["auc"] is not None]
            source_metrics[name] = {
                "photos": photos,
                "separation": sep,
                "auc_median": float(np.median(aucs)) if aucs else None,
                "auc_minimum": float(np.min(aucs)) if aucs else None,
                "acceptance": {
                    "peak_photos": len(referenced),
                    "peak_passed_photos": len(peak_passed),
                    "peak_failed": sorted(set(referenced) - set(peak_passed)),
                    "separation_passed": sep["margin"] > 0,
                    "usable": len(peak_passed) == len(referenced) and sep["margin"] > 0,
                },
            }
            acceptance = source_metrics[name]["acceptance"]
            print(f"[{source}] {name}: AUC median {fmt(source_metrics[name]['auc_median'])}, "
                  f"precision at 3 = 1 on {acceptance['peak_passed_photos']}/"
                  f"{acceptance['peak_photos']}, margin {sep['margin']:.3f}, "
                  f"usable {acceptance['usable']}")
        metrics["sources"][source] = source_metrics

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / "score_maps.npz", **score_maps)
    metrics["peak_working_set_bytes"] = fl.peak_working_set_bytes()
    metrics["machine"] = fl.machine_description()
    fl.write_json(out_dir / "metrics.json", metrics)
    with open(out_dir / "summary.md", "w", encoding="utf-8", newline="\n") as stream:
        stream.write(summary_tables(metrics))
    print(f"[done] {out_dir / 'metrics.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
