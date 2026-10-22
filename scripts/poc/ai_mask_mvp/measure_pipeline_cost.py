#!/usr/bin/env python3
"""Stage times and memory of the AI Mask MVP pipeline with all models loaded together (issue #5).

Input:  --photos (the quality photos at the application input size), --strokes-report (the E3
        strokes report: its `full_stroke_then_strokes` sequences are replayed as the stroke, add,
        and subtract interactions), the graphs of mask_pipeline.GraphPaths.default (or
        --subject-onnx), --provider (directml for the cost measurement).
Output: --out (JSON): graph load times, per stage the time of the first call, the median and the
        maximum over the photos, the warm p50 / p95 / maximum on the first photo, the peak working
        set and the private bytes. Run it under scripts/ai_models/sam2/run_with_gpu_memory.ps1
        for the dedicated and shared GPU memory of the process.

Sequence per photo (all graphs stay loaded): read and scale the photo, encoder, head, Sky,
People (one decoder run per listed person), the stroke sequence of E3 (one decoder run per
interaction, the previous mask as mask_input), Subject (BiRefNet, or S3OD with --subject-onnx).
Then --warm-runs repeated calls of each graph on the first photo: encoder, head, decoder with one
box, decoder with 10 points and mask_input, Subject. The Subject stage has the name of the graph
kind (`birefnet` or `s3od`, mask_pipeline.SUBJECT_OUTPUTS).
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import mask_pipeline as mp  # noqa: E402

fl = mp.fl


def spread(values: list[float]) -> dict:
    if not values:
        return {}
    ordered = sorted(values)
    return {"count": len(values), "p50_ms": round(statistics.median(ordered), 2),
            "p95_ms": round(ordered[min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1))))],
                            2),
            "max_ms": round(ordered[-1], 2)}


def memory() -> dict:
    import psutil

    info = psutil.Process().memory_info()
    return {"peak_working_set_bytes": int(getattr(info, "peak_wset", info.rss)),
            "working_set_bytes": int(info.rss),
            "private_bytes": int(getattr(info, "private", getattr(info, "vms", 0)))}


def replay_strokes(pipeline, state, image_entry: dict | None) -> int:
    if image_entry is None:
        return 0
    calls = 0
    for prompt in image_entry["prompts"]:
        sequence = prompt.get("sequences", {}).get("full_stroke_then_strokes", {})
        if "steps" not in sequence:
            continue
        selection = pipeline.new_selection(state)
        for step in sequence["steps"]:
            selection.add_points([list(p) for p in step["points"]], step["label"] == 0,
                                 step["interaction"])
            calls += 1
    return calls


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    defaults = mp.GraphPaths.default()
    parser.add_argument("--photos", required=True)
    parser.add_argument("--strokes-report", required=True)
    parser.add_argument("--subject-onnx", default=str(defaults.subject_onnx))
    parser.add_argument("--provider", choices=sorted(mp.PROVIDERS), default="directml")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--warm-runs", type=int, default=20)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    paths = mp.GraphPaths.default()
    paths.subject_onnx = Path(args.subject_onnx)
    memory_before = memory()
    pipeline = mp.MaskPipeline(paths, provider=args.provider, threads=args.threads)
    memory_after_load = memory()
    strokes = fl.read_json(Path(args.strokes_report))
    stroke_images = {Path(entry["file"]).stem: entry for entry in strokes["images"]}

    photos = sorted(Path(args.photos).glob("*.jpg"))
    per_photo = []
    first_state = None
    for path in photos:
        start_event = len(pipeline.events)
        started = time.perf_counter()
        state = pipeline.open_photo(path)
        sky = pipeline.select_sky(state)
        people = pipeline.select_people(state)
        stroke_calls = replay_strokes(pipeline, state, stroke_images.get(path.stem))
        pipeline.select_subject(state)
        wall_ms = (time.perf_counter() - started) * 1000
        events = pipeline.events[start_event:]
        record = {"photo": path.stem, "size": [state.photo.width, state.photo.height],
                  "sky_found": sky.found, "sky_regions": len(sky.regions),
                  "people_found": people.found, "people_listed": len(people.people),
                  "stroke_calls": stroke_calls, "wall_ms": round(wall_ms, 1),
                  "stage_ms": {}}
        for event in events:
            record["stage_ms"].setdefault(event["stage"], []).append(event["ms"])
        per_photo.append(record)
        print(f"[photo] {path.stem}: " + ", ".join(
            f"{k} {sum(v):.0f} ms ({len(v)})" for k, v in record["stage_ms"].items()), flush=True)
        if first_state is None:
            first_state = state
        else:
            state.subject_alpha = None

    # Warm runs on the first photo.
    state = first_state
    warm_start = len(pipeline.events)
    box = [[256.0, 256.0], [768.0, 768.0]]
    points = [[96.0 * (k + 1), 512.0] for k in range(10)]
    previous = pipeline.single_mask(pipeline.decode(state, box, [2.0, 3.0], None, "warm box"))
    for _ in range(args.warm_runs):
        pipeline.encode(state.photo)
        pipeline.decode(state, box, [2.0, 3.0], None, "warm box")
        pipeline.decode(state, points, [1.0] * 9 + [0.0], previous, "warm 10 points mask_input")
        state.subject_alpha = None
        pipeline.select_subject(state)
    warm_events = pipeline.events[warm_start + 1:]
    memory_at_end = memory()

    def stage_values(records, stage):
        return [v for r in records for v in r["stage_ms"].get(stage, [])]

    first = per_photo[0]["stage_ms"]
    stages = {}
    subject = pipeline.subject_kind
    for stage in ("read_photo", "encoder", "head", "decoder", subject):
        values = stage_values(per_photo, stage)
        other = stage_values(per_photo[1:], stage)
        stages[stage] = {"first_photo_first_call_ms": first[stage][0] if stage in first else None,
                         "all_calls": spread(values), "photos_after_the_first": spread(other)}
    warm = {
        "encoder": spread([e["ms"] for e in warm_events if e["stage"] == "encoder"]),
        "head": spread([e["ms"] for e in warm_events if e["stage"] == "head"]),
        "decoder_box": spread([e["ms"] for e in warm_events
                               if e["stage"] == "decoder" and e["purpose"] == "warm box"]),
        "decoder_10_points_mask_input": spread([e["ms"] for e in warm_events
                                                if e["stage"] == "decoder"
                                                and e["purpose"] != "warm box"]),
        subject: spread([e["ms"] for e in warm_events if e["stage"] == subject]),
    }
    decoder_by_purpose = {}
    sky_ms, people_ms = [], []
    for event in pipeline.events[:warm_start]:
        purpose = event.get("purpose", "")
        if event["stage"] == "decoder" and purpose.startswith("sky"):
            sky_ms.append(event["ms"])
        elif event["stage"] == "decoder" and purpose.startswith("person"):
            people_ms.append(event["ms"])
        elif event["stage"] == "decoder":
            decoder_by_purpose.setdefault(purpose, []).append(event["ms"])
    fl.write_json(Path(args.out), {
        "step": "issue #5: AI Mask MVP cost with all models loaded together",
        "provider": args.provider, "threads": args.threads, "warm_runs": args.warm_runs,
        "pipeline": {k: v for k, v in pipeline.log().items() if k != "events"},
        "memory_before_load": memory_before, "memory_after_load": memory_after_load,
        "memory_at_end": memory_at_end,
        "stages": stages, "warm_first_photo": warm,
        "decoder_sky_regions": spread(sky_ms), "decoder_people": spread(people_ms),
        "decoder_strokes_and_clicks": {k: spread(v) for k, v in decoder_by_purpose.items()},
        "per_photo": per_photo, "machine": fl.machine_description(),
    })
    print(f"[done] warm encoder {warm['encoder'].get('p50_ms')} ms, decoder box "
          f"{warm['decoder_box'].get('p50_ms')} ms, {subject} {warm[subject].get('p50_ms')} ms; "
          f"peak working set {memory_at_end['peak_working_set_bytes'] / 2**20:.0f} MB -> "
          f"{args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
