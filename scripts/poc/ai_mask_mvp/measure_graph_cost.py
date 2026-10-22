#!/usr/bin/env python3
"""Placement, cost, and first runs of one SAM 2.1 tiny or people and sky head graph on one ONNX
Runtime provider (issue #8; the S3OD graph has the scripts of scripts/ai_models/s3od).

Graphs (--graph-name; the files of mask_pipeline.GraphPaths.default under --explore; with
--provider coreml_native the compiled Core ML models of GraphPaths.coreml_models):
  point_encoder    SAM 2.1 tiny image encoder; input: the photo (sam2_graphs.preprocess_image)
  point_decoder    prompt encoder and mask decoder; inputs: the encoder outputs of the photo
                   (--features-dir) and the first box prompt of the photo in --prompts (labels 2
                   and 3, no mask_input), as mask_pipeline.MaskPipeline.decode
  people_sky_head  P1 run 2 head; input: the `image_embed` of the photo (--features-dir)

Subcommands:
  features    the encoder outputs of each photo on the CPU provider (<stem>.npz with
              image_embed, high_res_feats_0, high_res_feats_1), the inputs of the decoder and
              the head in the other subcommands
  placement   the record of provider_placement.measure_placement (Core ML capability log,
              executed nodes and kernel time per provider) on --warm-photo
  cost        in one process and one session: session load, the first run on --warm-photo,
              --warm-runs more runs (p50, p95, maximum), one run per photo of --prompts; for the
              decoder also --warm-runs runs with 10 points and the previous mask as mask_input
              (the measure_pipeline_cost.py warm runs); the memory after each phase
              (subject_runtime.memory_record: macOS physical footprint and its lifetime peak);
              the number of NaN and infinite output values
  first-runs  --count new processes, each creates the session and runs once on --warm-photo;
              the exit code (a negative value is the signal), the error, and on macOS the crash
              reports between the start and 10 s after the end of each process

Input:  --provider (cpu, coreml, coreml_native, directml), --coreml-model-format (provider
        coreml; default NeuralNetwork, the production format), --coreml-compute-units (provider
        coreml_native; default ALL), --photos, --prompts, --warm-photo (default DSC_1535.jpg).
        `placement` is for ONNX Runtime; the native models have
        scripts/ai_models/coreml/profile_compute_plan.py.
Output: --out (JSON). Sessions: mask_pipeline.create_session (4 CPU threads, CPU memory arena
off). Run one model job at a time (AGENTS.md). An ONNX Runtime error stops the script with that
error; the provider is not changed.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import mask_pipeline as mp  # noqa: E402
import macos_records  # noqa: E402  (scripts/ai_models/sam2, on sys.path by mask_pipeline)

GRAPH_NAMES = ("point_encoder", "point_decoder", "people_sky_head")
FEATURE_NAMES = ("image_embed", "high_res_feats_0", "high_res_feats_1")
CRASH_REPORT_MARGIN = timedelta(seconds=10)


def graph_path(explore: Path, name: str, provider: str = "cpu") -> Path:
    return mp.GraphPaths.default(explore).files(provider)[name]


def photo_boxes(prompts: Path) -> dict[str, list[float]]:
    """The first box prompt (normalized x0, y0, x1, y1) of each photo of the quality set."""
    data = json.loads(prompts.read_text(encoding="utf-8"))
    return {entry["file"]: entry["boxes"][0]["box"] for entry in data["images"]}


def decoder_inputs(features: dict, coords: list[list[float]], labels: list[float],
                   mask_input: np.ndarray | None) -> dict:
    """The decoder inputs of mask_pipeline.MaskPipeline.decode."""
    import sam2_graphs as graphs

    size = graphs.LOW_RES_MASK_SIZE
    return {**{name: features[name] for name in FEATURE_NAMES},
            "point_coords": np.array([coords], np.float32),
            "point_labels": np.array([labels], np.float32),
            "mask_input": (np.asarray(mask_input, np.float32)[None, None]
                           if mask_input is not None else np.zeros((1, 1, size, size), np.float32)),
            "has_mask_input": np.array([1.0 if mask_input is not None else 0.0], np.float32)}


def box_inputs(features: dict, box: list[float]) -> dict:
    import sam2_graphs as graphs

    x0, y0, x1, y1 = (v * graphs.IMAGE_SIZE for v in box)
    return decoder_inputs(features, [[x0, y0], [x1, y1]], [2.0, 3.0], None)


def graph_inputs(name: str, photo_file: str, args) -> dict:
    """The inputs of `name` for one photo."""
    if name == "point_encoder":
        import sam2_graphs as graphs

        photo = mp.load_photo(Path(args.photos) / photo_file)
        return {"pixel_values": graphs.preprocess_image(photo.image).numpy()}
    features = dict(np.load(Path(args.features_dir) / f"{Path(photo_file).stem}.npz"))
    if name == "people_sky_head":
        return {"image_embed": features["image_embed"]}
    return box_inputs(features, photo_boxes(Path(args.prompts))[photo_file])


def non_finite(outputs: list[np.ndarray]) -> int:
    return int(sum(np.count_nonzero(~np.isfinite(output)) for output in outputs))


def spread(values: list[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    return {"count": len(values), "p50": round(float(np.percentile(array, 50)), 2),
            "p95": round(float(np.percentile(array, 95)), 2),
            "max": round(float(array.max()), 2), "min": round(float(array.min()), 2)}


def run_features(args) -> int:
    import sam2_graphs as graphs

    out_dir = Path(args.features_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    graph = graph_path(Path(args.explore), "point_encoder")
    session = mp.create_session(graph, "cpu", args.threads)
    for photo_file in photo_boxes(Path(args.prompts)):
        photo = mp.load_photo(Path(args.photos) / photo_file)
        outputs = session.run(None, {"pixel_values": graphs.preprocess_image(photo.image).numpy()})
        np.savez(out_dir / f"{Path(photo_file).stem}.npz", **dict(zip(FEATURE_NAMES, outputs)))
        print(f"[features] {photo_file}", flush=True)
    mp.save_json(out_dir / "features.json", {"graph": mp.repository_path(graph),
                                              "sha256": mp.fl.file_sha256(graph),
                                              "provider": "cpu", "threads": args.threads})
    return 0


def run_placement(args) -> int:
    from provider_placement import measure_placement

    graph = graph_path(Path(args.explore), args.graph_name, args.provider)
    inputs = graph_inputs(args.graph_name, args.warm_photo, args)
    label = f"{args.graph_name}_{args.provider}"
    if args.provider == "coreml":
        label += f"_{args.coreml_model_format}"
    report = {"graph_name": args.graph_name, "photo": args.warm_photo,
              **measure_placement(graph, args.provider, lambda session: inputs, args.runs,
                                  Path(args.profile_dir), label,
                                  coreml_model_format=args.coreml_model_format)}
    mp.save_json(Path(args.out), report)
    print(json.dumps({k: report.get(k) for k in ("coreml_capability", "executed_nodes",
                                                 "kernel_ms_per_run", "cpu_nodes_per_operator")}),
          flush=True)
    return 0


def run_cost(args) -> int:
    import onnxruntime as ort

    pid = os.getpid()
    graph = graph_path(Path(args.explore), args.graph_name, args.provider)
    photos = list(photo_boxes(Path(args.prompts)))
    inputs = {photo: graph_inputs(args.graph_name, photo, args) for photo in photos}
    warm = inputs[args.warm_photo]
    phases: list[dict] = []

    def phase(name: str, **values) -> None:
        record = {"phase": name, **values, **macos_records.memory_record(pid),
                  "time": time.strftime("%Y-%m-%dT%H:%M:%S")}
        phases.append(record)
        print(json.dumps(record), flush=True)

    def timed_run(feed: dict) -> tuple[list[np.ndarray], float]:
        started = time.perf_counter()
        outputs = session.run(None, feed)
        return outputs, (time.perf_counter() - started) * 1000

    phase("before load")
    started = time.perf_counter()
    session = mp.create_session(graph, args.provider, args.threads, **mp.provider_options(args))
    phase("session load", ms=round((time.perf_counter() - started) * 1000, 1))
    outputs, milliseconds = timed_run(warm)
    phase("first run", ms=round(milliseconds, 1), non_finite_outputs=non_finite(outputs))
    warm_ms = [timed_run(warm)[1] for _ in range(args.warm_runs)]
    phase("warm runs", **spread(warm_ms))

    warm_mask_input_ms = []
    if args.graph_name == "point_decoder":
        # measure_pipeline_cost.py: 9 add points and 1 subtract point with the box mask.
        from compare_onnx_outputs import LOGIT_LIMIT, select_single_mask

        logits, scores, _ = outputs
        previous = np.clip(select_single_mask(logits[0], scores[0])[0], -LOGIT_LIMIT, LOGIT_LIMIT)
        features = {name: warm[name] for name in FEATURE_NAMES}
        points = decoder_inputs(features, [[96.0 * (k + 1), 512.0] for k in range(10)],
                                [1.0] * 9 + [0.0], previous)
        warm_mask_input_ms = [timed_run(points)[1] for _ in range(args.warm_runs)]
        phase("warm runs, 10 points with mask_input", **spread(warm_mask_input_ms))

    photo_records = []
    for photo in photos:
        outputs, milliseconds = timed_run(inputs[photo])
        photo_records.append({"image": photo, "ms": round(milliseconds, 1),
                              "non_finite_outputs": non_finite(outputs)})
    phase("photos", **spread([r["ms"] for r in photo_records]))
    del session
    gc.collect()
    phase("session released")
    mp.save_json(Path(args.out), {
        "graph_name": args.graph_name, "graph": mp.repository_path(graph),
        "graph_bytes": mp.graph_bytes(graph), "graph_sha256": mp.graph_sha256(graph),
        "provider": args.provider,
        "provider_settings": mp.provider_settings(args.provider, args.coreml_model_format,
                                                  args.coreml_compute_units),
        "onnxruntime": ort.__version__, "threads": args.threads, "warm_photo": args.warm_photo,
        "phases": phases, "warm_runs_ms": [round(v, 2) for v in warm_ms],
        "warm_runs_10_points_mask_input_ms": [round(v, 2) for v in warm_mask_input_ms],
        "photos": photo_records,
        "non_finite_outputs": sum(r["non_finite_outputs"] for r in photo_records),
        "machine": mp.fl.machine_description(),
    })
    return 0


def run_first_child(args) -> int:
    record: dict = {}
    try:
        graph = graph_path(Path(args.explore), args.graph_name, args.provider)
        inputs = graph_inputs(args.graph_name, args.warm_photo, args)
        started = time.perf_counter()
        session = mp.create_session(graph, args.provider, args.threads,
                                    **mp.provider_options(args))
        record["load_ms"] = round((time.perf_counter() - started) * 1000, 1)
        started = time.perf_counter()
        outputs = session.run(None, inputs)
        record["first_run_ms"] = round((time.perf_counter() - started) * 1000, 1)
        record["non_finite_outputs"] = non_finite(outputs)
        record.update(macos_records.memory_record())
    except Exception as error:  # noqa: BLE001 - the parent records the real ONNX Runtime error
        record["error"] = str(error)[:500]
        print(json.dumps(record), flush=True)
        return 1
    print(json.dumps(record), flush=True)
    return 0


def run_first_runs(args) -> int:
    import onnxruntime as ort

    runs = []
    window_start = datetime.now().replace(microsecond=0)
    for index in range(1, args.count + 1):
        if index > 1:
            time.sleep(args.pause_seconds)
        started = datetime.now()
        command = [sys.executable, "-I", os.path.abspath(__file__), "first-child",
                   "--graph-name", args.graph_name, "--provider", args.provider,
                   "--coreml-model-format", args.coreml_model_format,
                   "--coreml-compute-units", args.coreml_compute_units, "--explore", args.explore,
                   "--photos", args.photos, "--prompts", args.prompts,
                   "--warm-photo", args.warm_photo, "--threads", str(args.threads)]
        if args.features_dir:
            command += ["--features-dir", args.features_dir]
        process = subprocess.run(command, capture_output=True, text=True)
        ended = datetime.now()
        lines = [line for line in process.stdout.splitlines() if line.startswith("{")]
        record = json.loads(lines[-1]) if lines else {"error": process.stderr[-500:]}
        record.update({"index": index, "start": started.isoformat(timespec="seconds"),
                       "end": ended.isoformat(timespec="seconds"),
                       "seconds": round((ended - started).total_seconds(), 1),
                       "exit_code": process.returncode})
        runs.append(record)
        print(json.dumps(record), flush=True)
    time.sleep(CRASH_REPORT_MARGIN.total_seconds())
    window_end = datetime.now().replace(microsecond=0)
    reports = macos_records.crash_reports(window_start, window_end) \
        if sys.platform == "darwin" else []
    for record in runs:
        start = datetime.fromisoformat(record["start"])
        end = datetime.fromisoformat(record["end"]) + CRASH_REPORT_MARGIN
        record["crash_reports"] = [r for r in reports
                                   if start <= datetime.fromisoformat(r["time"]) <= end]
    report = {
        "graph_name": args.graph_name,
        "graph": mp.repository_path(graph_path(Path(args.explore), args.graph_name,
                                               args.provider)),
        "provider": args.provider,
        "provider_settings": mp.provider_settings(args.provider, args.coreml_model_format,
                                                  args.coreml_compute_units),
        "onnxruntime": ort.__version__, "photo": args.warm_photo,
        "pause_seconds": args.pause_seconds,
        "window": [window_start.isoformat(), window_end.isoformat()],
        "first_runs": len(runs), "failed_runs": sum(1 for r in runs if r["exit_code"] != 0),
        "runs_with_non_finite_outputs": sum(1 for r in runs if r.get("non_finite_outputs")),
        "runs_with_crash_reports": sum(1 for r in runs if r["crash_reports"]),
        "crash_reports_recorded": sys.platform == "darwin",
        "runs": runs, "crash_reports": reports,
    }
    mp.save_json(Path(args.out), report)
    print(f"[first runs] {args.graph_name} {args.provider}: {report['failed_runs']} failed of "
          f"{len(runs)}, runs with crash reports {report['runs_with_crash_reports']}", flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("features", "placement", "cost", "first-runs",
                                            "first-child"))
    parser.add_argument("--graph-name", choices=GRAPH_NAMES, default="point_encoder")
    mp.add_provider_arguments(parser)
    parser.add_argument("--explore", default=str(mp.DEFAULT_EXPLORE))
    parser.add_argument("--photos", required=True)
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--features-dir", default="")
    parser.add_argument("--warm-photo", default="DSC_1535.jpg")
    parser.add_argument("--warm-runs", type=int, default=20)
    parser.add_argument("--runs", type=int, default=2, help="placement: profiled runs")
    parser.add_argument("--profile-dir", default="")
    parser.add_argument("--count", type=int, default=9)
    parser.add_argument("--pause-seconds", type=float, default=5.0)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--out", default="")
    args = parser.parse_args()
    mp.fl.limit_threads(args.threads)
    if args.command != "features" and args.graph_name != "point_encoder" \
            and not args.features_dir:
        parser.error(f"{args.graph_name} needs --features-dir (the features subcommand)")
    if args.command == "features":
        if not args.features_dir:
            parser.error("features needs --features-dir")
        return run_features(args)
    if args.command == "first-child":
        return run_first_child(args)
    if not args.out:
        parser.error(f"{args.command} needs --out")
    if args.command == "placement":
        if not args.profile_dir:
            parser.error("placement needs --profile-dir")
        return run_placement(args)
    if args.command == "cost":
        return run_cost(args)
    return run_first_runs(args)


if __name__ == "__main__":
    raise SystemExit(main())
