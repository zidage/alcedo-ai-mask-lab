#!/usr/bin/env python3
"""Cost of one Subject graph on ONNX Runtime DirectML with the MVP settings (issue #6), and on
Core ML on macOS (issue #8, `--provider coreml`).

Input:  --model (s3od, birefnet, isnet, or universal_matting), --graph (ONNX file), the photos of
        --prompts in --photos (the quality set at a long edge of at most 4096 px), --warm-photo.
Output: --out (JSON), and with --alpha-dir the alpha of each photo (<stem>.npy, float16; see
        subject_runtime.SubjectModel) for probe_subject_masks.py. Each photo record has the number
        of NaN and infinite values in the graph outputs.

Phases, in one process and one session (the model calls only; the photo reading and the
preprocessing are not in the times):

  before load        GPU memory of the process before ONNX Runtime opens the device
  session load       session creation time
  first run          the first call on the warm photo
  warm runs          --warm-runs more calls on the warm photo (p50, p95, maximum)
  photos             one call per photo of the quality set (the alpha is stored)
  session released   the session is deleted

After each phase the script records the memory of this process (`subject_runtime.memory_record`):
on Windows the dedicated and the shared GPU memory (Windows "GPU Process Memory" counters,
`measure_birefnet_memory_phases.gpu_memory_mb`) and the working set; run it under
`run_with_gpu_memory.ps1` for the peak values between the phases. On macOS the physical footprint
and its lifetime peak (`macos_records.py`); run it under `run_with_memory_sampling.py` for the
system memory. `--coreml-model-format` selects the Core ML model format (default NeuralNetwork,
the production format). `--provider coreml_native`: --graph is the compiled native Core ML model
(scripts/ai_models/coreml/convert_models.py) and `--coreml-compute-units` its compute units
(default ALL). Run one model job at a time (AGENTS.md). An error of ONNX Runtime stops
the script with that error; the provider is not changed.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402

import subject_runtime as rt  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, choices=rt.MODELS)
    parser.add_argument("--graph", required=True)
    parser.add_argument("--photos", required=True)
    parser.add_argument("--prompts", required=True, help="the quality set; only the photo list "
                                                         "is used")
    parser.add_argument("--warm-photo", default="DSC_1535.jpg")
    parser.add_argument("--warm-runs", type=int, default=20)
    rt.mp.add_provider_arguments(parser, default="directml")
    parser.add_argument("--alpha-dir")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    import onnxruntime as ort

    pid = os.getpid()
    graph = Path(args.graph)
    model = rt.SubjectModel(args.model)
    photos = [entry["file"] for entry in
              json.loads(Path(args.prompts).read_text(encoding="utf-8"))["images"]]
    prepared = {name: model.prepare(Path(args.photos) / name) for name in photos}
    warm = prepared[args.warm_photo]
    alpha_dir = Path(args.alpha_dir) if args.alpha_dir else None
    if alpha_dir:
        alpha_dir.mkdir(parents=True, exist_ok=True)
    phases: list[dict] = []

    def phase(name: str, **values) -> None:
        record = {"phase": name, **values, **rt.memory_record(pid),
                  "time": time.strftime("%Y-%m-%dT%H:%M:%S")}
        phases.append(record)
        print(json.dumps(record), flush=True)

    phase("before load")
    started = time.perf_counter()
    session = rt.create_session(graph, args.provider, **rt.mp.provider_options(args))
    phase("session load", ms=round((time.perf_counter() - started) * 1000, 1))
    input_name = session.get_inputs()[0].name

    started = time.perf_counter()
    session.run(None, {input_name: warm.pixels})
    phase("first run", ms=round((time.perf_counter() - started) * 1000, 1))

    warm_ms = []
    for _ in range(args.warm_runs):
        started = time.perf_counter()
        session.run(None, {input_name: warm.pixels})
        warm_ms.append((time.perf_counter() - started) * 1000)
    phase("warm runs", **rt.percentiles(warm_ms))

    photo_records = []
    for name in photos:
        started = time.perf_counter()
        outputs = session.run(None, {input_name: prepared[name].pixels})
        milliseconds = (time.perf_counter() - started) * 1000
        alpha, selection = model.alpha(outputs, prepared[name])
        if alpha_dir:
            np.save(alpha_dir / f"{Path(name).stem}.npy", alpha)
        non_finite = int(sum(np.count_nonzero(~np.isfinite(output)) for output in outputs))
        photo_records.append({"image": name, "ms": round(milliseconds, 1),
                              "non_finite_outputs": non_finite, **selection})
    phase("photos", **rt.percentiles([r["ms"] for r in photo_records]))

    del session
    gc.collect()
    phase("session released")

    report = {
        "model": args.model, "provider": args.provider, "onnxruntime": ort.__version__,
        "graph": rt.mp.repository_path(graph), "graph_bytes": rt.mp.graph_bytes(graph),
        "graph_sha256": rt.mp.graph_sha256(graph), "warm_photo": args.warm_photo,
        "session_settings": rt.session_settings(args.provider, args.coreml_model_format,
                                                args.coreml_compute_units),
        "phases": phases, "warm_runs_ms": [round(v, 1) for v in warm_ms],
        "photos": photo_records,
        "non_finite_outputs": sum(r["non_finite_outputs"] for r in photo_records),
    }
    rt.mp.save_json(Path(args.out), report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
