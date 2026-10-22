#!/usr/bin/env python3
"""Cost of one Subject graph on ONNX Runtime DirectML with the MVP settings (issue #6).

Input:  --model (s3od, birefnet, isnet, or universal_matting), --graph (ONNX file), the photos of
        --prompts in --photos (the quality set at a long edge of at most 4096 px), --warm-photo.
Output: --out (JSON), and with --alpha-dir the alpha of each photo (<stem>.npy, float16; see
        subject_runtime.SubjectModel) for probe_subject_masks.py.

Phases, in one process and one session (the model calls only; the photo reading and the
preprocessing are not in the times):

  before load        GPU memory of the process before ONNX Runtime opens the device
  session load       session creation time
  first run          the first call on the warm photo
  warm runs          --warm-runs more calls on the warm photo (p50, p95, maximum)
  photos             one call per photo of the quality set (the alpha is stored)
  session released   the session is deleted

After each phase the script records the dedicated and the shared GPU memory of this process
(Windows "GPU Process Memory" counters, `measure_birefnet_memory_phases.gpu_memory_mb`) and the
working set. Run it under `run_with_gpu_memory.ps1` for the peak values between the phases. Run
one model job at a time (AGENTS.md). An error of ONNX Runtime stops the script with that error;
the provider is not changed.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402

import subject_runtime as rt  # noqa: E402


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


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
    parser.add_argument("--provider", default="directml", choices=sorted(rt.mp.PROVIDERS))
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
        record = {"phase": name, **values, **rt.gpu_memory_mb(pid), **rt.working_set(),
                  "time": time.strftime("%Y-%m-%dT%H:%M:%S")}
        phases.append(record)
        print(json.dumps(record), flush=True)

    phase("before load")
    started = time.perf_counter()
    session = rt.create_session(graph, args.provider)
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
        photo_records.append({"image": name, "ms": round(milliseconds, 1), **selection})
    phase("photos", **rt.percentiles([r["ms"] for r in photo_records]))

    del session
    gc.collect()
    phase("session released")

    report = {
        "model": args.model, "provider": args.provider, "onnxruntime": ort.__version__,
        "graph": rt.mp.repository_path(graph), "graph_bytes": graph.stat().st_size,
        "graph_sha256": sha256_of(graph), "warm_photo": args.warm_photo,
        "session_settings": {"intra_op_threads": 4, "inter_op_threads": 1,
                             "cpu_memory_arena": False, "memory_pattern": False,
                             "execution_mode": "sequential"},
        "phases": phases, "warm_runs_ms": [round(v, 1) for v in warm_ms],
        "photos": photo_records,
    }
    rt.mp.save_json(Path(args.out), report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
