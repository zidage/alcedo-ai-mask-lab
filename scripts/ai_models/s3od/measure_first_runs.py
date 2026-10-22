#!/usr/bin/env python3
"""First runs of one Subject graph in new processes: DirectML and the display driver resets
(issue #6, MVP report section 5.3), or Core ML on macOS and the crash reports (issue #8).

Input:  --model (s3od, birefnet, isnet, or universal_matting), --graph, --photo, --count
        (default 9), --provider (default directml), --coreml-model-format,
        --coreml-compute-units (provider coreml_native: --graph is the compiled Core ML model).
Output: --out (JSON): per process the start and end time, the session load time, the first run
        time or the ONNX Runtime error, the number of NaN and infinite output values, the exit
        code (a negative value is the signal that stopped the process), and the memory record.
        Windows: the System events of the time window: `nvlddmkm` event 153 and `Display` event
        4101 ("the display driver stopped responding and has recovered"). A process counts as
        reset when an event 4101 is between its start and 5 s after its end. macOS: the crash
        reports (`macos_records.crash_reports`, user and system processes) between the start of
        the process and 10 s after its end.

Each process creates a new session with the MVP settings, runs the graph once, and exits. The
processes run one after the other with --pause-seconds between them. The script does not retry a
failed run.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

RESET_EVENT = 4101
EVENT_MARGIN = timedelta(seconds=5)
CRASH_REPORT_MARGIN = timedelta(seconds=10)
PROVIDERS = ("coreml", "coreml_native", "cpu", "directml")
COREML_MODEL_FORMATS = ("NeuralNetwork", "MLProgram")
NATIVE_COMPUTE_UNITS = ("ALL", "CPU_AND_GPU", "CPU_AND_NE", "CPU_ONLY")


def child(args) -> int:
    import numpy as np

    import subject_runtime as rt

    model = rt.SubjectModel(args.model)
    prepared = model.prepare(Path(args.photo))
    record: dict = {}
    try:
        started = time.perf_counter()
        session = rt.create_session(Path(args.graph), args.provider,
                                    coreml_model_format=args.coreml_model_format,
                                    coreml_compute_units=args.coreml_compute_units)
        record["load_ms"] = round((time.perf_counter() - started) * 1000, 1)
        started = time.perf_counter()
        outputs = session.run(None, {session.get_inputs()[0].name: prepared.pixels})
        record["first_run_ms"] = round((time.perf_counter() - started) * 1000, 1)
        record["non_finite_outputs"] = int(sum(np.count_nonzero(~np.isfinite(output))
                                               for output in outputs))
        record.update(rt.memory_record(os.getpid()))
    except Exception as error:  # noqa: BLE001 - the parent records the real ONNX Runtime error
        record["error"] = str(error)[:500]
        print(json.dumps(record), flush=True)
        return 1
    print(json.dumps(record), flush=True)
    return 0


def system_events(start: datetime, end: datetime) -> list[dict]:
    command = (
        "$e = Get-WinEvent -FilterHashtable @{LogName='System'; Id=153,4101; "
        f"StartTime=[datetime]'{start:%Y-%m-%dT%H:%M:%S}'; EndTime=[datetime]'{end:%Y-%m-%dT%H:%M:%S}'}} "
        "-ErrorAction SilentlyContinue; "
        "@($e | ForEach-Object { [pscustomobject]@{time=$_.TimeCreated.ToString('s'); id=$_.Id; "
        "provider=$_.ProviderName} }) | ConvertTo-Json -Compress")
    text = subprocess.run(["powershell", "-NoProfile", "-Command", command], capture_output=True,
                          text=True).stdout.strip()
    if not text:
        return []
    events = json.loads(text)
    events = events if isinstance(events, list) else [events]
    return sorted(events, key=lambda e: e["time"])


def parent(args) -> int:
    import onnxruntime as ort

    import subject_runtime as rt

    runs = []
    window_start = datetime.now().replace(microsecond=0)
    for index in range(1, args.count + 1):
        if index > 1:
            time.sleep(args.pause_seconds)
        started = datetime.now()
        process = subprocess.run(
            [sys.executable, "-I", os.path.abspath(__file__), "--child", "--model", args.model,
             "--graph", args.graph, "--photo", args.photo, "--provider", args.provider,
             "--coreml-model-format", args.coreml_model_format,
             "--coreml-compute-units", args.coreml_compute_units],
            capture_output=True, text=True)
        ended = datetime.now()
        lines = [line for line in process.stdout.splitlines() if line.startswith("{")]
        record = json.loads(lines[-1]) if lines else {"error": process.stderr[-500:]}
        record.update({"index": index, "start": started.isoformat(timespec="seconds"),
                       "end": ended.isoformat(timespec="seconds"),
                       "seconds": round((ended - started).total_seconds(), 1),
                       "exit_code": process.returncode})
        runs.append(record)
        print(json.dumps(record), flush=True)
    window_end = datetime.now().replace(microsecond=0) + timedelta(seconds=10)
    time.sleep(10)
    report = {
        "model": args.model, "graph": Path(args.graph).as_posix(), "provider": args.provider,
        "provider_settings": rt.mp.provider_settings(args.provider, args.coreml_model_format,
                                                     args.coreml_compute_units),
        "onnxruntime": ort.__version__, "photo": Path(args.photo).name,
        "pause_seconds": args.pause_seconds,
        "window": [window_start.isoformat(), window_end.isoformat()],
        "first_runs": len(runs), "failed_runs": sum(1 for r in runs if r["exit_code"] != 0),
        "runs_with_non_finite_outputs": sum(1 for r in runs if r.get("non_finite_outputs")),
    }
    if sys.platform == "darwin":
        reports = rt.macos_records.crash_reports(window_start, window_end)
        for record in runs:
            start = datetime.fromisoformat(record["start"])
            end = datetime.fromisoformat(record["end"]) + CRASH_REPORT_MARGIN
            record["crash_reports"] = [r for r in reports
                                       if start <= datetime.fromisoformat(r["time"]) <= end]
        report.update({"runs_with_crash_reports": sum(1 for r in runs if r["crash_reports"]),
                       "runs": runs, "crash_reports": reports})
    else:
        events = system_events(window_start, window_end)
        for record in runs:
            start = datetime.fromisoformat(record["start"])
            end = datetime.fromisoformat(record["end"]) + EVENT_MARGIN
            inside = [e for e in events if start <= datetime.fromisoformat(e["time"]) <= end]
            record["events"] = inside
            record["reset"] = any(e["id"] == RESET_EVENT for e in inside)
        report.update({"resets": sum(1 for r in runs if r["reset"]), "runs": runs,
                       "events": events})
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"[first runs] {args.model} {Path(args.graph).parent.name} {args.provider}: "
          f"{report['failed_runs']} failed of {len(runs)}; resets {report.get('resets')}, "
          f"runs with crash reports {report.get('runs_with_crash_reports')}", flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, choices=("s3od", "birefnet", "isnet",
                                                         "universal_matting"))
    parser.add_argument("--graph", required=True)
    parser.add_argument("--photo", required=True)
    parser.add_argument("--count", type=int, default=9)
    parser.add_argument("--pause-seconds", type=float, default=5.0)
    parser.add_argument("--provider", default="directml", choices=PROVIDERS)
    parser.add_argument("--coreml-model-format", default=COREML_MODEL_FORMATS[0],
                        choices=COREML_MODEL_FORMATS)
    parser.add_argument("--coreml-compute-units", default=NATIVE_COMPUTE_UNITS[0],
                        choices=NATIVE_COMPUTE_UNITS)
    parser.add_argument("--out")
    parser.add_argument("--child", action="store_true")
    args = parser.parse_args()
    if args.child:
        return child(args)
    if not args.out:
        parser.error("--out is required")
    return parent(args)


if __name__ == "__main__":
    raise SystemExit(main())
