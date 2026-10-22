#!/usr/bin/env python3
"""First DirectML runs of one Subject graph in new processes, and the display driver resets
(issue #6, MVP report section 5.3).

Input:  --model (s3od, birefnet, or isnet), --graph, --photo, --count (default 9).
Output: --out (JSON): per process the start and end time, the session load time, the first run
        time or the ONNX Runtime error, the exit code; and the Windows System events of the time
        window: `nvlddmkm` event 153 and `Display` event 4101 ("the display driver stopped
        responding and has recovered"). A process counts as reset when an event 4101 is between
        its start and 5 s after its end.

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


def child(args) -> int:
    import subject_runtime as rt

    model = rt.SubjectModel(args.model)
    prepared = model.prepare(Path(args.photo))
    record: dict = {}
    try:
        started = time.perf_counter()
        session = rt.create_session(Path(args.graph), "directml")
        record["load_ms"] = round((time.perf_counter() - started) * 1000, 1)
        started = time.perf_counter()
        session.run(None, {session.get_inputs()[0].name: prepared.pixels})
        record["first_run_ms"] = round((time.perf_counter() - started) * 1000, 1)
        record.update(rt.gpu_memory_mb(os.getpid()))
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

    runs = []
    window_start = datetime.now().replace(microsecond=0)
    for index in range(1, args.count + 1):
        if index > 1:
            time.sleep(args.pause_seconds)
        started = datetime.now()
        process = subprocess.run(
            [sys.executable, "-I", os.path.abspath(__file__), "--child", "--model", args.model,
             "--graph", args.graph, "--photo", args.photo],
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
    events = system_events(window_start, window_end)
    for record in runs:
        start = datetime.fromisoformat(record["start"])
        end = datetime.fromisoformat(record["end"]) + EVENT_MARGIN
        inside = [e for e in events if start <= datetime.fromisoformat(e["time"]) <= end]
        record["events"] = inside
        record["reset"] = any(e["id"] == RESET_EVENT for e in inside)
    report = {
        "model": args.model, "graph": Path(args.graph).as_posix(),
        "onnxruntime": ort.__version__, "photo": Path(args.photo).name,
        "pause_seconds": args.pause_seconds,
        "window": [window_start.isoformat(), window_end.isoformat()],
        "first_runs": len(runs), "failed_runs": sum(1 for r in runs if r["exit_code"] != 0),
        "resets": sum(1 for r in runs if r["reset"]), "runs": runs, "events": events,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"[first runs] {args.model} {Path(args.graph).parent.name}: {report['failed_runs']} "
          f"failed, {report['resets']} resets of {len(runs)}; {len(events)} events", flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, choices=("s3od", "birefnet", "isnet"))
    parser.add_argument("--graph", required=True)
    parser.add_argument("--photo", required=True)
    parser.add_argument("--count", type=int, default=9)
    parser.add_argument("--pause-seconds", type=float, default=5.0)
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
