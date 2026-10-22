"""Memory and crash records on macOS (issue #8): the macOS records that replace the Windows
working set, the "GPU Process Memory" counters of `run_with_gpu_memory.ps1`, and the Windows
System events 153 and 4101 of `measure_first_runs.py`.

Input:  a process ID (default: this process), or a time window for the crash reports.
Output: memory values in MB (2**20 bytes); the crash reports of the window.

macOS has no per-process GPU memory counter. Apple silicon has unified memory: the Metal buffers
that Core ML allocates for the GPU are in the physical footprint of the process (the value of
Activity Monitor "Memory" and of `footprint`). The values come from `proc_pid_rusage` with
`RUSAGE_INFO_V4` (libproc):

  phys_footprint_mb               the physical footprint now (`ri_phys_footprint`)
  lifetime_max_phys_footprint_mb  the peak physical footprint since the process started
                                  (`ri_lifetime_max_phys_footprint`; the kernel keeps it, so no
                                  sample can miss the peak)
  resident_mb                     the resident size (`ri_resident_size`)

The Neural Engine runs the Core ML programs through the `aned` system process. Memory that the
system allocates for the Neural Engine outside this process is not in its footprint. The system
value `system_used_mb` (from `vm_stat`: anonymous pages - purgeable pages + wired pages + pages
in the compressor, the "Memory Used" of Activity Monitor) is sampled by
`run_with_memory_sampling.py` to show such memory as a difference to the value before the run.

Crash records: macOS writes a report (`.ips`) for each process that stops on a signal or an
exception, in ~/Library/Logs/DiagnosticReports (processes of the user, for example Python) and
/Library/Logs/DiagnosticReports (system processes, for example `aned` of the Neural Engine).
`crash_reports` lists the reports of a time window with the process name, the bug type, and the
exception type and signal.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import json
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

CRASH_REPORT_DIRS = (Path.home() / "Library" / "Logs" / "DiagnosticReports",
                     Path("/Library/Logs/DiagnosticReports"))

RUSAGE_INFO_V4 = 4
# Field indexes of struct rusage_info_v4 (sys/resource.h) after the 16-byte UUID.
RESIDENT_SIZE = 6
PHYS_FOOTPRINT = 7
LIFETIME_MAX_PHYS_FOOTPRINT = 28
RUSAGE_V4_FIELDS = 36
MB = 2**20

_libproc = None


def _proc_pid_rusage(pid: int) -> list[int]:
    global _libproc
    if sys.platform != "darwin":
        raise RuntimeError("proc_pid_rusage needs macOS")
    if _libproc is None:
        _libproc = ctypes.CDLL(ctypes.util.find_library("proc") or "/usr/lib/libproc.dylib",
                               use_errno=True)
    # The 16-byte UUID is the first two uint64 values.
    buffer = (ctypes.c_uint64 * (2 + RUSAGE_V4_FIELDS))()
    if _libproc.proc_pid_rusage(int(pid), RUSAGE_INFO_V4, ctypes.byref(buffer)) != 0:
        error = ctypes.get_errno()
        raise OSError(error, f"proc_pid_rusage({pid}): {os.strerror(error)}")
    return list(buffer)[2:]


def process_footprint(pid: int | None = None) -> dict:
    """Physical footprint, its lifetime peak, and the resident size of `pid` in MB."""
    fields = _proc_pid_rusage(os.getpid() if pid is None else pid)
    return {"phys_footprint_mb": round(fields[PHYS_FOOTPRINT] / MB, 1),
            "lifetime_max_phys_footprint_mb": round(fields[LIFETIME_MAX_PHYS_FOOTPRINT] / MB, 1),
            "resident_mb": round(fields[RESIDENT_SIZE] / MB, 1)}


def system_used_mb() -> float:
    """System memory in use (Activity Monitor "Memory Used") from `vm_stat`, in MB."""
    text = subprocess.run(["vm_stat"], capture_output=True, text=True, check=True).stdout
    page = int(re.search(r"page size of (\d+) bytes", text).group(1))
    values = {name.strip(): int(value) for name, value in
              re.findall(r"^([^:\n]+):\s+(\d+)\.", text, flags=re.MULTILINE)}
    pages = (values["Anonymous pages"] - values["Pages purgeable"] + values["Pages wired down"]
             + values["Pages occupied by compressor"])
    return round(pages * page / MB, 1)


def memory_record(pid: int | None = None) -> dict:
    """The memory record of a measurement phase: on macOS the footprint values above; on other
    systems the working set and its peak (psutil; `peak_wset` is Windows only)."""
    if sys.platform == "darwin":
        return process_footprint(pid)
    import psutil

    info = psutil.Process(pid).memory_info()
    return {"working_set_mb": round(info.rss / MB),
            "peak_working_set_mb": round(getattr(info, "peak_wset", info.rss) / MB)}


def crash_reports(start: datetime, end: datetime) -> list[dict]:
    """The `.ips` crash reports with a modification time from `start` to `end` (local time)."""
    reports = []
    for directory in CRASH_REPORT_DIRS:
        if not directory.is_dir():
            continue
        for path in directory.glob("*.ips"):
            try:
                modified = datetime.fromtimestamp(path.stat().st_mtime)
            except OSError:
                continue
            if not start <= modified <= end:
                continue
            record = {"file": str(path), "time": modified.isoformat(timespec="seconds")}
            try:
                header, _, body = path.read_text(encoding="utf-8", errors="replace").partition("\n")
                header_json = json.loads(header)
                record.update({"process": header_json.get("app_name"),
                               "bug_type": header_json.get("bug_type")})
                exception = json.loads(body).get("exception", {}) if body.strip() else {}
                record.update({"exception": exception.get("type"),
                               "signal": exception.get("signal")})
            except (OSError, ValueError) as error:
                record["read_error"] = str(error)
            reports.append(record)
    return sorted(reports, key=lambda r: r["time"])


if __name__ == "__main__":
    print({**process_footprint(), "system_used_mb": system_used_mb()})
