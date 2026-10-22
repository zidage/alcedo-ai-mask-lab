"""Shared helpers of the DirectML measurements of issue #6 (MVP environment, onnxruntime-directml)
and of the Core ML measurements of issue #8 (macOS, onnxruntime).

The measurements run the S3OD, IS-Net general-use, and universal matting graphs and, as the
reference, the BiRefNet
lite 1024 float16 graph of the MVP with the same code. `SubjectModel` gives the preprocessing of
each model and the alpha that the measurement stores:

  s3od      s3od_graphs.letterbox; alpha = the mask with the largest IoU score, unpadded (long
            edge 1024 px), float16
  birefnet  birefnet_graphs.preprocess (stretch to 1024 x 1024); alpha = the `alpha` output at
            1024 x 1024, float16 (the format of the E2 compare_onnx_outputs.py)
  isnet     isnet_graphs.preprocess (stretch to 1024 x 1024, the rule of the DIS `Inference.py`);
            alpha = the `d1` output scaled with its minimum and maximum
            (isnet_graphs.normalize_range) at 1024 x 1024, float16
  universal_matting
            universal_matting_graphs.model_input (the photo, [H, W, 3] float32, values 0 to 255;
            the graph does all resizing); alpha = channel 3 of the `rgba` output / 255
            (universal_matting_graphs.alpha_from_rgba) at the photo size, float16

Sessions: `mask_pipeline.create_session` of the MVP (one execution provider only, 4 CPU threads,
CPU memory arena off, sequential execution and no memory patterns on DirectML; on Core ML the
production provider settings of `mask_pipeline.provider_settings` with the model format
`coreml_model_format`).
Memory (`memory_record`): on Windows the "GPU Process Memory" counters of this process
(`measure_birefnet_memory_phases.gpu_memory_mb`) and the working set; on macOS the physical
footprint and its lifetime peak (`macos_records.py`; unified memory, no GPU counter).
"""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SCRIPTS = HERE.parents[1]
MVP = SCRIPTS / "poc" / "ai_mask_mvp"
for directory in (HERE, MVP):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import mask_pipeline as mp  # noqa: E402
import macos_records  # noqa: E402  (scripts/ai_models/sam2, on sys.path by mask_pipeline)
import s3od_graphs  # noqa: E402
from measure_birefnet_memory_phases import gpu_memory_mb  # noqa: E402,F401

MODELS = ("s3od", "birefnet", "isnet", "universal_matting")


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@dataclass
class Prepared:
    pixels: np.ndarray
    info: dict | None


class SubjectModel:
    """Preprocessing and output handling of one Subject graph."""

    def __init__(self, kind: str):
        if kind not in MODELS:
            raise ValueError(f"model {kind!r} is not one of {MODELS}")
        self.kind = kind
        self._birefnet = (load_module("birefnet_graphs", SCRIPTS / "ai_models" / "birefnet"
                                      / "birefnet_graphs.py") if kind == "birefnet" else None)
        self._isnet = (load_module("isnet_graphs", SCRIPTS / "ai_models" / "isnet"
                                   / "isnet_graphs.py") if kind == "isnet" else None)
        self._universal = (load_module("universal_matting_graphs",
                                       SCRIPTS / "ai_models" / "universal_matting"
                                       / "universal_matting_graphs.py")
                           if kind == "universal_matting" else None)

    def prepare(self, photo_path: Path) -> Prepared:
        """The photo as the comparison scripts read it (s3od: s3od_graphs.load_rgb, as the
        source; birefnet: birefnet_graphs.load_rgb, as E2; isnet: isnet_graphs.load_rgb;
        universal_matting: universal_matting_graphs.load_rgb, as the source) and the model
        input."""
        if self.kind == "s3od":
            pixels, info = s3od_graphs.letterbox(s3od_graphs.load_rgb(photo_path))
            return Prepared(pixels, info)
        if self.kind == "isnet":
            return Prepared(self._isnet.preprocess(self._isnet.load_rgb(photo_path)), None)
        if self.kind == "universal_matting":
            universal = self._universal
            return Prepared(universal.model_input(universal.load_rgb(photo_path)), None)
        image = self._birefnet.load_rgb(str(photo_path))
        return Prepared(self._birefnet.preprocess(image, 1024).numpy(), None)

    def alpha(self, outputs: list[np.ndarray], prepared: Prepared) -> tuple[np.ndarray, dict]:
        """The stored alpha (float16) and the selection record."""
        if self.kind == "s3od":
            masks, iou_scores = outputs
            alpha, index = s3od_graphs.select_mask(masks, iou_scores)
            unpadded = s3od_graphs.unpad(alpha[None], prepared.info)[0]
            return unpadded.astype(np.float16), {
                "selected_mask": index,
                "iou_scores": [round(float(v), 5) for v in iou_scores[0]]}
        if self.kind == "isnet":
            d1 = outputs[0][0, 0]
            return self._isnet.normalize_range(d1).astype(np.float16), {
                "d1_range": [round(float(d1.min()), 6), round(float(d1.max()), 6)]}
        if self.kind == "universal_matting":
            return self._universal.alpha_from_rgba(outputs[0]).astype(np.float16), {}
        return outputs[0][0, 0].astype(np.float16), {}


def create_session(graph: Path, provider: str, threads: int = 4, profile_prefix: str | None = None,
                   coreml_model_format: str = mp.COREML_MODEL_FORMATS[0],
                   coreml_compute_units: str = mp.NATIVE_COMPUTE_UNITS[0]):
    """`mask_pipeline.create_session`, with ONNX Runtime profiling when `profile_prefix` is set.
    Provider coreml_native: `graph` is the compiled Core ML model (convert_models.py)."""
    return mp.create_session(graph, provider, threads, coreml_model_format, profile_prefix,
                             coreml_compute_units=coreml_compute_units)


def memory_record(pid: int) -> dict:
    """The memory of this process after a phase: Windows GPU counters and working set, or the
    macOS physical footprint (see the module docstring)."""
    if sys.platform == "darwin":
        return macos_records.memory_record(pid)
    return {**gpu_memory_mb(pid), **macos_records.memory_record(pid)}


def session_settings(provider: str, coreml_model_format: str,
                     coreml_compute_units: str = mp.NATIVE_COMPUTE_UNITS[0]) -> dict:
    """The session settings of `create_session` for the measurement records."""
    settings = {"intra_op_threads": 4, "inter_op_threads": 1, "cpu_memory_arena": False,
                "graph_optimization": "ORT_ENABLE_ALL"}
    if provider == "directml":
        settings.update({"memory_pattern": False, "execution_mode": "sequential"})
    settings["providers"] = mp.provider_settings(provider, coreml_model_format,
                                                 coreml_compute_units)
    return settings


def percentiles(values: list[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    return {"count": len(values), "p50": round(float(np.percentile(array, 50)), 1),
            "p95": round(float(np.percentile(array, 95)), 1),
            "max": round(float(array.max()), 1), "min": round(float(array.min()), 1)}
