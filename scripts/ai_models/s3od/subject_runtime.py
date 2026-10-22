"""Shared helpers of the DirectML measurements of issue #6 (MVP environment, onnxruntime-directml).

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
CPU memory arena off, sequential execution and no memory patterns on DirectML).
GPU memory: the "GPU Process Memory" counters of this process
(`measure_birefnet_memory_phases.gpu_memory_mb`).
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


def create_session(graph: Path, provider: str, threads: int = 4, profile_prefix: str | None = None):
    """`mask_pipeline.create_session`, with ONNX Runtime profiling when `profile_prefix` is set."""
    if profile_prefix is None:
        return mp.create_session(graph, provider, threads)
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    options.enable_cpu_mem_arena = False
    options.log_severity_level = mp.ORT_ERROR_LOG_LEVEL
    options.enable_profiling = True
    options.profile_file_prefix = profile_prefix
    if provider == "directml":
        options.enable_mem_pattern = False
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    session = ort.InferenceSession(str(graph), options, providers=[mp.PROVIDERS[provider]])
    if session.get_providers()[0] != mp.PROVIDERS[provider]:
        raise RuntimeError(f"{graph.name}: ONNX Runtime selected {session.get_providers()}")
    return session


def working_set() -> dict:
    import psutil

    info = psutil.Process().memory_info()
    return {"working_set_mb": round(info.rss / 2**20),
            "peak_working_set_mb": round(getattr(info, "peak_wset", info.rss) / 2**20)}


def percentiles(values: list[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    return {"count": len(values), "p50": round(float(np.percentile(array, 50)), 1),
            "p95": round(float(np.percentile(array, 95)), 1),
            "max": round(float(array.max()), 1), "min": round(float(array.min()), 1)}
