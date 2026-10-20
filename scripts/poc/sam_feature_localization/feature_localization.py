"""Shared helpers of the proof of concept "locate people and sky from SAM 2.1 image features".

Issue #1 of this repository, README section 5. The scripts of this directory import this module:

  extract_cell_features.py : SAM 2.1 tiny encoder -> cell features on the 64 x 64 encoder grid
  make_label_grids.py      : reference masks + class_presence.json -> label grid per photo
  score_prototypes.py      : leave-one-photo-out prototypes -> score maps and measurements
  decode_top_cells.py      : top-scoring cells as prompts -> SAM 2.1 tiny decoder -> IoU
  make_score_sheets.py     : score map overlays (local only, not committed)

The cell grid is the grid of `image_embed`: the photo is stretched to 1024 x 1024 (the SAM 2.1
preprocessing), so one cell is 1/64 of the photo width by 1/64 of the photo height.
"""

from __future__ import annotations

import json
import os
import platform
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SAM2_SCRIPTS = HERE.parents[1] / "ai_models" / "sam2"

GRID = 64
IMAGE_SIZE = 1024
CELL_MODEL_PIXELS = IMAGE_SIZE // GRID
CLASSES = ("person", "sky")
# Label grid values.
LABEL_EXCLUDED = -1
LABEL_OTHER = 0
LABEL_PERSON = 1
LABEL_SKY = 2
CLASS_LABELS = {"person": LABEL_PERSON, "sky": LABEL_SKY}
# Feature sources. Each cell vector is normalized to unit length before any comparison.
FEATURE_SOURCES = ("image_embed", "high_res_feats_1", "high_res_feats_0", "concatenated")
FEATURE_SOURCE_TITLES = {
    "image_embed": "image_embed (256 ch, stride 16)",
    "high_res_feats_1": "high_res_feats_1 pooled 2 x 2 (64 ch, stride 8)",
    "high_res_feats_0": "high_res_feats_0 pooled 4 x 4 (32 ch, stride 4)",
    "concatenated": "concatenation of the three normalized sources (352 ch)",
}


def add_sam2_scripts_to_path() -> None:
    """Make `sam2_graphs` and `compare_onnx_outputs` of scripts/ai_models/sam2 importable."""
    if str(SAM2_SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SAM2_SCRIPTS))


def limit_threads(threads: int) -> None:
    """Limit the CPU threads of NumPy, OpenCV, and PyTorch (AGENTS.md: at most 4)."""
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[name] = str(threads)
    try:
        import cv2

        cv2.setNumThreads(threads)
    except ImportError:
        pass
    try:
        import torch

        torch.set_num_threads(threads)
    except ImportError:
        pass


def create_cpu_session(path: Path, threads: int):
    """ONNX Runtime CPU session with `threads` intra-op threads and the CPU memory arena off."""
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    options.enable_cpu_mem_arena = False
    return ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])


def peak_working_set_bytes() -> int:
    """Peak working set of this process (Windows `PeakWorkingSetSize`; else the maximum RSS)."""
    import psutil

    info = psutil.Process().memory_info()
    return int(getattr(info, "peak_wset", getattr(info, "rss", 0)))


def machine_description() -> dict:
    import psutil

    return {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "logical_cpus": psutil.cpu_count(logical=True),
        "memory_bytes": psutil.virtual_memory().total,
        "python": platform.python_version(),
    }


def file_sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as stream:
        json.dump(data, stream, indent=2)
        stream.write("\n")


def read_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def normalize_cells(features: np.ndarray) -> np.ndarray:
    """[C, 64, 64] -> [4096, C] with unit-length rows."""
    cells = features.reshape(features.shape[0], -1).T.astype(np.float64)
    norms = np.linalg.norm(cells, axis=1, keepdims=True)
    return cells / np.maximum(norms, 1e-12)


def cell_vectors(feature_file: Path, source: str) -> np.ndarray:
    """Unit-length cell vectors [4096, C] of one feature source of one photo."""
    with np.load(feature_file) as data:
        if source != "concatenated":
            return normalize_cells(data[source])
        parts = [normalize_cells(data[name]) for name in FEATURE_SOURCES[:3]]
    joined = np.concatenate(parts, axis=1)
    return joined / np.linalg.norm(joined, axis=1, keepdims=True)


def cell_center_model(row: int, column: int) -> list[float]:
    """Center of a cell in model-input pixels (the decoder prompt space)."""
    return [(column + 0.5) * CELL_MODEL_PIXELS, (row + 0.5) * CELL_MODEL_PIXELS]


def top_cells(score: np.ndarray, count: int) -> list[tuple[int, int]]:
    """(row, column) of the `count` highest-scoring cells, highest first."""
    order = np.argsort(score.ravel(), kind="stable")[::-1][:count]
    return [(int(i // GRID), int(i % GRID)) for i in order]


def best_multimask_index(compare_results: list[dict], file: str, prompt: str) -> int:
    """Index of the multimask output with the highest predicted IoU in the E1 compare.json."""
    entries = [r for r in compare_results
               if r["image"] == file and r["prompt"] == prompt and r["multimask"]]
    if not entries:
        raise KeyError(f"no multimask entry for {file} {prompt} in compare.json")
    return int(max(entries, key=lambda r: r["score_torch"])["mask_index"])


def reference_mask_file(stem: str, item: dict, compare_results: list[dict]) -> str:
    if item["rule"] == "single_mask":
        return f"{stem}__{item['prompt']}__single0.png"
    if item["rule"] == "best_multimask":
        index = best_multimask_index(compare_results, f"{stem}.jpg", item["prompt"])
        return f"{stem}__{item['prompt']}__multi{index}.png"
    raise ValueError(f"unknown reference rule {item['rule']}")


def class_reference_mask(stem: str, entry: dict, reference_dir: Path,
                         compare_results: list[dict]) -> tuple[np.ndarray, list[str]]:
    """Reference mask of one class at the photo size (union of the included references minus the
    excluded ones) and the file names that make it."""
    from PIL import Image

    mask = None
    files = []
    for item in entry["reference"]:
        name = reference_mask_file(stem, item, compare_results)
        files.append(("-" if item.get("exclude") else "+") + name)
        part = np.asarray(Image.open(Path(reference_dir) / name).convert("L")) > 127
        if mask is None:
            mask = np.zeros_like(part)
        mask = mask & ~part if item.get("exclude") else mask | part
    return mask, files


def box_mask(boxes: list[list[float]], width: int, height: int) -> np.ndarray:
    """Union of normalized [x0, y0, x1, y1] boxes as a boolean mask at the photo size."""
    mask = np.zeros((height, width), bool)
    for x0, y0, x1, y1 in boxes:
        mask[int(np.floor(y0 * height)):int(np.ceil(y1 * height)),
             int(np.floor(x0 * width)):int(np.ceil(x1 * width))] = True
    return mask


def cell_fraction(mask: np.ndarray) -> np.ndarray:
    """Share of the pixels of each cell of the 64 x 64 grid that are inside `mask` (area
    average; a cell is 1/64 of the photo width by 1/64 of the photo height)."""
    import cv2

    return cv2.resize(mask.astype(np.float32), (GRID, GRID), interpolation=cv2.INTER_AREA)


def cell_center_inside(mask: np.ndarray) -> np.ndarray:
    """True for the cells whose center pixel is inside `mask`."""
    height, width = mask.shape
    rows = np.minimum(((np.arange(GRID) + 0.5) * height / GRID).astype(int), height - 1)
    columns = np.minimum(((np.arange(GRID) + 0.5) * width / GRID).astype(int), width - 1)
    return mask[np.ix_(rows, columns)]


def area_under_roc(scores: np.ndarray, positive: np.ndarray) -> float | None:
    """Mann-Whitney estimate of the area under the ROC curve; ties count one half."""
    positives = int(positive.sum())
    negatives = positive.size - positives
    if positives == 0 or negatives == 0:
        return None
    from scipy.stats import rankdata

    ranks = rankdata(scores)
    return float((ranks[positive].sum() - positives * (positives + 1) / 2)
                 / (positives * negatives))
