"""AI Mask MVP pipeline: the selected model set of README section 4 on one photo (issue #5).

Input:  a photo (JPEG, PNG, or TIFF) and the ONNX graphs of the earlier steps (not committed):
          SAM 2.1 tiny `point_encoder.onnx` and `point_decoder.onnx` (E4, float32),
          `people_sky_head.onnx` and `people_sky_head.json` (P1 run 2, epoch 15),
          BiRefNet lite `subject_matte.onnx` at 1024 px (E2; float16 by default), or the S3OD
          `subject_masks.onnx` or the IS-Net general-use `subject_d1.onnx` of issue #6
          (scripts/ai_models/s3od, scripts/ai_models/isnet, `--subject-onnx`).
Output: per selection, the low-resolution logits or the alpha, the mask at the photo size, the
        prompts, and a timing record (stage, milliseconds, graph SHA-256) in an optional JSON log.

The module has no UI. `ai_mask_app.py` (the PySide6 UI), `compare_with_measurement_scripts.py`,
and `measure_pipeline_cost.py` use it. It reuses the code of the measurement scripts, so the MVP
and the scripts compute the same prompts and masks:

  photo scaling            make_input_set.scaled_size (long edge at most 4096 px, Lanczos)
  SAM 2.1 preprocessing    sam2_graphs.preprocess_image
  mask selection           compare_onnx_outputs.select_single_mask (box or several points);
                           best predicted IoU of tokens 1 to 3 for one point (E4 default)
  sky prompts              evaluate_quality_photos.sky_regions and region_points (P1 3.3)
  people prompts           head_model.decode_people with nms_iou = 0.5, box clipped to the photo
  stroke combination       E3 section 9: all points of the selection in one decoder run with the
                           previous selected logits as mask_input
  BiRefNet preprocessing   birefnet_graphs.preprocess and alpha_to_photo
  S3OD processing          s3od_graphs.letterbox, select_mask, unpad, and masks_to_photo (the
                           rules of the source; the graph has the outputs `masks`, `iou_scores`)
  IS-Net processing        isnet_graphs.preprocess, normalize_range, and alpha_to_photo (the
                           rules of the DIS `Inference.py`; the graph has the output `d1`)

Data flow (issue #5):

  open photo -> SAM 2.1 tiny encoder (once) -> image_embed, high_res_feats_0, high_res_feats_1
             -> head(image_embed) (once) -> presence, sky map, person centers and boxes
  Sky      sky regions -> 3 points per region -> decoder per region -> union
  People   decode_people -> one box per person above the list threshold -> decoder per person
  Stroke, box, click     decoder (E3 rules)
  Subject  BiRefNet lite 1024 (or S3OD 1024, or IS-Net 1024) -> alpha;  Background = 1 - alpha

Runtime: ONNX Runtime with DirectML (`directml`) or the CPU (`cpu`). The selected provider is the
only provider of every session; if it fails, the error stops the pipeline. CPU threads are
limited to `threads` (4) and the CPU memory arena is off.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SCRIPTS = HERE.parents[1]
REPOSITORY = SCRIPTS.parent
for directory in (SCRIPTS / "poc" / "people_sky_head", SCRIPTS / "poc" / "sam_feature_localization",
                  SCRIPTS / "ai_models" / "sam2"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import feature_localization as fl  # noqa: E402

DEFAULT_EXPLORE = REPOSITORY / "build" / "tmp" / "explore"
MAX_EDGE = 4096
STROKE_POINT_COUNT = 4
PROVIDERS = {"directml": "DmlExecutionProvider", "cpu": "CPUExecutionProvider"}
# Subject graph kinds, by the output names of the graph.
SUBJECT_OUTPUTS = {("alpha",): "birefnet", ("masks", "iou_scores"): "s3od", ("d1",): "isnet"}
ORT_ERROR_LOG_LEVEL = 3


def _load_birefnet_graphs():
    """`scripts/ai_models/birefnet/birefnet_graphs.py` by its file path. The BiRefNet directory is
    not put on sys.path: it has its own `compare_onnx_outputs.py`, which would hide the SAM 2.1
    module of the same name."""
    path = SCRIPTS / "ai_models" / "birefnet" / "birefnet_graphs.py"
    spec = importlib.util.spec_from_file_location("birefnet_graphs", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_s3od_graphs():
    """`scripts/ai_models/s3od/s3od_graphs.py` by its file path. The S3OD directory is not put on
    `sys.path`: its `compare_onnx_outputs.py` has the name of the SAM 2 module."""
    path = SCRIPTS / "ai_models" / "s3od" / "s3od_graphs.py"
    spec = importlib.util.spec_from_file_location("s3od_graphs", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_isnet_graphs():
    """`scripts/ai_models/isnet/isnet_graphs.py` by its file path. The IS-Net directory is not put
    on `sys.path`: its `compare_onnx_outputs.py` has the name of the SAM 2 module."""
    path = SCRIPTS / "ai_models" / "isnet" / "isnet_graphs.py"
    spec = importlib.util.spec_from_file_location("isnet_graphs", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@dataclass
class GraphPaths:
    """The ONNX graphs of the MVP."""

    sam2_dir: Path
    head_onnx: Path
    head_json: Path
    subject_onnx: Path

    @classmethod
    def default(cls, explore: Path = DEFAULT_EXPLORE) -> "GraphPaths":
        head = explore / "p1" / "colab_run2_coco_openimages" / "export"
        return cls(sam2_dir=explore / "e4" / "onnx_float32_batch16" / "tiny",
                   head_onnx=head / "people_sky_head.onnx",
                   head_json=head / "people_sky_head.json",
                   subject_onnx=explore / "e2" / "onnx_fp16" / "lite" / "subject_matte.onnx")

    def files(self) -> dict[str, Path]:
        return {"point_encoder": self.sam2_dir / "point_encoder.onnx",
                "point_decoder": self.sam2_dir / "point_decoder.onnx",
                "people_sky_head": self.head_onnx,
                "subject_matte": self.subject_onnx}


def repository_path(path: Path) -> str:
    """`path` relative to the repository (POSIX separators) when it is inside it."""
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(REPOSITORY).as_posix()
    except ValueError:
        return str(resolved)


def create_session(path: Path, provider: str, threads: int):
    """ONNX Runtime session on `provider` only. CPU threads `threads`, CPU memory arena off.
    DirectML needs sequential execution without memory patterns (ONNX Runtime documentation).
    A provider that cannot be created raises the ONNX Runtime error."""
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    options.enable_cpu_mem_arena = False
    options.log_severity_level = ORT_ERROR_LOG_LEVEL
    if provider == "directml":
        options.enable_mem_pattern = False
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    session = ort.InferenceSession(str(path), options, providers=[PROVIDERS[provider]])
    if session.get_providers()[0] != PROVIDERS[provider]:
        raise RuntimeError(f"{path.name}: ONNX Runtime selected {session.get_providers()}, "
                           f"not {PROVIDERS[provider]}")
    return session


@dataclass
class Photo:
    """A photo with the long edge scaled to at most MAX_EDGE px (8-bit RGB)."""

    path: Path
    image: object
    original_size: tuple[int, int]

    @property
    def width(self) -> int:
        return self.image.size[0]

    @property
    def height(self) -> int:
        return self.image.size[1]


def read_rgb8(path: Path):
    """8-bit RGB PIL image of a JPEG, PNG, or TIFF file, with the EXIF orientation applied.
    16-bit files (PNG, TIFF) are read with OpenCV and scaled to 8 bit (value / 257)."""
    import cv2
    from PIL import Image, ImageOps

    with Image.open(path) as image:
        if image.mode in ("RGB", "RGBA", "L", "P", "LA", "CMYK", "YCbCr"):
            return ImageOps.exif_transpose(image).convert("RGB")
    array = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if array is None:
        raise ValueError(f"{path}: OpenCV cannot read the file")
    if array.dtype == np.uint16:
        array = np.round(array.astype(np.float32) / 257.0).astype(np.uint8)
    elif array.dtype != np.uint8:
        raise ValueError(f"{path}: pixel type {array.dtype} is not supported")
    if array.ndim == 2:
        array = cv2.cvtColor(array, cv2.COLOR_GRAY2RGB)
    elif array.shape[2] == 4:
        array = cv2.cvtColor(array, cv2.COLOR_BGRA2RGB)
    else:
        array = cv2.cvtColor(array, cv2.COLOR_BGR2RGB)
    return Image.fromarray(array)


def load_photo(path: Path, max_edge: int = MAX_EDGE) -> Photo:
    """The photo scaled as `make_input_set.py` scales the quality photos (Lanczos, long edge at
    most `max_edge`). A photo that fits is not resampled."""
    from make_input_set import scaled_size
    from PIL import Image

    image = read_rgb8(Path(path))
    original = image.size
    size = scaled_size(*original, max_edge)
    if size != original:
        image = image.resize(size, Image.Resampling.LANCZOS)
    return Photo(path=Path(path), image=image, original_size=original)


def logits_to_mask(logits: np.ndarray, width: int, height: int) -> np.ndarray:
    """Boolean mask at width x height: bilinear upsampling of 256 x 256 logits, threshold 0 (the
    rule of evaluate_quality_photos.py)."""
    import cv2

    return cv2.resize(np.ascontiguousarray(logits, np.float32), (width, height),
                      interpolation=cv2.INTER_LINEAR) > 0


def stroke_points(polyline: list[list[float]], count: int, width: int,
                  height: int) -> list[list[float]]:
    """`count` points at equal spacing along a stroke, both ends included (E3 section 3.1).

    `polyline` is the pointer path in normalized photo coordinates. The spacing is measured in
    photo pixels along the path, so a straight stroke gives the points of the E3 Rust program
    (`start + k / (count - 1) * (end - start)`). One point (or a path without length) gives the
    first point: a click."""
    path = np.asarray(polyline, np.float64) * np.array([width, height], np.float64)
    if count <= 1 or len(path) == 1:
        return [list(map(float, polyline[0]))]
    segments = np.linalg.norm(np.diff(path, axis=0), axis=1)
    distance = np.concatenate([[0.0], np.cumsum(segments)])
    if distance[-1] <= 0.0:
        return [list(map(float, polyline[0]))]
    targets = np.linspace(0.0, distance[-1], count)
    xs = np.interp(targets, distance, path[:, 0]) / width
    ys = np.interp(targets, distance, path[:, 1]) / height
    return [[float(x), float(y)] for x, y in zip(xs, ys)]


@dataclass
class DecoderRun:
    """One decoder call: its inputs and its outputs."""

    inputs: dict[str, np.ndarray]
    logits: np.ndarray
    scores: np.ndarray
    milliseconds: float


@dataclass
class PhotoState:
    """Per photo: the encoder outputs and the head outputs (computed once)."""

    photo: Photo
    features: tuple[np.ndarray, np.ndarray, np.ndarray]
    head_outputs: dict[str, np.ndarray]
    subject_alpha: np.ndarray | None = None
    # S3OD: the alpha is on the unpadded letterbox grid (long edge 1024).
    subject_pad_info: dict | None = None


@dataclass
class SkyResult:
    """`present`: presence >= threshold. `found`: present and at least one sky region."""

    present: bool
    found: bool
    presence: float
    threshold: float
    message: str
    regions: list[dict] = field(default_factory=list)

    def mask(self, width: int, height: int) -> np.ndarray:
        """Union of the region masks."""
        union = np.zeros((height, width), bool)
        for region in self.regions:
            union |= logits_to_mask(region["logits"], width, height)
        return union


@dataclass
class PeopleResult:
    """`present`: presence >= threshold. `found`: present and at least one box in the People
    list (score >= list threshold)."""

    present: bool
    found: bool
    presence: float
    threshold: float
    list_threshold: float
    message: str
    people: list[dict] = field(default_factory=list)
    all_boxes: list[dict] = field(default_factory=list)

    def mask(self, width: int, height: int, ranks: list[int] | None = None) -> np.ndarray:
        """Mask of the people with the given ranks (all people when `ranks` is None)."""
        union = np.zeros((height, width), bool)
        for person in self.people:
            if ranks is None or person["rank"] in ranks:
                union |= logits_to_mask(person["logits"], width, height)
        return union


class MaskPipeline:
    """All graphs of the MVP, loaded together on one provider."""

    def __init__(self, paths: GraphPaths, provider: str = "directml", threads: int = 4,
                 log_path: Path | None = None) -> None:
        if provider not in PROVIDERS:
            raise ValueError(f"provider {provider!r}: use one of {sorted(PROVIDERS)}")
        fl.limit_threads(threads)
        import head_model as hm
        import sam2_graphs as graphs

        self.graphs = graphs
        self.hm = hm
        self.birefnet = _load_birefnet_graphs()
        self.paths = paths
        self.provider = provider
        self.threads = threads
        self.log_path = Path(log_path) if log_path else None
        self.events: list[dict] = []
        self.graph_records: dict[str, dict] = {}

        head_info = fl.read_json(paths.head_json)
        validation = head_info["validation"]
        self.thresholds = {
            "sky_presence": float(validation["presence_best_f1_sky"]["threshold"]),
            "person_presence": float(validation["presence_best_f1_person"]["threshold"]),
            "people_list": float(validation["person_list_best_f1_nms"]["threshold"]),
        }
        self.head_epoch = head_info["checkpoint_epoch"]
        self.sessions = {}
        for name, path in paths.files().items():
            sha256 = fl.file_sha256(path)
            if name == "people_sky_head" and sha256 != head_info["onnx"]["sha256"]:
                raise ValueError(f"{path}: SHA-256 {sha256}, {paths.head_json} names "
                                 f"{head_info['onnx']['sha256']}")
            started = time.perf_counter()
            self.sessions[name] = create_session(path, provider, threads)
            load_ms = (time.perf_counter() - started) * 1000
            self.graph_records[name] = {"path": repository_path(path),
                                        "bytes": path.stat().st_size,
                                        "sha256": sha256, "load_ms": round(load_ms, 1)}
            self._record("load", load_ms, graph=name)
        subject_input = self.sessions["subject_matte"].get_inputs()[0].shape
        self.subject_size = int(subject_input[-1])
        if self.subject_size != 1024:
            raise ValueError(f"{paths.subject_onnx}: input {subject_input}, the MVP uses a "
                             f"Subject graph at 1024 px")
        outputs = tuple(o.name for o in self.sessions["subject_matte"].get_outputs())
        if outputs not in SUBJECT_OUTPUTS:
            raise ValueError(f"{paths.subject_onnx}: outputs {outputs}, not one of "
                             f"{sorted(SUBJECT_OUTPUTS)}")
        self.subject_kind = SUBJECT_OUTPUTS[outputs]
        self.s3od = None
        self.isnet = None
        if self.subject_kind == "s3od":
            self.s3od = _load_s3od_graphs()
        elif self.subject_kind == "isnet":
            self.isnet = _load_isnet_graphs()

    # Records.

    def _record(self, stage: str, milliseconds: float, **details) -> None:
        self.events.append({"stage": stage, "ms": round(milliseconds, 2), **details})
        if self.log_path:
            fl.write_json(self.log_path, self.log())

    def log(self) -> dict:
        import onnxruntime as ort

        return {"step": "issue #5: AI Mask MVP", "provider": self.provider,
                "threads": self.threads, "onnxruntime": ort.__version__,
                "graphs": self.graph_records, "thresholds": self.thresholds,
                "head_checkpoint_epoch": self.head_epoch, "events": self.events}

    def _run(self, name: str, inputs: dict, stage: str, **details):
        started = time.perf_counter()
        outputs = self.sessions[name].run(None, inputs)
        milliseconds = (time.perf_counter() - started) * 1000
        self._record(stage, milliseconds, **details)
        return outputs, milliseconds

    # Photo.

    def open_photo(self, path: Path) -> PhotoState:
        """Read and scale the photo, run the encoder and the head once."""
        started = time.perf_counter()
        photo = load_photo(path)
        self._record("read_photo", (time.perf_counter() - started) * 1000, photo=photo.path.name,
                     size=[photo.width, photo.height], original_size=list(photo.original_size))
        return self.encode(photo)

    def encode(self, photo: Photo) -> PhotoState:
        pixels = self.graphs.preprocess_image(photo.image).numpy()
        features, _ = self._run("point_encoder", {"pixel_values": pixels}, "encoder",
                                photo=photo.path.name)
        head, _ = self._run("people_sky_head", {"image_embed": features[0]}, "head",
                            photo=photo.path.name)
        return PhotoState(photo=photo, features=tuple(features),
                          head_outputs=dict(zip(self.hm.OUTPUT_NAMES, head)))

    # Decoder.

    def decode(self, state: PhotoState, coords: list[list[float]], labels: list[float],
               mask_input: np.ndarray | None, purpose: str) -> DecoderRun:
        """One decoder call. `coords` are model-input pixels (normalized x 1024), box corners
        first with labels 2 and 3; `mask_input` is the previous selected 256 x 256 logits."""
        size = self.graphs.LOW_RES_MASK_SIZE
        image_embed, feats_0, feats_1 = state.features
        inputs = {
            "image_embed": image_embed, "high_res_feats_0": feats_0, "high_res_feats_1": feats_1,
            "point_coords": np.array([coords], np.float32),
            "point_labels": np.array([labels], np.float32),
            "mask_input": (np.asarray(mask_input, np.float32)[None, None] if mask_input is not None
                           else np.zeros((1, 1, size, size), np.float32)),
            "has_mask_input": np.array([1.0 if mask_input is not None else 0.0], np.float32),
        }
        (logits, scores, _), milliseconds = self._run(
            "point_decoder", inputs, "decoder", photo=state.photo.path.name, purpose=purpose,
            points=len(coords), mask_input=mask_input is not None)
        return DecoderRun(inputs=inputs, logits=logits[0], scores=scores[0],
                          milliseconds=milliseconds)

    def single_mask(self, run: DecoderRun) -> np.ndarray:
        """The E1 selection for a box or several points (token 0 or the stability fallback),
        clipped to +-32 as the measurement scripts do."""
        from compare_onnx_outputs import LOGIT_LIMIT, select_single_mask

        selected, _ = select_single_mask(run.logits, run.scores)
        return np.clip(selected, -LOGIT_LIMIT, LOGIT_LIMIT)

    # One-click selections.

    def select_sky(self, state: PhotoState) -> SkyResult:
        """P1 section 3.3: found when presence >= the run 2 threshold; the 3 most probable cells
        of each 8-connected region of >= 3 cells; the union of the region masks."""
        from evaluate_quality_photos import (SKY_POINTS_PER_REGION, region_points, sigmoid,
                                             sky_regions)

        presence = float(sigmoid(state.head_outputs["presence_logits"][0])[0])
        threshold = self.thresholds["sky_presence"]
        if presence < threshold:
            return SkyResult(False, False, presence, threshold,
                             f"No sky found: sky presence {presence:.3f} is below {threshold:.3f}.")
        probability = sigmoid(state.head_outputs["sky_logits"][0, 0])
        regions = sky_regions(probability, SKY_POINTS_PER_REGION)
        if not regions:
            return SkyResult(True, False, presence, threshold,
                             f"No sky found: presence {presence:.3f}, but no region of at least "
                             f"{SKY_POINTS_PER_REGION} cells with sky probability >= 0.5.")
        result = SkyResult(True, True, presence, threshold,
                           f"Sky: {len(regions)} region(s), presence {presence:.3f}.")
        for index, region in enumerate(regions):
            points = region_points(probability, region)
            run = self.decode(state, points, [1.0] * len(points), None, f"sky region {index}")
            result.regions.append({"cells": int(region.sum()), "points": points,
                                   "logits": self.single_mask(run)})
        return result

    def select_people(self, state: PhotoState) -> PeopleResult:
        """P1 section 3.3: found when presence >= the run 2 threshold; decode_people with
        nms_iou = 0.5; the People list holds the boxes with a score >= the list threshold (run 2
        COCO `val` best F1); one box prompt per person, clipped to the photo."""
        import torch
        from evaluate_quality_photos import sigmoid

        presence = float(sigmoid(state.head_outputs["presence_logits"][0])[1])
        threshold = self.thresholds["person_presence"]
        list_threshold = self.thresholds["people_list"]
        torch_outputs = {k: torch.from_numpy(v) for k, v in state.head_outputs.items()}
        decoded = self.hm.decode_people(torch_outputs, nms_iou=0.5)[0]
        all_boxes = [{"rank": rank, "score": float(score), "box_cells": [float(v) for v in box]}
                     for rank, (box, score) in enumerate(zip(decoded["boxes"].tolist(),
                                                             decoded["scores"].tolist()), start=1)]
        if presence < threshold:
            return PeopleResult(False, False, presence, threshold, list_threshold,
                                f"No people found: person presence {presence:.3f} is below "
                                f"{threshold:.3f}.", all_boxes=all_boxes)
        listed = [entry for entry in all_boxes if entry["score"] >= list_threshold]
        if not listed:
            return PeopleResult(True, False, presence, threshold, list_threshold,
                                f"No people found: presence {presence:.3f}, but no person box "
                                f"has a score >= {list_threshold:.3f}.", all_boxes=all_boxes)
        result = PeopleResult(True, True, presence, threshold, list_threshold,
                              f"People: {len(listed)} person(s), presence {presence:.3f}.",
                              all_boxes=all_boxes)
        for entry in listed:
            model_box = [min(max(v, 0.0), float(fl.GRID)) * fl.CELL_MODEL_PIXELS
                         for v in entry["box_cells"]]
            run = self.decode(state, [model_box[:2], model_box[2:]], [2.0, 3.0], None,
                              f"person rank {entry['rank']}")
            result.people.append({**entry, "box_model": model_box,
                                  "logits": self.single_mask(run)})
        return result

    def select_subject(self, state: PhotoState) -> np.ndarray:
        """Subject alpha in [0, 1] (no prompt). Background is 1 - alpha. BiRefNet lite: the
        [1024, 1024] alpha. S3OD: the mask with the largest IoU score, unpadded (the photo aspect
        with the long edge 1024). IS-Net: the [1024, 1024] d1 output scaled with its minimum and
        maximum. The stage name of the record is the graph kind."""
        if state.subject_alpha is None:
            if self.subject_kind == "s3od":
                pixels, info = self.s3od.letterbox(np.asarray(state.photo.image),
                                                   self.subject_size)
                (masks, iou_scores), _ = self._run("subject_matte", {"pixel_values": pixels},
                                                   "s3od", photo=state.photo.path.name)
                alpha, _ = self.s3od.select_mask(masks, iou_scores)
                state.subject_alpha = self.s3od.unpad(alpha[None], info)[0]
                state.subject_pad_info = info
            elif self.subject_kind == "isnet":
                pixels = self.isnet.preprocess(np.asarray(state.photo.image), self.subject_size)
                (d1,), _ = self._run("subject_matte", {"pixel_values": pixels}, "isnet",
                                     photo=state.photo.path.name)
                state.subject_alpha = self.isnet.normalize_range(d1[0, 0])
            else:
                pixels = self.birefnet.preprocess(state.photo.image, self.subject_size).numpy()
                (alpha,), _ = self._run("subject_matte", {"pixel_values": pixels}, "birefnet",
                                        photo=state.photo.path.name)
                state.subject_alpha = alpha[0, 0]
        return state.subject_alpha

    def subject_alpha_at_photo(self, state: PhotoState, width: int | None = None,
                               height: int | None = None) -> np.ndarray:
        alpha = self.select_subject(state)
        width, height = width or state.photo.width, height or state.photo.height
        if self.subject_kind == "s3od":
            return self.s3od.masks_to_photo(alpha[None], height, width)[0].clip(0.0, 1.0)
        if self.subject_kind == "isnet":
            return self.isnet.alpha_to_photo(alpha, height, width).clip(0.0, 1.0)
        return self.birefnet.alpha_to_photo(alpha, width, height)

    def new_selection(self, state: PhotoState,
                      stroke_point_count: int = STROKE_POINT_COUNT) -> "PromptSelection":
        return PromptSelection(self, state, stroke_point_count)


class NothingToSubtract(ValueError):
    """A subtract stroke or click without a selection."""


class PromptSelection:
    """Stroke, box, and click selection with add and subtract (F5, F6; E3 section 9).

    The selection keeps all points (normalized photo coordinates) and labels (1 add, 0 subtract),
    an optional box, and the selected logits of the previous decoder run. Each interaction
    decodes once: the box corners (labels 2, 3) first, then all points, with the previous selected
    logits as `mask_input`. One point without a box is a click: the multimask output with the best
    predicted IoU (E4 default), and the three tokens 1 to 3 are the levels from small to large for
    a level change without a model run. Otherwise the single-mask selection of E1.
    """

    def __init__(self, pipeline: MaskPipeline, state: PhotoState, stroke_point_count: int) -> None:
        self.pipeline = pipeline
        self.state = state
        self.stroke_point_count = stroke_point_count
        self.points: list[list[float]] = []
        self.labels: list[int] = []
        self.box: list[float] | None = None
        self.logits: np.ndarray | None = None
        self.levels: list[dict] = []
        self.level_index: int | None = None
        self.steps: list[dict] = []
        self.last_run: DecoderRun | None = None

    @property
    def empty(self) -> bool:
        return self.logits is None

    def add_stroke(self, polyline: list[list[float]], subtract: bool) -> np.ndarray:
        """A stroke (normalized pointer path) or a click (one point)."""
        photo = self.state.photo
        points = stroke_points(polyline, self.stroke_point_count, photo.width, photo.height)
        return self.add_points(points, subtract, "click" if len(points) == 1 else "stroke")

    def add_points(self, points: list[list[float]], subtract: bool, interaction: str) -> np.ndarray:
        if subtract and self.empty:
            raise NothingToSubtract("Nothing to subtract: make a selection first.")
        self.points += [list(p) for p in points]
        self.labels += [0 if subtract else 1] * len(points)
        return self._decode(interaction, {"points": [list(p) for p in points],
                                          "label": 0 if subtract else 1})

    def set_box(self, box: list[float]) -> np.ndarray:
        """A box (normalized x0, y0, x1, y1) replaces the box of the selection."""
        x0, y0, x1, y1 = box
        self.box = [min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)]
        return self._decode("box", {"box": list(self.box)})

    def _decode(self, interaction: str, step: dict) -> np.ndarray:
        prompt = {"points": self.points, "labels": self.labels, "box": self.box}
        coords, labels = self.pipeline.graphs.model_prompt(prompt)
        run = self.pipeline.decode(self.state, coords, [float(v) for v in labels], self.logits,
                                   interaction)
        self.last_run = run
        if self.box is None and len(self.points) == 1:
            from compare_onnx_outputs import LOGIT_LIMIT

            tokens = []
            for token in (1, 2, 3):
                logits = np.clip(run.logits[token], -LOGIT_LIMIT, LOGIT_LIMIT)
                tokens.append({"token": token, "score": float(run.scores[token]),
                               "area": int((logits > 0).sum()), "logits": logits})
            self.levels = sorted(tokens, key=lambda t: (t["area"], t["token"]))
            best = max(tokens, key=lambda t: t["score"])
            self.level_index = self.levels.index(best)
            self.logits = best["logits"]
            step["token"] = best["token"]
        else:
            self.levels, self.level_index = [], None
            self.logits = self.pipeline.single_mask(run)
        self.steps.append({"interaction": interaction, **step,
                           "decoder_points": len(coords),
                           "mask_input": bool(run.inputs["has_mask_input"][0]),
                           "decoder_ms": round(run.milliseconds, 2)})
        return self.logits

    def change_level(self, delta: int) -> bool:
        """Move a click selection to the next smaller (-1) or larger (+1) multimask output. No
        model run. Returns False when the selection has no levels or the end is reached."""
        if self.level_index is None:
            return False
        index = self.level_index + delta
        if not 0 <= index < len(self.levels):
            return False
        self.level_index = index
        self.logits = self.levels[index]["logits"]
        self.steps.append({"interaction": "level", "token": self.levels[index]["token"]})
        return True

    def record(self) -> dict:
        return {"points": self.points, "labels": self.labels, "box": self.box,
                "stroke_point_count": self.stroke_point_count, "steps": self.steps}


def adjust_mask(mask: np.ndarray, edge_px: float, feather_px: float) -> np.ndarray:
    """F8 Feather and Edge on a mask in [0, 1] (no model).

    Edge > 0 dilates and edge < 0 erodes by |edge| px: |edge| alternating 3 x 3 cross and square
    steps (an octagon, the dilation of a disk within 8 percent). Feather is a Gaussian blur with
    sigma = feather px. The values are in pixels of the mask that is adjusted."""
    import cv2

    result = np.ascontiguousarray(mask, np.float32)
    steps = int(round(abs(edge_px)))
    if steps:
        operation = cv2.dilate if edge_px > 0 else cv2.erode
        kernels = (cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3)),
                   cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
        for index in range(steps):
            result = operation(result, kernels[index % 2], borderType=cv2.BORDER_REPLICATE)
    if feather_px > 0:
        result = cv2.GaussianBlur(result, (0, 0), sigmaX=float(feather_px),
                                  borderType=cv2.BORDER_REPLICATE)
    return np.clip(result, 0.0, 1.0)


def save_mask_png(mask: np.ndarray, path: Path) -> None:
    """8-bit grayscale PNG (255 = inside) at the size of `mask`."""
    from PIL import Image

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.round(np.clip(mask, 0.0, 1.0) * 255).astype(np.uint8), mode="L").save(path)


def save_json(path: Path, data) -> None:
    fl.write_json(Path(path), data)


def to_jsonable(value):
    """Prompt records without the logit arrays."""
    if isinstance(value, dict):
        return {k: to_jsonable(v) for k, v in value.items() if k != "logits"}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


if __name__ == "__main__":
    print(json.dumps({name: str(path) for name, path in GraphPaths.default().files().items()},
                     indent=2))
