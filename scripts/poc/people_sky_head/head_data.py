"""Shared data layout of the people and sky head (README section 5, step 1).

The scripts of this directory import this module:

  prepare_coco_targets.py : COCO panoptic annotations -> per-cell targets of one split
  prepare_openimages_targets.py : Open Images V7 point labels and boxes -> per-cell targets
  encode_image_embed.py   : SAM 2.1 tiny encoder -> image_embed cache of one split
  train_head.py           : image_embed cache + targets -> trained head
  export_head_onnx.py     : trained head -> head.onnx

The cell grid is the grid of `image_embed`: the photo is stretched to 1024 x 1024 (the SAM 2.1
preprocessing), so one cell is 1/64 of the photo width by 1/64 of the photo height. All targets
use these stretched grid coordinates: x in [0, 64) along the width, y in [0, 64) along the height.

One split directory holds:

  manifest.json         images (id, file name, license), counts, sources
  targets/*.npy         the arrays of TARGET_ARRAYS, first axis = image index of the manifest
  image_embed.npy       float16 [N, 256, 64, 64], written by encode_image_embed.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

GRID = 64
IMAGE_SIZE = 1024
EMBED_CHANNELS = 256
# Persons per image with box targets; images with more persons keep the largest ones.
MAX_PERSONS = 64
# Crowd boxes per image (COCO `iscrowd` person segments), used only by the evaluation.
MAX_CROWDS = 16
# Presence labels: 1 present, 0 absent, -1 ignored (too small to require a "found" result).
PRESENCE_IGNORE = -1
# Class order of the presence output.
PRESENCE_CLASSES = ("sky", "person")
# A sky area of at least this share of the photo is "sky present".
SKY_PRESENT_MIN_AREA = 0.01
# A person segment of at least one cell area is "person present".
PERSON_PRESENT_MIN_AREA = 1.0 / (GRID * GRID)
# Cells with more than this share of unlabeled pixels have no sky target.
SKY_MAX_UNLABELED = 0.5
# Smallest box side in cells for the log-size target (a person of 1 model pixel).
MIN_BOX_CELLS = 1.0 / 16.0
# CenterNet Gaussian radius parameter (CornerNet `gaussian_radius`).
GAUSSIAN_MIN_OVERLAP = 0.7

TARGET_ARRAYS = {
    # name: (dtype, shape after the image axis)
    "sky_coverage": (np.uint8, (GRID, GRID)),            # sky share of each cell x 255
    "sky_valid": (np.bool_, (GRID, GRID)),               # cell has a sky target
    "center_heatmap": (np.float16, (GRID, GRID)),        # person center Gaussian, peak 1
    "heatmap_valid": (np.bool_, (GRID, GRID)),           # cell counts in the center loss
    "person_cell": (np.int32, (MAX_PERSONS,)),           # flat cell index y * 64 + x, -1 = none
    "person_offset": (np.float32, (MAX_PERSONS, 2)),     # center - cell corner, in [0, 1)
    "person_log_size": (np.float32, (MAX_PERSONS, 2)),   # log(box width, height in cells)
    "person_boxes": (np.float32, (MAX_PERSONS, 4)),      # x0, y0, x1, y1 in cells
    "crowd_boxes": (np.float32, (MAX_CROWDS, 4)),        # x0, y0, x1, y1 in cells
    "presence": (np.int8, (len(PRESENCE_CLASSES),)),     # 1, 0, or PRESENCE_IGNORE
    # True: every cell of `sky_valid` has a sky share from a segmentation (COCO). False: the
    # valid cells are only the cells of point labels (Open Images), so the image has no Dice term
    # and its cells count in the point measurements.
    "sky_dense": (np.bool_, ()),
}


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as file:
        json.dump(data, file, indent=2)
        file.write("\n")


def read_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as file:
        return json.load(file)


def gaussian_radius(height: float, width: float, min_overlap: float = GAUSSIAN_MIN_OVERLAP) -> float:
    """CornerNet / CenterNet radius: a corner shift of this size keeps IoU >= min_overlap."""
    a1 = 1.0
    b1 = height + width
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    r1 = (b1 + math.sqrt(b1 ** 2 - 4 * a1 * c1)) / 2
    a2 = 4.0
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    r2 = (b2 + math.sqrt(b2 ** 2 - 4 * a2 * c2)) / 2
    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    r3 = (b3 + math.sqrt(b3 ** 2 - 4 * a3 * c3)) / 2
    return min(r1, r2, r3)


def draw_gaussian(heatmap: np.ndarray, x: int, y: int, radius: int) -> None:
    """Element-wise maximum of `heatmap` and a Gaussian with peak 1 at cell (x, y)."""
    diameter = 2 * radius + 1
    sigma = diameter / 6.0
    offsets = np.arange(-radius, radius + 1, dtype=np.float32)
    gaussian = np.exp(-(offsets[None, :] ** 2 + offsets[:, None] ** 2) / (2 * sigma ** 2))
    height, width = heatmap.shape
    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)
    region = heatmap[y - top:y + bottom, x - left:x + right]
    patch = gaussian[radius - top:radius + bottom, radius - left:radius + right]
    np.maximum(region, patch, out=region)


def person_targets(boxes_cells: np.ndarray) -> dict[str, np.ndarray]:
    """Center heatmap and box targets of the non-crowd persons of one image.

    boxes_cells: [P, 4] x0, y0, x1, y1 in cells, sorted by decreasing area. Two persons with the
    same center cell share the heatmap peak; the box target of that cell is the larger person.
    """
    heatmap = np.zeros((GRID, GRID), np.float32)
    cells = np.full((MAX_PERSONS,), -1, np.int32)
    offsets = np.zeros((MAX_PERSONS, 2), np.float32)
    log_sizes = np.zeros((MAX_PERSONS, 2), np.float32)
    used_cells: set[int] = set()
    slot = 0
    for x0, y0, x1, y1 in boxes_cells[:MAX_PERSONS]:
        width, height = x1 - x0, y1 - y0
        center_x, center_y = (x0 + x1) / 2, (y0 + y1) / 2
        cell_x = min(int(center_x), GRID - 1)
        cell_y = min(int(center_y), GRID - 1)
        radius = max(0, int(gaussian_radius(math.ceil(height), math.ceil(width))))
        draw_gaussian(heatmap, cell_x, cell_y, radius)
        flat = cell_y * GRID + cell_x
        if flat in used_cells:
            continue
        used_cells.add(flat)
        cells[slot] = flat
        offsets[slot] = (center_x - cell_x, center_y - cell_y)
        log_sizes[slot] = (math.log(max(width, MIN_BOX_CELLS)), math.log(max(height, MIN_BOX_CELLS)))
        slot += 1
    return {"center_heatmap": heatmap, "person_cell": cells, "person_offset": offsets,
            "person_log_size": log_sizes}


def target_dir(split_dir: Path) -> Path:
    return Path(split_dir) / "targets"


def load_targets(split_dir: Path) -> dict[str, np.ndarray]:
    """All target arrays of one split, read into memory (about 50 KB per image)."""
    arrays = {}
    for name, (dtype, shape) in TARGET_ARRAYS.items():
        array = np.load(target_dir(split_dir) / f"{name}.npy")
        if array.dtype != dtype or array.shape[1:] != shape:
            raise ValueError(f"{split_dir}/{name}.npy: {array.dtype} {array.shape}, "
                             f"expected {np.dtype(dtype)} [N, {shape}]")
        arrays[name] = array
    counts = {name: len(array) for name, array in arrays.items()}
    if len(set(counts.values())) != 1:
        raise ValueError(f"{split_dir}: target arrays have different image counts {counts}")
    return arrays


def embed_path(split_dir: Path) -> Path:
    return Path(split_dir) / "image_embed.npy"


class EmbedDataset:
    """image_embed (float32 [256, 64, 64]) and the targets of one image, as torch tensors.

    The image_embed cache is a float16 memory map; each worker opens it on first access.
    """

    def __init__(self, split_dir: Path, require_complete: bool = True) -> None:
        self.split_dir = Path(split_dir)
        self.targets = load_targets(self.split_dir)
        self.count = len(self.targets["presence"])
        progress = read_json(self.split_dir / "image_embed_progress.json")
        if require_complete and progress["done"] != self.count:
            raise ValueError(f"{self.split_dir}: image_embed has {progress['done']} of "
                             f"{self.count} images; run encode_image_embed.py first")
        self._embeds: np.ndarray | None = None

    def __len__(self) -> int:
        return self.count

    def __getitem__(self, index: int) -> dict:
        import torch

        if self._embeds is None:
            self._embeds = np.load(embed_path(self.split_dir), mmap_mode="r")
            if self._embeds.shape != (self.count, EMBED_CHANNELS, GRID, GRID):
                raise ValueError(f"{embed_path(self.split_dir)}: shape {self._embeds.shape}")
        item = {"image_embed": torch.from_numpy(np.asarray(self._embeds[index], np.float32)),
                "index": index}
        for name, array in self.targets.items():
            value = array[index]
            if value.dtype == np.float16:
                value = value.astype(np.float32)
            # np.array keeps a 0-d value (sky_dense) 0-d; np.ascontiguousarray makes it [1].
            item[name] = torch.from_numpy(np.array(value))
        return item


class BlockShuffleSampler:
    """Random order with local disk reads: shuffle blocks of consecutive images, then the images
    inside each block. The images of a split are already in random order (prepare step), so a
    block is a random subset of the split."""

    def __init__(self, count: int, block: int, seed: int) -> None:
        self.count, self.block, self.seed, self.epoch = count, block, seed, 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return self.count

    def __iter__(self):
        generator = np.random.default_rng(self.seed + self.epoch)
        starts = np.arange(0, self.count, self.block)
        generator.shuffle(starts)
        for start in starts:
            indices = np.arange(start, min(start + self.block, self.count))
            generator.shuffle(indices)
            yield from indices.tolist()
