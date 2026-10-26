"""Shared layout of the sky matte label step: panoptic decoding, trimap, and file names.

The sky matte labels are the fine sky training targets of the sky head (README decision U17).
The coarse COCO 2017 panoptic sky segment gives a trimap; a classical matting algorithm without
learned weights (PyMatting) computes the sky alpha in the unknown area of the trimap.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage
from scipy.spatial import cKDTree

SKY_CATEGORY = "sky-other-merged"
TREE_CATEGORIES = ("tree-merged",)
BUILDING_CATEGORIES = ("building-other-merged", "house", "roof", "bridge")
# Photos with these segments get no sky color rule for the trees: snow on branches has the color
# of an overcast sky.
SKY_COLORED_CATEGORIES = ("snow",)

# COCO license ids that permit commercial use (the same rule as P1, decision U2):
# 4 CC BY 2.0, 5 CC BY-SA 2.0, 7 no known copyright restrictions, 8 United States Government Work.
ALLOWED_LICENSES = (4, 5, 7, 8)

# Trimap widths as a fraction of the long edge of the photo (16 px and 48 px at 1024 px).
BOUNDARY_BAND_FRACTION = 1.0 / 64.0
VOID_DEPTH_FRACTION = 3.0 / 64.0

# Sky color rule. The sky color distance of a pixel is the RGB distance (in [0, 1]) to its
# SKY_COLOR_NEIGHBORS-th nearest of SKY_COLOR_SAMPLES definite sky pixels. The k-th neighbor, not
# the nearest, because the coarse COCO sky also covers branches and wires: a color is a sky color
# only when many sky samples are near it. More than SKY_COLOR_FAR is not sky; less than
# SKY_COLOR_NEAR is sky.
SKY_COLOR_FAR = 0.25
SKY_COLOR_NEAR = 0.03
SKY_COLOR_SAMPLES = 4096
SKY_COLOR_NEIGHBORS = 400
SKY_COLOR_SEED = 0
# The sky samples are only coarse sky pixels with a flat neighborhood (standard deviation of the
# luminance in a SKY_FLAT_WINDOW x SKY_FLAT_WINDOW window below SKY_FLAT_STD): the sky is smooth,
# branches and foliage under the coarse sky polygons are not.
SKY_FLAT_WINDOW = 5
SKY_FLAT_STD = 0.02
# A dark sky (median luminance of the samples below DARK_SKY_LUMINANCE, for example at night) has
# the colors of dark foliage, so the tree components get no sky color rule.
DARK_SKY_LUMINANCE = 0.2
# Pixels of the coarse sky with a color that is not sky, and this ring around them (as a fraction
# of the long edge, 4 px at 1024 px), are unknown.
SKY_OUTLIER_RING_FRACTION = 1.0 / 256.0

# add_enclosed_void_to_sky: largest unlabeled component (share of the photo) that becomes sky.
VOID_TO_SKY_MAX_SHARE = 0.01

TRIMAP_BACKGROUND = 0.0
TRIMAP_UNKNOWN = 0.5
TRIMAP_SKY = 1.0


def panoptic_segment_ids(png: Image.Image) -> np.ndarray:
    """COCO panoptic PNG -> segment id per pixel (id = R + 256 G + 256^2 B)."""
    rgb = np.asarray(png.convert("RGB"), dtype=np.uint32)
    return rgb[..., 0] + 256 * rgb[..., 1] + 256 * 256 * rgb[..., 2]


def read_panoptic_png(archive: zipfile.ZipFile, file_name: str) -> np.ndarray:
    """Segment ids of one photo from panoptic_train2017.zip."""
    with archive.open(f"panoptic_train2017/{file_name}") as file:
        return panoptic_segment_ids(Image.open(io.BytesIO(file.read())))


def category_mask(segment_ids: np.ndarray, segments_info: list[dict],
                  category_ids: set[int]) -> np.ndarray:
    ids = [s["id"] for s in segments_info if s["category_id"] in category_ids]
    return np.isin(segment_ids, ids)


def resize_mask(mask: np.ndarray, width: int, height: int) -> np.ndarray:
    """Nearest-neighbor resize of a boolean annotation mask to the photo size."""
    image = Image.fromarray(mask.astype(np.uint8) * 255)
    return np.asarray(image.resize((width, height), Image.Resampling.NEAREST)) > 127


def add_enclosed_void_to_sky(sky: np.ndarray, void: np.ndarray) -> np.ndarray:
    """Coarse sky + each small unlabeled component whose outer ring is at least half sky.

    COCO-Stuff leaves unlabeled strokes inside the sky (thin things, label gaps). As unknown
    area they give alpha artifacts in clear sky; as sky they follow the annotation intent. A
    component larger than VOID_TO_SKY_MAX_SHARE of the photo is an object without a COCO class
    (for example a balloon) and stays unlabeled."""
    components, count = ndimage.label(void, structure=np.ones((3, 3), dtype=bool))
    if count == 0:
        return sky
    ring = ndimage.binary_dilation(void, structure=np.ones((3, 3), dtype=bool)) & ~void
    ring_labels = ndimage.grey_dilation(components, size=(3, 3))
    indexes = np.arange(1, count + 1)
    ring_sky = ndimage.sum(sky & ring, labels=ring_labels * ring, index=indexes)
    ring_size = ndimage.sum(ring, labels=ring_labels * ring, index=indexes)
    area = ndimage.sum(void, labels=components, index=indexes)
    small = area <= VOID_TO_SKY_MAX_SHARE * void.size
    enclosed = indexes[(ring_size > 0) & (ring_sky >= 0.5 * ring_size) & small]
    return sky | np.isin(components, enclosed)


def disk(radius: int) -> np.ndarray:
    y, x = np.ogrid[-radius:radius + 1, -radius:radius + 1]
    return x * x + y * y <= radius * radius


def sky_color_samples(image: np.ndarray, sky: np.ndarray) -> np.ndarray:
    """Up to SKY_COLOR_SAMPLES colors of the flat pixels of the sky mask (see SKY_FLAT_STD)."""
    luminance = image.mean(axis=2)
    mean = ndimage.uniform_filter(luminance, size=SKY_FLAT_WINDOW)
    mean_square = ndimage.uniform_filter(luminance * luminance, size=SKY_FLAT_WINDOW)
    flat = np.sqrt(np.maximum(mean_square - mean * mean, 0.0)) < SKY_FLAT_STD
    sky_colors = image[sky & flat]
    generator = np.random.default_rng(SKY_COLOR_SEED)
    count = min(SKY_COLOR_SAMPLES, len(sky_colors))
    return sky_colors[generator.choice(len(sky_colors), size=count, replace=False)]


def sky_color_distance(samples: np.ndarray, colors: np.ndarray) -> np.ndarray:
    """Sky color distance (see SKY_COLOR_NEIGHBORS) of the colors [N, 3]."""
    if len(samples) < SKY_COLOR_NEIGHBORS:
        return np.full(len(colors), np.inf)
    distance, _ = cKDTree(samples).query(colors, k=[SKY_COLOR_NEIGHBORS])
    return distance[:, 0]


def sky_trimap(image: np.ndarray, sky: np.ndarray, tree: np.ndarray, void: np.ndarray,
               sky_colored_material: bool) -> np.ndarray:
    """Trimap of the sky alpha at the photo size (1 sky, 0 not sky, 0.5 unknown).

    image is RGB in [0, 1]. COCO-Stuff often labels a whole bare or sparse tree, with the sky
    between the branches, as one tree segment, and its sky polygons cover branches and wires.
    So the trimap also uses the sky color distance (SKY_COLOR_NEIGHBORS):

    sky     = coarse sky outside the band of BOUNDARY_BAND_FRACTION
    unknown = the band on both sides of the coarse sky edge
            + unlabeled pixels within VOID_DEPTH_FRACTION of the coarse sky
            + the pixels of each tree component that comes within the band of the coarse sky
            + coarse sky pixels with a distance > SKY_COLOR_FAR, and a ring of
              SKY_OUTLIER_RING_FRACTION around them
    sky_colored_material is True when the photo has a SKY_COLORED_CATEGORIES segment.
    sky color samples: flat coarse sky pixels (SKY_FLAT_STD). With sky_colored_material or a dark
    sky (DARK_SKY_LUMINANCE), the tree components next to the sky get no color rule and stay not
    sky outside the band. Otherwise, in the tree components next to the sky:
      not sky   distance > SKY_COLOR_FAR (the cores of branches and leaves; no erosion, so thin
                branches keep a constraint)
      sky       distance < SKY_COLOR_NEAR, eroded by 1 px (sky between the branches)
    """
    long_edge = max(sky.shape)
    band = max(1, round(BOUNDARY_BAND_FRACTION * long_edge))
    depth = max(band, round(VOID_DEPTH_FRACTION * long_edge))
    distance_to_sky = ndimage.distance_transform_edt(~sky)
    sky_definite = ndimage.binary_erosion(sky, structure=disk(band), border_value=1)
    band_area = (distance_to_sky <= band) & ~sky_definite
    void_area = void & (distance_to_sky <= depth)

    components, count = ndimage.label(tree, structure=np.ones((3, 3), dtype=bool))
    touching = np.unique(components[(distance_to_sky <= band) & tree])
    tree_near_sky = np.isin(components, touching[touching > 0]) if count else tree
    samples = sky_color_samples(image, sky_definite)
    color_rule = len(samples) >= SKY_COLOR_NEIGHBORS
    if (not color_rule or sky_colored_material
            or np.median(samples.mean(axis=1)) < DARK_SKY_LUMINANCE):
        tree_near_sky = np.zeros_like(tree)
    checked = tree_near_sky | sky_definite
    color_distance = np.zeros(sky.shape)
    if color_rule:
        color_distance[checked] = sky_color_distance(samples, image[checked])
    far = color_distance > SKY_COLOR_FAR
    not_sky_color = tree_near_sky & far
    sky_color = ndimage.binary_erosion(tree_near_sky & (color_distance < SKY_COLOR_NEAR),
                                       structure=np.ones((3, 3), dtype=bool))
    ring = max(1, round(SKY_OUTLIER_RING_FRACTION * long_edge))
    sky_outliers = sky_definite & ndimage.binary_dilation(sky_definite & far,
                                                          structure=disk(ring))

    trimap = np.full(sky.shape, TRIMAP_BACKGROUND, dtype=np.float64)
    trimap[sky_definite & ~sky_outliers] = TRIMAP_SKY
    trimap[band_area | void_area | tree_near_sky | sky_outliers] = TRIMAP_UNKNOWN
    trimap[not_sky_color] = TRIMAP_BACKGROUND
    trimap[sky_color] = TRIMAP_SKY
    return trimap


def photo_path(photo_dir: Path, image_id: int) -> Path:
    return photo_dir / f"{image_id:012d}.jpg"


def label_dir(out_dir: Path, image_id: int) -> Path:
    return out_dir / "labels" / f"{image_id:012d}"
