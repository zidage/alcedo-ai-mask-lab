"""Measurements of a sky alpha against the target at 1024 x 1024 (issue #10).

Per photo, only on the pixels with weight 1. A sky photo has target >= 0.5 at one or more
pixels; a photo without sky has none.

  iou              IoU of (alpha >= 0.5) and (target >= 0.5) (sky photos only)
  mae              mean |alpha - target|
  edge_sad         sum |alpha - target| / 1000 in the edge band (the matting SAD)
  edge_mse         mean (alpha - target)^2 in the edge band
  edge_gradient    mean |grad(alpha) - grad(target)| in the edge band
  boundary_f1      F1 of the boundary pixels of (alpha >= 0.5) and (target >= 0.5) with a match
                   distance of BOUNDARY_TOLERANCE px (sky photos only)
  false_sky_share  share of the pixels with alpha >= 0.5 (photos without sky only)

The edge band is the target pixels with 0 < target < 1 and the pixels within EDGE_BAND px of
a boundary pixel of (target >= 0.5): the area where the fine edge is.

IoU and boundary F1 are not used on photos without sky: one false sky pixel would set them to 0
and the mean would show the count of such photos, not the sky edge. false_sky_share measures
these photos.

The summary has the mean of each value over the counted photos.
"""

from __future__ import annotations

import numpy as np

EDGE_BAND = 8
BOUNDARY_TOLERANCE = 3


def boundary(mask: np.ndarray) -> np.ndarray:
    from scipy import ndimage

    return mask & ~ndimage.binary_erosion(mask, border_value=1)


def photo_measurements(alpha: np.ndarray, target: np.ndarray, weight: np.ndarray) -> dict:
    from scipy import ndimage

    use = weight >= 0.5
    pred, true = (alpha >= 0.5) & use, (target >= 0.5) & use
    has_sky = bool(true.any())
    result = {"iou": (np.count_nonzero(pred & true) / np.count_nonzero(pred | true)
                      if has_sky else None),
              "mae": float(np.abs(alpha - target)[use].mean()) if use.any() else None,
              "false_sky_share": (None if has_sky or not use.any()
                                  else np.count_nonzero(pred) / np.count_nonzero(use))}

    true_edge = boundary(target >= 0.5)
    distance_to_true = ndimage.distance_transform_edt(~true_edge) if true_edge.any() else None
    band = ((target > 0.0) & (target < 1.0)) | (
        distance_to_true <= EDGE_BAND if distance_to_true is not None else False)
    band &= use
    if band.any():
        difference = (alpha - target)[band]
        gy_a, gx_a = np.gradient(alpha)
        gy_t, gx_t = np.gradient(target)
        result["edge_sad"] = float(np.abs(difference).sum() / 1000.0)
        result["edge_mse"] = float((difference ** 2).mean())
        result["edge_gradient"] = float(
            (np.abs(gx_a - gx_t) + np.abs(gy_a - gy_t))[band].mean())
    else:
        result.update(edge_sad=None, edge_mse=None, edge_gradient=None)

    pred_edge = boundary(alpha >= 0.5) & use
    true_edge &= use
    if has_sky and (pred_edge.any() or true_edge.any()):
        if pred_edge.any() and true_edge.any():
            distance_to_pred = ndimage.distance_transform_edt(~pred_edge)
            precision = np.count_nonzero(
                distance_to_true[pred_edge] <= BOUNDARY_TOLERANCE) / np.count_nonzero(pred_edge)
            recall = np.count_nonzero(
                distance_to_pred[true_edge] <= BOUNDARY_TOLERANCE) / np.count_nonzero(true_edge)
            result["boundary_f1"] = (2 * precision * recall / (precision + recall)
                                     if precision + recall else 0.0)
        else:
            result["boundary_f1"] = 0.0
    else:
        result["boundary_f1"] = None
    return result


def summary(measurements: list[dict]) -> dict:
    keys = ("iou", "mae", "edge_sad", "edge_mse", "edge_gradient", "boundary_f1",
            "false_sky_share")
    out = {"photos": len(measurements)}
    for key in keys:
        values = [m[key] for m in measurements if m[key] is not None]
        out[key] = float(np.mean(values)) if values else None
        out[f"{key}_photos"] = len(values)
    return out
