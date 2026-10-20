"""Validation measurements of the people and sky head (README section 5, step 1).

  sky_cell_iou          IoU over all valid cells of the split: predicted sky probability >= 0.5
                        against a sky share >= 0.5
  presence_auc_*        area under the ROC curve of the presence output (ignored labels skipped)
  map_max_auc_*         the same with the map maximum as the score (sky probability, center
                        heatmap), for comparison with the presence output
  presence_best_f1_*    the threshold with the highest F1 of "found", with its precision and recall
  person_ap50           average precision of the person boxes at IoU 0.5 (all-point interpolation).
                        A detection without a match that has at least half of its area inside a
                        crowd box is not counted
  person_top1_hit       share of the images with person presence 1 whose highest-scoring box has
                        IoU >= 0.5 with a person (the "select one person" case of decision U5)
"""

from __future__ import annotations

import numpy as np
import torch

import head_model as hm

IOU_MATCH = 0.5
CROWD_AREA_SHARE = 0.5


def roc_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Mann-Whitney AUC; ties count one half."""
    positives, negatives = int((labels == 1).sum()), int((labels == 0).sum())
    if positives == 0 or negatives == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    _, first, counts = np.unique(scores[order], return_index=True, return_counts=True)
    ranks = np.empty(len(scores))
    ranks[order] = np.repeat(first + (counts + 1) / 2, counts)
    return float((ranks[labels == 1].sum() - positives * (positives + 1) / 2)
                 / (positives * negatives))


def best_f1(scores: np.ndarray, labels: np.ndarray) -> dict:
    """Threshold on sigmoid scores with the highest F1 of the positive class."""
    order = np.argsort(-scores, kind="mergesort")
    sorted_scores, sorted_labels = scores[order], labels[order]
    true_positive = np.cumsum(sorted_labels == 1)
    predicted = np.arange(1, len(scores) + 1)
    total_positive = max(int((labels == 1).sum()), 1)
    precision = true_positive / predicted
    recall = true_positive / total_positive
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    best = int(np.argmax(f1))
    return {"threshold": float(sorted_scores[best]), "f1": float(f1[best]),
            "precision": float(precision[best]), "recall": float(recall[best])}


def box_iou(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """[N, 4] x [M, 4] -> [N, M]."""
    top_left = torch.maximum(a[:, None, :2], b[None, :, :2])
    bottom_right = torch.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = (bottom_right - top_left).clamp(min=0).prod(2)
    area_a = (a[:, 2:] - a[:, :2]).clamp(min=0).prod(1)
    area_b = (b[:, 2:] - b[:, :2]).clamp(min=0).prod(1)
    return inter / (area_a[:, None] + area_b[None, :] - inter).clamp(min=1e-9)


def inside_share(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Share of the area of each box of a [N, 4] inside each box of b [M, 4] -> [N, M]."""
    top_left = torch.maximum(a[:, None, :2], b[None, :, :2])
    bottom_right = torch.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = (bottom_right - top_left).clamp(min=0).prod(2)
    area_a = (a[:, 2:] - a[:, :2]).clamp(min=0).prod(1)
    return inter / area_a[:, None].clamp(min=1e-9)


def average_precision(records: list[tuple[float, int]], total_ground_truth: int) -> float:
    """records: (score, 1 true positive / 0 false positive), ignored detections removed."""
    if total_ground_truth == 0:
        return float("nan")
    records = sorted(records, key=lambda r: -r[0])
    hits = np.array([r[1] for r in records], np.float64)
    true_positive = np.cumsum(hits)
    precision = true_positive / np.arange(1, len(hits) + 1)
    recall = true_positive / total_ground_truth
    precision = np.concatenate([[0.0], precision, [0.0]])
    recall = np.concatenate([[0.0], recall, [recall[-1] if len(recall) else 0.0]])
    for i in range(len(precision) - 2, -1, -1):
        precision[i] = max(precision[i], precision[i + 1])
    steps = np.where(recall[1:] != recall[:-1])[0]
    return float(np.sum((recall[steps + 1] - recall[steps]) * precision[steps + 1]))


class Evaluation:
    """Accumulates the measurements over the validation batches."""

    def __init__(self) -> None:
        self.sky_intersection = 0
        self.sky_union = 0
        self.presence_scores: list[np.ndarray] = []
        self.map_max_scores: list[np.ndarray] = []
        self.presence_labels: list[np.ndarray] = []
        self.box_records: list[tuple[float, int]] = []
        self.total_persons = 0
        self.top1_hits = 0
        self.top1_images = 0

    def add(self, outputs: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> None:
        valid = batch["sky_valid"]
        predicted = torch.sigmoid(outputs["sky_logits"][:, 0]) >= 0.5
        target = batch["sky_coverage"] >= 128
        self.sky_intersection += int((predicted & target & valid).sum())
        self.sky_union += int(((predicted | target) & valid).sum())

        self.presence_scores.append(torch.sigmoid(outputs["presence_logits"]).cpu().numpy())
        self.map_max_scores.append(torch.stack([
            torch.sigmoid(outputs["sky_logits"]).amax((1, 2, 3)),
            torch.sigmoid(outputs["center_logits"]).amax((1, 2, 3))], dim=1).cpu().numpy())
        self.presence_labels.append(batch["presence"].cpu().numpy())

        detections = hm.decode_people(outputs)
        for b, found in enumerate(detections):
            persons = batch["person_boxes"][b]
            persons = persons[(persons[:, 2:] > persons[:, :2]).all(1)]
            crowds = batch["crowd_boxes"][b]
            crowds = crowds[(crowds[:, 2:] > crowds[:, :2]).all(1)]
            self.total_persons += len(persons)
            matched = torch.zeros(len(persons), dtype=torch.bool, device=persons.device)
            ious = box_iou(found["boxes"], persons) if len(persons) else None
            crowd_share = inside_share(found["boxes"], crowds) if len(crowds) else None
            for d in range(len(found["boxes"])):
                score = float(found["scores"][d])
                if ious is not None:
                    best = int(ious[d].argmax())
                    if ious[d, best] >= IOU_MATCH and not matched[best]:
                        matched[best] = True
                        self.box_records.append((score, 1))
                        continue
                if crowd_share is not None and crowd_share[d].max() >= CROWD_AREA_SHARE:
                    continue
                self.box_records.append((score, 0))
            if int(batch["presence"][b, 1]) == 1:
                self.top1_images += 1
                if ious is not None and len(found["boxes"]) and ious[0].max() >= IOU_MATCH:
                    self.top1_hits += 1

    def result(self) -> dict:
        presence_scores = np.concatenate(self.presence_scores)
        map_max_scores = np.concatenate(self.map_max_scores)
        labels = np.concatenate(self.presence_labels)
        result = {"sky_cell_iou": self.sky_intersection / max(self.sky_union, 1)}
        for k, name in enumerate(("sky", "person")):
            keep = labels[:, k] >= 0
            result[f"presence_auc_{name}"] = roc_auc(presence_scores[keep, k], labels[keep, k])
            result[f"map_max_auc_{name}"] = roc_auc(map_max_scores[keep, k], labels[keep, k])
            result[f"presence_best_f1_{name}"] = best_f1(presence_scores[keep, k], labels[keep, k])
        result["person_ap50"] = average_precision(self.box_records, self.total_persons)
        result["person_top1_hit"] = self.top1_hits / max(self.top1_images, 1)
        result["counts"] = {"persons": self.total_persons, "top1_images": self.top1_images,
                            "sky_union_cells": self.sky_union}
        return result


def selection_score(result: dict) -> float:
    """Checkpoint selection: mean of the sky cell IoU and the person AP50."""
    return 0.5 * (result["sky_cell_iou"] + result["person_ap50"])
