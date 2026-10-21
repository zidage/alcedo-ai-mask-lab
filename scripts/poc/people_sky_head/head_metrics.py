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
  person_ap50_nms       the same after the suppression of overlapping boxes (head_model.decode_people
                        with nms_iou = NMS_IOU). The top-1 box is never suppressed, so
                        person_top1_hit does not change
  person_list_best_f1_nms  the box score threshold with the highest F1 of the matched boxes after
                        the suppression (the threshold of the People list), with its precision and
                        recall. The recall counts all non-crowd persons
  person_box_fit_nms    per person height (box height in cells: small < 8, medium 8 to 24, large
                        >= 24): the count, the mean of the best IoU of a box after the suppression
                        with the person, the share with a best IoU >= 0.75, and, for the persons
                        with a best IoU >= 0.5, the median of log(box height / person height) and
                        log(box width / person width) (negative: the box is too small)
"""

from __future__ import annotations

import numpy as np
import torch

import head_model as hm

IOU_MATCH = 0.5
CROWD_AREA_SHARE = 0.5
NMS_IOU = 0.5
# Person height buckets in cells (64 cells = the photo height).
HEIGHT_BUCKETS = (("small", 0.0, 8.0), ("medium", 8.0, 24.0), ("large", 24.0, float("inf")))
TIGHT_IOU = 0.75


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


box_iou = hm.box_iou


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


def box_best_f1(records: list[tuple[float, int]], total_ground_truth: int) -> dict:
    """Box score threshold with the highest F1; records as in average_precision."""
    if not records or total_ground_truth == 0:
        return {"threshold": float("nan"), "f1": 0.0, "precision": 0.0, "recall": 0.0}
    records = sorted(records, key=lambda r: -r[0])
    scores = np.array([r[0] for r in records])
    true_positive = np.cumsum([r[1] for r in records])
    precision = true_positive / np.arange(1, len(records) + 1)
    recall = true_positive / total_ground_truth
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    best = int(np.argmax(f1))
    return {"threshold": float(scores[best]), "f1": float(f1[best]),
            "precision": float(precision[best]), "recall": float(recall[best])}


def match_boxes(found: dict[str, torch.Tensor], persons: torch.Tensor, crowds: torch.Tensor,
                records: list[tuple[float, int]]) -> bool:
    """Append (score, 1 matched / 0 false) per detection of one image, highest score first; a
    person matches once at IoU >= IOU_MATCH, and an unmatched detection inside a crowd box is
    skipped. Returns True when the highest-scoring detection has IoU >= IOU_MATCH with a person."""
    matched = torch.zeros(len(persons), dtype=torch.bool, device=persons.device)
    ious = box_iou(found["boxes"], persons) if len(persons) else None
    crowd_share = inside_share(found["boxes"], crowds) if len(crowds) else None
    for d in range(len(found["boxes"])):
        score = float(found["scores"][d])
        if ious is not None:
            best = int(ious[d].argmax())
            if ious[d, best] >= IOU_MATCH and not matched[best]:
                matched[best] = True
                records.append((score, 1))
                continue
        if crowd_share is not None and crowd_share[d].max() >= CROWD_AREA_SHARE:
            continue
        records.append((score, 0))
    return bool(ious is not None and len(found["boxes"]) and ious[0].max() >= IOU_MATCH)


class Evaluation:
    """Accumulates the measurements over the validation batches."""

    def __init__(self) -> None:
        self.sky_intersection = 0
        self.sky_union = 0
        self.presence_scores: list[np.ndarray] = []
        self.map_max_scores: list[np.ndarray] = []
        self.presence_labels: list[np.ndarray] = []
        self.box_records: list[tuple[float, int]] = []
        self.box_records_nms: list[tuple[float, int]] = []
        self.total_persons = 0
        self.top1_hits = 0
        self.top1_images = 0
        self.fit = {name: {"best_iou": [], "log_height": [], "log_width": []}
                    for name, _, _ in HEIGHT_BUCKETS}

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

        all_maxima = hm.decode_people(outputs)
        suppressed = hm.decode_people(outputs, nms_iou=NMS_IOU)
        for b in range(len(all_maxima)):
            persons = batch["person_boxes"][b]
            persons = persons[(persons[:, 2:] > persons[:, :2]).all(1)]
            crowds = batch["crowd_boxes"][b]
            crowds = crowds[(crowds[:, 2:] > crowds[:, :2]).all(1)]
            self.total_persons += len(persons)
            top1_hit = match_boxes(all_maxima[b], persons, crowds, self.box_records)
            match_boxes(suppressed[b], persons, crowds, self.box_records_nms)
            self.add_box_fit(suppressed[b]["boxes"], persons)
            if int(batch["presence"][b, 1]) == 1:
                self.top1_images += 1
                self.top1_hits += int(top1_hit)

    def add_box_fit(self, boxes: torch.Tensor, persons: torch.Tensor) -> None:
        if len(persons) == 0:
            return
        if len(boxes):
            best_iou, best = box_iou(persons, boxes).max(1)
        else:
            best_iou = torch.zeros(len(persons), device=persons.device)
            best = torch.zeros(len(persons), dtype=torch.long, device=persons.device)
        person_size = (persons[:, 2:] - persons[:, :2]).clamp(min=1e-6)
        for p in range(len(persons)):
            height = float(person_size[p, 1])
            name = next(n for n, low, high in HEIGHT_BUCKETS if low <= height < high)
            self.fit[name]["best_iou"].append(float(best_iou[p]))
            if float(best_iou[p]) >= IOU_MATCH:
                box_size = (boxes[best[p], 2:] - boxes[best[p], :2]).clamp(min=1e-6)
                ratio = torch.log(box_size / person_size[p])
                self.fit[name]["log_width"].append(float(ratio[0]))
                self.fit[name]["log_height"].append(float(ratio[1]))

    def box_fit(self) -> dict:
        result = {}
        for name, values in self.fit.items():
            best_iou = np.array(values["best_iou"])
            some = len(best_iou) > 0
            result[name] = {
                "persons": len(best_iou),
                "mean_best_iou": float(best_iou.mean()) if some else float("nan"),
                "share_best_iou_ge_0_75": float((best_iou >= TIGHT_IOU).mean()) if some
                else float("nan"),
                "matched": len(values["log_height"]),
                "median_log_height_ratio": float(np.median(values["log_height"]))
                if values["log_height"] else float("nan"),
                "median_log_width_ratio": float(np.median(values["log_width"]))
                if values["log_width"] else float("nan"),
            }
        return result

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
        result["person_ap50_nms"] = average_precision(self.box_records_nms, self.total_persons)
        result["person_list_best_f1_nms"] = box_best_f1(self.box_records_nms, self.total_persons)
        result["person_box_fit_nms"] = self.box_fit()
        result["counts"] = {"persons": self.total_persons, "top1_images": self.top1_images,
                            "sky_union_cells": self.sky_union}
        return result


def selection_score(result: dict) -> float:
    """Checkpoint selection: mean of the sky cell IoU and the person AP50."""
    return 0.5 * (result["sky_cell_iou"] + result["person_ap50"])
