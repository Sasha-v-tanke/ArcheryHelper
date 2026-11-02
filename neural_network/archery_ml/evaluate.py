"""Impact detection, localization and scoring metrics with one-to-one matching."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from statistics import median
from typing import Sequence

from neural_network.archery_ml.postprocess import ImpactPrediction
from neural_network.archery_ml.scoring import ImpactPoint, score
from neural_network.archery_ml.targets import TargetConfig, TargetFormat, TargetTemplate, TenRingMode, TripleLayout


@dataclass(frozen=True)
class EvaluationResult:
    precision: float
    recall: float
    f1: float
    true_positives: int
    false_positives: int
    false_negatives: int
    localization_median_ring: float | None
    localization_p95_ring: float | None
    exact_score_accuracy_non_line_call: float | None
    score_eligible_count: int
    false_positives_per_empty_face: float | None
    empty_faces: int
    correction_rate_proxy: float | None

    def to_dict(self) -> dict:
        return asdict(self)


def _match(
    predictions: Sequence[ImpactPrediction],
    truths: Sequence[tuple[float, float]],
    radius: float,
) -> list[tuple[int, int, float]]:
    """Maximum-cardinality bipartite matching; each prediction/GT used at most once."""
    edges: list[list[tuple[int, float]]] = []
    for prediction in predictions:
        neighbors = []
        for i, (x, y) in enumerate(truths):
            distance = math.hypot(prediction.x_norm - x, prediction.y_norm - y)
            if distance <= radius:
                neighbors.append((i, distance))
        edges.append(sorted(neighbors, key=lambda pair: (pair[1], pair[0])))
    owners: dict[int, int] = {}

    def augment(pred_index: int, seen: set[int]) -> bool:
        for truth_index, _ in edges[pred_index]:
            if truth_index in seen:
                continue
            seen.add(truth_index)
            if truth_index not in owners or augment(owners[truth_index], seen):
                owners[truth_index] = pred_index
                return True
        return False

    for index in range(len(predictions)):
        augment(index, set())
    return sorted(
        ((pred, truth, math.hypot(predictions[pred].x_norm - truths[truth][0],
                                  predictions[pred].y_norm - truths[truth][1]))
         for truth, pred in owners.items()),
        key=lambda entry: entry[1],
    )


def evaluate_faces(
    faces: Sequence[tuple[Sequence[ImpactPrediction], Sequence[tuple[float, float]], object | None]],
    *,
    match_radius_ring: float = 0.5,
    line_call_uncertainty_norm: float = 0.015,
    correction_distance_ring: float = 0.15,
) -> EvaluationResult:
    if match_radius_ring <= 0 or line_call_uncertainty_norm < 0 or correction_distance_ring < 0:
        raise ValueError("invalid evaluation tolerances")
    tp = fp = fn = empty = empty_fp = eligible = correct = correction = 0
    distances: list[float] = []
    for predictions, truths, metadata in faces:
        matched = _match(predictions, truths, radius=match_radius_ring * 0.1)
        tp += len(matched)
        missed = len(truths) - len(matched)
        extra = len(predictions) - len(matched)
        fn += missed
        fp += extra
        correction += missed + extra + sum(distance > correction_distance_ring * 0.1
                                           for _, _, distance in matched)
        if not truths:
            empty += 1
            empty_fp += len(predictions)
        distances.extend(distance / 0.1 for _, _, distance in matched)
        if metadata is None or metadata.face_diameter_mm is None:
            continue
        config = TargetConfig(
            format=TargetFormat(metadata.format),
            minimum_scoring_zone=metadata.minimum_scoring_zone,
            ten_ring_mode=TenRingMode(metadata.ten_ring_mode),
            face_diameter_mm=metadata.face_diameter_mm,
            triple_layout=None if metadata.triple_layout is None else TripleLayout(metadata.triple_layout),
        )
        template = TargetTemplate.from_config(config)
        matched_truths = {truth: pred for pred, truth, _ in matched}
        for i, (x, y) in enumerate(truths):
            expected = score(template, ImpactPoint(x, y), localization_error=line_call_uncertainty_norm)
            if expected.line_call:
                continue
            eligible += 1
            if i in matched_truths:
                pred = predictions[matched_truths[i]]
                actual = score(template, ImpactPoint(pred.x_norm, pred.y_norm))
                correct += actual.final_candidate == expected.final_candidate
    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / (tp + fn) if tp + fn else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    rank = max(0, math.ceil(0.95 * len(distances)) - 1)
    return EvaluationResult(
        precision=precision, recall=recall, f1=f1, true_positives=tp,
        false_positives=fp, false_negatives=fn,
        localization_median_ring=median(distances) if distances else None,
        localization_p95_ring=sorted(distances)[rank] if distances else None,
        exact_score_accuracy_non_line_call=correct / eligible if eligible else None,
        score_eligible_count=eligible,
        false_positives_per_empty_face=empty_fp / empty if empty else None,
        empty_faces=empty,
        correction_rate_proxy=correction / max(1, sum(len(truths) for _, truths, _ in faces)),
    )
