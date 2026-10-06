from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Sequence

from neural_network.contracts import ShotAnnotation


@dataclass(frozen=True)
class EvaluationReport:
    true_positive: int
    false_positive: int
    false_negative: int
    precision: float
    recall: float
    localization_mae: float


def evaluate_detections(
    predictions: Sequence[ShotAnnotation],
    targets: Sequence[ShotAnnotation],
    match_threshold: float = 0.08,
) -> EvaluationReport:
    matches = _match_predictions(predictions, targets, match_threshold)
    true_positive = len(matches)
    false_positive = len(predictions) - true_positive
    false_negative = len(targets) - true_positive
    precision = true_positive / len(predictions) if predictions else 0.0
    recall = true_positive / len(targets) if targets else 0.0
    localization_mae = (
        sum(distance(predictions[pred_idx], targets[target_idx]) for pred_idx, target_idx in matches) / true_positive
        if true_positive
        else 0.0
    )
    return EvaluationReport(
        true_positive=true_positive,
        false_positive=false_positive,
        false_negative=false_negative,
        precision=precision,
        recall=recall,
        localization_mae=localization_mae,
    )


def distance(a: ShotAnnotation, b: ShotAnnotation) -> float:
    dx = a.x_norm - b.x_norm
    dy = a.y_norm - b.y_norm
    return sqrt(dx * dx + dy * dy)


def _match_predictions(
    predictions: Sequence[ShotAnnotation],
    targets: Sequence[ShotAnnotation],
    match_threshold: float,
) -> list[tuple[int, int]]:
    candidates = []
    for pred_idx, prediction in enumerate(predictions):
        for target_idx, target in enumerate(targets):
            candidate_distance = distance(prediction, target)
            if candidate_distance <= match_threshold:
                candidates.append((candidate_distance, pred_idx, target_idx))

    used_predictions = set()
    used_targets = set()
    matches = []
    for _, pred_idx, target_idx in sorted(candidates):
        if pred_idx in used_predictions or target_idx in used_targets:
            continue
        used_predictions.add(pred_idx)
        used_targets.add(target_idx)
        matches.append((pred_idx, target_idx))
    return matches
