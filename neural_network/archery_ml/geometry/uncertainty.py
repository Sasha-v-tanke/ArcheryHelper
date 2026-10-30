"""Deterministic ring/sector sensitivity and held-segment checks, not a statistical CI."""

from dataclasses import dataclass

import cv2
import numpy as np

from neural_network.archery_ml.geometry.arc_extraction import SECTORS, describe_arcs
from neural_network.archery_ml.geometry.projective_solver import (
    SolverPolicy,
    fit,
    hypotheses,
    pack,
    residual_jacobian,
    safe_transform,
)
from neural_network.archery_ml.geometry.geometry_math import project


@dataclass(frozen=True)
class RobustEstimate:
    transform: np.ndarray
    diagnostics: dict


def estimate(initial, rings, policy=SolverPolicy()):
    reasons = []
    solutions = []
    if len(rings) < 2:
        reasons.append("insufficient_independent_rings")
    if not rings:
        return RobustEstimate(
            initial, {"solver_status": "REJECTED", "rejection_reasons": tuple(reasons)}
        )
    # Entire angular sectors are held out, rather than randomly adjacent pixels.
    training = [r.subset(r.sectors % 4 != 0) for r in rings]
    validation = [r.subset(r.sectors % 4 == 0) for r in rings]
    training = [r for r in training if len(r.points) >= 8]
    validation = [r for r in validation if len(r.points) >= 4]
    if len(training) != len(rings) or len(validation) != len(rings):
        reasons.append("insufficient_held_sector_evidence")
    if not training:
        training = rings
    for anchor, seed in hypotheses(initial, training):
        try:
            transform, loss = fit(seed, training, anchor=anchor, policy=policy)
            solutions.append((loss, transform, anchor))
        except (np.linalg.LinAlgError, ValueError, FloatingPointError):
            continue
    if not solutions:
        return RobustEstimate(
            initial,
            {"solver_status": "REJECTED", "rejection_reasons": ("optimizer_failed",)},
        )
    solutions.sort(key=lambda item: item[0])
    loss, best, anchor = solutions[0]
    validation_error = 1.0
    if validation:
        p, r, w = pack(validation)
        residual, _ = residual_jacobian(best, p, r)
        validation_error = float(np.sqrt(np.sum(w * residual**2)))
    if validation_error > policy.maximum_validation_error:
        reasons.append("held_sector_disagreement")
    # Only refit after recording independent-segment validation of the seed.
    best, _ = fit(best, rings, anchor=anchor, policy=policy)

    def distance(transform):
        candidate = project([[0.0, 0.0]], np.linalg.inv(transform))[0]
        return float(np.linalg.norm(project([candidate], best)[0]))

    ring_shifts = []
    sector_shifts = []
    failed = 0
    for index in range(len(rings)):
        subset = [r for j, r in enumerate(rings) if j != index]
        # One ring cannot determine projective center; do not report a falsely
        # precise leave-one-out fit. Single-ring ellipse disagreement is recorded.
        if len(subset) < 2:
            points = project(rings[index].points, best)
            ellipse_center, _, _ = cv2.fitEllipse(points.astype(np.float32))
            ring_shifts.append(float(np.linalg.norm(ellipse_center)))
            continue
        try:
            other, _ = fit(best, subset, anchor=anchor, policy=policy)
            ring_shifts.append(distance(other))
        except (np.linalg.LinAlgError, ValueError, FloatingPointError):
            failed += 1
    for omit in range(4):
        subset = [r.subset(r.sectors % 4 != omit) for r in rings]
        try:
            other, _ = fit(best, subset, anchor=anchor, policy=policy)
            sector_shifts.append(distance(other))
        except (np.linalg.LinAlgError, ValueError, FloatingPointError):
            failed += 1
    disagreement = max(
        (distance(h) for score, h, _ in solutions if score <= loss * 1.25 + 1e-8),
        default=0.0,
    )
    stability = max(ring_shifts + sector_shifts + [disagreement], default=1.0)
    occupied = np.unique(np.concatenate([r.sectors for r in rings]))
    coverage = len(occupied) / SECTORS
    if coverage < policy.minimum_angular_fraction:
        reasons.append("insufficient_angular_coverage")
    if stability > policy.maximum_center_spread:
        reasons.append("unstable_center")
    if failed:
        reasons.append("resampling_failed")
    p, r, w = pack(rings)
    residual, jacobian = residual_jacobian(best, p, r)
    jacobian /= np.maximum(np.linalg.norm(jacobian, axis=0), 1e-12)
    singular = np.linalg.svd(jacobian, compute_uv=False)
    # Circle equations leave one rotational gauge. Additional null directions
    # mean that perspective/center cannot be identified by this evidence.
    conditioning = float(singular[0] / max(singular[-2], 1e-12))
    if conditioning > 1e5 or not safe_transform(anchor, best, policy):
        reasons.append("ill_conditioned_geometry")
    angles = np.linspace(0, 2 * np.pi, 64, endpoint=False)
    boundary = np.column_stack((np.cos(angles), np.sin(angles)))
    deviation = np.linalg.norm(
        project(project(boundary, np.linalg.inv(anchor)), best) - boundary, axis=1
    )
    if np.max(deviation) >= policy.maximum_boundary_shift * 0.99:
        reasons.append("extrapolation_limit_reached")
    diagnostics = {
        "diagnostics_version": 1,
        "solver_method": "multi_ring_cauchy_irls_v1",
        "solver_status": "NEEDS_REVIEW" if reasons else "AUTO_READY",
        "rejection_reasons": tuple(dict.fromkeys(reasons)),
        "center_uncertainty_norm": stability,
        "center_stability_norm": max(sector_shifts, default=1.0),
        "observed_ring_count": len(rings),
        "observed_arc_fraction": float(
            np.mean([len(np.unique(r.sectors)) / SECTORS for r in rings])
        ),
        "angular_coverage": coverage,
        "cross_ring_consistency": max(ring_shifts, default=1.0),
        "hypothesis_disagreement": disagreement,
        "numerical_conditioning": conditioning,
        "held_sector_error": validation_error,
        "observed_arcs": tuple(arc for ring in rings for arc in describe_arcs(ring)),
        "leave_one_ring_out_status": "MEASURED"
        if len(rings) >= 3
        else "NOT_IDENTIFIABLE_WITH_ONE_REMAINING_RING",
        "ring_leave_out_shifts": tuple(ring_shifts) if len(rings) >= 3 else (),
        "sector_resample_shifts": tuple(sector_shifts),
        "diagnostic_note": "deterministic sensitivity; not a confidence interval or calibrated probability",
    }
    return RobustEstimate(best, diagnostics)
