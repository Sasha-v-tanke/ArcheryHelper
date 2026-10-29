"""Bounded multi-ring Cauchy IRLS with a shared projective center and fixed radii."""

from dataclasses import dataclass

import cv2
import numpy as np

from neural_network.archery_ml.geometry.geometry_math import project


@dataclass(frozen=True)
class SolverPolicy:
    """Development safeguards; these are not calibrated production probabilities."""

    loss_scale: float = 0.01
    maximum_iterations: int = 32
    maximum_boundary_shift: float = 0.12
    maximum_center_spread: float = 0.025
    maximum_validation_error: float = 0.03
    minimum_angular_fraction: float = 0.5


def safe_transform(initial, proposed, policy=SolverPolicy()):
    if not np.isfinite(proposed).all():
        return False
    try:
        inverse = np.linalg.inv(proposed)
        if np.linalg.cond(proposed) > 1e12 or abs(inverse[2, 2]) <= np.linalg.norm(
            inverse[2, :2]
        ):
            return False
        angles = np.linspace(0, 2 * np.pi, 64, endpoint=False)
        boundary = np.column_stack((np.cos(angles), np.sin(angles)))
        mapped = project(project(boundary, np.linalg.inv(initial)), proposed)
        return bool(
            np.max(np.linalg.norm(mapped - boundary, axis=1))
            <= policy.maximum_boundary_shift
        )
    except np.linalg.LinAlgError:
        return False


def pack(rings):
    if not rings:
        raise ValueError("no observed rings")
    points = np.concatenate([r.points for r in rings])
    radii = np.concatenate([np.full(len(r.points), r.radius) for r in rings])
    weights = np.concatenate(
        [r.weights / max(float(r.weights.sum()), 1e-12) / len(rings) for r in rings]
    )
    if (
        points.ndim != 2
        or points.shape[1] != 2
        or not np.isfinite(points).all()
        or not np.isfinite(radii).all()
        or np.any(radii <= 0)
        or not np.isfinite(weights).all()
        or np.any(weights < 0)
        or weights.sum() <= 0
    ):
        raise ValueError("invalid ring observations")
    return points, radii, weights


def residual_jacobian(transform, points, radii):
    mapped = project(points, transform)
    x, y = mapped.T
    distance = np.maximum(np.linalg.norm(mapped, axis=1), 1e-9)
    jacobian = np.column_stack(
        (
            x * x / distance,
            x * y / distance,
            x / distance,
            y * x / distance,
            y * y / distance,
            y / distance,
            -distance * x,
            -distance * y,
        )
    )
    return distance - radii, jacobian


def fit(initial, rings, *, anchor=None, policy=SolverPolicy()):
    points, radii, weights = pack(rings)
    transform = np.array(initial, dtype=np.float64, copy=True)
    anchor = transform if anchor is None else anchor
    if not safe_transform(anchor, transform, policy):
        raise np.linalg.LinAlgError("unsafe initial hypothesis")
    scale = policy.loss_scale

    def loss(error):
        return float(np.sum(weights * scale * scale * np.log1p((error / scale) ** 2)))

    for _ in range(policy.maximum_iterations):
        residual, jacobian = residual_jacobian(transform, points, radii)
        robust = weights / (1 + (residual / scale) ** 2)
        root = np.sqrt(robust)
        weighted = jacobian * root[:, None]
        regularizer = 0.1 * np.diag(np.linalg.norm(weighted, axis=0))
        step = np.linalg.lstsq(
            np.vstack((weighted, regularizer)),
            np.concatenate((-residual * root, np.zeros(8))),
            rcond=1e-8,
        )[0]
        previous = loss(residual)
        accepted = False
        for fraction in (1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625):
            delta = step * fraction
            correction = np.array(
                [
                    [1 + delta[0], delta[1], delta[2]],
                    [delta[3], 1 + delta[4], delta[5]],
                    [delta[6], delta[7], 1.0],
                ]
            )
            proposed = correction @ transform
            if not safe_transform(anchor, proposed, policy):
                continue
            error, _ = residual_jacobian(proposed, points, radii)
            if np.isfinite(error).all() and loss(error) < previous - 1e-13:
                transform = proposed
                accepted = True
                break
        if not accepted:
            break
    return transform, loss(residual_jacobian(transform, points, radii)[0])


def hypotheses(initial, rings):
    """Independent measured-ring affine seeds plus bounded perspective seeds."""
    seeds = [(initial, initial)]
    for ring in rings:
        if len(ring.points) < 12 or len(np.unique(ring.sectors)) < 24:
            continue
        (cx, cy), (dx, dy), angle = cv2.fitEllipse(ring.points.astype(np.float32))
        if min(dx, dy) <= 0:
            continue
        a = np.deg2rad(angle)
        rotation = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
        linear = (
            rotation
            @ np.diag([2 * ring.radius / dx, 2 * ring.radius / dy])
            @ rotation.T
        )
        seed = np.eye(3)
        seed[:2, :2] = linear
        seed[:2, 2] = -linear @ np.array([cx, cy])
        # The radial objective has an unobservable in-plane rotation. Align
        # that gauge to the caller's declared image/layout axes.
        relative = initial[:2, :2] @ np.linalg.inv(seed[:2, :2])
        u, _, vt = np.linalg.svd(relative)
        gauge = np.eye(3)
        gauge[:2, :2] = u @ vt
        measured = gauge @ seed
        # A complete independently observed inner boundary supplies its own
        # affine anchor when a damaged outer boundary was a poor proposal.
        seeds.append((measured, measured))
    for skew in (-0.04, 0.04):
        correction = np.eye(3)
        correction[2, :2] = (skew, -skew * 0.5)
        seeds.append((initial, correction @ initial))
    return seeds[:6]
