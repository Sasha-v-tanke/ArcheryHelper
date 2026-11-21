"""Measured arc grouping and equal-sector sampling, without invented contour pixels."""

from dataclasses import dataclass

import numpy as np

from neural_network.archery_ml.geometry.geometry_math import project


SECTORS = 48


@dataclass(frozen=True)
class RingObservation:
    radius: float
    color: str
    points: np.ndarray
    weights: np.ndarray
    sectors: np.ndarray

    def subset(self, keep):
        return RingObservation(
            self.radius,
            self.color,
            self.points[keep],
            self.weights[keep],
            self.sectors[keep],
        )


def angular_sectors(points, transform):
    mapped = project(points, transform)
    angle = np.arctan2(mapped[:, 1], mapped[:, 0])
    return np.minimum(
        SECTORS - 1, ((angle + np.pi) * SECTORS / (2 * np.pi)).astype(int)
    )


def balance_ring(points, weights, radius, color, transform):
    """Keep at most four observed pixels per sector; each occupied sector has equal mass."""
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    weights = np.asarray(weights, dtype=np.float64)
    sectors = angular_sectors(points, transform)
    chosen = []
    for sector in np.unique(sectors):
        indices = np.flatnonzero(sectors == sector)
        # Stable ordering makes equal scores deterministic across runs.
        indices = indices[np.argsort(-weights[indices], kind="stable")[:4]]
        chosen.extend(indices.tolist())
    chosen = np.asarray(chosen, dtype=int)
    points, weights, sectors = points[chosen], weights[chosen], sectors[chosen]
    for sector in np.unique(sectors):
        keep = sectors == sector
        weights[keep] /= max(float(weights[keep].sum()), 1e-12)
    weights /= max(float(weights.sum()), 1e-12)
    return RingObservation(radius, color, points, weights, sectors)


def describe_arcs(ring):
    """Separate measured groups across angular gaps; no convex hull or interpolation."""
    occupied = np.unique(ring.sectors)
    groups = np.split(occupied, np.flatnonzero(np.diff(occupied) > 1) + 1)
    arcs = []
    for group in groups:
        keep = np.isin(ring.sectors, group)
        order = np.argsort(ring.sectors[keep], kind="stable")
        points = ring.points[keep][order]
        arcs.append(
            {
                "radius_norm": ring.radius,
                "color": ring.color,
                "points": points.tolist(),
                "weight": float(ring.weights[keep].sum()),
                "angular_fraction": len(group) / SECTORS,
                "length_px": float(
                    np.linalg.norm(np.diff(points, axis=0), axis=1).sum()
                ),
            }
        )
    return tuple(arcs)
