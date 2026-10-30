"""Conservative target grouping; appearance never proves ownership."""

from itertools import combinations

import numpy as np


def select_group(candidates, template, order, candidate_ids=None):
    candidates = sorted(
        candidates, key=lambda c: (c.ellipse.center_y, c.ellipse.center_x)
    )
    count = template.expected_face_count
    if candidate_ids is not None:
        labels = {f"f{i}": c for i, c in enumerate(candidates)}
        if (
            len(candidate_ids) != count
            or len(set(candidate_ids)) != count
            or any(cid not in labels for cid in candidate_ids)
        ):
            raise ValueError("selection must identify exactly the configured faces")
        return order([labels[cid] for cid in candidate_ids], template)
    if len(candidates) <= count:
        return order(candidates, template)
    if count == 1 or len(candidates) > 18:
        return candidates, False
    groups = []
    for subset in combinations(candidates, count):
        diameters = [c.ellipse.major_axis_px for c in subset]
        if max(diameters) > 1.6 * min(diameters):
            continue
        ordered, valid = order(list(subset), template)
        if not valid:
            continue
        centers = np.array([[c.ellipse.center_x, c.ellipse.center_y] for c in ordered])
        distances = [
            np.linalg.norm(a - b) / np.median(diameters)
            for a, b in combinations(centers, 2)
        ]
        if min(distances) < 0.85 or max(distances) > 4.2:
            continue
        if (
            str(template.triple_layout.value) == "VERTICAL"
            and abs(centers[-1, 0] - centers[0, 0])
            > abs(centers[-1, 1] - centers[0, 1]) * 0.5
        ):
            continue
        groups.append(ordered)
    # Multiple plausible groups require explicit selection, regardless of area.
    return (groups[0], True) if len(groups) == 1 else (candidates, False)
