"""Fail-closed validation of frozen independent geometry evaluation reports."""
from __future__ import annotations

import numpy as np


def session_bootstrap(samples: list[dict], *, seed: int = 42, repetitions: int = 2000) -> dict:
    """95% cluster bootstrap CI for equal-image mean landmark error."""
    sessions: dict[str, list[float]] = {}
    for sample in samples:
        value = sample["point_error_2d"]["mean"]
        session = sample.get("capture_metadata", {}).get("capture_session_id")
        if value is not None and session:
            sessions.setdefault(session, []).append(value)
    if len(sessions) < 2:
        return {"low": None, "high": None, "sessions": len(sessions)}
    groups = [sessions[key] for key in sorted(sessions)]
    rng = np.random.default_rng(seed)
    means = [float(np.mean([v for index in rng.integers(0, len(groups), len(groups)) for v in groups[index]]))
             for _ in range(repetitions)]
    low, high = np.percentile(means, [2.5, 97.5])
    return {"low": float(low), "high": float(high), "sessions": len(sessions),
            "seed": seed, "repetitions": repetitions, "statistic": "equal-image mean 2D error"}


def validate_holdout(report: dict, policy: dict) -> dict:
    reasons = []
    if policy.get("status") != "frozen":
        reasons.append("quality policy has not been frozen after development calibration")
    samples = report["sample_results"]
    if not samples or any(s["split"] != "mobile_real_test" for s in samples):
        reasons.append("independent mobile_real_test required")
    provenance = report.get("provenance", {})
    if provenance.get("working_tree_dirty", True):
        reasons.append("algorithm working tree is not frozen")
    for key in ("algorithm_sha", "manifest_sha256", "policy_sha256", "gt_revision", "dependencies"):
        if not provenance.get(key):
            reasons.append(f"missing frozen provenance: {key}")
    ready = [s for s in samples if s["status"] == "READY"]
    if not ready or any(s["point_error_2d"]["count"] < 4 * s["detected_face_count"] for s in ready):
        reasons.append("READY faces lack independent landmarks")
    if any(s.get("annotation_issues") for s in samples):
        reasons.append("unresolved GT issues")
    for key, limit in (("median", policy["maximum_median_2d"]),
                       ("p95", policy["maximum_p95_2d"]), ("max", policy["maximum_ready_error_2d"])):
        value = report.get("breakdowns", {}).get("acceptance", {}).get("ready", {}).get("point_error_2d", {}).get(key)
        if value is None or not np.isfinite(value) or value > limit:
            reasons.append(f"READY {key} 2D error missing or exceeds {limit}")
    coverage = report["coverage"]
    for key, limit in (("false_accept_fraction", policy["maximum_false_accept_fraction"]),
                       ("false_reject_fraction", policy["maximum_false_reject_fraction"])):
        value = coverage[key]
        if value is None or value > limit:
            reasons.append(f"{key} missing or exceeds {limit}")
    sessions = {s.get("capture_metadata", {}).get("capture_session_id") for s in samples}
    originals = {s.get("capture_metadata", {}).get("original_id") for s in samples}
    devices = {s.get("capture_metadata", {}).get("device_model") for s in samples}
    for name, values, minimum in (("sessions", sessions, policy["minimum_sessions"]),
                                  ("originals", originals, policy["minimum_originals"]),
                                  ("devices", devices, policy["minimum_devices"])):
        if None in values or len(values) < minimum:
            reasons.append(f"insufficient independent {name}")
    return {"status": "FAIL" if reasons else "PASS", "reasons": reasons,
            "session_bootstrap_mean_2d": session_bootstrap(ready)}
