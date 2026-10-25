"""Fail-closed validation of frozen independent geometry evaluation reports."""
from __future__ import annotations

import hashlib
import json

import numpy as np

from neural_network.archery_ml.geometry.evaluate import _equal_group_metrics


METRICS_VERSION = "stage3-2d-v3-equal-image"

def configuration_key(config: dict) -> str:
    return ":".join(str(config.get(k) or "SINGLE") for k in
                    ("format", "triple_layout", "minimum_scoring_zone", "ten_ring_mode"))


def aggregate_reports(reports: list[dict], policy: dict) -> dict:
    """Consolidate frozen evaluations, refusing incompatible or overlapping inputs."""
    reasons = []
    samples = []
    seen = set()
    first = reports[0].get("provenance", {}) if reports else {}
    for report in reports:
        provenance = report.get("provenance", {})
        for key in ("algorithm_sha", "working_tree_dirty", "manifest_sha256", "policy_sha256",
                    "gt_revision", "gt_sources", "dependencies", "quality_thresholds"):
            if provenance.get(key) != first.get(key):
                reasons.append(f"incompatible provenance: {key}")
        if report.get("metrics_version") != METRICS_VERSION:
            reasons.append("incompatible metrics version")
        if report.get("coverage", {}).get("unknown_suitability_samples", 0) > 0:
            reasons.append("unknown suitability in input report")
        for sample in report.get("sample_results", []):
            meta = sample.get("capture_metadata", {})
            identities = {(k, meta[k]) for k in ("original_id", "original_sha256") if meta.get(k)}
            if seen & identities:
                reasons.append("duplicate original across reports")
                continue
            seen.update(identities)
            samples.append({**sample, "target_config": report.get("target_config", sample.get("target_config", {}))})
    result = {"sample_results": samples, "provenance": first, "metrics_version": METRICS_VERSION}
    result["validation"] = validate_holdout(result, policy)
    result["validation"]["reasons"].extend(reasons)
    if reasons:
        result["validation"]["status"] = "FAIL"
    return result


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
    policy_sha = hashlib.sha256(json.dumps(policy, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if provenance.get("policy_sha256") != policy_sha:
        reasons.append("policy checksum mismatch")
    if report.get("metrics_version") != METRICS_VERSION:
        reasons.append("missing or incompatible metrics version")
    if report.get("coverage", {}).get("unknown_suitability_samples", 0) > 0:
        reasons.append("unknown suitability in coverage")
    ready = [s for s in samples if s["status"] == "READY"]
    for identity in ("original_id", "original_sha256"):
        values = [s.get("capture_metadata", {}).get(identity) for s in samples]
        if None in values or len(values) != len(set(values)):
            reasons.append(f"missing or duplicate independent {identity}")
    if not ready:
        reasons.append("READY faces lack independent landmarks")
    for sample in samples:
        meta = sample.get("capture_metadata", {})
        gt = sample.get("gt_provenance", {})
        if (not gt.get("annotation_manifest_sha256") or not gt.get("annotations_sha256")
                or gt.get("revision") != meta.get("gt_revision")
                or gt.get("revision") != provenance.get("gt_revision")
                or gt.get("annotations_sha256") != meta.get("gt_annotations_sha256")
                or gt.get("annotation_manifest_sha256") not in provenance.get("gt_sources", [])):
            reasons.append("missing or inconsistent verified GT provenance")
        if meta.get("suitability") not in {"usable", "unusable"}:
            reasons.append("unknown suitability")
        if sample["status"] != "READY":
            continue
        config = sample.get("target_config", report.get("target_config", {}))
        count = 1 if config.get("format") == "SINGLE" else 3
        faces = sample.get("face_error", {})
        if sample.get("detected_face_count") != count or set(faces) != {str(i) for i in range(count)}:
            reasons.append("READY face_index/count mismatch")
        for face in faces.values():
            if face.get("independent_landmarks", 0) < 4 or face.get("point_error_2d", {}).get("count", 0) < 4:
                reasons.append("READY face lacks independent landmarks")
            _check_errors(face.get("point_error_2d", {}), policy, reasons, "READY face")
            if policy.get("require_physical_orientation", False) and face.get("orientation_reference") not in {
                    "physical_marker", "board", "manual_calibration"}:
                reasons.append("physical orientation not established by algorithm")
    if any(s.get("annotation_issues") for s in samples):
        reasons.append("unresolved GT issues")
    _check_errors(_sample_metrics(ready), policy, reasons, "READY")
    usable = [s for s in samples if s.get("capture_metadata", {}).get("suitability") == "usable"]
    unusable = [s for s in samples if s.get("capture_metadata", {}).get("suitability") == "unusable"]
    coverage = {
        "false_reject_fraction": None if not usable else sum(s["status"] != "READY" for s in usable) / len(usable),
        "false_accept_fraction": None if not unusable else sum(s["status"] == "READY" for s in unusable) / len(unusable),
    }
    for key, limit in (("false_accept_fraction", policy["maximum_false_accept_fraction"]),
                       ("false_reject_fraction", policy["maximum_false_reject_fraction"])):
        value = coverage[key]
        if value is None or value > limit:
            reasons.append(f"{key} missing or exceeds {limit}")
    matrix = policy.get("required_configurations", [])
    if not matrix:
        reasons.append("missing mandatory configuration matrix")
    groups = {}
    for requirement in matrix:
        key = configuration_key(requirement)
        group = [s for s in samples if configuration_key(s.get("target_config", report.get("target_config", {}))) == key]
        accepted = [s for s in group if s["status"] == "READY"]
        usable = [s for s in group if s.get("capture_metadata", {}).get("suitability") == "usable"]
        unusable = [s for s in group if s.get("capture_metadata", {}).get("suitability") == "unusable"]
        originals = {s.get("capture_metadata", {}).get("original_id") for s in group} - {None}
        sessions = {s.get("capture_metadata", {}).get("capture_session_id") for s in group} - {None}
        rate = sum(s["status"] == "READY" for s in usable) / len(usable) if usable else None
        false_accept = sum(s["status"] == "READY" for s in unusable) / len(unusable) if unusable else None
        metrics = _sample_metrics(accepted)
        groups[key] = {"originals": len(originals), "sessions": len(sessions), "ready": len(accepted),
                       "acceptance_rate": rate, "usable_acceptance_rate": rate,
                       "usable_samples": len(usable), "unusable_samples": len(unusable),
                       "false_accept_rate": false_accept, "point_error_2d": metrics,
                       "landmark_weighted_point_error_2d": _landmark_metrics(accepted)}
        if (len(originals) < requirement["minimum_originals"] or len(sessions) < requirement["minimum_sessions"]
                or len(accepted) < requirement["minimum_ready"] or rate is None
                or rate < requirement["minimum_acceptance_rate"]):
            reasons.append(f"insufficient configuration coverage: {key}")
        _check_errors(metrics, policy, reasons, key)
    sessions = {s.get("capture_metadata", {}).get("capture_session_id") for s in samples}
    originals = {s.get("capture_metadata", {}).get("original_id") for s in samples}
    devices = {s.get("capture_metadata", {}).get("device_model") for s in samples}
    for name, values, minimum in (("sessions", sessions, policy["minimum_sessions"]),
                                  ("originals", originals, policy["minimum_originals"]),
                                  ("devices", devices, policy["minimum_devices"])):
        if None in values or len(values) < minimum:
            reasons.append(f"insufficient independent {name}")
    return {"status": "FAIL" if reasons else "PASS", "reasons": reasons,
            "configurations": groups,
            "ready_point_error_2d": _sample_metrics(ready),
            "ready_landmark_weighted_point_error_2d": _landmark_metrics(ready),
            "metric_weighting": "equal image mass; equal landmark mass within image; inverse empirical CDF quantiles",
            "session_bootstrap_mean_2d": session_bootstrap(ready)}


def _check_errors(metrics: dict, policy: dict, reasons: list, label: str) -> None:
    for key, limit in (("median", policy["maximum_median_2d"]), ("p95", policy["maximum_p95_2d"]),
                       ("max", policy["maximum_ready_error_2d"])):
        value = metrics.get(key)
        if value is None or not np.isfinite(value) or value > limit:
            reasons.append(f"{label} {key} 2D error missing or exceeds {limit}")


def _sample_metrics(samples: list[dict]) -> dict:
    """Each independent READY photo has mass 1; its n points each have mass 1/n.

    Quantiles are the smallest error whose weighted empirical CDF reaches q.
    Missing/non-finite READY points invalidate metrics instead of losing a photo.
    """
    groups = [s.get("point_values_2d", []) for s in samples]
    if not groups or any(not values or not np.all(np.isfinite(values)) for values in groups):
        return {}
    return _equal_group_metrics(groups)


def _landmark_metrics(samples: list[dict]) -> dict:
    """Unweighted diagnostic only; never used to approve a holdout."""
    values = [v for s in samples for v in s.get("point_values_2d", [])]
    if not values or not np.all(np.isfinite(values)):
        return {}
    return {"median": float(np.median(values)), "p95": float(np.percentile(values, 95)), "max": max(values)}
