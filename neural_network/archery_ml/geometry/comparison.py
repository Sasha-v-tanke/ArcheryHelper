"""Paired fixed-input comparison, retaining rejections and unmatched references."""

import argparse
import html
import json
import os
from pathlib import Path

import numpy as np

from neural_network.archery_ml.geometry.benchmark import summary
from neural_network.archery_ml.geometry.geometry_math import project
from neural_network.archery_ml.geometry.synthetic import scenarios


def synthetic_metrics(records):
    by_id = {r["id"]: r for r in records if r["dataset"] == "independent_synthetic"}
    center = []
    points = []
    ready_center = []
    ready_points = []
    negative = 0
    false_accept = 0
    for index, scene in enumerate(scenarios()):
        row = by_id[f"synthetic-{index:02}-{scene.scenario}"]
        if scene.expected_reject:
            negative += 1
            false_accept += row["status"] == "READY"
            continue
        for face, matrix in zip(row["faces"], scene.norm_to_image):
            center_error = float(
                np.linalg.norm(project([face["center"]], np.linalg.inv(matrix))[0])
            )
            # External synthetic orientation reference uses a dedicated short
            # x-axis segment. Validation probes below are disjoint; neither
            # centers nor radii are recalibrated using their validation truth.
            reference = project(
                project([[0.0, 0.0], [0.1, 0.0]], matrix),
                np.asarray(face["homography"]),
            )
            axis = reference[1] - reference[0]
            angle = -np.arctan2(axis[1], axis[0])
            rotation = np.array(
                [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
            )
            angles = np.linspace(0, 2 * np.pi, 32, endpoint=False)
            probes = 0.45 * np.column_stack((np.cos(angles), np.sin(angles)))
            predicted = (
                project(project(probes, matrix), np.asarray(face["homography"])) - 255.5
            ) / 256
            errors = np.linalg.norm(predicted @ rotation.T - probes, axis=1).tolist()
            center.append(center_error)
            points.extend(errors)
            if row["status"] == "READY":
                ready_center.append(center_error)
                ready_points.extend(errors)
    return {
        "center_error_norm_all_detected": summary(center),
        "point_error_2d_external_axes_all_detected": summary(points),
        "center_error_norm_ready": summary(ready_center),
        "point_error_2d_external_axes_ready": summary(ready_points),
        "false_accept_fraction_on_negative_scenes": false_accept / max(1, negative),
        "negative_scenes": negative,
        "orientation_scope": "dedicated synthetic external x-axis reference; real physical axes remain unanchored",
    }


def compare(baseline, candidate, output):
    baseline, candidate, output = Path(baseline), Path(candidate), Path(output)
    old = json.loads((baseline / "report.json").read_text())
    new = json.loads((candidate / "report.json").read_text())
    first = {r["id"]: r for r in old["records"]}
    second = {r["id"]: r for r in new["records"]}
    if set(first) != set(second) or old["provenance"].get("review_sha256") != new[
        "provenance"
    ].get("review_sha256"):
        raise ValueError("comparison requires identical samples and review revision")
    paired_before = []
    paired_after = []
    missing = []
    cases = []
    for key, row in second.items():
        previous = first[key]
        if previous.get("roi") != row.get("roi") or previous.get(
            "input_sha256"
        ) != row.get("input_sha256"):
            raise ValueError("comparison inputs or selection ROI changed")
        old_points = {p["id"]: p for p in previous.get("pairs", [])}
        new_points = {p["id"]: p for p in row.get("pairs", [])}
        for cid in sorted(set(old_points) | set(new_points)):
            if cid not in old_points or cid not in new_points:
                missing.append(
                    {
                        "image": key,
                        "point_id": cid,
                        "baseline_measured": cid in old_points,
                        "candidate_measured": cid in new_points,
                    }
                )
            else:
                paired_before.append(old_points[cid]["error_px"])
                paired_after.append(new_points[cid]["error_px"])
        regression = previous["status"] == "READY" and row["status"] != "READY"
        cases.append(
            {
                "id": key,
                "number": row.get("number"),
                "baseline_status": previous["status"],
                "new_status": row["status"],
                "acceptance_regression": regression,
                "new_max_center_error_px": max(
                    (p["error_px"] for p in row.get("pairs", [])), default=None
                ),
                "reasons": sorted(
                    {
                        reason
                        for f in row.get("faces", [])
                        for reason in f.get("rejection_reasons", [])
                    }
                ),
                "baseline_preview": str((baseline / previous["preview"]).resolve()),
                "new_preview": str((candidate / row["preview"]).resolve()),
            }
        )
    cases.sort(
        key=lambda c: (
            not c["acceptance_regression"],
            c["new_status"] == "READY",
            -(c["new_max_center_error_px"] or 0),
        )
    )
    result = {
        "schema_version": 1,
        "baseline_provenance": old["provenance"],
        "candidate_provenance": new["provenance"],
        "development_real_paired_centers": {
            "baseline": summary(paired_before),
            "new": summary(paired_after),
            "unmatched_references": missing,
            "independent_gt": False,
        },
        "baseline_synthetic": synthetic_metrics(old["records"]),
        "new_synthetic": synthetic_metrics(new["records"]),
        "baseline": {
            k: v for k, v in old.items() if k not in ("records", "provenance")
        },
        "new": {k: v for k, v in new.items() if k not in ("records", "provenance")},
        "cases": cases,
        "production_validation": "NOT_VALIDATED",
        "center_gate_proposal": {
            "maximum_median_center_error_norm": 0.005,
            "maximum_p95_center_error_norm": 0.01,
            "maximum_ready_center_error_norm": 0.02,
            "maximum_center_stability_norm": 0.025,
            "angular_fraction_minimum": 0.5,
            "status": "development proposal based on synthetic center calibration; real center accuracy gate NOT_VALIDATED",
        },
    }
    output.mkdir(parents=True, exist_ok=False)
    (output / "comparison.json").write_text(
        json.dumps(result, indent=2, allow_nan=False)
    )
    rows = []
    for case in cases:
        before = html.escape(os.path.relpath(case["baseline_preview"], output))
        after = html.escape(os.path.relpath(case["new_preview"], output))
        rows.append(
            f"<article><h3>{html.escape(str(case['number'] or case['id']))}: {case['baseline_status']} → {case['new_status']}</h3>"
            f"<p>{html.escape(', '.join(case['reasons']))}</p><a href='{before}'><img src='{before}'></a>"
            f"<a href='{after}'><img src='{after}'></a></article>"
        )
    (output / "index.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>Target geometry comparison</title>'
        "<style>body{background:#202124;color:#eee;font:16px sans-serif}img{width:48%;vertical-align:top}article{margin:30px 0}</style>"
        "<h1>Baseline / Robust geometry</h1><p>Green: prediction. Red: reviewed center. Orange: observed arcs. Magenta: excluded pixels.</p>"
        "<p>Development feedback is not independent real holdout. Acceptance regressions are shown first.</p>"
        + "".join(rows)
    )
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = compare(args.baseline, args.candidate, args.output)
    print(
        json.dumps(
            {
                k: v
                for k, v in result.items()
                if k not in ("cases", "baseline_provenance", "candidate_provenance")
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
