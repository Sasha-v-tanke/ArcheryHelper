from __future__ import annotations

import argparse
import json
import hashlib
import platform
import subprocess
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

from neural_network.archery_ml.data.view import DatasetView
from neural_network.archery_ml.geometry.evaluate import GeometryEvaluator
from neural_network.archery_ml.geometry.validation import validate_holdout, session_bootstrap, aggregate_reports
from neural_network.archery_ml.data.geometry_provenance import verify_mobile_gt
from neural_network.archery_ml.targets import TargetConfig, TargetFormat, TenRingMode, TripleLayout


DEFAULT_MANIFEST = Path("data/manifests/dataset_manifest.json")
DEFAULT_OUTPUT = Path("data/reports/geometry_evaluation.json")


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.aggregate_report:
        if args.policy is None:
            parser.error("--aggregate-report requires --policy")
        result = aggregate_reports([json.loads(path.read_text()) for path in args.aggregate_report],
                                   json.loads(args.policy.read_text()))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
        print(json.dumps(result["validation"], indent=2))
        return int(result["validation"]["status"] != "PASS")
    full_view = DatasetView.load(args.manifest)
    gt = [verify_mobile_gt(s) for s in full_view if (s.capture_metadata or {}).get("gt_annotations_sha256")]
    revisions = {item["revision"] for item in gt}
    if len(revisions) > 1:
        raise ValueError("mixed GT revisions are not supported for consolidated validation")
    revision = next(iter(revisions), None)
    if args.gt_revision is not None and args.gt_revision != revision:
        raise ValueError("declared GT revision differs from verified annotations")
    view = full_view.select(
        splits=args.split,
        source_ids=args.source,
    )
    config = TargetConfig(
        format=TargetFormat(args.format),
        triple_layout=None if args.triple_layout is None else TripleLayout(args.triple_layout),
        minimum_scoring_zone=args.minimum_scoring_zone,
        ten_ring_mode=TenRingMode(args.ten_ring_mode),
        face_diameter_mm=args.face_diameter_mm,
    )
    evaluator = GeometryEvaluator()
    report = evaluator.evaluate(view, config, preview_dir=args.preview_dir)
    result = report.to_dict()
    algorithm_sha, working_tree_dirty = _git_state()
    result["provenance"] = {
        "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        "sources": args.source,
        "splits": args.split,
        "metric_weighting": "radial: annotation vertices; center: images; no rejected samples excluded",
        "center_reference": "explicit target_center, otherwise outer GT ellipse center (a perspective-biased proxy)",
        "algorithm_sha": algorithm_sha,
        "working_tree_dirty": working_tree_dirty,
        "dependencies": {"python": platform.python_version(), "numpy": np.__version__, "opencv": cv2.__version__},
        "quality_thresholds": asdict(evaluator.quality_gate.thresholds),
        "gt_revision": revision,
        "gt_sources": sorted({item["annotation_manifest_sha256"] for item in gt}),
        "policy_sha256": None if args.policy is None else hashlib.sha256(json.dumps(
            json.loads(args.policy.read_text()), sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
    }
    result["session_bootstrap_mean_2d"] = session_bootstrap(result["sample_results"])
    if args.policy is not None:
        result["validation"] = validate_holdout(result, json.loads(args.policy.read_text()))
    payload = json.dumps(result, indent=2, sort_keys=True, allow_nan=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    return 1 if result.get("validation", {}).get("status") == "FAIL" else 0


def _git_state() -> tuple[str | None, bool]:
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
        diff = subprocess.check_output(["git", "diff", "HEAD", "--", "neural_network", "dataset_tools"], stderr=subprocess.DEVNULL)
        untracked = subprocess.check_output(
            ["git", "ls-files", "--others", "--exclude-standard", "--", "neural_network", "dataset_tools"],
            text=True, stderr=subprocess.DEVNULL,
        ).splitlines()
        return sha, bool(diff) or any(Path(path).suffix in {".py", ".json"} for path in untracked)
    except (OSError, subprocess.CalledProcessError):
        # Evaluation also works from an installed package; final validation still needs a frozen SHA.
        return None, True


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m archery_ml.geometry")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--preview-dir", type=Path, default=None)
    parser.add_argument("--source", action="append", default=None)
    parser.add_argument("--split", action="append", default=None)
    parser.add_argument("--format", choices=("SINGLE", "TRIPLE"), default="SINGLE")
    parser.add_argument("--triple-layout", choices=("VERTICAL", "TRIANGULAR"))
    parser.add_argument("--policy", type=Path)
    parser.add_argument("--gt-revision")
    parser.add_argument("--aggregate-report", type=Path, action="append",
                        help="combine configuration reports for one independent holdout decision")
    parser.add_argument(
        "--minimum-scoring-zone",
        type=int,
        choices=(1, 5, 6),
        default=1,
    )
    parser.add_argument(
        "--ten-ring-mode",
        choices=tuple(mode.value for mode in TenRingMode),
        default=TenRingMode.RECURVE.value,
    )
    parser.add_argument("--face-diameter-mm", type=int, default=1)
    return parser
