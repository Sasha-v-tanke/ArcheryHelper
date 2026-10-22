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
from neural_network.archery_ml.geometry.validation import validate_holdout, session_bootstrap
from neural_network.archery_ml.targets import TargetConfig, TargetFormat, TenRingMode, TripleLayout


DEFAULT_MANIFEST = Path("data/manifests/dataset_manifest.json")
DEFAULT_OUTPUT = Path("data/reports/geometry_evaluation.json")


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    view = DatasetView.load(args.manifest).select(
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
        "gt_revision": args.gt_revision,
        "policy_sha256": None if args.policy is None else hashlib.sha256(args.policy.read_bytes()).hexdigest(),
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
