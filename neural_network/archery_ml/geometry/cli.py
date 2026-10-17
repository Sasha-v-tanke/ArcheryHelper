from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from neural_network.archery_ml.data.view import DatasetView
from neural_network.archery_ml.geometry.evaluate import GeometryEvaluator
from neural_network.archery_ml.targets import TargetConfig, TargetFormat, TenRingMode


DEFAULT_MANIFEST = Path("data/manifests/dataset_manifest.json")
DEFAULT_OUTPUT = Path("data/reports/geometry_evaluation.json")


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    view = DatasetView.load(args.manifest).select(
        splits=args.split,
        source_ids=args.source,
        require_geometry=True,
    )
    config = TargetConfig(
        format=TargetFormat.SINGLE,
        minimum_scoring_zone=args.minimum_scoring_zone,
        ten_ring_mode=TenRingMode(args.ten_ring_mode),
        face_diameter_mm=args.face_diameter_mm,
    )
    report = GeometryEvaluator().evaluate(view, config, preview_dir=args.preview_dir)
    payload = json.dumps(report.to_dict(), indent=2, sort_keys=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m archery_ml.geometry")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--preview-dir", type=Path, default=None)
    parser.add_argument("--source", action="append", default=None)
    parser.add_argument("--split", action="append", default=None)
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
