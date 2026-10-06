from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from neural_network.archery_ml.contracts import load_manifest, save_manifest
from neural_network.archery_ml.data.deduplicate import deduplicate_samples
from neural_network.archery_ml.data.download import materialize_source
from neural_network.archery_ml.data.importers import import_dataset
from neural_network.archery_ml.data.registry import DEFAULT_REGISTRY_PATH, DatasetRegistry
from neural_network.archery_ml.data.report import build_report
from neural_network.archery_ml.data.split import assign_splits
from neural_network.archery_ml.data.validate import validate_dataset


DEFAULT_MANIFEST = Path("data/dataset_manifest.json")
DEFAULT_WORKSPACE = Path("data/dataset_pipeline")


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "import":
        return _run_import(args)
    if args.command == "validate":
        return _run_validate(args)
    if args.command == "deduplicate":
        return _run_deduplicate(args)
    if args.command == "split":
        return _run_split(args)
    if args.command == "report":
        return _run_report(args)
    parser.error(f"unsupported command: {args.command}")
    return 2


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m archery_ml.data")
    subparsers = parser.add_subparsers(dest="command", required=True)

    import_parser = subparsers.add_parser("import")
    import_parser.add_argument("--source", required=True)
    import_parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY_PATH)
    import_parser.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE)
    import_parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)

    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY_PATH)
    validate_parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)

    deduplicate_parser = subparsers.add_parser("deduplicate")
    deduplicate_parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    deduplicate_parser.add_argument("--perceptual-threshold", type=int, default=4)

    split_parser = subparsers.add_parser("split")
    split_parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    split_parser.add_argument("--seed", type=int, default=42)
    split_parser.add_argument("--train-ratio", type=float, default=0.8)
    split_parser.add_argument("--val-ratio", type=float, default=0.1)
    split_parser.add_argument("--test-ratio", type=float, default=0.1)

    report_parser = subparsers.add_parser("report")
    report_parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    report_parser.add_argument("--perceptual-threshold", type=int, default=4)

    return parser


def _run_import(args: argparse.Namespace) -> int:
    registry = DatasetRegistry.load(args.registry)
    source = registry.get(args.source)
    source_root = materialize_source(source, args.workspace)
    imported = import_dataset(source_root, source)
    existing = load_manifest(args.manifest) if args.manifest.exists() else []
    combined = [sample for sample in existing if sample.source_id != source.id] + imported
    save_manifest(args.manifest, combined)
    print(json.dumps({"source": source.id, "imported": len(imported), "manifest": str(args.manifest)}))
    return 0


def _run_validate(args: argparse.Namespace) -> int:
    samples = load_manifest(args.manifest)
    registry = DatasetRegistry.load(args.registry)
    report = validate_dataset(samples, registry)
    if report.is_valid:
        print(json.dumps({"valid": True, "samples": len(samples)}))
        return 0
    print(report.format_errors())
    return 1


def _run_deduplicate(args: argparse.Namespace) -> int:
    samples = load_manifest(args.manifest)
    result = deduplicate_samples(samples, perceptual_threshold=args.perceptual_threshold)
    save_manifest(args.manifest, result.samples)
    print(
        json.dumps(
            {
                "samples": len(result.samples),
                "exact_duplicates_removed": len(result.exact_duplicates_removed),
                "near_duplicate_groups": len(result.near_duplicate_groups),
            }
        )
    )
    return 0


def _run_split(args: argparse.Namespace) -> int:
    samples = load_manifest(args.manifest)
    updated = assign_splits(
        samples,
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
    )
    save_manifest(args.manifest, updated)
    print(json.dumps({"samples": len(updated), "seed": args.seed}))
    return 0


def _run_report(args: argparse.Namespace) -> int:
    samples = load_manifest(args.manifest)
    report = build_report(samples, perceptual_threshold=args.perceptual_threshold)
    print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    return 0
