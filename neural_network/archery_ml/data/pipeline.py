from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from neural_network.archery_ml.contracts import save_manifest
from neural_network.archery_ml.data.deduplicate import deduplicate_samples
from neural_network.archery_ml.data.download import materialize_source
from neural_network.archery_ml.data.importers import import_dataset
from neural_network.archery_ml.data.registry import (
    DEFAULT_REGISTRY_PATH,
    DatasetRegistry,
)
from neural_network.archery_ml.data.report import DatasetReport, build_report
from neural_network.archery_ml.data.snapshot import (
    DatasetSnapshot,
    build_snapshot,
    save_snapshot,
)
from neural_network.archery_ml.data.split import assign_splits
from neural_network.archery_ml.data.validate import validate_dataset
from neural_network.archery_ml.data.workspace import DatasetWorkspace


@dataclass(frozen=True)
class DatasetRebuildResult:
    manifest_path: Path
    report_path: Path
    snapshot_path: Path
    report: DatasetReport
    snapshot: DatasetSnapshot


def rebuild_dataset(
    *,
    registry_path: str | Path = DEFAULT_REGISTRY_PATH,
    data_root: str | Path = "data",
    source_ids: Sequence[str] | None = None,
    manifest_path: str | Path | None = None,
    report_path: str | Path | None = None,
    snapshot_path: str | Path | None = None,
    seed: int = 42,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    perceptual_threshold: int = 4,
) -> DatasetRebuildResult:
    registry_file = Path(registry_path)
    registry = DatasetRegistry.load(registry_file)
    selected_ids = _select_source_ids(registry, source_ids)
    selected_registry = DatasetRegistry(tuple(registry.get(source_id) for source_id in selected_ids))

    workspace = DatasetWorkspace(Path(data_root))
    workspace.ensure_output_dirs()
    manifest_output = Path(manifest_path) if manifest_path is not None else workspace.manifest_path
    report_output = Path(report_path) if report_path is not None else workspace.report_path
    snapshot_output = Path(snapshot_path) if snapshot_path is not None else workspace.snapshot_path

    imported = []
    for source_id in selected_ids:
        source = registry.get(source_id)
        source_root = materialize_source(source, workspace.root)
        imported.extend(import_dataset(source_root, source))

    validation = validate_dataset(imported, selected_registry)
    if not validation.is_valid:
        raise ValueError(validation.format_errors())

    deduplication = deduplicate_samples(
        imported,
        perceptual_threshold=perceptual_threshold,
    )
    split_samples = assign_splits(
        deduplication.samples,
        seed=seed,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
    )
    validation = validate_dataset(split_samples, selected_registry)
    if not validation.is_valid:
        raise ValueError(validation.format_errors())

    save_manifest(manifest_output, split_samples)
    report = build_report(
        split_samples,
        perceptual_threshold=perceptual_threshold,
        deduplication=deduplication,
    )
    report_output.parent.mkdir(parents=True, exist_ok=True)
    report_output.write_text(
        json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    snapshot = build_snapshot(
        manifest_path=manifest_output,
        registry_path=registry_file,
        registry=registry,
        source_ids=selected_ids,
        report=report,
        split_seed=seed,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        perceptual_threshold=perceptual_threshold,
    )
    save_snapshot(snapshot_output, snapshot)

    return DatasetRebuildResult(
        manifest_path=manifest_output,
        report_path=report_output,
        snapshot_path=snapshot_output,
        report=report,
        snapshot=snapshot,
    )


def _select_source_ids(
    registry: DatasetRegistry,
    source_ids: Sequence[str] | None,
) -> tuple[str, ...]:
    if source_ids is None:
        selected = tuple(source.id for source in registry.sources)
    else:
        selected = tuple(dict.fromkeys(source_ids))
        for source_id in selected:
            registry.get(source_id)
    if not selected:
        raise ValueError("dataset registry contains no selected sources")
    return selected
