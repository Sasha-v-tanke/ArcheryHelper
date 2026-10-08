from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from neural_network.archery_ml.contracts import SCHEMA_VERSION
from neural_network.archery_ml.data.registry import DatasetRegistry, DatasetSource
from neural_network.archery_ml.data.report import DatasetReport


SNAPSHOT_VERSION = 1
PIPELINE_VERSION = 1


@dataclass(frozen=True)
class DatasetSourceSnapshot:
    id: str
    version: str
    license: str
    author: str
    checksum_sha256: str | None
    annotation_space: str
    allowed_tasks: tuple[str, ...]
    homepage_url: str | None

    @staticmethod
    def from_source(source: DatasetSource) -> "DatasetSourceSnapshot":
        return DatasetSourceSnapshot(
            id=source.id,
            version=source.version,
            license=source.license,
            author=source.author,
            checksum_sha256=source.checksum_sha256,
            annotation_space=source.annotation_space,
            allowed_tasks=tuple(sorted(source.allowed_tasks)),
            homepage_url=source.homepage_url,
        )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "version": self.version,
            "license": self.license,
            "author": self.author,
            "checksum_sha256": self.checksum_sha256,
            "annotation_space": self.annotation_space,
            "allowed_tasks": list(self.allowed_tasks),
            "homepage_url": self.homepage_url,
        }

    @staticmethod
    def from_dict(data: dict) -> "DatasetSourceSnapshot":
        return DatasetSourceSnapshot(
            id=str(data["id"]),
            version=str(data["version"]),
            license=str(data["license"]),
            author=str(data["author"]),
            checksum_sha256=data.get("checksum_sha256"),
            annotation_space=str(data["annotation_space"]),
            allowed_tasks=tuple(str(value) for value in data.get("allowed_tasks", ())),
            homepage_url=data.get("homepage_url"),
        )


@dataclass(frozen=True)
class DatasetSnapshot:
    snapshot_version: int
    pipeline_version: int
    dataset_schema_version: int
    manifest_sha256: str
    registry_sha256: str
    split_seed: int
    train_ratio: float
    val_ratio: float
    test_ratio: float
    perceptual_threshold: int
    sources: tuple[DatasetSourceSnapshot, ...]
    report: DatasetReport

    def to_dict(self) -> dict:
        return {
            "snapshot_version": self.snapshot_version,
            "pipeline_version": self.pipeline_version,
            "dataset_schema_version": self.dataset_schema_version,
            "manifest_sha256": self.manifest_sha256,
            "registry_sha256": self.registry_sha256,
            "split": {
                "seed": self.split_seed,
                "train_ratio": self.train_ratio,
                "val_ratio": self.val_ratio,
                "test_ratio": self.test_ratio,
            },
            "perceptual_threshold": self.perceptual_threshold,
            "sources": [source.to_dict() for source in self.sources],
            "report": self.report.to_dict(),
        }

    @staticmethod
    def from_dict(data: dict) -> "DatasetSnapshot":
        split = data["split"]
        report = data["report"]
        return DatasetSnapshot(
            snapshot_version=int(data["snapshot_version"]),
            pipeline_version=int(data["pipeline_version"]),
            dataset_schema_version=int(data["dataset_schema_version"]),
            manifest_sha256=str(data["manifest_sha256"]),
            registry_sha256=str(data["registry_sha256"]),
            split_seed=int(split["seed"]),
            train_ratio=float(split["train_ratio"]),
            val_ratio=float(split["val_ratio"]),
            test_ratio=float(split["test_ratio"]),
            perceptual_threshold=int(data["perceptual_threshold"]),
            sources=tuple(
                DatasetSourceSnapshot.from_dict(item) for item in data.get("sources", ())
            ),
            report=DatasetReport(
                images=int(report["images"]),
                impacts=int(report["impacts"]),
                canonical_impacts=int(report.get("canonical_impacts", report["impacts"])),
                raw_impacts=int(report.get("raw_impacts", 0)),
                geometry_annotations=int(report.get("geometry_annotations", 0)),
                source_distribution={
                    str(key): int(value)
                    for key, value in report.get("source_distribution", {}).items()
                },
                target_distribution={
                    str(key): int(value)
                    for key, value in report.get("target_distribution", {}).items()
                },
                split_sizes={
                    str(key): int(value)
                    for key, value in report.get("split_sizes", {}).items()
                },
                exact_duplicate_count=int(report["exact_duplicate_count"]),
                near_duplicate_group_count=int(report["near_duplicate_group_count"]),
                duplicate_rate=float(report["duplicate_rate"]),
            ),
        )


def build_snapshot(
    *,
    manifest_path: Path,
    registry_path: Path,
    registry: DatasetRegistry,
    source_ids: Sequence[str],
    report: DatasetReport,
    split_seed: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    perceptual_threshold: int,
) -> DatasetSnapshot:
    selected = tuple(
        sorted(
            (DatasetSourceSnapshot.from_source(registry.get(source_id)) for source_id in source_ids),
            key=lambda source: source.id,
        )
    )
    return DatasetSnapshot(
        snapshot_version=SNAPSHOT_VERSION,
        pipeline_version=PIPELINE_VERSION,
        dataset_schema_version=SCHEMA_VERSION,
        manifest_sha256=_sha256(manifest_path),
        registry_sha256=_sha256(registry_path),
        split_seed=split_seed,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        perceptual_threshold=perceptual_threshold,
        sources=selected,
        report=report,
    )


def save_snapshot(path: str | Path, snapshot: DatasetSnapshot) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(snapshot.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def load_snapshot(path: str | Path) -> DatasetSnapshot:
    return DatasetSnapshot.from_dict(
        json.loads(Path(path).read_text(encoding="utf-8"))
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
