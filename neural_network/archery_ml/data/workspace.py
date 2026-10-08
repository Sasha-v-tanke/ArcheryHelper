from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


DEFAULT_DATA_ROOT = Path("data")


@dataclass(frozen=True)
class DatasetWorkspace:
    root: Path = DEFAULT_DATA_ROOT

    @property
    def sources_dir(self) -> Path:
        return self.root / "sources"

    @property
    def pipeline_dir(self) -> Path:
        return self.root / "dataset_pipeline"

    @property
    def cache_dir(self) -> Path:
        return self.pipeline_dir / "cache"

    @property
    def manifests_dir(self) -> Path:
        return self.root / "manifests"

    @property
    def reports_dir(self) -> Path:
        return self.root / "reports"

    @property
    def snapshots_dir(self) -> Path:
        return self.root / "snapshots"

    @property
    def manifest_path(self) -> Path:
        return self.manifests_dir / "dataset_manifest.json"

    @property
    def report_path(self) -> Path:
        return self.reports_dir / "dataset_report.json"

    @property
    def snapshot_path(self) -> Path:
        return self.snapshots_dir / "dataset_snapshot.json"

    def ensure_output_dirs(self) -> None:
        self.sources_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.manifests_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.snapshots_dir.mkdir(parents=True, exist_ok=True)
