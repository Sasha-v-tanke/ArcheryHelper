from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


DEFAULT_REGISTRY_PATH = Path("dataset_tools/sources.json")
SUPPORTED_IMPORTERS = frozenset({"legacy", "yolo", "coco", "roboflow"})
SUPPORTED_TASKS = frozenset({"impact_detection", "geometry_evaluation"})
SUPPORTED_ANNOTATION_SPACES = frozenset({"canonical", "image"})
SUPPORTED_CLASS_ROLES = frozenset(
    {"impact", "target_center", "target_face", "ring", "ignore"}
)
_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")


@dataclass(frozen=True)
class DatasetSource:
    id: str
    importer: str
    url: str
    version: str
    license: str
    author: str
    checksum_sha256: str | None
    allowed_tasks: tuple[str, ...]
    class_mapping: dict[str, str] = field(default_factory=dict)
    options: dict[str, Any] = field(default_factory=dict)
    annotation_space: str = "canonical"
    homepage_url: str | None = None

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "DatasetSource":
        source = DatasetSource(
            id=str(data["id"]),
            importer=str(data["importer"]),
            url=str(data["url"]),
            version=str(data["version"]),
            license=str(data["license"]),
            author=str(data["author"]),
            checksum_sha256=data.get("checksum_sha256", data.get("sha256")),
            allowed_tasks=tuple(str(value) for value in data.get("allowed_tasks", ())),
            class_mapping={str(key): str(value) for key, value in data.get("class_mapping", {}).items()},
            options=dict(data.get("options", {})),
            annotation_space=str(data.get("annotation_space", "canonical")),
            homepage_url=None if data.get("homepage_url") is None else str(data["homepage_url"]),
        )
        source.validate()
        return source

    def validate(self) -> None:
        if not self.id or any(char.isspace() for char in self.id):
            raise ValueError("dataset source id must be non-empty and contain no whitespace")
        if self.importer not in SUPPORTED_IMPORTERS:
            raise ValueError(f"unsupported importer for {self.id}: {self.importer}")
        if not self.url:
            raise ValueError(f"dataset source {self.id} must define url")
        if not self.version:
            raise ValueError(f"dataset source {self.id} must define version")
        if not self.license:
            raise ValueError(f"dataset source {self.id} must define license")
        if not self.author:
            raise ValueError(f"dataset source {self.id} must define author")
        if not self.allowed_tasks:
            raise ValueError(f"dataset source {self.id} must define at least one allowed task")
        unsupported_tasks = set(self.allowed_tasks) - SUPPORTED_TASKS
        if unsupported_tasks:
            raise ValueError(f"unsupported tasks for {self.id}: {sorted(unsupported_tasks)}")
        if self.annotation_space not in SUPPORTED_ANNOTATION_SPACES:
            raise ValueError(
                f"unsupported annotation_space for {self.id}: {self.annotation_space}"
            )
        if self.url.startswith(("http://", "https://")):
            if self.checksum_sha256 is None:
                raise ValueError(f"remote dataset source {self.id} must define checksum_sha256")
            if not _SHA256_RE.fullmatch(self.checksum_sha256):
                raise ValueError(f"invalid checksum_sha256 for {self.id}")
        elif self.checksum_sha256 is not None and not _SHA256_RE.fullmatch(self.checksum_sha256):
            raise ValueError(f"invalid checksum_sha256 for {self.id}")
        invalid_targets = sorted(set(self.class_mapping.values()) - SUPPORTED_CLASS_ROLES)
        if invalid_targets:
            raise ValueError(f"unsupported class mapping targets for {self.id}: {invalid_targets}")
        geometry_roles = {"target_center", "target_face", "ring"} & set(self.class_mapping.values())
        if geometry_roles and self.annotation_space != "image":
            raise ValueError(
                f"geometry annotations for {self.id} require annotation_space=image"
            )


@dataclass(frozen=True)
class DatasetRegistry:
    sources: tuple[DatasetSource, ...]

    @staticmethod
    def load(path: str | Path = DEFAULT_REGISTRY_PATH) -> "DatasetRegistry":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        version = int(payload.get("registry_version", 1))
        if version != 1:
            raise ValueError(f"unsupported dataset registry version: {version}")
        sources = tuple(DatasetSource.from_dict(item) for item in payload.get("sources", ()))
        ids = [source.id for source in sources]
        duplicates = sorted({source_id for source_id in ids if ids.count(source_id) > 1})
        if duplicates:
            raise ValueError(f"duplicate dataset source ids: {duplicates}")
        return DatasetRegistry(sources)

    def get(self, source_id: str) -> DatasetSource:
        for source in self.sources:
            if source.id == source_id:
                return source
        raise KeyError(f"unknown dataset source: {source_id}")
