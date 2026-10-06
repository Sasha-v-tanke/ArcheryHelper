from __future__ import annotations

import os
import re
from pathlib import Path

from neural_network.archery_ml.contracts import SplitName, TargetMetadata
from neural_network.archery_ml.data.registry import DatasetSource


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def image_files(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def sample_id(source: DatasetSource, root: Path, image_path: Path) -> str:
    relative = image_path.relative_to(root).with_suffix("").as_posix()
    return f"{source.id}:{relative}"


def portable_path(path: Path) -> Path:
    return Path(os.path.relpath(path.resolve(), Path.cwd().resolve()))


def group_id(source: DatasetSource, root: Path, image_path: Path) -> str:
    relative = image_path.relative_to(root).as_posix()
    pattern = source.options.get("group_regex")
    if pattern:
        match = re.search(str(pattern), relative)
        if not match:
            raise ValueError(f"group_regex did not match {relative} for source {source.id}")
        raw = match.group(1) if match.groups() else match.group(0)
    else:
        raw = image_path.stem
    return f"{source.id}:{raw}"


def split_name(source: DatasetSource, image_path: Path) -> SplitName:
    default = str(source.options.get("default_split", "train"))
    if default == "mobile_real_test":
        return "mobile_real_test"

    parts = {part.lower() for part in image_path.parts}
    if "test" in parts:
        return "test"
    if "valid" in parts or "validation" in parts or "val" in parts:
        return "val"
    if "train" in parts:
        return "train"
    if default not in {"train", "val", "test", "mobile_real_test"}:
        raise ValueError(f"unsupported default split for {source.id}: {default}")
    return default  # type: ignore[return-value]


def target_metadata(source: DatasetSource) -> TargetMetadata | None:
    value = source.options.get("target_metadata")
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError(f"target_metadata must be an object for source {source.id}")
    return TargetMetadata.from_dict(value)


def class_role(source: DatasetSource, class_id: str, class_name: str | None = None) -> str:
    if class_name is not None and class_name in source.class_mapping:
        return source.class_mapping[class_name]
    if class_id in source.class_mapping:
        return source.class_mapping[class_id]
    if not source.class_mapping:
        return "impact"
    return "ignore"


def image_to_canonical(x_fraction: float, y_fraction: float) -> tuple[float, float]:
    return (x_fraction - 0.5) * 2.0, (y_fraction - 0.5) * 2.0
