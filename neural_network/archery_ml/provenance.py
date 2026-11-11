"""Fail-closed split and checkpoint provenance for canonical Stage 4 data."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from neural_network.archery_ml.data.snapshot import load_snapshot
from neural_network.archery_ml.data.view import DatasetView


SPLITS = ("train", "val", "test", "mobile_dev", "mobile_real_test")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fingerprint(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode("utf-8")).hexdigest()


def verify_negative_faces(sample, face_indices: list[int]) -> None:
    """Negative labels require explicit independent empty-face review.

    A missing canonical impact does not prove that a face has no arrow. Stage 4
    materialization records the reviewer-confirmed empty face IDs separately.
    """
    annotated = {point.face_index for point in sample.annotations}
    expected_empty = set(face_indices) - annotated
    review = (sample.capture_metadata or {}).get("stage4_review") or {}
    declared = review.get("verified_empty_faces")
    if expected_empty or declared is not None:
        if (not isinstance(declared, list)
                or any(type(index) is not int for index in declared)
                or len(declared) != len(set(declared))
                or set(declared) != expected_empty):
            raise ValueError(f"missing or inconsistent verified_empty_faces: {sample.id}")


def verify_dataset(config: dict, *, snapshot=None) -> dict:
    """Validate all split boundaries, including the exact canonical model inputs.

    Returns sorted provenance suitable for embedding in an immutable checkpoint.
    Unprepared raw-only samples do not count as model inputs, but groups are still
    checked across the entire manifest. Every prepared crop is checked on disk.
    """
    manifest = Path(config["dataset"]["manifest"])
    if snapshot is None:
        snapshot = load_snapshot(config["dataset"]["snapshot"])
    manifest_hash = sha256(manifest)
    if snapshot.manifest_sha256 != manifest_hash:
        raise ValueError("manifest SHA256 differs from immutable dataset snapshot")
    workspace = manifest.parent.parent.resolve()
    groups: dict[tuple[str | None, str], str] = {}
    sample_ids: set[str] = set()
    image_splits: dict[Path, str] = {}
    original_splits: dict[str, str] = {}
    crop_splits: dict[str, str] = {}
    per_split = {split: {"original_sha256": set(), "canonical_crop_sha256": set(),
                         "groups": set(), "annotation_records": []} for split in SPLITS}
    for sample in DatasetView.load(manifest):
        if sample.id in sample_ids:
            raise ValueError(f"duplicate manifest sample ID: {sample.id}")
        sample_ids.add(sample.id)
        if sample.split not in per_split:
            raise ValueError(f"unknown split: {sample.split}")
        if sample.group_id is not None:
            key = (sample.source_id, sample.group_id)
            previous = groups.setdefault(key, sample.split)
            if previous != sample.split:
                raise ValueError(f"group leakage between {previous} and {sample.split}: {key}")
        entries = (sample.capture_metadata or {}).get("canonical_faces")
        if entries is None:
            continue
        if not sample.group_id or not sample.source_id:
            raise ValueError(f"canonical face requires source_id and group_id: {sample.id}")
        image_path = sample.image_path.resolve()
        if not image_path.is_relative_to(workspace) or not image_path.is_file():
            raise ValueError(f"missing or external original source image: {sample.id}")
        previous = image_splits.setdefault(image_path, sample.split)
        if previous != sample.split:
            raise ValueError(f"raw image leakage between {previous} and {sample.split}")
        original_hash = sha256(image_path)
        reviewed_hash = ((sample.capture_metadata or {}).get("stage4_review") or {}).get("source_image_sha256")
        if reviewed_hash is not None and reviewed_hash != original_hash:
            raise ValueError(f"original image SHA256 differs from Stage 4 review: {sample.id}")
        previous = original_splits.setdefault(original_hash, sample.split)
        if previous != sample.split:
            raise ValueError(f"original image SHA256 leakage between {previous} and {sample.split}")
        split = per_split[sample.split]
        split["original_sha256"].add(original_hash)
        split["groups"].add((sample.source_id, sample.group_id))
        if not isinstance(entries, list) or not entries:
            raise ValueError(f"invalid canonical_faces for {sample.id}")
        face_indices: list[int] = []
        for entry in entries:
            if not isinstance(entry, dict) or entry.get("annotation_status") != "verified":
                raise ValueError(f"unverified canonical crop: {sample.id}")
            name = entry.get("image_path")
            if not isinstance(name, str) or not name.strip():
                raise ValueError(f"missing canonical crop path: {sample.id}")
            path = Path(name)
            path = (path if path.is_absolute() else manifest.parent / path).resolve()
            if not path.is_relative_to(workspace) or not path.is_file():
                raise ValueError(f"missing or external canonical crop: {sample.id}")
            crop_hash = sha256(path)
            if entry.get("sha256") != crop_hash:
                raise ValueError(f"canonical crop SHA256 mismatch: {sample.id}")
            previous = crop_splits.setdefault(crop_hash, sample.split)
            if previous != sample.split:
                raise ValueError(f"canonical crop SHA256 leakage between {previous} and {sample.split}")
            split["canonical_crop_sha256"].add(crop_hash)
            face_index = entry.get("face_index")
            if type(face_index) is not int or face_index < 0:
                raise ValueError(f"invalid canonical face_index in {sample.id}")
            face_indices.append(face_index)
            labels = [annotation.to_dict() for annotation in sample.annotations
                      if annotation.face_index == face_index]
            labels.sort(key=lambda item: json.dumps(item, sort_keys=True, separators=(", ", ":"),
                                                    allow_nan=False))
            split["annotation_records"].append({
                "sample_id": sample.id, "source_id": sample.source_id,
                "group_id": sample.group_id, "original_sha256": original_hash,
                "canonical_crop_sha256": crop_hash, "face_index": face_index,
                "annotations": labels,
                "target_metadata": (None if sample.target_metadata is None
                                    else sample.target_metadata.to_dict()),
            })
        if (len(set(face_indices)) != len(face_indices)
                or any(annotation.face_index not in face_indices
                       for annotation in sample.annotations)):
            raise ValueError(f"canonical impact/face assignment mismatch: {sample.id}")
        verify_negative_faces(sample, face_indices)
    return {
        "version": 2,
        "manifest_sha256": manifest_hash,
        "splits": {
            name: {
                "original_sha256": sorted(values["original_sha256"]),
                "canonical_crop_sha256": sorted(values["canonical_crop_sha256"]),
                "groups": [list(group) for group in sorted(values["groups"])],
                "annotation_sha256": fingerprint(sorted(
                    values["annotation_records"],
                    key=lambda record: (record["sample_id"], record["face_index"],
                                        record["canonical_crop_sha256"]))),
            }
            for name, values in per_split.items()
        },
    }


def training_provenance(dataset: dict) -> dict:
    """Only train + best-checkpoint-selection (val) may influence checkpoint."""
    return {
        "version": 2,
        "manifest_sha256": dataset["manifest_sha256"],
        "splits": {split: dataset["splits"][split] for split in ("train", "val")},
    }


def verify_checkpoint_holdout(checkpoint: dict, metadata: dict, dataset: dict,
                              *, split: str) -> None:
    saved = checkpoint.get("training_provenance")
    if not isinstance(saved, dict) or saved.get("version") != 2:
        raise ValueError("checkpoint lacks versioned training/validation annotation provenance")
    saved_hash = fingerprint(saved)
    if metadata.get("training_provenance_sha256") != saved_hash:
        raise ValueError("checkpoint and metadata training provenance differ")
    if (not isinstance(saved.get("splits"), dict)
            or set(saved["splits"]) != {"train", "val"}):
        raise ValueError("incomplete checkpoint training/validation provenance")
    for source_split in ("train", "val"):
        recorded = saved["splits"][source_split]
        if not isinstance(recorded, dict) or not isinstance(recorded.get("annotation_sha256"), str):
            raise ValueError(f"checkpoint lacks {source_split} annotation SHA256")
        current = dataset["splits"][source_split]
        # A holdout-only manifest is allowed; when train/val is supplied it
        # must be exactly the data and GT used for fitting/checkpoint selection.
        if current["original_sha256"] and current != recorded:
            raise ValueError(f"{source_split} data/annotations differ from checkpoint provenance")
    selected = dataset["splits"][split]
    if split == "val":
        if selected != saved["splits"]["val"]:
            raise ValueError("validation data differs from checkpoint selection provenance")
        return
    for source_split in ("train", "val"):
        used = saved["splits"][source_split]
        for key in ("original_sha256", "canonical_crop_sha256", "groups"):
            previous = {tuple(v) if isinstance(v, list) else v for v in used[key]}
            current = {tuple(v) if isinstance(v, list) else v for v in selected[key]}
            if previous & current:
                raise ValueError(f"checkpoint {source_split} overlaps {split} by {key}")
