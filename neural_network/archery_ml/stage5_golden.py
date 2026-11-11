"""Derive a reproducible real golden set from the verified Stage 4 manifest.

This tool selects existing reviewed canonical crops. It does NOT claim the Stage 4
labels are independently audited; that remains an explicit prerequisite.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from pathlib import Path

from neural_network.archery_ml.data.view import DatasetView
from neural_network.archery_ml.impact_dataset import CanonicalImpactDataset
from neural_network.archery_ml.inference import load_detector
from neural_network.archery_ml.provenance import verify_dataset
from neural_network.archery_ml.stage5_parity import (
    MAX_GOLDEN, MIN_GOLDEN, read_golden_manifest, sha256_file,
)


def select_goldens(records: list, count: int) -> list:
    """Prefer both a verified-empty and a positive face with distinct originals."""
    if not MIN_GOLDEN <= count <= MAX_GOLDEN:
        raise ValueError("golden count must be between 20 and 50")

    # Deterministic selection independent of manifest order.
    ordered = sorted(
        records,
        key=lambda record: (
            hashlib.sha256((record.source_image_sha256 + record.sha256).encode()).hexdigest(),
            record.sample_id, record.face_index,
        ),
    )
    selected = []
    origins: set[str] = set()
    hashes: set[str] = set()

    def add(record) -> bool:
        if record.source_image_sha256 in origins or record.sha256 in hashes:
            return False
        origins.add(record.source_image_sha256)
        hashes.add(record.sha256)
        selected.append(record)
        return True

    # Both negative and positive examples are required for a useful golden set.
    positives = [record for record in ordered if record.points]
    negatives = [record for record in ordered if not record.points]
    if not positives or not negatives:
        raise ValueError("golden set requires reviewed positive and verified-empty faces")
    if not add(positives[0]) or not any(add(item) for item in negatives):
        raise ValueError("golden positive and negative faces must have distinct originals")
    for record in ordered:
        if len(selected) == count:
            break
        add(record)
    if len(selected) != count:
        raise ValueError(f"need {count} distinct golden originals, found {len(selected)}")
    return sorted(selected, key=lambda item: (item.sample_id, item.face_index))


def prepare_golden(
    checkpoint: Path,
    metadata_path: Path,
    manifest: Path,
    snapshot: Path,
    output_directory: Path,
    *,
    count: int = 32,
) -> Path:
    checkpoint, metadata_path = Path(checkpoint), Path(metadata_path)
    manifest, snapshot = Path(manifest), Path(snapshot)
    output_directory = Path(output_directory).absolute()
    if output_directory.exists():
        raise FileExistsError("golden destination already exists")
    # Stage 4 loader refuses stale metadata and tampered checkpoint provenance.
    _, metadata = load_detector(checkpoint, metadata_path, device="cpu")
    metadata_document = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata_document.get("training_dataset_sha256") != sha256_file(manifest):
        raise ValueError("Stage 4 manifest is not the one bound to the trained checkpoint")
    # Verify snapshot and split integrity for the ENTIRE dataset, not only the chosen crops.
    verify_dataset({"dataset": {"manifest": str(manifest), "snapshot": str(snapshot)}})
    view = DatasetView.load(manifest).select(splits=("val", "test"))
    sample_splits = {sample.id: sample.split for sample in view}
    if len(sample_splits) != len(view):
        raise ValueError("ambiguous sample IDs")
    faces = CanonicalImpactDataset(
        view,
        manifest_dir=manifest.parent,
        input_size=metadata.input.width,
        stride=metadata.output.stride,
        augment=False,
    )
    selected = select_goldens(faces.records, count)
    output_directory.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".golden-", dir=output_directory.parent))
    try:
        samples = []
        for index, record in enumerate(selected):
            if sha256_file(record.image_path) != record.sha256:
                raise ValueError(f"source golden crop changed: {record.sample_id}")
            relative = Path("images") / f"canonical-{index:03d}.png"
            destination = staging / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(record.image_path, destination)
            if sha256_file(destination) != record.sha256:
                raise ValueError("copied golden crop SHA256 mismatch")
            samples.append({
                "sample_id": f"golden-{index:03d}",
                "image_path": relative.as_posix(),
                "crop_sha256": record.sha256,
                "original_sha256": record.source_image_sha256,
                "face_index": record.face_index,
                "split": sample_splits[record.sample_id],
                "stage4_sample_id": record.sample_id,
                "reviewed_gt_impact_count": len(record.points),
            })
        golden = staging / "golden.json"
        golden.write_text(json.dumps({
            "schema_version": 1,
            "source_manifest_sha256": sha256_file(manifest),
            "source_snapshot_sha256": sha256_file(snapshot),
            "source_checkpoint_sha256": sha256_file(checkpoint),
            "samples": samples,
        }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        read_golden_manifest(golden, metadata)
        if output_directory.exists():
            raise FileExistsError("golden destination appeared during preparation")
        staging.rename(output_directory)
        return output_directory / "golden.json"
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Select 20–50 reviewed canonical crops from Stage 4 for ExecuTorch parity"
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--stage4-manifest", type=Path, required=True)
    parser.add_argument("--stage4-snapshot", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--count", type=int, default=32)
    options = parser.parse_args()
    print(prepare_golden(options.checkpoint, options.metadata,
                         options.stage4_manifest, options.stage4_snapshot,
                         options.output_directory, count=options.count))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
