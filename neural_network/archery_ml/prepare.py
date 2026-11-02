"""Convert independently reviewed Stage 3 crops and raw impacts into Stage 4 data.

This bridge never guesses face ownership or invents points for an empty target.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
from PIL import Image

from neural_network.archery_ml.contracts import ImpactAnnotation, load_manifest, save_manifest
# Stage 3's canonical geometry defines the 512x512 frame and 255.5/256
# center/radius convention. Keep this lightweight bridge independent of
# importing Stage 3's OpenCV geometry package at runtime.
CANONICAL_SIZE = 512
CANONICAL_CENTER_PX = (CANONICAL_SIZE - 1) / 2
CANONICAL_RADIUS_PX = CANONICAL_SIZE / 2


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _verified_path(value: str, *, root: Path, relative_to: Path) -> Path:
    candidate = Path(value)
    resolved = (candidate if candidate.is_absolute() else relative_to / candidate).resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError(f"file escapes dataset workspace: {value}")
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return resolved


def _raw_labels_digest(sample) -> str:
    labels = [annotation.to_dict() for annotation in sample.raw_annotations]
    return hashlib.sha256(json.dumps(labels, sort_keys=True, separators=(",", ":"),
                                      allow_nan=False).encode()).hexdigest()


def reviewed_sample(sample, review: dict, *, data_root: Path, review_dir: Path, manifest_dir: Path):
    """Return verified per-face coordinates and crop paths for one sample.

    All review assertions must come from an independently prepared mapping, not
    from automatic target detection or a model prediction.
    """
    if not sample.source_id or not sample.group_id:
        raise ValueError(f"source_id/group_id absent: {sample.id}")
    if sample.annotations:
        raise ValueError(f"mixed existing canonical and raw impact labels: {sample.id}")
    if review.get("annotator") == review.get("reviewer") or not all(
        isinstance(review.get(key), str) and review[key].strip()
        for key in ("annotator", "reviewer")
    ):
        raise ValueError(f"independent annotation review required: {sample.id}")
    if review.get("face_ownership_reviewed") is not True:
        raise ValueError(f"unreviewed target face ownership: {sample.id}")
    if review.get("raw_annotations_sha256") != _raw_labels_digest(sample):
        raise ValueError(f"raw impact annotations differ from independent review: {sample.id}")
    original = sample.image_path.resolve()
    if not original.is_relative_to(data_root.resolve()) or not original.is_file():
        raise ValueError(f"source image unavailable or outside data workspace: {sample.id}")
    original_sha = _sha256(original)
    if review.get("original_sha256") != original_sha:
        raise ValueError(f"original image SHA256 mismatch: {sample.id}")
    geometry_path = _verified_path(review["geometry_path"], root=data_root, relative_to=review_dir)
    if _sha256(geometry_path) != review.get("geometry_sha256"):
        raise ValueError(f"Stage 3 geometry SHA256 mismatch: {sample.id}")
    geometry = json.loads(geometry_path.read_text(encoding="utf-8"))
    if geometry.get("status") != "READY" or geometry.get("quality_status") != "READY":
        raise ValueError(f"Stage 3 geometry is not READY: {sample.id}")
    if not geometry.get("artifact_revision"):
        raise ValueError(f"unversioned geometry: {sample.id}")
    faces = geometry.get("faces")
    if not isinstance(faces, list):
        raise ValueError(f"missing face geometry: {sample.id}")
    count = 3 if sample.target_metadata and sample.target_metadata.format == "TRIPLE" else 1
    if count == 3 and geometry.get("physical_face_order_verified") is not True:
        raise ValueError(f"triple face order not independently confirmed: {sample.id}")
    if len(faces) != count or {f.get("face_index") for f in faces} != set(range(count)):
        raise ValueError(f"incorrect number/identity of selected faces: {sample.id}")
    empty_indices = review.get("verified_empty_faces", [])
    if (not isinstance(empty_indices, list) or any(type(i) is not int for i in empty_indices)
            or len(empty_indices) != len(set(empty_indices))
            or not set(empty_indices) <= set(range(count))):
        raise ValueError(f"invalid verified_empty_faces: {sample.id}")
    with Image.open(original) as original_image:
        original_width, original_height = original_image.size

    entries = []
    transforms = {}
    for face in sorted(faces, key=lambda f: f["face_index"]):
        index = face["face_index"]
        crop = _verified_path(face["crop_path"], root=data_root, relative_to=geometry_path.parent)
        with Image.open(crop) as image:
            if image.mode != "RGB" or image.size != (CANONICAL_SIZE, CANONICAL_SIZE):
                raise ValueError(f"Stage 3 canonical crop has wrong format/dimensions: {sample.id}")
        h = np.asarray(face["original_to_canonical"], dtype=np.float64)
        if h.shape != (3, 3) or not np.isfinite(h).all() or abs(np.linalg.det(h)) < 1e-12:
            raise ValueError(f"invalid original-to-canonical homography: {sample.id}")
        transforms[index] = h
        # Rechecking crop SHA256 at dataset construction and every epoch closes
        # the gap between this review map and what training consumes.
        entries.append({"face_index": index,
                        "image_path": Path(os.path.relpath(crop, manifest_dir)).as_posix(),
                        "sha256": _sha256(crop), "annotation_status": "verified",
                        "geometry_revision": geometry["artifact_revision"]})

    assigned_indices = review.get("impact_face_indices")
    if (not isinstance(assigned_indices, list)
            or len(assigned_indices) != len(sample.raw_annotations)
            or any(item is not None and (type(item) is not int or item not in transforms)
                   for item in assigned_indices)):
        raise ValueError(f"impact_face_indices must explicitly review every raw point: {sample.id}")
    transformed = []
    annotated_counts = {index: 0 for index in range(count)}
    for point, index in zip(sample.raw_annotations, assigned_indices):
        if index is None:
            # This raw annotation belongs to a foreign/irrelevant face, and
            # cannot be treated as a supervised target on the selected board.
            continue
        vec = transforms[index] @ np.asarray([point.x_fraction * original_width,
                                               point.y_fraction * original_height, 1.0])
        if not np.isfinite(vec).all() or abs(vec[2]) < 1e-12:
            raise ValueError(f"impact transform produced infinity: {sample.id}")
        norm_x = (vec[0] / vec[2] - CANONICAL_CENTER_PX) / CANONICAL_RADIUS_PX
        norm_y = (vec[1] / vec[2] - CANONICAL_CENTER_PX) / CANONICAL_RADIUS_PX
        if not all(math.isfinite(v) and -1 <= v <= 1 for v in (norm_x, norm_y)):
            raise ValueError(f"reviewed impact lies outside canonical crop: {sample.id}")
        transformed.append(ImpactAnnotation(x_norm=float(norm_x), y_norm=float(norm_y),
                                            face_index=index))
        annotated_counts[index] += 1
    if any((annotated_counts[index] == 0) != (index in empty_indices) for index in range(count)):
        raise ValueError(f"empty-face assertions disagree with reviewed impact labels: {sample.id}")
    metadata = dict(sample.capture_metadata or {})
    if "canonical_faces" in metadata:
        raise ValueError(f"existing canonical faces cannot be overwritten: {sample.id}")
    metadata["canonical_faces"] = entries
    metadata["stage4_review"] = {"source_image_sha256": original_sha,
                                 "source_annotations_sha256": _raw_labels_digest(sample),
                                 "geometry_sha256": review["geometry_sha256"],
                                 "impact_face_indices": assigned_indices,
                                 "annotator": review["annotator"], "reviewer": review["reviewer"]}
    return replace(sample, annotations=tuple(transformed), capture_metadata=metadata)


def materialize(*, manifest: Path, snapshot: Path, review_map: Path,
                output_manifest: Path, output_snapshot: Path, data_root: Path) -> dict:
    """Create new immutable manifest/snapshot; never mutate the Stage 2.5 inputs."""
    root = data_root.resolve()
    source_files = (manifest.resolve(), snapshot.resolve(), review_map.resolve())
    target_files = (output_manifest.resolve(), output_snapshot.resolve())
    if len(set((*source_files, *target_files))) != 5 or any(p.exists() for p in target_files):
        raise ValueError("outputs must be new files distinct from source inputs")
    if any(not p.is_relative_to(root) for p in (*source_files, *target_files)):
        raise ValueError("Stage 4 inputs/outputs must stay inside data workspace")
    source_snapshot = json.loads(snapshot.read_text(encoding="utf-8"))
    old_digest = _sha256(manifest)
    if source_snapshot.get("manifest_sha256") != old_digest:
        raise ValueError("source manifest does not match dataset snapshot")
    review_bytes = review_map.read_bytes()
    mappings = json.loads(review_bytes)
    if not isinstance(mappings, dict) or not mappings:
        raise ValueError("review mapping is empty or invalid")
    source_samples = load_manifest(manifest)
    ids = [sample.id for sample in source_samples]
    if len(ids) != len(set(ids)) or not set(mappings) <= set(ids):
        raise ValueError("duplicate or unknown reviewed sample IDs")
    prepared = []
    for sample in source_samples:
        prepared.append(reviewed_sample(sample, mappings[sample.id], data_root=root,
                         review_dir=review_map.parent, manifest_dir=output_manifest.parent)
                        if sample.id in mappings else sample)
    if not any(sample.annotations and (sample.capture_metadata or {}).get("canonical_faces")
               for sample in prepared):
        raise ValueError("no verified impact coordinates in reviewed samples")
    # Prepare complete manifest first; update its corresponding derived snapshot
    # only after the source snapshot and review artifacts have been verified.
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    output_snapshot.parent.mkdir(parents=True, exist_ok=True)
    save_manifest(output_manifest, prepared)
    try:
        derived = dict(source_snapshot)
        derived["manifest_sha256"] = _sha256(output_manifest)
        additional_canonical = sum(len(s.annotations) for s in prepared) - sum(
            len(s.annotations) for s in source_samples
        )
        if isinstance(derived.get("report"), dict):
            original_report = source_snapshot["report"]
            derived["report"] = dict(original_report)
            derived["report"]["canonical_impacts"] = int(original_report.get("canonical_impacts", 0)) + additional_canonical
            derived["report"]["impacts"] = int(original_report["impacts"]) + additional_canonical
        derived["stage4_materialization"] = {"source_manifest_sha256": old_digest,
                                              "source_snapshot_sha256": _sha256(snapshot),
                                              "review_map_sha256": hashlib.sha256(review_bytes).hexdigest(),
                                              "reviewed_samples": len(mappings)}
        output_snapshot.write_text(json.dumps(derived, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    except Exception:
        output_manifest.unlink(missing_ok=True)
        raise
    return {"manifest": str(output_manifest), "snapshot": str(output_snapshot),
            "reviewed_samples": len(mappings),
            "canonical_impacts": sum(len(sample.annotations) for sample in prepared)}


def write_review_draft(*, manifest: Path, output: Path, data_root: Path,
                       include_unlabeled: bool = False) -> dict:
    """Generate an unapproved review queue from imported raw annotations.

    A draft is never fed into training as verified data, and no point is
    automatically associated with an owned target face.
    """
    root = data_root.resolve()
    destination = output.resolve()
    if output.exists() or not destination.is_relative_to(root):
        raise ValueError("draft review path must be new and under data workspace")
    if not manifest.resolve().is_relative_to(root):
        raise ValueError("manifest must be under data workspace")
    mapping = {}
    for sample in load_manifest(manifest):
        if not sample.raw_annotations and not include_unlabeled:
            continue
        original = sample.image_path.resolve()
        if not original.is_relative_to(root) or not original.is_file():
            raise ValueError(f"unavailable original for review draft: {sample.id}")
        if sample.id in mapping:
            raise ValueError(f"duplicate sample id: {sample.id}")
        mapping[sample.id] = {
            "annotator": "", "reviewer": "", "face_ownership_reviewed": False,
            "original_sha256": _sha256(original),
            "raw_annotations_sha256": _raw_labels_digest(sample),
            "geometry_path": "", "geometry_sha256": "",
            "raw_impact_count": len(sample.raw_annotations),
            "source_face_index_hints": [point.face_index for point in sample.raw_annotations],
            "impact_face_indices": [None] * len(sample.raw_annotations),
            "verified_empty_faces": [],
        }
    if not mapping:
        raise ValueError("no candidate raw samples; use --include-unlabeled to review empty candidates")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(mapping, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"draft_review_map": str(destination), "unapproved_samples": len(mapping)}


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage 3 reviewed faces -> immutable Stage 4 manifest")
    parser.add_argument("--manifest", type=Path, default=Path("data/manifests/dataset_manifest.json"))
    parser.add_argument("--snapshot", type=Path, default=Path("data/snapshots/dataset_snapshot.json"))
    parser.add_argument("--review-map", type=Path)
    parser.add_argument("--draft-review-map", type=Path,
                        help="Create an unapproved raw-impact review queue, do not materialize")
    parser.add_argument("--include-unlabeled", action="store_true",
                        help="In review draft, include samples without any raw impact annotations")
    parser.add_argument("--output-manifest", type=Path, default=Path("data/manifests/stage4_manifest.json"))
    parser.add_argument("--output-snapshot", type=Path, default=Path("data/snapshots/stage4_snapshot.json"))
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    args = parser.parse_args()
    if args.draft_review_map is not None:
        if args.review_map is not None:
            parser.error("--draft-review-map and --review-map are mutually exclusive")
        print(json.dumps(write_review_draft(manifest=args.manifest,
                            output=args.draft_review_map, data_root=args.data_root,
                            include_unlabeled=args.include_unlabeled), indent=2))
        return 0
    if args.review_map is None or args.include_unlabeled:
        parser.error("--review-map is required for materialization; --include-unlabeled is draft-only")
    print(json.dumps(materialize(manifest=args.manifest, snapshot=args.snapshot,
                                 review_map=args.review_map, output_manifest=args.output_manifest,
                                 output_snapshot=args.output_snapshot, data_root=args.data_root), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
