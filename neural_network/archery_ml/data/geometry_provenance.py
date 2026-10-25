"""Content binding shared by mobile import and independent evaluation."""
import hashlib
import json

from neural_network.archery_ml.contracts import DatasetSample
from pathlib import Path
from PIL import Image, ImageOps


def verify_landmark_ids(sample: DatasetSample) -> None:
    meta = sample.capture_metadata or {}
    actual = []
    for annotation in sample.geometry_annotations:
        if not annotation.canonical_points:
            continue
        ids = annotation.landmark_ids
        if not ids and len(annotation.canonical_points) == 1 and annotation.source_label:
            ids = (annotation.source_label,)
        if len(ids) != len(annotation.canonical_points):
            raise ValueError("evaluation landmarks require actual per-point annotation IDs")
        actual.extend(ids)
    declared = meta.get("evaluation_landmark_ids", [])
    if (not isinstance(declared,list) or any(not isinstance(v,str) or not v.strip() for v in declared)
            or len(set(declared)) != len(declared)
            or len(set(actual)) != len(actual) or set(declared) != set(actual)):
        raise ValueError("evaluation_landmark_ids differ from actual annotations")
    if set(actual) & set(meta.get("calibration_landmark_ids", [])):
        raise ValueError("calibration and evaluation landmarks must be independent")


def verify_image_content(image_path: Path, meta: dict, root: Path) -> None:
    if hashlib.sha256(image_path.read_bytes()).hexdigest() != meta.get("image_sha256", meta.get("original_sha256")):
        raise ValueError("mobile original checksum mismatch")
    derivative = meta.get("image_sha256", meta.get("original_sha256")) != meta.get("original_sha256")
    if derivative and not meta.get("original_path"):
        raise ValueError("derivative requires an original source path")
    if meta.get("original_path"):
        original = (root / meta["original_path"]).resolve()
        if not original.is_relative_to(root.resolve()) or hashlib.sha256(original.read_bytes()).hexdigest() != meta.get("original_sha256"):
            raise ValueError("mobile source original checksum/path mismatch")
        if derivative:
            if meta.get("image_transform") not in {"PIL.ImageOps.exif_transpose", "PIL.ImageOps.exif_transpose+RGB"}:
                raise ValueError("unsupported derivative transform")
            with Image.open(original) as source, Image.open(image_path) as actual:
                expected = ImageOps.exif_transpose(source).convert("RGB")
                if (source.getexif().get(274,1) != meta.get("exif_orientation") or expected.size != actual.size
                        or expected.tobytes() != actual.convert("RGB").tobytes()):
                    raise ValueError("derivative pixels do not match declared original transformation")


def annotation_digest(sample: DatasetSample) -> str:
    payload = {
        "gt_revision": (sample.capture_metadata or {}).get("gt_revision"),
        "target_metadata": None if sample.target_metadata is None else sample.target_metadata.to_dict(),
        "geometry_annotations": [a.to_dict() for a in sample.geometry_annotations],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def verify_mobile_gt(sample: DatasetSample) -> dict:
    """Recheck manifest annotations against the versioned source, not a CLI label."""
    meta = sample.capture_metadata or {}
    if "evaluation_landmark_ids" in meta:
        verify_landmark_ids(sample)
    if not meta.get("annotator") or not meta.get("reviewer") or meta["annotator"] == meta["reviewer"]:
        raise ValueError("GT requires independent annotation review")
    digest = annotation_digest(sample)
    if not meta.get("gt_revision") or meta.get("gt_annotations_sha256") != digest:
        raise ValueError("GT revision/annotation checksum mismatch")
    if sample.annotation_path is None:
        raise ValueError("missing versioned GT annotation manifest")
    path = sample.annotation_path
    manifest_digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if meta.get("gt_manifest_sha256") != manifest_digest:
        raise ValueError("GT annotation manifest checksum mismatch")
    records = json.loads(path.read_text(encoding="utf-8"))["samples"]
    matches = [DatasetSample.from_dict(r) for r in records if r["id"] == sample.id]
    if len(matches) != 1 or annotation_digest(matches[0]) != digest:
        raise ValueError("GT differs from versioned annotation manifest")
    derived = {"gt_annotations_sha256", "gt_manifest_sha256", "source_root"}
    if ({k: v for k, v in (matches[0].capture_metadata or {}).items() if k not in derived}
            != {k: v for k, v in meta.items() if k not in derived}):
        raise ValueError("GT capture provenance differs from annotation manifest")
    verify_image_content(sample.image_path, meta, Path(meta.get("source_root", path.parent)))
    return {"revision": meta["gt_revision"], "annotations_sha256": digest,
            "annotation_manifest_sha256": manifest_digest}
