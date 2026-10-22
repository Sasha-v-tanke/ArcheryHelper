"""Content binding shared by mobile import and independent evaluation."""
import hashlib
import json

from neural_network.archery_ml.contracts import DatasetSample


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
    derived = {"gt_annotations_sha256", "gt_manifest_sha256"}
    if ({k: v for k, v in (matches[0].capture_metadata or {}).items() if k not in derived}
            != {k: v for k, v in meta.items() if k not in derived}):
        raise ValueError("GT capture provenance differs from annotation manifest")
    if hashlib.sha256(sample.image_path.read_bytes()).hexdigest() != meta.get("original_sha256"):
        raise ValueError("mobile original checksum mismatch")
    return {"revision": meta["gt_revision"], "annotations_sha256": digest,
            "annotation_manifest_sha256": manifest_digest}
