"""Explicit, checksum-pinned corrections; external exports remain untouched."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from neural_network.archery_ml.data.registry import DatasetSource


def corrected_coco(root: Path, path: Path, source: DatasetSource) -> dict:
    raw = path.read_bytes()
    payload = json.loads(raw)
    configured = source.options.get("corrections_file")
    if configured is None:
        return payload
    corrections = json.loads(Path(configured).read_text(encoding="utf-8"))
    if corrections["source_id"] != source.id or corrections["source_version"] != source.version:
        raise ValueError("GT corrections source/version mismatch")
    categories = {item["name"]: item["id"] for item in payload["categories"]}
    images = {item["id"]: item for item in payload["images"]}
    annotations = {item["id"]: item for item in payload["annotations"]}
    seen = set()
    for change in corrections["changes"]:
        if change["annotation_file"] != path.relative_to(root).as_posix():
            continue
        if hashlib.sha256(raw).hexdigest() != change["annotation_sha256"]:
            raise ValueError("GT correction input checksum mismatch")
        key = change["annotation_id"]
        if key in seen:
            raise ValueError("duplicate GT correction")
        seen.add(key)
        if (key not in annotations or change["image_id"] not in images
                or change["old_label"] not in categories or change["new_label"] not in categories):
            raise ValueError("GT correction identity/label missing")
        annotation = annotations[key]
        image = images[change["image_id"]]
        if (annotation["image_id"] != image["id"] or image["file_name"] != change["file_name"]
                or annotation["category_id"] != categories[change["old_label"]]):
            raise ValueError("GT correction identity/label mismatch")
        annotation["category_id"] = categories[change["new_label"]]
    return payload
