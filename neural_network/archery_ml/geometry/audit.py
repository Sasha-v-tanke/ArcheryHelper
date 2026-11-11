"""Read-only split audit; exported folders are provenance, not split authority."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from neural_network.archery_ml.data.deduplicate import _dhash
from neural_network.archery_ml.data.view import DatasetView
from neural_network.archery_ml.geometry.lineage import original_family


def audit_splits(view: DatasetView) -> dict:
    groups: dict[str, list] = defaultdict(list)
    originals: dict[str, list] = defaultdict(list)
    exact: dict[str, list] = defaultdict(list)
    sessions: dict[str, list] = defaultdict(list)
    original_hashes: dict[str, list] = defaultdict(list)
    hashes = []
    folder_splits: Counter[str] = Counter()
    for sample in view:
        meta = sample.capture_metadata or {}
        if meta.get("capture_session_id"):
            sessions[meta["capture_session_id"]].append(sample)
        if meta.get("original_sha256"):
            original_hashes[meta["original_sha256"]].append(sample)
        groups[sample.group_id or sample.id].append(sample)
        originals[f"{sample.source_id}:{original_family(sample.image_path)}"].append(sample)
        exact[hashlib.sha256(sample.image_path.read_bytes()).hexdigest()].append(sample)
        hashes.append((sample, _dhash(sample.image_path)))
        folder_splits[f"{sample.image_path.parent.name}->{sample.split}"] += 1

    def leaks(buckets):
        return {
            key: [sample.id for sample in members]
            for key, members in sorted(buckets.items())
            if len({sample.split for sample in members}) > 1
        }

    near_leaks = []
    for index, (left, left_hash) in enumerate(hashes):
        for right, right_hash in hashes[index + 1:]:
            if left.split != right.split and (left_hash ^ right_hash).bit_count() <= 4:
                near_leaks.append([left.id, right.id])
    return {
        "samples": len(view),
        "original_families": len(originals),
        "group_count": len(groups),
        "split_counts": dict(sorted(Counter(sample.split for sample in view).items())),
        "export_folder_to_manifest_split": dict(sorted(folder_splits.items())),
        "group_leaks": leaks(groups),
        "capture_session_leaks": leaks(sessions),
        "original_sha256_leaks": leaks(original_hashes),
        "original_family_leaks": leaks(originals),
        "exact_image_leaks": leaks(exact),
        "near_image_leaks_dhash_4": near_leaks,
        "limitations": [
            "Filename lineage and dHash cannot certify capture-session or cross-source independence.",
            "A previously diagnosed test split is development data regardless of its name.",
        ],
    }


def require_holdout_isolation(view: DatasetView) -> dict:
    """Audit the entire supplied manifest before selecting test/configuration rows."""
    audit = audit_splits(view)
    if any(audit[key] for key in ("group_leaks", "capture_session_leaks", "original_sha256_leaks",
            "original_family_leaks", "exact_image_leaks", "near_image_leaks_dhash_4")):
        raise ValueError("cross-split leakage in evaluation manifest; holdout isolation failed")
    return audit


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--source", action="append")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    view = DatasetView.load(args.manifest).select(source_ids=args.source)
    report = audit_splits(view)
    report["manifest_sha256"] = hashlib.sha256(args.manifest.read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return int(any(report[key] for key in (
        "group_leaks", "capture_session_leaks", "original_sha256_leaks",
        "original_family_leaks", "exact_image_leaks", "near_image_leaks_dhash_4",
    )))


if __name__ == "__main__":
    raise SystemExit(main())
