"""Local capture registration, independent review, export and coverage. No detector GT."""
import argparse
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path

from PIL import Image, ImageOps, ImageDraw

from neural_network.archery_ml.contracts import DatasetSample, TargetMetadata
from neural_network.archery_ml.data.geometry_provenance import annotation_digest
from neural_network.archery_ml.data.importers.mobile_geometry import import_mobile_geometry
from neural_network.archery_ml.data.registry import DatasetSource
from neural_network.archery_ml.geometry.validation import configuration_key


DEFAULT_POLICY = Path("dataset_tools/stage3_geometry_policy.json")


def write_new(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")


def load(path):
    return json.loads(path.read_text())


def capture_plan(policy):
    return {"schema_version": 1, "status": "BLOCKED: EXTERNAL CAPTURE REQUIRED", "revision": "fill-before-capture",
            "policy_sha256": hashlib.sha256(json.dumps(policy, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
            "sessions": [], "requirements": {key: policy[key] for key in ("minimum_originals", "minimum_sessions", "minimum_devices")},
            "configurations": [{**c, "planned_usable": math.ceil(c["minimum_originals"] * 1.5),
                                "planned_unusable": math.ceil(c["minimum_originals"] * .5)} for c in policy["required_configurations"]],
            "instructions": "Assign real session_id, device_model and split before registration. Keep a never-used reserve holdout. READY counts are measured, never promised by this plan."}


def register(root, photo, plan_path, session_id, metadata, annotator, revision, normalize_exif=False, normalize_rgb=False):
    plan = load(plan_path)
    sessions = [s for s in plan["sessions"] if s["capture_session_id"] == session_id]
    if (plan.get("status") != "approved-before-capture" or len(sessions) != 1
            or not plan.get("revision") or plan["revision"] == "fill-before-capture"):
        raise ValueError("a preassigned approved capture session is required")
    session = sessions[0]
    if not session.get("device_model") or session["split"] not in {"mobile_dev", "mobile_real_test"}:
        raise ValueError("invalid device or split in capture plan")
    if not annotator or not revision:
        raise ValueError("annotator and GT revision required")
    frozen_plan = root / "capture-plan.json"
    plan_bytes = plan_path.read_bytes()
    if frozen_plan.exists() and frozen_plan.read_bytes() != plan_bytes:
        raise ValueError("capture plan changed after registration")
    if not frozen_plan.exists():
        frozen_plan.parent.mkdir(parents=True, exist_ok=True)
        with frozen_plan.open("xb") as stream:
            stream.write(plan_bytes)
    raw = photo.read_bytes()
    checksum = hashlib.sha256(raw).hexdigest()
    original = root / "originals" / (checksum + photo.suffix.lower())
    with Image.open(photo) as image:
        orientation = image.getexif().get(274, 1)
        if orientation not in range(1,9):
            raise ValueError("invalid EXIF orientation")
        if orientation != 1 and not normalize_exif:
            raise ValueError("nontrivial EXIF requires explicit --normalize-exif")
        source_mode = image.mode
        if source_mode != "RGB" and not normalize_rgb:
            raise ValueError("non-RGB original requires explicit --normalize-rgb")
        rgb = ImageOps.exif_transpose(image).convert("RGB")
        size = list(rgb.size)
    record_path = root / "drafts" / (checksum + ".json")
    if record_path.exists():
        raise ValueError("original already registered")
    original.parent.mkdir(parents=True, exist_ok=True)
    if original.exists() and original.read_bytes() != raw:
        raise ValueError("existing original checksum mismatch")
    if not original.exists():
        with original.open("xb") as stream:
            stream.write(raw)
    image_path = original
    transform = "identity"
    if orientation != 1 or source_mode != "RGB":
        image_path = root / "derivatives" / (checksum + ".png")
        image_path.parent.mkdir(parents=True, exist_ok=True)
        if image_path.exists():
            raise ValueError("EXIF derivative already exists")
        rgb.save(image_path)
        transform = "PIL.ImageOps.exif_transpose+RGB"
    meta = {**session, "original_id": checksum, "original_sha256": checksum,
            "original_path": original.relative_to(root).as_posix(), "image_size": size,
            "image_sha256": hashlib.sha256(image_path.read_bytes()).hexdigest(),
            "exif_orientation": orientation, "original_mode": source_mode, "image_transform": transform,
            "capture_plan_sha256": hashlib.sha256(plan_path.read_bytes()).hexdigest(),
            "gt_revision": revision, "annotator": annotator, "reviewer": None,
            "orientation_reference": "unknown", "suitability": "unknown", "annotation_state": "draft",
            "independent_gt_method": None, "review_notes": None,
            "calibration_landmark_ids": [], "evaluation_landmark_ids": []}
    sample = DatasetSample(checksum, image_path.relative_to(root), None, split=session["split"],
                           target_metadata=metadata, capture_metadata=meta)
    write_new(record_path, sample.to_dict())
    return record_path


def source_for(root, annotation_file="geometry.json"):
    return DatasetSource("mobile_geometry", "mobile_geometry", str(root.resolve()), "1", "private", "project owner",
                         None, ("geometry_evaluation",), options={"annotation_file": annotation_file}, annotation_space="image")


def review(root, record_path, reviewer, notes, policy):
    sample = DatasetSample.from_dict(load(record_path))
    meta = dict(sample.capture_metadata or {})
    if meta.get("annotation_state") != "draft" or not reviewer or reviewer == meta.get("annotator") or not notes.strip():
        raise ValueError("independent reviewer, review notes and a draft record required")
    methods = {"physical_measurement", "surveyed_print_template", "independent_calibration"}
    if meta.get("suitability") == "unusable":
        methods.add("independent_suitability_review")
    if meta.get("independent_gt_method") not in methods:
        raise ValueError("declare an independent GT method; detector predictions are prohibited")
    evaluation_ids = meta.get("evaluation_landmark_ids", [])
    calibration_ids = meta.get("calibration_landmark_ids", [])
    if set(evaluation_ids) & set(calibration_ids):
        raise ValueError("calibration and evaluation landmarks must be independent")
    if meta.get("suitability") == "usable" and not evaluation_ids:
        raise ValueError("independent evaluation landmark identities required")
    maximum = policy["maximum_gt_uncertainty_norm"]
    for annotation in sample.geometry_annotations:
        if annotation.canonical_points or annotation.kind == "target_center":
            if annotation.uncertainty_norm is None or annotation.uncertainty_norm > maximum:
                raise ValueError("GT uncertainty missing or above policy budget")
    meta.update(reviewer=reviewer, review_notes=notes, annotation_state="reviewed")
    annotations = tuple(replace(a, annotation_quality="reviewed") for a in sample.geometry_annotations)
    sample = replace(sample, capture_metadata=meta, geometry_annotations=annotations)
    meta["gt_annotations_sha256"] = annotation_digest(sample)
    # Exercise the same importer as production, without claiming real validation.
    candidate = root / ".review-candidate.json"
    write_new(candidate, {"samples": [sample.to_dict()]})
    try:
        import_mobile_geometry(root, source_for(root, candidate.name))
    finally:
        candidate.unlink()
    destination = root / "reviewed" / record_path.name
    write_new(destination, sample.to_dict())
    write_new(root / "audits" / record_path.name, {
        "reviewed_record_sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
        "original_sha256": meta["original_sha256"], "gt_revision": meta["gt_revision"],
        "annotator": meta["annotator"], "reviewer": reviewer, "review_notes": notes,
        "independent_gt_method": meta["independent_gt_method"], "capture_plan_sha256": meta["capture_plan_sha256"]})
    return destination


def export(root):
    records = sorted((root / "reviewed").glob("*.json"))
    if not records:
        raise ValueError("no reviewed records")
    samples = [DatasetSample.from_dict(load(path)) for path in records]
    for path in records:
        audit = load(root / "audits" / path.name)
        if audit["reviewed_record_sha256"] != hashlib.sha256(path.read_bytes()).hexdigest():
            raise ValueError("reviewed record changed after independent review")
    if any((s.capture_metadata or {}).get("annotation_state") != "reviewed" for s in samples):
        raise ValueError("unreviewed export record")
    if len({(s.capture_metadata or {}).get("gt_revision") for s in samples}) != 1:
        raise ValueError("mixed GT revisions")
    plan_sha = hashlib.sha256((root / "capture-plan.json").read_bytes()).hexdigest()
    if any(s.capture_metadata.get("capture_plan_sha256") != plan_sha for s in samples):
        raise ValueError("capture plan checksum mismatch")
    sessions = {s["capture_session_id"]: s for s in load(root / "capture-plan.json")["sessions"]}
    for sample in samples:
        meta = sample.capture_metadata
        session = sessions.get(meta["capture_session_id"], {})
        if sample.split != session.get("split") or meta["device_model"] != session.get("device_model"):
            raise ValueError("record differs from preassigned capture session/split")
    candidate = root / ".export-candidate.json"
    write_new(candidate, {"samples": [s.to_dict() for s in samples]})
    try:
        imported = import_mobile_geometry(root, source_for(root, candidate.name))
        output = root / "geometry.json"
        with output.open("xb") as stream:
            stream.write(candidate.read_bytes())
    finally:
        candidate.unlink()
    source = {"id": "mobile_geometry", "importer": "mobile_geometry", "url": str(root.resolve()), "version": samples[0].capture_metadata["gt_revision"],
              "license": "private", "author": "project owner", "allowed_tasks": ["geometry_evaluation"], "annotation_space": "image"}
    write_new(root / "sources.json", {"registry_version": 1, "sources": [source]})
    return {"samples": len(imported), "geometry_sha256": hashlib.sha256(output.read_bytes()).hexdigest()}


def coverage(root, policy):
    records = list((root / "drafts").glob("*.json"))
    reviewed = {p.name: p for p in (root / "reviewed").glob("*.json")}
    samples = [DatasetSample.from_dict(load(reviewed.get(p.name, p))) for p in records]
    def summary(selected):
        meta = [s.capture_metadata for s in selected]
        return {"originals": len({m["original_id"] for m in meta}), "sessions": len({m["capture_session_id"] for m in meta}),
                "devices": len({m["device_model"] for m in meta}),
                "usable": sum(m["suitability"] == "usable" for m in meta), "unusable": sum(m["suitability"] == "unusable" for m in meta),
                "unknown": sum(m["suitability"] not in {"usable", "unusable"} for m in meta),
                "reviewed": sum(m["annotation_state"] == "reviewed" for m in meta), "ready": None}
    splits = {}
    for split in ("mobile_dev", "mobile_real_test"):
        selected = [s for s in samples if s.split == split]
        groups = {}
        for required in policy["required_configurations"]:
            key = configuration_key(required)
            group = [s for s in selected if configuration_key(s.target_metadata.to_dict()) == key]
            groups[key] = {**summary(group), "requirements": required}
        splits[split] = {**summary(selected), "configurations": groups}
    return {"status": "BLOCKED: EXTERNAL CAPTURE REQUIRED" if not samples else "COLLECTED: NOT EVALUATED",
            **summary(samples), "splits": splits}


def preview(root, record, output):
    sample = DatasetSample.from_dict(load(record))
    with Image.open(root / sample.image_path) as raw:
        image = raw.convert("RGB")
    draw = ImageDraw.Draw(image)
    for annotation in sample.geometry_annotations:
        for i, (x,y) in enumerate(annotation.points):
            px, py = x * image.width, y * image.height
            draw.ellipse((px-4,py-4,px+4,py+4), outline="lime", width=2)
            coordinate = annotation.canonical_points[i] if annotation.canonical_points else "center"
            draw.text((px+5,py), f"face {annotation.face_index}: {coordinate}", fill="lime")
    image.save(output)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["plan", "register", "review", "export", "coverage", "preview", "annotate"])
    parser.add_argument("--root", type=Path, default=Path("data/mobile"))
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--record", type=Path)
    parser.add_argument("--image", type=Path)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--session")
    parser.add_argument("--annotator")
    parser.add_argument("--reviewer")
    parser.add_argument("--review-notes", default="")
    parser.add_argument("--gt-revision")
    parser.add_argument("--normalize-exif", action="store_true")
    parser.add_argument("--normalize-rgb", action="store_true")
    parser.add_argument("--format", choices=["SINGLE", "TRIPLE"], default="SINGLE")
    parser.add_argument("--triple-layout", choices=["VERTICAL", "TRIANGULAR"])
    parser.add_argument("--minimum-scoring-zone", type=int, choices=[1,5,6], default=1)
    parser.add_argument("--ten-ring-mode", choices=["RECURVE", "COMPOUND"], default="RECURVE")
    args = parser.parse_args(argv)
    policy = load(args.policy)
    if args.command == "plan":
        write_new(args.output or args.root / "capture-plan.json", capture_plan(policy))
    elif args.command == "register":
        if not all([args.image, args.plan, args.session, args.annotator, args.gt_revision]):
            parser.error("register requires --image --plan --session --annotator --gt-revision")
        print(register(args.root, args.image, args.plan, args.session,
            TargetMetadata(args.format, args.minimum_scoring_zone, args.ten_ring_mode, triple_layout=args.triple_layout),
            args.annotator, args.gt_revision, args.normalize_exif, args.normalize_rgb))
    elif args.command == "review":
        if not args.record or not args.reviewer:
            parser.error("review requires --record --reviewer --review-notes")
        print(review(args.root, args.record, args.reviewer, args.review_notes, policy))
    elif args.command == "export":
        print(json.dumps(export(args.root), sort_keys=True))
    elif args.command == "coverage":
        result = coverage(args.root, policy)
        if args.output:
            write_new(args.output, result)
        print(json.dumps(result, indent=2, sort_keys=True))
    elif args.command == "preview":
        if not args.record or not args.output:
            parser.error("preview requires --record --output")
        preview(args.root, args.record, args.output)
    elif args.command == "annotate":
        if not args.record or not args.output:
            parser.error("annotate requires --record --output")
        from neural_network.archery_ml.data.annotation_editor import write_editor
        write_editor(args.root,args.record,args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
