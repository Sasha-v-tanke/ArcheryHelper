"""Stage 5 golden selection through REAL Stage 4 manifest/snapshot readers.

No dataset/checkpoint access is assumed. Synthesizes reviewed metadata and
deliberately mocks ONLY the trained checkpoint loader; dataset provenance,
negative-face verification, split checks, crop checks and serialization are real.
"""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from PIL import Image

from neural_network.archery_ml.contracts import (
    DatasetSample, ImpactAnnotation, ImagePointAnnotation,
    ModelInputSpec, ModelMetadata, ModelOutputSpec, ModelPostprocessSpec,
    save_manifest,
)
from neural_network.archery_ml.stage5_golden import prepare_golden
from neural_network.archery_ml.stage5_parity import read_golden_manifest, sha256_file


def _stage4_dataset(root: Path):
    data = root / "data"
    manifests = data / "manifests"
    originals = data / "originals"
    crops = data / "crops"
    for path in (manifests, originals, crops):
        path.mkdir(parents=True)
    samples = []
    for index in range(24):
        positive = index % 2 == 1
        original = originals / f"photo-{index:02}.png"
        crop = crops / f"crop-{index:02}.png"
        Image.new("RGB", (512, 512), (40 + index, 77, 125)).save(original)
        Image.new("RGB", (512, 512), (80 + index, 66, 133)).save(crop)
        original_hash = sha256_file(original)
        raw = (ImagePointAnnotation(x_fraction=.55, y_fraction=.6, face_index=0),) if positive else ()
        canonical = (ImpactAnnotation(x_norm=.1, y_norm=.2, face_index=0),) if positive else ()
        raw_digest = hashlib.sha256(json.dumps(
            [point.to_dict() for point in raw],
            sort_keys=True, separators=(",", ":"), allow_nan=False,
        ).encode()).hexdigest()
        samples.append(DatasetSample(
            id=f"real-source-{index:02}", image_path=original,
            annotation_path=None, split="val" if index % 3 else "test",
            source_id="synthetic-reviewed-source", group_id=f"original-{index:02}",
            annotations=canonical, raw_annotations=raw,
            capture_metadata={
                "canonical_faces": [{
                    "face_index": 0, "image_path": f"../crops/crop-{index:02}.png",
                    "sha256": sha256_file(crop), "annotation_status": "verified",
                }],
                "stage4_review": {
                    "annotator": "review-person-A", "reviewer": "review-person-B",
                    "source_image_sha256": original_hash,
                    "source_annotations_sha256": raw_digest,
                    "geometry_sha256": hashlib.sha256(f"geometry-{index}".encode()).hexdigest(),
                    "verified_empty_faces": [] if positive else [0],
                    "impact_face_indices": [0] if positive else [],
                },
            },
        ))
    manifest = manifests / "manifest.json"
    save_manifest(manifest, samples)
    snapshot = manifests / "snapshot.json"
    snapshot.write_text(json.dumps({
        "snapshot_version": 1, "pipeline_version": 1, "dataset_schema_version": 2,
        "manifest_sha256": sha256_file(manifest),
        "registry_sha256": hashlib.sha256(b"synthetic-registry").hexdigest(),
        "split": {"seed": 42, "train_ratio": .8, "val_ratio": .1, "test_ratio": .1},
        "perceptual_threshold": 4, "sources": [],
        "report": {
            "images": 24, "impacts": 12,
            "source_distribution": {"synthetic-reviewed-source": 24},
            "target_distribution": {},
            "split_sizes": {"val": 16, "test": 8},
            "exact_duplicate_count": 0, "near_duplicate_group_count": 0,
            "duplicate_rate": 0.0,
        },
    }), encoding="utf-8")
    checkpoint = root / "fixture-checkpoint.pt"
    checkpoint.write_bytes(b"synthetic-checkpoint-for-mocked-loader")
    metadata = ModelMetadata(
        contract_version=2, model_version="stage5-generator-integration",
        input=ModelInputSpec(width=512, height=512),
        output=ModelOutputSpec(type="impact_heatmap_offset", stride=2),
        postprocess=ModelPostprocessSpec(confidence_threshold=.5, nms_radius=4),
    )
    metadata_file = root / "metadata.json"
    metadata_file.write_text(json.dumps({
        **metadata.to_dict(),
        "training_dataset_sha256": sha256_file(manifest),
    }), encoding="utf-8")
    return samples, manifest, snapshot, checkpoint, metadata, metadata_file


class TestRealStage4GoldenSelection(unittest.TestCase):
    def test_real_manifest_snapshot_and_reviewed_faces(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            samples, manifest, snapshot, checkpoint, metadata, metadata_file = _stage4_dataset(root)
            destination = root / "golden-release"
            with mock.patch("neural_network.archery_ml.stage5_golden.load_detector",
                            return_value=(None, metadata)):
                result = prepare_golden(checkpoint, metadata_file, manifest, snapshot,
                                        destination, count=20)
            selected = read_golden_manifest(result, metadata)
            self.assertEqual(len(selected), 20)
            self.assertEqual(len({r.original_sha256 for r in selected}), 20)
            doc = json.loads(result.read_text())
            self.assertEqual(doc["source_manifest_sha256"], sha256_file(manifest))
            self.assertEqual(doc["source_snapshot_sha256"], sha256_file(snapshot))
            self.assertTrue(any(item["reviewed_gt_impact_count"] == 0 for item in doc["samples"]))
            self.assertTrue(any(item["reviewed_gt_impact_count"] == 1 for item in doc["samples"]))
            self.assertTrue(all(item["split"] in ("val", "test") for item in doc["samples"]))

    def test_reject_tampered_crop_in_real_stage4_manifest(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            samples, manifest, snapshot, checkpoint, metadata, metadata_file = _stage4_dataset(root)
            crop = root / "data/crops/crop-00.png"
            crop.write_bytes(b"changed")
            with mock.patch("neural_network.archery_ml.stage5_golden.load_detector",
                            return_value=(None, metadata)):
                with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                    prepare_golden(checkpoint, metadata_file, manifest, snapshot,
                                   root / "golden-release")
            self.assertFalse((root / "golden-release").exists())

    def test_reject_missing_review_provenance_in_real_manifest(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            samples, manifest, snapshot, checkpoint, metadata, metadata_file = _stage4_dataset(root)
            # Tamper dataset but re-seal its manifest, snapshot and model metadata:
            # the Stage 5 reviewer audit must STILL reject the missing reviewer.
            document = json.loads(manifest.read_text())
            document["samples"][0]["capture_metadata"]["stage4_review"]["reviewer"] = ""
            manifest.write_text(json.dumps(document))
            snapshot_doc = json.loads(snapshot.read_text())
            snapshot_doc["manifest_sha256"] = sha256_file(manifest)
            snapshot.write_text(json.dumps(snapshot_doc))
            metadata_doc = json.loads(metadata_file.read_text())
            metadata_doc["training_dataset_sha256"] = sha256_file(manifest)
            metadata_file.write_text(json.dumps(metadata_doc))
            with mock.patch("neural_network.archery_ml.stage5_golden.load_detector",
                            return_value=(None, metadata)):
                with self.assertRaisesRegex(ValueError, "independent Stage 4"):
                    prepare_golden(checkpoint, metadata_file, manifest, snapshot,
                                   root / "golden-release")

    def test_reject_cross_split_group_leakage(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            samples, manifest, snapshot, checkpoint, metadata, metadata_file = _stage4_dataset(root)
            document = json.loads(manifest.read_text())
            # sample 0 = test, sample 1 = val, same group+source must fail.
            document["samples"][1]["group_id"] = document["samples"][0]["group_id"]
            manifest.write_text(json.dumps(document))
            snapshot_doc = json.loads(snapshot.read_text())
            snapshot_doc["manifest_sha256"] = sha256_file(manifest)
            snapshot.write_text(json.dumps(snapshot_doc))
            metadata_doc = json.loads(metadata_file.read_text())
            metadata_doc["training_dataset_sha256"] = sha256_file(manifest)
            metadata_file.write_text(json.dumps(metadata_doc))
            with mock.patch("neural_network.archery_ml.stage5_golden.load_detector",
                            return_value=(None, metadata)):
                with self.assertRaisesRegex(ValueError, "group leakage"):
                    prepare_golden(checkpoint, metadata_file, manifest, snapshot,
                                   root / "golden-release")


if __name__ == "__main__":
    unittest.main()
