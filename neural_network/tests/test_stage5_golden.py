"""Deterministic, original-separated golden selection regression coverage."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from PIL import Image

from neural_network.archery_ml.contracts import (
    ModelInputSpec, ModelMetadata, ModelOutputSpec, ModelPostprocessSpec,
)
from neural_network.archery_ml.stage5_golden import prepare_golden, select_goldens
from neural_network.archery_ml.stage5_parity import read_golden_manifest, sha256_file


def _metadata() -> ModelMetadata:
    return ModelMetadata(
        contract_version=2, model_version="fixture",
        input=ModelInputSpec(width=512, height=512),
        output=ModelOutputSpec(type="impact_heatmap_offset", stride=2),
        postprocess=ModelPostprocessSpec(confidence_threshold=.35, nms_radius=4),
    )


def _records(root: Path, count: int = 28):
    records = []
    for index in range(count):
        image = root / f"input-{index:03}.png"
        Image.new("RGB", (512, 512), color=(20+index, 23, 42)).save(image)
        records.append(SimpleNamespace(
            sample_id=f"sample-{index:03}",
            face_index=0,
            image_path=image,
            sha256=sha256_file(image),
            source_image_sha256=hashlib.sha256(f"camera-{index}".encode()).hexdigest(),
            points=((0.0, 0.0),) if index % 2 else (),
        ))
    return records


class TestGoldenSelection(unittest.TestCase):
    def test_selects_distinct_originals_with_positive_and_negative(self):
        with tempfile.TemporaryDirectory() as folder:
            records = _records(Path(folder))
            ordered = select_goldens(records, 20)
            reversed_order = select_goldens(list(reversed(records)), 20)
            self.assertEqual(ordered, reversed_order)
            self.assertEqual(len({r.source_image_sha256 for r in ordered}), 20)
            self.assertTrue(any(r.points for r in ordered))
            self.assertTrue(any(not r.points for r in ordered))

    def test_rejects_lack_of_verified_negatives(self):
        with tempfile.TemporaryDirectory() as folder:
            records = _records(Path(folder))
            for record in records:
                record.points = ((0., 0.),)
            with self.assertRaisesRegex(ValueError, "positive and verified-empty"):
                select_goldens(records, 20)

    def test_fails_with_insufficient_independent_sources(self):
        with tempfile.TemporaryDirectory() as folder:
            records = _records(Path(folder))
            for record in records:
                record.source_image_sha256 = "a" * 64
            with self.assertRaisesRegex(ValueError, "distinct|different"):
                select_goldens(records, 20)

    def test_preparation_binds_source_and_is_atomic(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            records = _records(root)
            manifest = root / "stage4.json"
            manifest.write_text('{"manifest":"v1"}')
            snapshot = root / "snapshot.json"
            snapshot.write_text('{"snapshot":"v1"}')
            checkpoint = root / "best.pt"
            checkpoint.write_bytes(b"checked checkpoint fixture")
            metadata_path = root / "metadata.json"
            metadata_path.write_text(json.dumps({
                **_metadata().to_dict(),
                "training_dataset_sha256": sha256_file(manifest),
            }))
            view = SimpleNamespace(
                __len__=lambda: 28,
                select=lambda **_: None,
            )
            dataset = SimpleNamespace(records=records)
            sample_dict = []
            for index in range(28):
                raw_labels = ([{"x_fraction": .5, "y_fraction": .5, "face_index": 0}]
                              if index % 2 else [])
                raw = tuple(SimpleNamespace(to_dict=lambda value=value: value)
                            for value in raw_labels)
                raw_digest = hashlib.sha256(json.dumps(
                    raw_labels, sort_keys=True, separators=(",", ":"),
                ).encode()).hexdigest()
                sample_dict.append(SimpleNamespace(
                    id=f"sample-{index:03}", split="val", raw_annotations=raw,
                    capture_metadata={
                        "canonical_faces": [{"annotation_status": "verified"}],
                        "stage4_review": {
                            "annotator": "annotator-1", "reviewer": "reviewer-2",
                            "source_annotations_sha256": raw_digest,
                        },
                    },
                ))

            class View:
                def __iter__(self): return iter(sample_dict)
                def __len__(self): return len(sample_dict)
                def select(self, **_): return self

            with (
                mock.patch("neural_network.archery_ml.stage5_golden.load_detector",
                           return_value=(None, _metadata())),
                mock.patch("neural_network.archery_ml.stage5_golden.verify_dataset"),
                mock.patch("neural_network.archery_ml.stage5_golden.DatasetView.load",
                           return_value=View()),
                mock.patch("neural_network.archery_ml.stage5_golden.CanonicalImpactDataset",
                           return_value=dataset),
            ):
                result = prepare_golden(checkpoint, metadata_path, manifest, snapshot,
                                        root / "derived", count=20)
            self.assertTrue(result.exists())
            items = read_golden_manifest(result, _metadata())
            self.assertEqual(len(items), 20)
            document = json.loads(result.read_text())
            self.assertEqual(document["source_checkpoint_sha256"], sha256_file(checkpoint))
            self.assertEqual(document["source_manifest_sha256"], sha256_file(manifest))

    def test_preparation_rejects_changed_checkpoint_manifest(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            manifest = root / "stage4.json"
            manifest.write_text("{}")
            snapshot = root / "snapshot.json"
            snapshot.write_text("{}")
            checkpoint = root / "model.pt"
            checkpoint.write_bytes(b"fixture")
            metadata = root / "metadata.json"
            metadata.write_text(json.dumps({
                **_metadata().to_dict(), "training_dataset_sha256": "a" * 64,
            }))
            with mock.patch("neural_network.archery_ml.stage5_golden.load_detector",
                            return_value=(None, _metadata())):
                with self.assertRaisesRegex(ValueError, "not the one"):
                    prepare_golden(checkpoint, metadata, manifest, snapshot, root / "missing")
            self.assertFalse((root / "missing").exists())


if __name__ == "__main__":
    unittest.main()
