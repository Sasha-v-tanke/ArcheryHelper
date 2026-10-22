import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from neural_network.archery_ml.contracts import DatasetSample
from neural_network.archery_ml.data.view import DatasetView
from neural_network.archery_ml.geometry.audit import audit_splits
from neural_network.archery_ml.geometry.lineage import original_family


class GeometrySplitAuditTest(unittest.TestCase):
    def test_augmentation_lineage_survives_hash_and_extension_changes(self):
        self.assertEqual("original", original_family(Path("train/Original_jpg.rf.aaa.jpg")))
        self.assertEqual("original", original_family(Path("test/original_PNG.rf.bbb.jpg")))
        self.assertEqual("target", original_family(Path("target.png")))
        self.assertNotEqual(original_family(Path("Fig11.jpg")), original_family(Path("10ring-fig11.jpg")))

    def test_audit_catches_leaks_even_when_manifest_group_ids_disagree(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            left = root / "original_jpg.rf.aaa.jpg"
            right = root / "original_jpg.rf.bbb.jpg"
            pixels = np.random.default_rng(42).integers(0, 256, (64, 64, 3), dtype=np.uint8)
            Image.fromarray(pixels).save(left)
            right.write_bytes(left.read_bytes())
            view = DatasetView.from_samples((
                DatasetSample(id="a", image_path=left, annotation_path=None, source_id="source", group_id="a", split="train"),
                DatasetSample(id="b", image_path=right, annotation_path=None, source_id="source", group_id="b", split="test"),
            ))
            report = audit_splits(view)
            self.assertFalse(report["group_leaks"])
            self.assertEqual({"source:original": ["a", "b"]}, report["original_family_leaks"])
            self.assertEqual(1, len(report["exact_image_leaks"]))
            self.assertEqual([["a", "b"]], report["near_image_leaks_dhash_4"])

    def test_export_folder_is_recorded_without_overriding_manifest_split(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "train" / "original.png"
            path.parent.mkdir()
            Image.new("RGB", (32, 32), (100, 120, 150)).save(path)
            report = audit_splits(DatasetView.from_samples((DatasetSample(
                id="a", image_path=path, annotation_path=None, split="test", group_id="original",
            ),)))
            self.assertEqual({"test": 1}, report["split_counts"])
            self.assertEqual({"train->test": 1}, report["export_folder_to_manifest_split"])
            self.assertFalse(report["original_family_leaks"])
