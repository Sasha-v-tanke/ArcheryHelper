"""Review desk keeps real photo/geometry identity and cannot approve by itself."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from neural_network.archery_ml.contracts import (
    DatasetSample, ImagePointAnnotation, save_manifest,
)
from neural_network.archery_ml.prepare import _raw_labels_digest
from neural_network.archery_ml.review_stage4 import build_queue, PAGE


class ReviewDeskTest(unittest.TestCase):
    def setUp(self):
        workspace = tempfile.TemporaryDirectory()
        self.addCleanup(workspace.cleanup)
        self.root = Path(workspace.name) / "data"
        self.root.mkdir()
        self.original = self.root / "original.png"
        self.crop = self.root / "crop.png"
        Image.new("RGB", (512, 512), (40, 60, 80)).save(self.original)
        Image.new("RGB", (512, 512), (80, 60, 40)).save(self.crop)
        self.sample = DatasetSample(
            id="sample-1", image_path=self.original, annotation_path=None,
            split="val", source_id="source", group_id="session",
            raw_annotations=(ImagePointAnnotation(0.5, 0.25),),
        )
        self.manifest = self.root / "manifest.json"
        save_manifest(self.manifest, [self.sample])
        self.sha = hashlib.sha256(self.original.read_bytes()).hexdigest()
        self.geometry = self.root / "geometry.json"
        self.geometry.write_text(json.dumps({self.sha: {
            "original_id": self.sha, "status": "READY", "quality_status": "READY",
            "selection_status": "USER_CONFIRMED", "ownership_confirmed": True,
            "candidate_ids": ["f0"], "artifact_revision": "rev-1",
            "faces": [{"face_index": 0, "crop_path": "crop.png",
                       "original_to_canonical": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]}],
        }}))

    def test_queue_binds_exact_source_and_suggests_no_review(self):
        queue, assets = build_queue(self.manifest, self.geometry, self.root)
        self.assertEqual(1, len(queue))
        self.assertEqual([], queue[0]["reasons"])
        self.assertEqual([{"index": 0, "url": "/asset/crop-0-0"}], queue[0]["faces"])
        self.assertEqual(self.original.resolve(), assets["image-0"])
        self.assertEqual(self.crop.resolve(), assets["crop-0-0"])
        self.assertEqual(_raw_labels_digest(self.sample), queue[0]["raw_annotations_sha256"])
        self.assertNotIn("reviewer", queue[0])
        self.assertNotIn("face_ownership_reviewed", queue[0])
        self.assertIn("s.checked", PAGE)

    def test_unmatched_geometry_blocks_export_suggestion(self):
        self.geometry.write_text("{}")
        queue, assets = build_queue(self.manifest, self.geometry, self.root)
        self.assertEqual([], queue[0]["faces"])
        self.assertEqual(1, len(assets))
        self.assertIn("No selected Stage 3 geometry", queue[0]["reasons"][0])

    def test_escaping_crop_path_blocks_preview(self):
        payload = json.loads(self.geometry.read_text())
        payload[self.sha]["faces"][0]["crop_path"] = "../outside.png"
        self.geometry.write_text(json.dumps(payload))
        queue, assets = build_queue(self.manifest, self.geometry, self.root)
        self.assertTrue(queue[0]["reasons"])
        self.assertNotIn("crop-0-0", assets)


if __name__ == "__main__":
    unittest.main()
