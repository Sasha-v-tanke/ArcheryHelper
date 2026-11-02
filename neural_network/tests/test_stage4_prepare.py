"""Stage 3 -> Stage 4 materialization regression tests, without downloads."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from neural_network.archery_ml.contracts import (
    DatasetSample, ImagePointAnnotation, TargetMetadata, load_manifest, save_manifest,
)
from neural_network.archery_ml.prepare import _raw_labels_digest, reviewed_sample, materialize


class MaterializerTest(unittest.TestCase):
    def setUp(self):
        work = tempfile.TemporaryDirectory()
        self.addCleanup(work.cleanup)
        self.root = Path(work.name) / "data"
        self.root.mkdir()
        self.manifests = self.root / "manifests"
        self.manifests.mkdir()
        self.artifacts = self.root / "stage3"
        self.artifacts.mkdir()
        self.original = self.root / "original.png"
        self.crop = self.artifacts / "face0.png"
        Image.new("RGB", (512, 512), (20, 30, 50)).save(self.original)
        Image.new("RGB", (512, 512), (40, 60, 80)).save(self.crop)
        self.geometry = self.artifacts / "selected.json"
        self.geometry.write_text(json.dumps({
            "status": "READY", "quality_status": "READY", "artifact_revision": "rev-1",
            "physical_face_order_verified": True,
            "faces": [{"face_index": 0, "crop_path": "face0.png",
                       "original_to_canonical": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]}]
        }))

    def sample(self, raw=None):
        if raw is None:
            raw = (ImagePointAnnotation(x_fraction=0.5, y_fraction=0.5, face_index=0),)
        return DatasetSample(id="example", image_path=self.original, annotation_path=None,
                             split="train", source_id="verified-source", group_id="session-1",
                             raw_annotations=tuple(raw))

    def review(self, sample):
        return {"annotator": "person1", "reviewer": "person2",
                "face_ownership_reviewed": True, "raw_annotations_sha256": _raw_labels_digest(sample),
                "original_sha256": hashlib.sha256(self.original.read_bytes()).hexdigest(),
                "geometry_path": "stage3/selected.json",
                "geometry_sha256": hashlib.sha256(self.geometry.read_bytes()).hexdigest(),
                "verified_empty_faces": []}

    def prepare_one(self, sample=None, review=None):
        sample = self.sample() if sample is None else sample
        review = self.review(sample) if review is None else review
        return reviewed_sample(sample, review, data_root=self.root,
                               review_dir=self.root, manifest_dir=self.manifests)

    def test_reviewed_homography_points_and_crop_binding(self):
        prepared = self.prepare_one()
        self.assertEqual(1, len(prepared.annotations))
        point = prepared.annotations[0]
        self.assertAlmostEqual((256 - 255.5) / 256, point.x_norm)
        self.assertAlmostEqual((256 - 255.5) / 256, point.y_norm)
        face = prepared.capture_metadata["canonical_faces"][0]
        self.assertEqual(0, face["face_index"])
        self.assertEqual("verified", face["annotation_status"])
        self.assertEqual(hashlib.sha256(self.crop.read_bytes()).hexdigest(), face["sha256"])
        self.assertEqual(self.crop.resolve(), (self.manifests / face["image_path"]).resolve())

    def test_unreviewed_owner_and_mismatched_digests_fail_closed(self):
        sample = self.sample()
        for edit, error in [({"face_ownership_reviewed": False}, "ownership"),
                            ({"annotator": "person2"}, "independent"),
                            ({"raw_annotations_sha256": "0" * 64}, "raw impact annotations"),
                            ({"original_sha256": "0" * 64}, "original image"),
                            ({"geometry_sha256": "0" * 64}, "geometry SHA256")]:
            review = {**self.review(sample), **edit}
            with self.subTest(edit=edit), self.assertRaisesRegex(ValueError, error):
                self.prepare_one(sample, review)

    def test_empty_negative_requires_explicit_attestation(self):
        sample = self.sample(raw=())
        with self.assertRaisesRegex(ValueError, "empty-face assertions"):
            self.prepare_one(sample)
        valid = {**self.review(sample), "verified_empty_faces": [0]}
        prepared = self.prepare_one(sample, valid)
        self.assertFalse(prepared.annotations)
        self.assertEqual(1, len(prepared.capture_metadata["canonical_faces"]))

    def test_outside_crop_and_missing_face_owner_rejected(self):
        sample = self.sample(raw=(ImagePointAnnotation(x_fraction=0.5, y_fraction=0.5),))
        with self.assertRaisesRegex(ValueError, "lacks reviewed target face index"):
            self.prepare_one(sample)
        sample = self.sample(raw=(ImagePointAnnotation(x_fraction=0.5, y_fraction=0.5, face_index=0),))
        geom = json.loads(self.geometry.read_text())
        geom["faces"][0]["original_to_canonical"][0][2] = 1000
        self.geometry.write_text(json.dumps(geom))
        with self.assertRaisesRegex(ValueError, "outside canonical crop"):
            self.prepare_one(sample)

    def test_rejected_stage3_status_never_materialized(self):
        sample = self.sample()
        geom = json.loads(self.geometry.read_text())
        geom["quality_status"] = "GEOMETRY_UNCERTAIN"
        self.geometry.write_text(json.dumps(geom))
        with self.assertRaisesRegex(ValueError, "not READY"):
            self.prepare_one(sample)

    def test_derived_manifest_snapshot_are_new_and_bound(self):
        sample = self.sample()
        manifest = self.manifests / "dataset_manifest.json"
        save_manifest(manifest, [sample])
        snapshot = self.root / "dataset_snapshot.json"
        old_hash = hashlib.sha256(manifest.read_bytes()).hexdigest()
        snapshot.write_text(json.dumps({"manifest_sha256": old_hash}))
        review_map = self.root / "review.json"
        review_map.write_text(json.dumps({sample.id: self.review(sample)}))
        out_manifest = self.manifests / "stage4_manifest.json"
        out_snapshot = self.root / "stage4_snapshot.json"
        result = materialize(manifest=manifest, snapshot=snapshot, review_map=review_map,
                             output_manifest=out_manifest, output_snapshot=out_snapshot,
                             data_root=self.root)
        self.assertEqual(1, result["canonical_impacts"])
        self.assertEqual(1, len(load_manifest(out_manifest)[0].annotations))
        self.assertEqual(old_hash, hashlib.sha256(manifest.read_bytes()).hexdigest())
        derived = json.loads(out_snapshot.read_text())
        self.assertEqual(hashlib.sha256(out_manifest.read_bytes()).hexdigest(), derived["manifest_sha256"])
        self.assertEqual(old_hash, derived["stage4_materialization"]["source_manifest_sha256"])
        with self.assertRaisesRegex(ValueError, "outputs must be new"):
            materialize(manifest=manifest, snapshot=snapshot, review_map=review_map,
                        output_manifest=out_manifest, output_snapshot=out_snapshot,
                        data_root=self.root)


if __name__ == "__main__":
    unittest.main()
