"""Stage 3 -> Stage 4 materialization regression tests, without downloads."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from neural_network.archery_ml.data.view import DatasetView
from neural_network.archery_ml.impact_dataset import CanonicalImpactDataset
from neural_network.archery_ml.contracts import (
    DatasetSample, ImagePointAnnotation, TargetMetadata, load_manifest, save_manifest,
)
from neural_network.archery_ml.prepare import (
    _raw_labels_digest, reviewed_sample, materialize, write_review_draft, export_stage3_geometry,
)


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
            "original_id": hashlib.sha256(self.original.read_bytes()).hexdigest(),
            "selection_status": "USER_CONFIRMED", "ownership_confirmed": True,
            "candidate_ids": ["c1"],
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
                "selected_candidate_ids": ["c1"],
                "impact_face_indices": [p.face_index for p in sample.raw_annotations],
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

    def test_swapped_geometry_from_another_original_is_rejected(self):
        foreign = self.root / "foreign.png"
        Image.new("RGB", (512, 512), (10, 11, 12)).save(foreign)
        geom = json.loads(self.geometry.read_text())
        geom["original_id"] = hashlib.sha256(foreign.read_bytes()).hexdigest()
        self.geometry.write_text(json.dumps(geom))
        sample = self.sample()
        # Geometry SHA is correct for the foreign record, so the original
        # binding (not file integrity alone) must reject the substitution.
        review = self.review(sample)
        with self.assertRaisesRegex(ValueError, "original_id"):
            self.prepare_one(sample, review)

    def test_stage3_ownership_and_selected_candidate_ids_are_required(self):
        sample = self.sample()
        for edit, message in [({"selected_candidate_ids": ["foreign"]}, "candidate_ids"),
                              ({"selected_candidate_ids": []}, "candidate_ids")]:
            with self.subTest(edit=edit), self.assertRaisesRegex(ValueError, message):
                self.prepare_one(sample, {**self.review(sample), **edit})
        geom = json.loads(self.geometry.read_text())
        geom["selection_status"] = "AUTO_CONFIDENT"
        geom["ownership_confirmed"] = False
        self.geometry.write_text(json.dumps(geom))
        with self.assertRaisesRegex(ValueError, "USER_CONFIRMED"):
            self.prepare_one(sample, self.review(sample))

    def test_real_stage3_index_and_export_preserve_crop_binding(self):
        sample = self.sample()
        original_sha = hashlib.sha256(self.original.read_bytes()).hexdigest()
        index = self.artifacts / "geometry-results.json"
        index.write_text(json.dumps({original_sha: json.loads(self.geometry.read_text())}))
        review = {**self.review(sample), "geometry_path": "stage3/geometry-results.json",
                  "geometry_sha256": hashlib.sha256(index.read_bytes()).hexdigest()}
        direct = self.prepare_one(sample, review)
        self.assertEqual(1, len(direct.annotations))
        destination = self.root / "exports" / "original.json"
        result = export_stage3_geometry(geometry_results=index, original_sha256=original_sha,
                                        output=destination, data_root=self.root)
        self.assertEqual(original_sha, result["original_id"])
        self.assertEqual(["c1"], result["candidate_ids"])
        converted = {**review, "geometry_path": "exports/original.json",
                     "geometry_sha256": result["geometry_sha256"]}
        exported = self.prepare_one(sample, converted)
        self.assertEqual(direct.annotations, exported.annotations)
        self.assertEqual(direct.capture_metadata["canonical_faces"][0]["sha256"],
                         exported.capture_metadata["canonical_faces"][0]["sha256"])
        with self.assertRaisesRegex(ValueError, "output must be new"):
            export_stage3_geometry(geometry_results=index, original_sha256=original_sha,
                                   output=destination, data_root=self.root)

    def test_empty_negative_requires_explicit_attestation(self):
        sample = self.sample(raw=())
        with self.assertRaisesRegex(ValueError, "empty-face assertions"):
            self.prepare_one(sample)
        valid = {**self.review(sample), "verified_empty_faces": [0]}
        prepared = self.prepare_one(sample, valid)
        self.assertFalse(prepared.annotations)
        self.assertEqual(1, len(prepared.capture_metadata["canonical_faces"]))
        self.assertEqual([0], prepared.capture_metadata["stage4_review"]["verified_empty_faces"])

    def test_outside_crop_and_missing_face_owner_rejected(self):
        sample = self.sample(raw=(ImagePointAnnotation(x_fraction=0.5, y_fraction=0.5),))
        review_missing = self.review(sample)
        del review_missing["impact_face_indices"]
        with self.assertRaisesRegex(ValueError, "impact_face_indices"):
            self.prepare_one(sample, review_missing)
        reviewed = {**self.review(sample), "impact_face_indices": [0]}
        self.assertEqual(1, len(self.prepare_one(sample, reviewed).annotations))
        sample = self.sample(raw=(ImagePointAnnotation(x_fraction=0.5, y_fraction=0.5, face_index=0),))
        geom = json.loads(self.geometry.read_text())
        geom["faces"][0]["original_to_canonical"][0][2] = 1000
        self.geometry.write_text(json.dumps(geom))
        with self.assertRaisesRegex(ValueError, "outside canonical crop"):
            self.prepare_one(sample)

    def test_foreign_board_impacts_are_explicitly_excluded(self):
        sample = self.sample(raw=(ImagePointAnnotation(x_fraction=0.5, y_fraction=0.5),
                                  ImagePointAnnotation(x_fraction=0.8, y_fraction=0.8)))
        reviewed = {**self.review(sample), "impact_face_indices": [0, None]}
        prepared = self.prepare_one(sample, reviewed)
        self.assertEqual(1, len(prepared.annotations))
        self.assertEqual([0, None], prepared.capture_metadata["stage4_review"]["impact_face_indices"])
        with self.assertRaisesRegex(ValueError, "impact_face_indices"):
            self.prepare_one(sample, {**reviewed, "impact_face_indices": [0]})
        with self.assertRaisesRegex(ValueError, "empty-face assertions"):
            self.prepare_one(sample, {**reviewed, "impact_face_indices": [None, None]})

    def test_rejected_stage3_status_never_materialized(self):
        sample = self.sample()
        geom = json.loads(self.geometry.read_text())
        geom["quality_status"] = "GEOMETRY_UNCERTAIN"
        self.geometry.write_text(json.dumps(geom))
        with self.assertRaisesRegex(ValueError, "not READY"):
            self.prepare_one(sample)

    def test_draft_never_approves_or_assigns_foreign_images(self):
        sample = self.sample(raw=(ImagePointAnnotation(x_fraction=0.5, y_fraction=0.5),
                                  ImagePointAnnotation(x_fraction=0.7, y_fraction=0.7)))
        manifest = self.manifests / "raw.json"
        save_manifest(manifest, [sample])
        draft = self.root / "review_draft.json"
        info = write_review_draft(manifest=manifest, output=draft, data_root=self.root)
        self.assertEqual(1, info["unapproved_samples"])
        record = json.loads(draft.read_text())[sample.id]
        self.assertIs(record["face_ownership_reviewed"], False)
        self.assertEqual([None, None], record["impact_face_indices"])
        self.assertEqual("", record["reviewer"])
        self.assertEqual(hashlib.sha256(self.original.read_bytes()).hexdigest(),
                         record["original_sha256"])
        with self.assertRaisesRegex(ValueError, "path must be new"):
            write_review_draft(manifest=manifest, output=draft, data_root=self.root)
        with self.assertRaisesRegex(ValueError, "independent annotation review"):
            self.prepare_one(sample, record)

    def test_source_snapshot_rejects_inconsistent_impact_count(self):
        sample = self.sample()
        manifest = self.manifests / "raw.json"
        save_manifest(manifest, [sample])
        snapshot = self.root / "inconsistent_snapshot.json"
        snapshot.write_text(json.dumps({"manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
                                        "report": {"impacts": 5, "canonical_impacts": 0,
                                                   "raw_impacts": 1}}))
        review_map = self.root / "review.json"
        review_map.write_text(json.dumps({sample.id: self.review(sample)}))
        out_manifest = self.manifests / "prepared.json"
        out_snapshot = self.root / "prepared_snapshot.json"
        with self.assertRaisesRegex(ValueError, "source snapshot impact count"):
            materialize(manifest=manifest, snapshot=snapshot, review_map=review_map,
                        output_manifest=out_manifest, output_snapshot=out_snapshot,
                        data_root=self.root)
        self.assertFalse(out_manifest.exists())
        self.assertFalse(out_snapshot.exists())

    def test_materialized_empty_face_reaches_canonical_loader(self):
        positive = self.sample()
        alternate = self.root / "empty-original.png"
        Image.new("RGB", (512, 512), (9, 18, 27)).save(alternate)
        negative = replace(self.sample(raw=()), id="confirmed-empty",
                           image_path=alternate, group_id="different-session")
        original_sha = hashlib.sha256(alternate.read_bytes()).hexdigest()
        geometry = json.loads(self.geometry.read_text(encoding="utf-8"))
        geometry["original_id"] = original_sha
        other_geometry = self.artifacts / "empty-selected.json"
        other_geometry.write_text(json.dumps(geometry), encoding="utf-8")
        negative_review = {**self.review(negative),
                           "original_sha256": original_sha,
                           "geometry_path": "stage3/empty-selected.json",
                           "geometry_sha256": hashlib.sha256(other_geometry.read_bytes()).hexdigest(),
                           "verified_empty_faces": [0]}
        manifest = self.manifests / "source.json"
        save_manifest(manifest, [positive, negative])
        snapshot = self.root / "source-snapshot.json"
        snapshot.write_text(json.dumps({
            "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
            "report": {"impacts": 1, "raw_impacts": 1, "canonical_impacts": 0}
        }), encoding="utf-8")
        review_map = self.root / "review-map.json"
        review_map.write_text(json.dumps({
            positive.id: self.review(positive), negative.id: negative_review
        }), encoding="utf-8")
        output_manifest = self.manifests / "prepared.json"
        output_snapshot = self.root / "prepared-snapshot.json"
        materialize(manifest=manifest, snapshot=snapshot, review_map=review_map,
                    output_manifest=output_manifest, output_snapshot=output_snapshot,
                    data_root=self.root)
        data = CanonicalImpactDataset(DatasetView.load(output_manifest),
                                      manifest_dir=output_manifest.parent, input_size=512)
        self.assertEqual(2, len(data))
        self.assertEqual([0, 1], sorted(len(record.points) for record in data.records))
        prepared = load_manifest(output_manifest)
        empty = next(sample for sample in prepared if sample.id == negative.id)
        self.assertEqual([0], empty.capture_metadata["stage4_review"]["verified_empty_faces"])

    def test_partial_snapshot_failure_cleans_both_outputs(self):
        sample = self.sample()
        manifest = self.manifests / "source.json"
        save_manifest(manifest, [sample])
        snapshot = self.root / "source-snapshot.json"
        snapshot.write_text(json.dumps({
            "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
            "report": {"impacts": 1, "raw_impacts": 1, "canonical_impacts": 0}
        }), encoding="utf-8")
        reviews = self.root / "review-map.json"
        reviews.write_text(json.dumps({sample.id: self.review(sample)}), encoding="utf-8")
        output_manifest = self.manifests / "prepared.json"
        output_snapshot = self.root / "broken-snapshot.json"
        original_write = Path.write_text

        def fail_snapshot_write(path, content, *args, **kwargs):
            if path == output_snapshot:
                original_write(path, "{\"partial\": ", encoding="utf-8")
                raise OSError("simulated snapshot write failure")
            return original_write(path, content, *args, **kwargs)

        with patch.object(Path, "write_text", autospec=True,
                          side_effect=fail_snapshot_write):
            with self.assertRaisesRegex(OSError, "simulated snapshot write failure"):
                materialize(manifest=manifest, snapshot=snapshot, review_map=reviews,
                            output_manifest=output_manifest, output_snapshot=output_snapshot,
                            data_root=self.root)
        self.assertFalse(output_manifest.exists())
        self.assertFalse(output_snapshot.exists())

    def test_derived_manifest_snapshot_are_new_and_bound(self):
        sample = self.sample()
        manifest = self.manifests / "dataset_manifest.json"
        save_manifest(manifest, [sample])
        snapshot = self.root / "dataset_snapshot.json"
        old_hash = hashlib.sha256(manifest.read_bytes()).hexdigest()
        snapshot.write_text(json.dumps({"manifest_sha256": old_hash,
                                        "report": {"impacts": 1, "raw_impacts": 1, "canonical_impacts": 0}}))
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
        self.assertEqual(1, derived["report"]["canonical_impacts"])
        self.assertEqual(1, derived["report"]["impacts"])
        self.assertEqual(1, derived["report"]["raw_impacts"])
        with self.assertRaisesRegex(ValueError, "outputs must be new"):
            materialize(manifest=manifest, snapshot=snapshot, review_map=review_map,
                        output_manifest=out_manifest, output_snapshot=out_snapshot,
                        data_root=self.root)


if __name__ == "__main__":
    unittest.main()
