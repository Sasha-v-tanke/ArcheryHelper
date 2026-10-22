import hashlib
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import cv2
from PIL import Image

from neural_network.archery_ml.contracts import DatasetSample, ImageGeometryAnnotation, TargetMetadata
from neural_network.archery_ml.data.importers.coco import import_coco
from neural_network.archery_ml.data.importers.mobile_geometry import import_mobile_geometry
from neural_network.archery_ml.data.registry import DatasetRegistry, DatasetSource
from neural_network.archery_ml.data.split import assign_splits
from neural_network.archery_ml.data.view import DatasetView
from neural_network.archery_ml.geometry.detector import TargetGeometryDetector, _stable_transform
from neural_network.archery_ml.geometry.evaluate import GeometryEvaluator, _annotation_issues, _equal_group_metrics
from neural_network.archery_ml.geometry.normalize import TargetNormalizer
from neural_network.archery_ml.geometry.quality import ImageQualityGate, QualityStatus
from neural_network.archery_ml.geometry.validation import session_bootstrap, validate_holdout
from neural_network.archery_ml.targets import TargetConfig, TargetFormat, TenRingMode, TripleLayout
from neural_network.tests import test_geometry_evaluation as fixtures
from neural_network.tests import test_geometry_diagnostics as diagnostics


class Stage3CompletionTest(unittest.TestCase):
    def test_contract_round_trip_and_invalid_landmarks(self):
        annotation = ImageGeometryAnnotation("landmark", ((0.2, 0.3),), face_index=2,
                                             canonical_points=((0.4, -0.2),), annotation_quality="reviewed")
        self.assertEqual(annotation, ImageGeometryAnnotation.from_dict(annotation.to_dict()))
        for kwargs in ({"face_index": -1}, {"face_index": 1.5},
                       {"canonical_points": ((float("nan"), 0),)},
                       {"canonical_points": ((0, 0), (1, 0))}):
            with self.assertRaises(ValueError):
                ImageGeometryAnnotation("ring", ((0.2, 0.3),), **kwargs)

    def test_singular_refinement_falls_back_and_singular_initial_raises(self):
        initial = np.eye(3)
        for refined in (np.zeros((3, 3)), np.diag([1, 1, 1e-14]), np.full((3, 3), np.nan)):
            np.testing.assert_equal(initial, _stable_transform(initial, refined, []))
        with self.assertRaises(np.linalg.LinAlgError):
            _stable_transform(np.zeros((3, 3)), initial, [])

    def test_invalid_geometry_cannot_normalize(self):
        fixture = fixtures.GeometryEvaluationTest()
        image, _ = fixture._fixture()
        result = TargetGeometryDetector().detect(image, fixture._config())
        for change in ({"homography": ((0, 0, 0),) * 3},
                       {"inverse_homography": ((1, 0, 0), (0, 1, 0), (0, 0, 1))},
                       {"geometry_confidence": float("nan")}):
            invalid = replace(result, faces=(replace(result.faces[0], **change),))
            self.assertEqual(QualityStatus.GEOMETRY_UNCERTAIN, ImageQualityGate().evaluate(image, invalid).status)
            with self.assertRaisesRegex(ValueError, "requires READY"):
                TargetNormalizer().normalize(image, invalid)

    def test_numerical_fitting_failure_is_a_public_rejection(self):
        fixture = fixtures.GeometryEvaluationTest()
        image, _ = fixture._fixture()
        with patch.object(TargetGeometryDetector, "_fit_face", side_effect=np.linalg.LinAlgError):
            result = TargetGeometryDetector().detect(image, fixture._config())
        self.assertEqual(QualityStatus.TARGET_NOT_FOUND, ImageQualityGate().evaluate(image, result).status)

    def test_2d_error_detects_rotation_invisible_to_radial_metric_and_associates_faces(self):
        fixture = fixtures.GeometryEvaluationTest()
        image, _ = fixture._fixture()
        base = TargetGeometryDetector().detect(image, fixture._config())
        # Three explicit independent faces; deliberately rotate only the third GT point.
        faces = tuple(replace(base.faces[0], face_index=i) for i in range(3))
        annotations = tuple(ImageGeometryAnnotation(
            "landmark", ((460 / 700, 350 / 700),), face_index=i,
            canonical_points=(((0.5, 0) if i < 2 else (0, 0.5)),),
        ) for i in range(3))
        class Detector:
            def detect(self, image, config):
                return replace(base, faces=faces, expected_face_count=3, layout_valid=True)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "image.png"
            Image.fromarray(image).save(path)
            sample = DatasetSample("fixture", path, None, geometry_annotations=annotations)
            config = TargetConfig(TargetFormat.TRIPLE, 1, TenRingMode.RECURVE, 400, triple_layout=TripleLayout.VERTICAL)
            report = GeometryEvaluator(detector=Detector()).evaluate(DatasetView.from_samples([sample]), config)
        self.assertEqual(3, report.to_dict()["point_error_2d"]["count"])
        self.assertLess(report.sample_results[0].face_error["0"]["point_error_2d"]["max"], 0.01)
        self.assertGreater(report.sample_results[0].face_error["2"]["point_error_2d"]["max"], 0.7)
        self.assertEqual(0, report.radial_error.count)

    def test_mobile_dev_split_is_preserved_and_cannot_mix_with_holdout(self):
        sample = DatasetSample("mobile", Path("unused"), None, split="mobile_dev", group_id="session")
        self.assertEqual("mobile_dev", assign_splits([sample])[0].split)
        with self.assertRaises(ValueError):
            assign_splits([sample, replace(sample, id="other", split="mobile_real_test")])

    def test_versioned_gt_changes_exactly_six_labels_and_no_points(self):
        root = Path("data/sources/archery_aiscore_ounar/v9")
        if not root.exists():
            self.skipTest("external v9 export unavailable")
        source = DatasetRegistry.load().get("archery_aiscore_ounar")
        original = import_coco(root, replace(source, options={k: v for k, v in source.options.items() if k != "corrections_file"}))
        corrected = import_coco(root, source)
        changed = []
        for before, after in zip(original, corrected):
            self.assertEqual(before.raw_annotations, after.raw_annotations)
            for a, b in zip(before.geometry_annotations, after.geometry_annotations):
                self.assertEqual(a.points, b.points)
                if a.source_label != b.source_label:
                    changed.append(before.id)
            self.assertFalse(_annotation_issues(after.geometry_annotations))
        self.assertEqual(6, len(changed))
        self.assertEqual(3, len(set(changed)))

    def test_holdout_gate_cannot_pass_regression_data_or_draft_policy(self):
        policy = json.loads(Path("dataset_tools/stage3_geometry_policy.json").read_text())
        report = {"sample_results": [], "coverage": {"false_accept_fraction": None, "false_reject_fraction": None}}
        result = validate_holdout(report, policy)
        self.assertEqual("FAIL", result["status"])
        self.assertGreater(len(result["reasons"]), 5)

    def test_session_bootstrap_is_reproducible(self):
        samples = [{"point_error_2d": {"mean": 0.01 * i}, "capture_metadata": {"capture_session_id": str(i)}} for i in range(3)]
        self.assertEqual(session_bootstrap(samples, repetitions=100), session_bootstrap(samples, repetitions=100))
        self.assertIsNone(session_bootstrap(samples[:1])["low"])

    def test_mobile_import_validates_independent_gt_hash_and_session_isolation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = root / "target.png"
            Image.fromarray(np.zeros((40, 40, 3), dtype=np.uint8)).save(image)
            meta = {"capture_session_id": "session-1", "original_id": "original-1",
                    "device_model": "camera", "original_sha256": hashlib.sha256(image.read_bytes()).hexdigest(),
                    "gt_revision": "manual-v1", "annotator": "a", "reviewer": "b",
                    "orientation_reference": "physical_marker", "suitability": "usable", "image_size": [40, 40]}
            annotation = ImageGeometryAnnotation("landmark", ((0.2, 0.2), (0.8, 0.2), (0.8, 0.8), (0.2, 0.8)),
                face_index=0, canonical_points=((-0.6, -0.6), (0.6, -0.6), (0.6, 0.6), (-0.6, 0.6)),
                annotation_quality="reviewed", uncertainty_norm=0.002)
            center = ImageGeometryAnnotation("target_center", ((0.5, 0.5),), face_index=0,
                annotation_quality="reviewed", uncertainty_norm=0.002)
            sample = DatasetSample("test", Path("target.png"), None, split="mobile_dev",
                target_metadata=TargetMetadata("SINGLE", 1, "RECURVE"),
                geometry_annotations=(center, annotation), capture_metadata=meta)
            source = DatasetSource("mobile", "mobile_geometry", str(root), "1", "private", "test", None, ("geometry_evaluation",), annotation_space="image")
            path = root / "geometry.json"
            path.write_text(json.dumps({"samples": [sample.to_dict()]}))
            imported = import_mobile_geometry(root, source)
            self.assertEqual("mobile:session-1", imported[0].group_id)
            path.write_text(json.dumps({"samples": [sample.to_dict(), replace(sample, id="other", split="mobile_real_test").to_dict()]}))
            with self.assertRaisesRegex(ValueError, "cross-split"):
                import_mobile_geometry(root, source)
            path.write_text(json.dumps({"samples": [replace(sample, capture_metadata={**meta, "original_sha256": "0" * 64}).to_dict()]}))
            with self.assertRaisesRegex(ValueError, "checksum"):
                import_mobile_geometry(root, source)

    def test_equal_image_weighting_does_not_depend_on_landmark_density(self):
        sparse = _equal_group_metrics([(0.01, 0.02), (0.2,)])
        dense = _equal_group_metrics([(0.01, 0.02) * 100, (0.2,)])
        self.assertAlmostEqual(sparse["mean"], dense["mean"])
        self.assertEqual(sparse["p95"], dense["p95"])
        self.assertEqual(2, dense["groups"])

    def test_both_triple_layouts_evaluate_independent_projected_landmarks(self):
        fixture = diagnostics.GeometryDiagnosticsTest()
        cases = ((TripleLayout.VERTICAL, ((350, 170), (350, 450), (350, 730))),
                 (TripleLayout.TRIANGULAR, ((350, 170), (215, 470), (485, 470))))
        with tempfile.TemporaryDirectory() as tmp:
            for layout, centers in cases:
                photo, transform, _, radius = fixture._scenario(
                    image_size=(900, 700), centers=centers, radius=115,
                    output_size=(700, 900),
                    dst_corners=((20, 20), (679, 5), (685, 879), (5, 885)), minimum_zone=6)
                height, width = photo.shape[:2]
                annotations = []
                canonical = np.array([[-0.3, -0.3], [0.3, -0.3], [0.3, 0.3], [-0.3, 0.3]], dtype=np.float32)
                for index, center in enumerate(centers):
                    source = canonical * radius + center
                    projected = cv2.perspectiveTransform(source.astype(np.float32).reshape(-1, 1, 2), transform).reshape(-1, 2)
                    annotations.append(ImageGeometryAnnotation("landmark", tuple(map(tuple, projected / (width, height))),
                        face_index=index, canonical_points=tuple(map(tuple, canonical))))
                path = Path(tmp) / f"{layout.value}.png"
                Image.fromarray(photo).save(path)
                sample = DatasetSample(layout.value, path, None, geometry_annotations=tuple(annotations))
                config = fixture._triple_config(layout, 6)
                report = GeometryEvaluator().evaluate(DatasetView.from_samples([sample]), config).to_dict()
                self.assertEqual(12, report["point_error_2d"]["count"])
                self.assertLess(report["point_error_2d"]["p95"], 0.08)
                self.assertEqual({"0", "1", "2"}, set(report["sample_results"][0]["face_error"]))

    def test_negative_without_geometry_remains_in_false_accept_denominator(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "blank.png"
            Image.fromarray(np.full((100, 100, 3), 120, dtype=np.uint8)).save(path)
            sample = DatasetSample("negative", path, None, capture_metadata={"suitability": "unusable"})
            report = GeometryEvaluator().evaluate(DatasetView.from_samples([sample]), fixtures.GeometryEvaluationTest()._config()).to_dict()
            self.assertEqual(1, report["samples"])
            self.assertEqual(1, report["coverage"]["unusable_samples"])
            self.assertEqual(0, report["coverage"]["false_accept_fraction"])
            self.assertIsNone(report["point_error_2d"]["p95"])


if __name__ == "__main__":
    unittest.main()
