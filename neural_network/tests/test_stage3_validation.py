"""Controlled fixtures exercise final gates; they are not real-photo validation."""
import copy
import hashlib
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from neural_network.archery_ml.geometry.validation import aggregate_reports, validate_holdout
from neural_network.archery_ml.geometry.cli import main
from neural_network.archery_ml.targets import TargetConfig, TargetTemplate, TargetFormat, TenRingMode, TripleLayout
from neural_network.archery_ml.data.corrections import corrected_coco
from neural_network.archery_ml.data.registry import DatasetRegistry
from neural_network.archery_ml.data.geometry_provenance import annotation_digest, verify_mobile_gt
from neural_network.archery_ml.contracts import DatasetSample, ImageGeometryAnnotation, TargetMetadata
from neural_network.archery_ml.data.view import DatasetView
from neural_network.archery_ml.geometry.evaluate import _independent_landmarks


class HoldoutValidationTest(unittest.TestCase):
    def setUp(self):
        self.policy = json.loads(Path("dataset_tools/stage3_geometry_policy.json").read_text())
        self.policy.update(status="frozen", minimum_sessions=2, minimum_devices=1, minimum_originals=1)
        for config in self.policy["required_configurations"]:
            config.update(minimum_originals=5, minimum_sessions=2, minimum_ready=4)
        sha = hashlib.sha256(json.dumps(self.policy, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        self.reports = []
        for index, config in enumerate(self.policy["required_configurations"]):
            count = 1 if config["format"] == "SINGLE" else 3
            metric = dict(mean=.001, median=.001, p95=.001, max=.001, count=4)
            samples = []
            for j in range(5):
                identity = f"{index}:{j}"
                samples.append(dict(sample_id=identity, split="mobile_real_test", status="READY" if j < 4 else "REJECTED",
                    detected_face_count=count if j < 4 else 0, annotation_issues=[],
                    point_error_2d={**metric, "count": 4 * count}, point_values_2d=[.001] * (4 * count) if j < 4 else [],
                    face_error={str(i): dict(point_error_2d=metric.copy(), independent_landmarks=4,
                        orientation_reference="manual_calibration") for i in range(count)} if j < 4 else {},
                    capture_metadata=dict(original_id=identity, original_sha256=hashlib.sha256(identity.encode()).hexdigest(),
                        capture_session_id=f"session-{j % 2}", device_model="camera", suitability="usable" if j < 4 else "unusable",
                        gt_revision="reviewed-v1", gt_annotations_sha256="annotation-checksum"),
                    gt_provenance=dict(revision="reviewed-v1", annotations_sha256="annotation-checksum", annotation_manifest_sha256="gt-manifest")))
            self.reports.append(dict(target_config=config.copy(), metrics_version="stage3-2d-v2", sample_results=samples,
                provenance=dict(algorithm_sha="algorithm", working_tree_dirty=False, manifest_sha256="manifest",
                    policy_sha256=sha, gt_revision="reviewed-v1", gt_sources=["gt-manifest"], dependencies={"python": "test"})))

    def result(self):
        return aggregate_reports(self.reports, self.policy)["validation"]

    def assertFails(self, text):
        result = self.result()
        self.assertEqual("FAIL", result["status"])
        self.assertTrue(any(text in r for r in result["reasons"]), result)

    def test_complete_matrix_passes_controlled_fixture_and_contract(self):
        for c in self.policy["required_configurations"]:
            config = TargetConfig(TargetFormat(c["format"]), c["minimum_scoring_zone"], TenRingMode(c["ten_ring_mode"]),
                                  400, None if c["triple_layout"] is None else TripleLayout(c["triple_layout"]))
            self.assertTrue(TargetTemplate.from_config(config).ring_boundaries)
        self.assertEqual("PASS", self.result()["status"])

    def test_200_single_cannot_pass_without_triple(self):
        report = self.reports[0]
        report["sample_results"] = [copy.deepcopy(report["sample_results"][i % 5]) for i in range(200)]
        for i, sample in enumerate(report["sample_results"]):
            sample["capture_metadata"].update(original_id=str(i), original_sha256=hashlib.sha256(str(i).encode()).hexdigest())
        self.reports = [report]
        self.assertFails("TRIPLE")

    def test_missing_triangular_fails(self):
        self.reports = [r for r in self.reports if r["target_config"]["triple_layout"] != "TRIANGULAR"]
        self.assertFails("TRIANGULAR")

    def test_one_underpopulated_configuration_fails(self):
        self.reports[0]["sample_results"].pop()
        self.assertFails("insufficient configuration")

    def test_landmarks_from_one_face_cannot_cover_another(self):
        sample = self.reports[-1]["sample_results"][0]
        sample["face_error"]["0"]["independent_landmarks"] = 12
        sample["face_error"]["1"]["independent_landmarks"] = 0
        self.assertFails("lacks independent landmarks")

    def test_missing_or_invalid_face_index_fails(self):
        sample = self.reports[-1]["sample_results"][0]
        sample["face_error"]["3"] = sample["face_error"].pop("2")
        self.assertFails("face_index")

    def test_subgroup_p95_cannot_hide_in_global_metrics(self):
        for s in self.reports[0]["sample_results"][:4]:
            s["point_values_2d"] = [.02] * 4
        self.assertFails("SINGLE:SINGLE:1:RECURVE p95")

    def test_face_metrics_are_required_and_bounded(self):
        self.reports[-1]["sample_results"][0]["face_error"]["2"]["point_error_2d"]["p95"] = .02
        self.assertFails("READY face p95")

    def test_duplicate_original_is_rejected_not_counted_twice(self):
        self.reports.append(copy.deepcopy(self.reports[0]))
        result = aggregate_reports(self.reports, self.policy)
        self.assertEqual(90, len(result["sample_results"]))
        self.assertEqual("FAIL", result["validation"]["status"])

    def test_incompatible_provenance_fails(self):
        for key in ("algorithm_sha", "manifest_sha256", "policy_sha256", "gt_revision", "gt_sources", "dependencies"):
            with self.subTest(key=key):
                reports = copy.deepcopy(self.reports)
                reports[0]["provenance"][key] = "different"
                self.assertEqual("FAIL", aggregate_reports(reports, self.policy)["validation"]["status"])

    def test_unknown_and_absent_suitability_fail(self):
        meta = self.reports[0]["sample_results"][0]["capture_metadata"]
        for value in (None, "unknown"):
            meta["suitability"] = value
            self.assertFails("unknown suitability")

    def test_unanchored_orientation_cannot_attest_global_hit_map(self):
        for value in (None, "unknown", "unanchored_image_or_layout_axes"):
            self.reports[0]["sample_results"][0]["face_error"]["0"]["orientation_reference"] = value
            self.assertFails("physical orientation")

    def test_draft_policy_and_changed_policy_checksum_fail(self):
        self.policy["status"] = "draft"
        self.assertFails("not been frozen")
        self.assertFails("policy checksum")

    def test_gt_revision_and_checksum_mismatch_fail(self):
        sample = self.reports[0]["sample_results"][0]
        sample["gt_provenance"]["revision"] = "wrong"
        self.assertFails("GT provenance")
        sample["gt_provenance"]["revision"] = "reviewed-v1"
        sample["capture_metadata"]["gt_annotations_sha256"] = "changed"
        self.assertFails("GT provenance")

    def test_legacy_development_report_cannot_pass_final_gate(self):
        self.reports[0]["sample_results"][0]["gt_provenance"] = {}
        self.assertFails("GT provenance")

    def test_aggregate_cli_serialization_and_exit_code(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            policy = root / "policy.json"
            policy.write_text(json.dumps(self.policy))
            args = ["--policy", str(policy), "--output", str(root / "result.json")]
            for i, report in enumerate(self.reports):
                path = root / f"{i}.json"
                path.write_text(json.dumps(report))
                args.extend(["--aggregate-report", str(path)])
            with patch("builtins.print"):
                self.assertEqual(0, main(args))
            self.assertEqual("PASS", json.loads((root / "result.json").read_text())["validation"]["status"])


class GTBindingTest(unittest.TestCase):
    def test_cli_revision_is_only_an_assertion_and_mixed_revisions_fail(self):
        sample = DatasetSample("test", Path("unused"), None, capture_metadata={"gt_annotations_sha256": "test"})
        view = DatasetView.from_samples([sample])
        with patch("neural_network.archery_ml.geometry.cli.DatasetView.load", return_value=view), patch(
                "neural_network.archery_ml.geometry.cli.verify_mobile_gt", return_value={"revision": "v1"}):
            with self.assertRaisesRegex(ValueError, "declared GT revision"):
                main(["--gt-revision", "wrong"])
        with patch("neural_network.archery_ml.geometry.cli.DatasetView.load", return_value=DatasetView.from_samples([sample, sample])), patch(
                "neural_network.archery_ml.geometry.cli.verify_mobile_gt", side_effect=[{"revision": "v1"}, {"revision": "v2"}]):
            with self.assertRaisesRegex(ValueError, "mixed GT revisions"):
                main([])

    def test_repeated_or_collinear_landmarks_are_not_independent(self):
        square = {(0, 0), (1, 0), (1, 1), (0, 1)}
        self.assertEqual(4, _independent_landmarks(square, square))
        self.assertEqual(0, _independent_landmarks(square, {(0, 0)}))
        self.assertEqual(0, _independent_landmarks(square, {(i, i) for i in range(4)}))

    def test_source_binding_detects_stale_checksums_and_revision(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = root / "image"
            image.write_bytes(b"original")
            path = root / "geometry.json"
            sample = DatasetSample("test", image, path, target_metadata=TargetMetadata("SINGLE", 1, "RECURVE"),
                geometry_annotations=(ImageGeometryAnnotation("landmark", ((.2,.2),), face_index=0, canonical_points=((.1,.1),)),),
                capture_metadata={"gt_revision": "v1", "annotator": "a", "reviewer": "b",
                                  "original_sha256": hashlib.sha256(b"original").hexdigest()})
            sample.capture_metadata["gt_annotations_sha256"] = annotation_digest(sample)
            path.write_text(json.dumps({"samples": [sample.to_dict()]}))
            sample.capture_metadata["gt_manifest_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
            baseline = verify_mobile_gt(sample)
            for changed in (replace(sample, capture_metadata={**sample.capture_metadata, "gt_revision": "wrong"}),
                            replace(sample, geometry_annotations=())):
                with self.assertRaisesRegex(ValueError, "checksum"):
                    verify_mobile_gt(changed)
            revised = replace(sample, capture_metadata={**sample.capture_metadata, "gt_revision": "v2"})
            revised.capture_metadata["gt_annotations_sha256"] = annotation_digest(revised)
            with self.assertRaisesRegex(ValueError, "manifest"):
                verify_mobile_gt(revised)
            path.write_text(json.dumps({"samples": [revised.to_dict()]}))
            revised.capture_metadata["gt_manifest_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
            self.assertNotEqual(baseline, verify_mobile_gt(revised))
            image.write_bytes(b"modified")
            with self.assertRaisesRegex(ValueError, "original checksum"):
                verify_mobile_gt(revised)

    def test_real_coco_corrections_exact_identity_repeatability_and_checksum(self):
        root = Path("data/sources/archery_aiscore_ounar/v9")
        if not root.exists():
            self.skipTest("external v9 export unavailable")
        source = DatasetRegistry.load().get("archery_aiscore_ounar")
        changes = json.loads(Path(source.options["corrections_file"]).read_text())
        path = root / "train/_annotations.coco.json"
        raw = json.loads(path.read_bytes())
        expected = copy.deepcopy(raw)
        categories = {c["name"]: c["id"] for c in raw["categories"]}
        for change in changes["changes"]:
            annotation = next(a for a in expected["annotations"] if a["id"] == change["annotation_id"])
            self.assertEqual(change["image_id"], annotation["image_id"])
            annotation["category_id"] = categories[change["new_label"]]
        self.assertEqual(expected, corrected_coco(root, path, source))
        self.assertEqual(expected, corrected_coco(root, path, source))
        with tempfile.TemporaryDirectory() as tmp:
            testroot = Path(tmp)
            (testroot / "train").mkdir()
            bad = testroot / "train/_annotations.coco.json"
            bad.write_text(json.dumps(raw))
            with self.assertRaisesRegex(ValueError, "checksum"):
                corrected_coco(testroot, bad, source)
            correction_path = testroot / "corrections.json"
            for key, value in (("annotation_id", -1), ("image_id", -1), ("image_id", 40),
                               ("old_label", "Ring-3-Polygon"), ("annotation_sha256", "0" * 64)):
                with self.subTest(key=key, value=value):
                    changed = copy.deepcopy(changes)
                    changed["changes"][0][key] = value
                    correction_path.write_text(json.dumps(changed))
                    configured = replace(source, options={**source.options, "corrections_file": str(correction_path)})
                    with self.assertRaises(ValueError):
                        corrected_coco(root, path, configured)


if __name__ == "__main__":
    unittest.main()
