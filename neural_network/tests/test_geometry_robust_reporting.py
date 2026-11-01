"""Report denominators, unmatched centers and reproducible CLI outputs."""

from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from neural_network.archery_ml.geometry.benchmark import (
    RGB_PIXEL_HASH_SPEC,
    REAL_PREPROCESSING_VERSION,
    rgb_pixel_sha256,
    run,
)
from neural_network.archery_ml.geometry.comparison import compare, synthetic_metrics
from neural_network.archery_ml.geometry.synthetic import render_scene


class RobustReportingTest(unittest.TestCase):
    def test_rejected_samples_and_unmatched_centers_are_retained(self):
        scenes = [render_scene(), render_scene("blank")]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch(
                "neural_network.archery_ml.geometry.benchmark.scenarios",
                return_value=scenes,
            ):
                old = run(root / "old", "legacy")
                new = run(root / "new", "robust")
            self.assertEqual(2, new["synthetic"]["samples"])
            self.assertEqual(1, new["synthetic"]["ready"])
            for report, folder in ((old, "old"), (new, "new")):
                row = {
                    "id": "reviewed",
                    "input_sha256": "a" * 64,
                    "cached_rgb_file_sha256": "c" * 64,
                    "upright_rgb_sha256": "a" * 64,
                    "effective_detector_input_sha256": "b" * 64,
                    "pixel_hash_spec": RGB_PIXEL_HASH_SPEC,
                    "target_config": asdict(scenes[0].config),
                    "candidate_ids": ["f0", "f1"],
                    "preprocessing_version": REAL_PREPROCESSING_VERSION,
                    "dataset": "development_real",
                    "status": "READY",
                    "roi": [0, 0, 800, 800],
                    "pairs": [{"id": "f0", "error_px": 2.0}],
                    "preview": report["records"][0]["preview"],
                }
                if folder == "old":
                    row["pairs"].append({"id": "f1", "error_px": 20.0})
                else:
                    row["status"] = "GEOMETRY_UNCERTAIN"
                report["records"].append(row)
                (root / folder / "report.json").write_text(json.dumps(report))
            with patch(
                "neural_network.archery_ml.geometry.comparison.scenarios",
                return_value=scenes,
            ):
                result = compare(root / "old", root / "new", root / "comparison")
            paired = result["development_real_paired_centers"]
            self.assertEqual(1, paired["baseline"]["count"])
            self.assertEqual(1, len(paired["unmatched_references"]))
            self.assertTrue(result["cases"][0]["acceptance_regression"])
            self.assertEqual("NOT_VALIDATED", result["production_validation"])
            for path in (
                root / "new/contact-sheet.jpg",
                root / "new/metrics.csv",
                root / "comparison/index.html",
            ):
                self.assertTrue(path.is_file())

    def test_output_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileExistsError):
                run(Path(directory), "robust")

    def test_false_accept_uses_negative_scene_denominator(self):
        from dataclasses import replace
        from types import SimpleNamespace

        # One accepted negative among 99 accepted positives is 100% FPR.
        positive = render_scene(size=48)
        negative = replace(positive, expected_reject=True)
        with tempfile.TemporaryDirectory() as directory:
            with (
                patch(
                    "neural_network.archery_ml.geometry.benchmark.scenarios",
                    return_value=[positive] * 99 + [negative],
                ),
                patch(
                    "neural_network.archery_ml.geometry.benchmark.ImageQualityGate.evaluate",
                    return_value=SimpleNamespace(status=SimpleNamespace(value="READY")),
                ),
            ):
                report = run(Path(directory) / "report", "robust")
        self.assertEqual(100, report["synthetic"]["ready"])
        self.assertEqual(1, report["synthetic"]["negative_scenes"])
        self.assertEqual(1.0, report["synthetic"]["false_accept_fraction"])

    def test_real_hash_binding_and_rejected_input_comparison(self):
        import hashlib
        from PIL import Image

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pilot = root / "pilot"
            pilot.mkdir()
            image = render_scene().image.copy()
            path = pilot / "image.png"
            Image.fromarray(image).save(path)
            row = {
                "original_id": "photo",
                "number": 1,
                "read_status": "OK",
                "rgb_path": "image.png",
                "rgb_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            rejected = {"original_id": "missing", "number": 2, "read_status": "MISSING"}
            (pilot / "inventory.json").write_text(
                json.dumps({"images": [row, rejected]})
            )
            (pilot / "detections.json").write_text(
                json.dumps(
                    {
                        "photo": {
                            "candidates": [
                                {
                                    "id": "f0",
                                    "center": [400, 400],
                                    "diameter": 580,
                                    "bbox": [0, 0, 800, 800],
                                },
                                {
                                    "id": "foreign",
                                    "center": [210, 225],
                                    "diameter": 180,
                                    "bbox": [120, 135, 300, 315],
                                    "ellipse": {
                                        "center_x": 210,
                                        "center_y": 225,
                                        "diameter_x": 180,
                                        "diameter_y": 180,
                                        "angle_deg": 0,
                                    },
                                },
                            ]
                        }
                    }
                )
            )
            (pilot / "target-selections.json").write_text(
                json.dumps(
                    {
                        "photo": {
                            "candidate_ids": ["f0"],
                            "format": "SINGLE",
                            "layout": None,
                        }
                    }
                )
            )
            with patch(
                "neural_network.archery_ml.geometry.benchmark.scenarios",
                return_value=[],
            ):
                old = run(root / "old", "robust", pilot=pilot)
                run(root / "new", "robust", pilot=pilot)
            self.assertEqual(
                hashlib.sha256(image.tobytes()).hexdigest(),
                old["records"][0]["input_sha256"],
            )
            self.assertEqual(
                rgb_pixel_sha256(image), old["records"][0]["upright_rgb_sha256"]
            )
            self.assertEqual(
                64, len(old["records"][0]["effective_detector_input_sha256"])
            )
            with patch(
                "neural_network.archery_ml.geometry.comparison.scenarios",
                return_value=[],
            ):
                result = compare(root / "old", root / "new", root / "comparison")
            missing = next(r for r in result["cases"] if r["id"] == "missing")
            self.assertIsNone(missing["new_preview"])
            self.assertFalse(missing["input_identity_verified"])
            self.assertIn(
                "Preview unavailable", (root / "comparison/index.html").read_text()
            )
            # The original RGB and ROI stay fixed; only the foreign mask moves.
            detections_path = pilot / "detections.json"
            detections_data = json.loads(detections_path.read_text())
            detections_data["photo"]["candidates"][1]["ellipse"]["center_x"] = 300
            detections_path.write_text(json.dumps(detections_data))
            with patch(
                "neural_network.archery_ml.geometry.benchmark.scenarios",
                return_value=[],
            ):
                masked = run(root / "mask-changed", "robust", pilot=pilot)
            before = old["records"][0]
            after = masked["records"][0]
            self.assertEqual(before["input_sha256"], after["input_sha256"])
            self.assertEqual(before["upright_rgb_sha256"], after["upright_rgb_sha256"])
            self.assertEqual(before["roi"], after["roi"])
            self.assertNotEqual(
                before["effective_detector_input_sha256"],
                after["effective_detector_input_sha256"],
            )
            with self.assertRaisesRegex(
                ValueError, "image photo: effective_detector_input_sha256 changed"
            ):
                compare(root / "old", root / "mask-changed", root / "bad-mask")

            image[0, 0] = [1, 2, 3]
            Image.fromarray(image).save(path)
            row["rgb_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
            (pilot / "inventory.json").write_text(
                json.dumps({"images": [row, rejected]})
            )
            with patch(
                "neural_network.archery_ml.geometry.benchmark.scenarios",
                return_value=[],
            ):
                changed = run(root / "changed", "robust", pilot=pilot)
            with self.assertRaisesRegex(ValueError, "image photo: input_sha256 changed"):
                compare(root / "old", root / "changed", root / "bad")
            del changed["records"][0]["input_sha256"]
            (root / "changed/report.json").write_text(json.dumps(changed))
            with self.assertRaisesRegex(ValueError, "image photo: missing input_sha256"):
                compare(root / "old", root / "changed", root / "missing-hash")


    def test_comparison_rejects_changes_to_real_preprocessing_contract(self):
        from neural_network.archery_ml.targets import (
            TargetConfig,
            TargetFormat,
            TenRingMode,
        )

        config = asdict(TargetConfig(TargetFormat.SINGLE, 5, TenRingMode.RECURVE, 1))
        record = {
            "id": "photo",
            "dataset": "development_real",
            "status": "READY",
            "input_sha256": "1" * 64,
            "cached_rgb_file_sha256": "2" * 64,
            "upright_rgb_sha256": "3" * 64,
            "effective_detector_input_sha256": "4" * 64,
            "pixel_hash_spec": RGB_PIXEL_HASH_SPEC,
            "target_config": config,
            "candidate_ids": ["face-0", "face-1"],
            "roi": [10, 20, 200, 210],
            "preprocessing_version": REAL_PREPROCESSING_VERSION,
            "pairs": [],
        }
        changes = {
            "minimum_scoring_zone": ("target_config", {"minimum_scoring_zone": 6}),
            "format": ("target_config", {"format": "TRIPLE", "triple_layout": "VERTICAL"}),
            "layout": ("target_config", {"triple_layout": "VERTICAL"}),
            "candidate_ids": ("candidate_ids", ["face-1", "face-0"]),
            "preprocessing_version": ("preprocessing_version", "new-mask-algorithm"),
            "detector_pixels": ("effective_detector_input_sha256", "f" * 64),
            "missing_pixel_hash": ("effective_detector_input_sha256", None),
            "missing_target_config": ("target_config", None),
            "partial_target_config": ("target_config", {"drop": "ten_ring_mode"}),
            "missing_selection": ("candidate_ids", None),
            "missing_roi": ("roi", None),
            "missing_preprocessing": ("preprocessing_version", None),
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "old").mkdir()
            (root / "new").mkdir()
            report = {"records": [record], "provenance": {}}
            (root / "old/report.json").write_text(json.dumps(report))
            (root / "new/report.json").write_text(json.dumps(report))
            with patch(
                "neural_network.archery_ml.geometry.comparison.scenarios",
                return_value=[],
            ):
                valid = compare(root / "old", root / "new", root / "valid")
                self.assertTrue(valid["cases"][0]["input_identity_verified"])
                for name, (field, value) in changes.items():
                    with self.subTest(change=name):
                        altered = deepcopy(record)
                        if field == "target_config" and isinstance(value, dict):
                            altered[field].update(
                                {k: v for k, v in value.items() if k != "drop"}
                            )
                            if "drop" in value:
                                del altered[field][value["drop"]]
                        elif value is None:
                            del altered[field]
                        else:
                            altered[field] = value
                        (root / "new/report.json").write_text(
                            json.dumps({"records": [altered], "provenance": {}})
                        )
                        with self.assertRaisesRegex(ValueError, "image photo:"):
                            compare(root / "old", root / "new", root / f"bad-{name}")

    def test_synthetic_comparison_negative_rates(self):
        from types import SimpleNamespace

        for expected, statuses, negative_count, fraction in (
            ([False, False], ["READY", "READY"], 0, None),
            ([True], ["READY"], 1, 1.0),
            ([True, True, True], ["REJECTED"] * 3, 3, 0.0),
            ([False, True, False, True], ["READY", "READY", "READY", "REJECTED"], 2, 0.5),
        ):
            scenes = [
                SimpleNamespace(scenario=f"case-{i}", expected_reject=flag, norm_to_image=[])
                for i, flag in enumerate(expected)
            ]
            records = [
                {
                    "id": f"synthetic-{i:02}-case-{i}",
                    "dataset": "independent_synthetic",
                    "status": status,
                    "faces": [],
                }
                for i, status in enumerate(statuses)
            ]
            with patch(
                "neural_network.archery_ml.geometry.comparison.scenarios",
                return_value=scenes,
            ):
                result = synthetic_metrics(records)
            self.assertEqual(negative_count, result["negative_scenes"])
            self.assertEqual(fraction, result["false_accept_fraction_on_negative_scenes"])
            self.assertEqual(
                fraction,
                json.loads(json.dumps(result))["false_accept_fraction_on_negative_scenes"],
            )
