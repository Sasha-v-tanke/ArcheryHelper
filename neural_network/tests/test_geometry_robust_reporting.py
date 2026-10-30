"""Report denominators, unmatched centers and reproducible CLI outputs."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from neural_network.archery_ml.geometry.benchmark import run
from neural_network.archery_ml.geometry.comparison import compare
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
                                }
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
            with self.assertRaisesRegex(ValueError, "inputs or selection ROI changed"):
                compare(root / "old", root / "changed", root / "bad")
            del changed["records"][0]["input_sha256"]
            (root / "changed/report.json").write_text(json.dumps(changed))
            with self.assertRaisesRegex(ValueError, "requires input_sha256"):
                compare(root / "old", root / "changed", root / "missing-hash")
