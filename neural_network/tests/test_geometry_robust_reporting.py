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
