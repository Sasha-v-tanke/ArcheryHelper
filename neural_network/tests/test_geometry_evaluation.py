import json
import math
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from neural_network.archery_ml.contracts import (
    DatasetSample,
    ImageGeometryAnnotation,
    save_manifest,
)
from neural_network.archery_ml.data.view import DatasetView
from neural_network.archery_ml.geometry.cli import main
from neural_network.archery_ml.geometry.evaluate import GeometryEvaluator
from neural_network.archery_ml.targets import (
    TargetConfig,
    TargetFormat,
    TenRingMode,
    TripleLayout,
)


class GeometryEvaluationTest(unittest.TestCase):
    def test_evaluator_measures_real_manifest_geometry_in_canonical_units(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_path = root / "target.jpg"
            image, annotations = self._fixture()
            Image.fromarray(image).save(image_path, quality=100)
            sample = DatasetSample(
                id="geometry",
                image_path=image_path,
                annotation_path=None,
                split="test",
                source_id="fixture",
                group_id="fixture",
                geometry_annotations=annotations,
            )

            report = GeometryEvaluator().evaluate(
                DatasetView.from_samples((sample,)),
                self._config(),
            )

            self.assertEqual(1, report.samples)
            self.assertEqual(1, report.evaluated_samples)
            self.assertEqual(1, report.ready_samples)
            self.assertEqual({"READY": 1}, report.status_counts)
            self.assertIsNotNone(report.center_error.p95)
            self.assertIsNotNone(report.radial_error.p95)
            self.assertLess(report.center_error.p95, 0.03)
            self.assertLess(report.radial_error.p95, 0.04)
            self.assertEqual(72, report.radial_error.count)

    def test_cli_filters_geometry_samples_and_writes_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_path = root / "target.jpg"
            image, annotations = self._fixture()
            Image.fromarray(image).save(image_path, quality=100)
            manifest = root / "manifest.json"
            output = root / "geometry-report.json"
            save_manifest(
                manifest,
                [
                    DatasetSample(
                        id="geometry",
                        image_path=image_path,
                        annotation_path=None,
                        split="test",
                        source_id="fixture",
                        group_id="fixture",
                        geometry_annotations=annotations,
                    ),
                    DatasetSample(
                        id="no-geometry",
                        image_path=image_path,
                        annotation_path=None,
                        split="train",
                        source_id="other",
                        group_id="other",
                    ),
                ],
            )

            result = main(
                [
                    "--manifest",
                    str(manifest),
                    "--output",
                    str(output),
                    "--source",
                    "fixture",
                    "--split",
                    "test",
                ]
            )

            self.assertEqual(0, result)
            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(1, payload["samples"])
            self.assertEqual(1, payload["evaluated_samples"])
            self.assertEqual(1, payload["ready_samples"])
            self.assertEqual("SINGLE", payload["target_config"]["format"])

    def test_evaluator_rejects_triple_until_geometry_annotations_have_face_index(self):
        image, annotations = self._fixture()
        with tempfile.TemporaryDirectory() as tmp:
            image_path = Path(tmp) / "target.jpg"
            Image.fromarray(image).save(image_path, quality=100)
            sample = DatasetSample(
                id="geometry",
                image_path=image_path,
                annotation_path=None,
                geometry_annotations=annotations,
            )
            config = TargetConfig(
                format=TargetFormat.TRIPLE,
                minimum_scoring_zone=6,
                ten_ring_mode=TenRingMode.RECURVE,
                face_diameter_mm=400,
                triple_layout=TripleLayout.VERTICAL,
            )

            with self.assertRaisesRegex(ValueError, "requires SINGLE"):
                GeometryEvaluator().evaluate(DatasetView.from_samples((sample,)), config)

    def _fixture(self):
        size = 700
        center = (350, 350)
        radius = 220
        image = np.full((size, size, 3), 118, dtype=np.uint8)
        cv2.circle(image, center, radius, (235, 235, 228), -1, lineType=cv2.LINE_AA)
        cv2.circle(image, center, int(round(radius * 0.8)), (28, 28, 28), -1, lineType=cv2.LINE_AA)
        cv2.circle(image, center, int(round(radius * 0.6)), (28, 92, 210), -1, lineType=cv2.LINE_AA)
        cv2.circle(image, center, int(round(radius * 0.4)), (220, 30, 45), -1, lineType=cv2.LINE_AA)
        cv2.circle(image, center, int(round(radius * 0.2)), (250, 214, 20), -1, lineType=cv2.LINE_AA)
        annotations = (
            ImageGeometryAnnotation(
                "target_center",
                ((center[0] / size, center[1] / size),),
                source_label="target-center",
            ),
            self._ring_annotation("Ring-9-Polygon", center, radius * 0.2, size),
            self._ring_annotation("Ring-7-Polygon", center, radius * 0.4, size),
            self._ring_annotation("Ring-5-Polygon", center, radius * 0.6, size),
        )
        return image, annotations

    def _ring_annotation(self, label, center, radius, size):
        points = []
        for angle_degrees in range(0, 360, 15):
            angle = math.radians(angle_degrees)
            points.append(
                (
                    (center[0] + radius * math.cos(angle)) / size,
                    (center[1] + radius * math.sin(angle)) / size,
                )
            )
        return ImageGeometryAnnotation(
            "ring",
            tuple(points),
            source_label=label,
        )

    def _config(self):
        return TargetConfig(
            format=TargetFormat.SINGLE,
            minimum_scoring_zone=1,
            ten_ring_mode=TenRingMode.RECURVE,
            face_diameter_mm=1220,
        )


if __name__ == "__main__":
    unittest.main()
