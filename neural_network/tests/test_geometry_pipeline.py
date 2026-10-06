import math
import unittest
from dataclasses import replace

import cv2
import numpy as np

from neural_network.archery_ml.geometry import (
    CANONICAL_CENTER_PX,
    CANONICAL_RADIUS_PX,
    ImageQualityGate,
    QualityStatus,
    QualityThresholds,
    TargetGeometryDetector,
    TargetNormalizer,
)
from neural_network.archery_ml.targets import TargetConfig, TargetFormat, TenRingMode, TripleLayout


class GeometryPipelineTest(unittest.TestCase):
    def setUp(self):
        self.detector = TargetGeometryDetector()
        self.normalizer = TargetNormalizer()

    def test_single_face_geometry_and_normalization_under_projective_transform(self):
        source = np.full((700, 700, 3), 118, dtype=np.uint8)
        center = (350, 350)
        nominal_radius = 220
        self._draw_face(source, center, nominal_radius, minimum_zone=1)
        source_corners = np.float32([[0, 0], [699, 0], [699, 699], [0, 699]])
        target_corners = np.float32([[42, 65], [654, 34], [675, 650], [28, 625]])
        camera_transform = cv2.getPerspectiveTransform(source_corners, target_corners)
        photo = cv2.warpPerspective(
            source,
            camera_transform,
            (700, 700),
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(118, 118, 118),
        )

        result = self.detector.detect(photo, self._single_config())

        self.assertEqual(1, len(result.faces))
        self.assertTrue(result.layout_valid)
        self.assertGreaterEqual(result.faces[0].ring_count, 2)
        self.assertGreater(result.geometry_confidence, 0.2)

        point_norm = np.array([0.28, -0.16], dtype=np.float64)
        source_point = np.array(
            [
                [
                    center[0] + point_norm[0] * nominal_radius,
                    center[1] + point_norm[1] * nominal_radius,
                ]
            ],
            dtype=np.float32,
        ).reshape(1, 1, 2)
        photo_point = cv2.perspectiveTransform(source_point, camera_transform)
        canonical_point = cv2.perspectiveTransform(
            photo_point,
            np.asarray(result.faces[0].homography, dtype=np.float64),
        ).reshape(2)
        recovered_norm = np.array(
            [
                (canonical_point[0] - CANONICAL_CENTER_PX) / CANONICAL_RADIUS_PX,
                (canonical_point[1] - CANONICAL_CENTER_PX) / CANONICAL_RADIUS_PX,
            ]
        )
        error = float(np.linalg.norm(recovered_norm - point_norm))
        self.assertLess(error, 0.04)

        canonical_faces = self.normalizer.normalize(photo, result)
        self.assertEqual(1, len(canonical_faces))
        self.assertEqual((512, 512, 3), canonical_faces[0].image.shape)
        center_photo = cv2.perspectiveTransform(
            np.array([[[center[0], center[1]]]], dtype=np.float32),
            camera_transform,
        )
        normalized_center = cv2.perspectiveTransform(
            center_photo,
            np.asarray(canonical_faces[0].original_to_canonical, dtype=np.float64),
        ).reshape(2)
        self.assertLess(
            math.hypot(
                normalized_center[0] - CANONICAL_CENTER_PX,
                normalized_center[1] - CANONICAL_CENTER_PX,
            ),
            5.0,
        )

    def test_vertical_and_triangular_triple_layouts(self):
        vertical = np.full((800, 600, 3), 118, dtype=np.uint8)
        for center in ((300, 170), (300, 400), (300, 630)):
            self._draw_face(vertical, center, 100, minimum_zone=6)
        vertical_result = self.detector.detect(
            vertical,
            self._triple_config(TripleLayout.VERTICAL),
        )

        self.assertEqual(3, len(vertical_result.faces))
        self.assertTrue(vertical_result.layout_valid)
        vertical_y = [face.center[1] for face in vertical_result.faces]
        self.assertEqual(vertical_y, sorted(vertical_y))

        triangular = np.full((620, 700, 3), 118, dtype=np.uint8)
        for center in ((350, 155), (220, 405), (480, 405)):
            self._draw_face(triangular, center, 110, minimum_zone=6)
        triangular_result = self.detector.detect(
            triangular,
            self._triple_config(TripleLayout.TRIANGULAR),
        )

        self.assertEqual(3, len(triangular_result.faces))
        self.assertTrue(triangular_result.layout_valid)
        self.assertLess(
            triangular_result.faces[0].center[1],
            triangular_result.faces[1].center[1],
        )
        self.assertLess(
            triangular_result.faces[1].center[0],
            triangular_result.faces[2].center[0],
        )

    def test_quality_gate_returns_all_finite_statuses(self):
        image = np.full((700, 700, 3), 118, dtype=np.uint8)
        self._draw_face(image, (350, 350), 220, minimum_zone=1)
        geometry = self.detector.detect(image, self._single_config())
        gate = ImageQualityGate()

        self.assertEqual(QualityStatus.READY, gate.evaluate(image, geometry).status)

        blank = np.full((700, 700, 3), 118, dtype=np.uint8)
        not_found = self.detector.detect(blank, self._single_config())
        self.assertEqual(
            QualityStatus.TARGET_NOT_FOUND,
            gate.evaluate(blank, not_found).status,
        )

        wrong_count = self.detector.detect(
            image,
            self._triple_config(TripleLayout.VERTICAL),
        )
        self.assertEqual(
            QualityStatus.WRONG_FACE_COUNT,
            gate.evaluate(image, wrong_count).status,
        )

        good_face = geometry.faces[0]
        clipped_face = replace(good_face, coverage=0.75)
        clipped = replace(
            geometry,
            faces=(clipped_face,),
            geometry_confidence=0.9,
        )
        self.assertEqual(
            QualityStatus.TARGET_CLIPPED,
            gate.evaluate(image, clipped).status,
        )

        ellipse = good_face.ellipse
        perspective_face = replace(
            good_face,
            ellipse=replace(
                ellipse,
                diameter_y=ellipse.diameter_x * 0.4,
            ),
            coverage=1.0,
        )
        perspective = replace(
            geometry,
            faces=(perspective_face,),
            geometry_confidence=0.9,
        )
        self.assertEqual(
            QualityStatus.EXCESSIVE_PERSPECTIVE,
            gate.evaluate(image, perspective).status,
        )

        small_face = replace(
            good_face,
            ellipse=replace(ellipse, diameter_x=60.0, diameter_y=60.0),
            coverage=1.0,
        )
        too_small = replace(
            geometry,
            faces=(small_face,),
            geometry_confidence=0.9,
        )
        self.assertEqual(
            QualityStatus.TARGET_TOO_SMALL,
            gate.evaluate(image, too_small).status,
        )

        blurred = np.full_like(image, 128)
        self.assertEqual(
            QualityStatus.TOO_BLURRY,
            gate.evaluate(blurred, geometry).status,
        )

        exposure_gate = ImageQualityGate(
            QualityThresholds(minimum_laplacian_variance=0.0)
        )
        overexposed = np.full_like(image, 240)
        self.assertEqual(
            QualityStatus.BAD_EXPOSURE,
            exposure_gate.evaluate(overexposed, geometry).status,
        )

    def test_detector_rejects_invalid_image_contract(self):
        with self.assertRaisesRegex(ValueError, "H x W x 3"):
            self.detector.detect(np.zeros((64, 64), dtype=np.uint8), self._single_config())

    def _single_config(self):
        return TargetConfig(
            format=TargetFormat.SINGLE,
            minimum_scoring_zone=1,
            ten_ring_mode=TenRingMode.RECURVE,
            face_diameter_mm=1220,
        )

    def _triple_config(self, layout):
        return TargetConfig(
            format=TargetFormat.TRIPLE,
            minimum_scoring_zone=6,
            ten_ring_mode=TenRingMode.RECURVE,
            face_diameter_mm=400,
            triple_layout=layout,
        )

    def _draw_face(self, image, center, nominal_radius, minimum_zone):
        visible_radius = (11 - minimum_zone) * 0.1
        if visible_radius > 0.6:
            cv2.circle(
                image,
                center,
                int(round(nominal_radius * visible_radius)),
                (235, 235, 228),
                thickness=-1,
                lineType=cv2.LINE_AA,
            )
            cv2.circle(
                image,
                center,
                int(round(nominal_radius * 0.8)),
                (28, 28, 28),
                thickness=-1,
                lineType=cv2.LINE_AA,
            )
        cv2.circle(
            image,
            center,
            int(round(nominal_radius * min(0.6, visible_radius))),
            (28, 92, 210),
            thickness=-1,
            lineType=cv2.LINE_AA,
        )
        cv2.circle(
            image,
            center,
            int(round(nominal_radius * 0.4)),
            (220, 30, 45),
            thickness=-1,
            lineType=cv2.LINE_AA,
        )
        cv2.circle(
            image,
            center,
            int(round(nominal_radius * 0.2)),
            (250, 214, 20),
            thickness=-1,
            lineType=cv2.LINE_AA,
        )


if __name__ == "__main__":
    unittest.main()
