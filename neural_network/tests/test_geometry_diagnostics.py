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


class GeometryDiagnosticsTest(unittest.TestCase):
    def setUp(self):
        self.detector = TargetGeometryDetector()
        self.normalizer = TargetNormalizer()
        self.gate = ImageQualityGate()

    def test_single_face_diagnostic_metrics_and_round_trip(self):
        scenarios = (
            self._scenario(
                image_size=(720, 720),
                centers=((360, 360),),
                radius=220,
                output_size=(720, 720),
                dst_corners=((0, 0), (719, 0), (719, 719), (0, 719)),
            ),
            self._scenario(
                image_size=(720, 720),
                centers=((360, 360),),
                radius=220,
                output_size=(720, 720),
                dst_corners=((45, 68), (673, 35), (691, 668), (29, 639)),
            ),
            self._scenario(
                image_size=(760, 620),
                centers=((245, 310),),
                radius=210,
                output_size=(640, 800),
                dst_corners=((52, 28), (598, 64), (613, 746), (31, 711)),
            ),
            self._scenario(
                image_size=(680, 680),
                centers=((265, 265),),
                radius=220,
                output_size=(680, 680),
                dst_corners=((0, 0), (679, 0), (679, 679), (0, 679)),
            ),
        )

        for photo, source_to_photo, centers, radius in scenarios:
            with self.subTest(shape=photo.shape):
                result = self.detector.detect(photo, self._single_config(1, TenRingMode.RECURVE))
                self.assertEqual(1, len(result.faces))
                self.assertEqual(QualityStatus.READY, self.gate.evaluate(photo, result).status)
                metrics = self._coordinate_metrics(
                    result.faces[0],
                    source_to_photo,
                    centers[0],
                    radius,
                    self._sample_points(),
                )
                self.assertLess(metrics["mean"], 0.035)
                self.assertLess(metrics["median"], 0.03)
                self.assertLess(metrics["p95"], 0.07)
                self.assertLess(metrics["max"], 0.09)
                self.assertLess(
                    self._round_trip_pixel_error(result.faces[0], source_to_photo, centers[0], radius),
                    1e-3,
                )
                canonical = self.normalizer.normalize(photo, result)
                self.assertEqual((512, 512, 3), canonical[0].image.shape)

    def test_vertical_and_triangular_triples_have_stable_face_indices_and_metrics(self):
        vertical = self._scenario(
            image_size=(900, 700),
            centers=((350, 180), (350, 450), (350, 720)),
            radius=115,
            output_size=(790, 960),
            dst_corners=((115, 0), (789, 143), (600, 959), (0, 817)),
            minimum_zone=6,
        )
        triangular = self._scenario(
            image_size=(700, 760),
            centers=((380, 160), (245, 445), (515, 445)),
            radius=120,
            output_size=(780, 740),
            dst_corners=((51, 38), (718, 16), (754, 708), (19, 725)),
            minimum_zone=6,
        )

        for layout, scenario in (
            (TripleLayout.VERTICAL, vertical),
            (TripleLayout.TRIANGULAR, triangular),
        ):
            photo, source_to_photo, centers, radius = scenario
            with self.subTest(layout=layout):
                result = self.detector.detect(photo, self._triple_config(layout, 6))
                self.assertEqual(3, len(result.faces))
                self.assertTrue(result.layout_valid)
                self.assertEqual([0, 1, 2], [face.face_index for face in result.faces])
                for face, center in zip(result.faces, centers):
                    metrics = self._coordinate_metrics(
                        face,
                        source_to_photo,
                        center,
                        radius,
                        self._sample_points(),
                    )
                    self.assertLess(metrics["mean"], 0.08)
                    self.assertLess(metrics["p95"], 0.15)
                    self.assertLess(metrics["max"], 0.18)

    def test_quality_status_precedence_and_contract_edges(self):
        image, _, _, _ = self._scenario(
            image_size=(700, 700),
            centers=((350, 350),),
            radius=220,
            output_size=(700, 700),
            dst_corners=((0, 0), (699, 0), (699, 699), (0, 699)),
        )
        geometry = self.detector.detect(image, self._single_config(1, TenRingMode.RECURVE))
        face = geometry.faces[0]

        clipped_and_perspective = replace(
            geometry,
            faces=(
                replace(
                    face,
                    coverage=0.65,
                    ellipse=replace(face.ellipse, diameter_y=face.ellipse.diameter_x * 0.35),
                ),
            ),
            geometry_confidence=0.9,
        )
        self.assertEqual(
            QualityStatus.TARGET_CLIPPED,
            self.gate.evaluate(image, clipped_and_perspective).status,
        )

        small_and_blurry = replace(
            geometry,
            faces=(
                replace(
                    face,
                    ellipse=replace(face.ellipse, diameter_x=58.0, diameter_y=58.0),
                    coverage=1.0,
                ),
            ),
            geometry_confidence=0.9,
        )
        self.assertEqual(
            QualityStatus.TARGET_TOO_SMALL,
            self.gate.evaluate(np.full_like(image, 128), small_and_blurry).status,
        )

        with self.assertRaises(ValueError):
            self.detector.detect(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY), self._single_config(1))
        with self.assertRaises(ValueError):
            self.detector.detect(np.dstack((image, image[:, :, 0])), self._single_config(1))
        with self.assertRaises(ValueError):
            self.detector.detect(np.zeros((12, 12, 3), dtype=np.uint8), self._single_config(1))

        float_result = self.detector.detect(image.astype(np.float32), self._single_config(1))
        self.assertEqual(1, len(float_result.faces))

    def test_detection_failures_and_perturbations_are_finite(self):
        blank = np.full((640, 640, 3), 118, dtype=np.uint8)
        not_found = self.detector.detect(blank, self._single_config(1))
        self.assertEqual(QualityStatus.TARGET_NOT_FOUND, self.gate.evaluate(blank, not_found).status)

        one_face = self._triple_photo(((320, 320),))
        two_faces = self._triple_photo(((320, 200), (320, 440)))
        four_faces = self._triple_photo(((200, 200), (440, 200), (200, 440), (440, 440)))
        for photo in (one_face, two_faces, four_faces):
            result = self.detector.detect(photo, self._triple_config(TripleLayout.VERTICAL, 6))
            self.assertEqual(QualityStatus.WRONG_FACE_COUNT, self.gate.evaluate(photo, result).status)

        wrong_layout = self.detector.detect(
            self._triple_photo(((320, 140), (320, 320), (320, 500))),
            self._triple_config(TripleLayout.TRIANGULAR, 6),
        )
        self.assertEqual(QualityStatus.TARGET_NOT_FOUND, self.gate.evaluate(one_face, wrong_layout).status)

        base = self._triple_photo(((320, 320),), minimum_zone=1)
        for perturbed in (
            self._brightness(base, 28),
            self._brightness(base, -24),
            self._contrast(base, 1.18),
            self._gaussian_noise(base, 4.0),
            self._jpeg_round_trip(base, 72),
            self._saturation(base, 0.82),
        ):
            result = self.detector.detect(perturbed, self._single_config(1))
            status = self.gate.evaluate(perturbed, result).status
            self.assertIn(status, set(QualityStatus))
            if status is QualityStatus.READY:
                self.assertEqual(1, len(self.normalizer.normalize(perturbed, result)))

        weak_yellow = self._triple_photo(((320, 320),), minimum_zone=1, weak_colors={"yellow"})
        weak_result = self.detector.detect(weak_yellow, self._single_config(1))
        weak_status = self.gate.evaluate(weak_yellow, weak_result).status
        self.assertIn(weak_status, set(QualityStatus))

    def test_supported_scoring_zone_configs_share_physical_geometry(self):
        for minimum_zone in (1, 5, 6):
            image, _, _, _ = self._scenario(
                image_size=(700, 700),
                centers=((350, 350),),
                radius=220,
                output_size=(700, 700),
                dst_corners=((0, 0), (699, 0), (699, 699), (0, 699)),
                minimum_zone=minimum_zone,
            )
            recurve = self.detector.detect(
                image,
                self._single_config(minimum_zone, TenRingMode.RECURVE),
            )
            compound = self.detector.detect(
                image,
                self._single_config(minimum_zone, TenRingMode.COMPOUND),
            )
            self.assertEqual(1, len(recurve.faces))
            self.assertEqual(1, len(compound.faces))
            self.assertLess(
                np.linalg.norm(
                    np.asarray(recurve.faces[0].homography)
                    - np.asarray(compound.faces[0].homography)
                ),
                1e-9,
            )

    def _scenario(
        self,
        image_size,
        centers,
        radius,
        output_size,
        dst_corners,
        minimum_zone=1,
    ):
        height, width = image_size
        source = np.full((height, width, 3), 118, dtype=np.uint8)
        for center in centers:
            self._draw_face(source, center, radius, minimum_zone)
        source_corners = np.float32(
            [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]]
        )
        source_to_photo = cv2.getPerspectiveTransform(source_corners, np.float32(dst_corners))
        output_width, output_height = output_size
        photo = cv2.warpPerspective(
            source,
            source_to_photo,
            (output_width, output_height),
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(118, 118, 118),
        )
        return photo, source_to_photo, centers, radius

    def _triple_photo(self, centers, minimum_zone=6, weak_colors=frozenset()):
        image = np.full((640, 640, 3), 118, dtype=np.uint8)
        for center in centers:
            self._draw_face(image, center, 95, minimum_zone, weak_colors=weak_colors)
        return image

    def _draw_face(self, image, center, nominal_radius, minimum_zone, weak_colors=frozenset()):
        visible_radius = (11 - minimum_zone) * 0.1
        if visible_radius > 0.6:
            cv2.circle(image, center, int(round(nominal_radius * visible_radius)), (235, 235, 228), -1)
            cv2.circle(image, center, int(round(nominal_radius * 0.8)), (28, 28, 28), -1)
        colors = {
            "blue": (28, 92, 210),
            "red": (220, 30, 45),
            "yellow": (250, 214, 20),
        }
        for color in weak_colors:
            colors[color] = tuple(int(0.55 * value + 0.45 * 118) for value in colors[color])
        cv2.circle(
            image,
            center,
            int(round(nominal_radius * min(0.6, visible_radius))),
            colors["blue"],
            -1,
            lineType=cv2.LINE_AA,
        )
        cv2.circle(image, center, int(round(nominal_radius * 0.4)), colors["red"], -1, lineType=cv2.LINE_AA)
        cv2.circle(image, center, int(round(nominal_radius * 0.2)), colors["yellow"], -1, lineType=cv2.LINE_AA)

    def _sample_points(self):
        points = [(0.0, 0.0)]
        for radius in (0.2, 0.4, 0.6, 0.8):
            for angle_deg in range(0, 360, 45):
                angle = math.radians(angle_deg)
                points.append((radius * math.cos(angle), radius * math.sin(angle)))
        return points

    def _coordinate_metrics(self, face, source_to_photo, source_center, radius, normalized_points):
        errors = []
        transform = np.asarray(face.homography, dtype=np.float64)
        for x_norm, y_norm in normalized_points:
            source_point = np.array(
                [[[source_center[0] + x_norm * radius, source_center[1] + y_norm * radius]]],
                dtype=np.float32,
            )
            photo_point = cv2.perspectiveTransform(source_point, source_to_photo)
            canonical_point = cv2.perspectiveTransform(photo_point, transform).reshape(2)
            recovered = np.array(
                [
                    (canonical_point[0] - CANONICAL_CENTER_PX) / CANONICAL_RADIUS_PX,
                    (canonical_point[1] - CANONICAL_CENTER_PX) / CANONICAL_RADIUS_PX,
                ]
            )
            errors.append(float(np.linalg.norm(recovered - np.array([x_norm, y_norm]))))
        values = np.asarray(errors, dtype=np.float64)
        return {
            "mean": float(values.mean()),
            "median": float(np.median(values)),
            "p95": float(np.percentile(values, 95)),
            "max": float(values.max()),
        }

    def _round_trip_pixel_error(self, face, source_to_photo, source_center, radius):
        original_to_canonical = np.asarray(face.homography, dtype=np.float64)
        canonical_to_original = np.asarray(face.inverse_homography, dtype=np.float64)
        errors = []
        for x_norm, y_norm in self._sample_points():
            source_point = np.array(
                [[[source_center[0] + x_norm * radius, source_center[1] + y_norm * radius]]],
                dtype=np.float32,
            )
            original = cv2.perspectiveTransform(source_point, source_to_photo)
            canonical = cv2.perspectiveTransform(original, original_to_canonical)
            recovered = cv2.perspectiveTransform(canonical, canonical_to_original)
            errors.append(float(np.linalg.norm(recovered.reshape(2) - original.reshape(2))))
        return max(errors)

    def _single_config(self, minimum_zone, ten_mode=TenRingMode.RECURVE):
        return TargetConfig(
            format=TargetFormat.SINGLE,
            minimum_scoring_zone=minimum_zone,
            ten_ring_mode=ten_mode,
            face_diameter_mm=1220,
        )

    def _triple_config(self, layout, minimum_zone):
        return TargetConfig(
            format=TargetFormat.TRIPLE,
            minimum_scoring_zone=minimum_zone,
            ten_ring_mode=TenRingMode.RECURVE,
            face_diameter_mm=400,
            triple_layout=layout,
        )

    def _brightness(self, image, delta):
        return np.clip(image.astype(np.int16) + delta, 0, 255).astype(np.uint8)

    def _contrast(self, image, factor):
        return np.clip((image.astype(np.float32) - 128.0) * factor + 128.0, 0, 255).astype(np.uint8)

    def _gaussian_noise(self, image, sigma):
        rng = np.random.default_rng(42)
        noisy = image.astype(np.float32) + rng.normal(0.0, sigma, image.shape)
        return np.clip(noisy, 0, 255).astype(np.uint8)

    def _jpeg_round_trip(self, image, quality):
        ok, encoded = cv2.imencode(
            ".jpg",
            cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
            [int(cv2.IMWRITE_JPEG_QUALITY), quality],
        )
        self.assertTrue(ok)
        decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        return cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)

    def _saturation(self, image, factor):
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * factor, 0, 255)
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)


if __name__ == "__main__":
    unittest.main()
