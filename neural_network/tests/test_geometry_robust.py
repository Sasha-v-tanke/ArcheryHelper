"""Independent truth, occlusions, ambiguity and fail-closed integration."""

import unittest
from dataclasses import replace
from unittest.mock import patch

import numpy as np

from neural_network.archery_ml.geometry.arc_extraction import balance_ring
from neural_network.archery_ml.geometry.detector import TargetGeometryDetector
from neural_network.archery_ml.geometry.geometry_math import project
from neural_network.archery_ml.geometry.normalize import TargetNormalizer
from neural_network.archery_ml.geometry.projective_solver import fit, safe_transform
from neural_network.archery_ml.geometry.quality import ImageQualityGate, QualityStatus
from neural_network.archery_ml.geometry.synthetic import render_scene
from neural_network.archery_ml.geometry.uncertainty import estimate
from neural_network.archery_ml.targets import TripleLayout


class RobustGeometryTest(unittest.TestCase):
    def fixture(self, fraction=0.0, corrupt=False):
        truth = np.array(
            [[350.0, 25.0, 450.0], [15.0, 300.0, 430.0], [0.04, -0.025, 1.0]]
        )
        affine = truth.copy()
        affine[2] = [0.0, 0.0, 1.0]
        initial = np.linalg.inv(affine)
        rings = []
        for radius, color in ((0.2, "yellow"), (0.4, "red"), (0.6, "blue")):
            angle = np.linspace(fraction * 2 * np.pi, 2 * np.pi, 360, endpoint=False)
            points = project(
                radius * np.column_stack((np.cos(angle), np.sin(angle))), truth
            )
            if corrupt and radius == 0.4:
                points[:30] += [8.0, -10.0]
            rings.append(
                balance_ring(points, np.ones(len(points)), radius, color, initial)
            )
        return initial, rings, truth

    def test_exact_projective_truth_without_central_pixels(self):
        initial, rings, truth = self.fixture()
        result = estimate(initial, rings)
        self.assertEqual("AUTO_READY", result.diagnostics["solver_status"])
        center = project([[0.0, 0.0]], np.linalg.inv(result.transform))[0]
        self.assertLess(np.linalg.norm(center - project([[0.0, 0.0]], truth)[0]), 0.1)
        self.assertTrue(safe_transform(initial, result.transform))
        probes = np.array([[0.2, 0.1], [-0.3, 0.1], [0.0, -0.4]])
        mapped = project(project(probes, truth), result.transform)
        np.testing.assert_allclose(
            np.linalg.norm(mapped, axis=1), np.linalg.norm(probes, axis=1), atol=0.001
        )

    def test_partial_arcs_and_outliers(self):
        initial, rings, truth = self.fixture(0.25, True)
        result = estimate(initial, rings)
        center = project([[0.0, 0.0]], np.linalg.inv(result.transform))[0]
        self.assertLess(np.linalg.norm(center - project([[0.0, 0.0]], truth)[0]), 3.0)
        self.assertGreater(result.diagnostics["angular_coverage"], 0.6)
        self.assertTrue(np.isfinite(result.diagnostics["held_sector_error"]))

    def test_one_ring_and_insufficient_sectors_never_ready(self):
        initial, rings, _ = self.fixture(0.7)
        for evidence in (rings[:1], rings):
            self.assertNotEqual(
                "AUTO_READY", estimate(initial, evidence).diagnostics["solver_status"]
            )

    def test_permuted_radii_and_optimizer_failure_never_ready(self):
        initial, rings, _ = self.fixture()
        wrong = [
            replace(
                r,
                radius=0.4 if r.radius == 0.2 else 0.2 if r.radius == 0.4 else r.radius,
            )
            for r in rings
        ]
        self.assertNotEqual(
            "AUTO_READY", estimate(initial, wrong).diagnostics["solver_status"]
        )
        with patch(
            "neural_network.archery_ml.geometry.uncertainty.fit",
            side_effect=np.linalg.LinAlgError("failure"),
        ):
            self.assertEqual(
                "REJECTED", estimate(initial, rings).diagnostics["solver_status"]
            )

    def test_singular_nonfinite_and_horizon_rejected(self):
        initial, rings, _ = self.fixture()
        for matrix in (
            np.zeros((3, 3)),
            np.full((3, 3), np.nan),
            np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [2.0, 0.0, 1.0]]),
        ):
            self.assertFalse(safe_transform(initial, matrix))
        bad = replace(rings[0], points=np.full_like(rings[0].points, np.inf))
        with self.assertRaises(ValueError):
            fit(initial, [bad])

    def test_safety_boundary_cannot_masquerade_as_stable_solution(self):
        _, rings, _ = self.fixture()
        poor_seed = np.array(
            [[1 / 350, 0.0, -450 / 350], [0.0, 1 / 300, -430 / 300], [0.0, 0.0, 1.0]]
        )
        with patch(
            "neural_network.archery_ml.geometry.uncertainty.hypotheses",
            return_value=[(poor_seed, poor_seed)],
        ):
            result = estimate(poor_seed, rings)
        self.assertNotEqual("AUTO_READY", result.diagnostics["solver_status"])
        self.assertIn(
            "extrapolation_limit_reached", result.diagnostics["rejection_reasons"]
        )

    def test_determinism_and_coordinate_scaling(self):
        initial, rings, _ = self.fixture(0.1, True)
        first = estimate(initial, rings)
        second = estimate(initial, rings)
        np.testing.assert_array_equal(first.transform, second.transform)
        scale = np.array([[2.0, 0.0, 100.0], [0.0, 2.0, -50.0], [0.0, 0.0, 1.0]])
        transformed = [replace(r, points=project(r.points, scale)) for r in rings]
        other = estimate(initial @ np.linalg.inv(scale), transformed)
        np.testing.assert_allclose(first.transform, other.transform @ scale, atol=1e-7)

    def test_rendered_damage_and_scale(self):
        detector = TargetGeometryDetector()
        gate = ImageQualityGate()
        for scenario in (
            "clean",
            "center_destroyed",
            "arrows",
            "damaged_ring",
            "lighting",
        ):
            for zone in (1, 5, 6):
                with self.subTest(scenario=scenario, zone=zone):
                    scene = render_scene(scenario, zone=zone, perspective=0.15)
                    geometry = detector.detect(scene.image, scene.config)
                    self.assertEqual(
                        QualityStatus.READY, gate.evaluate(scene.image, geometry).status
                    )
                    center = project(
                        [geometry.faces[0].center],
                        np.linalg.inv(scene.norm_to_image[0]),
                    )[0]
                    self.assertLess(np.linalg.norm(center), 0.015)
                    self.assertEqual(
                        "unanchored_image_or_layout_axes",
                        geometry.orientation_reference,
                    )

    def test_negative_clipped_and_uncertainty_cannot_normalize(self):
        for scenario in ("blank", "false_colors", "insufficient", "clipped"):
            scene = render_scene(scenario, perspective=0.15)
            geometry = TargetGeometryDetector().detect(scene.image, scene.config)
            self.assertNotEqual(
                QualityStatus.READY,
                ImageQualityGate().evaluate(scene.image, geometry).status,
            )
            with self.assertRaises(ValueError):
                TargetNormalizer().normalize(scene.image, geometry)
        scene = render_scene()
        geometry = TargetGeometryDetector().detect(scene.image, scene.config)
        for face in (
            replace(geometry.faces[0], solver_status="NEEDS_REVIEW"),
            replace(geometry.faces[0], center_uncertainty_norm=float("nan")),
            replace(geometry.faces[0], center_uncertainty_norm=0.5),
            replace(geometry.faces[0], angular_coverage=0.1),
            replace(geometry.faces[0], center=(1.0, 1.0)),
        ):
            altered = replace(geometry, faces=(face,))
            self.assertEqual(
                QualityStatus.GEOMETRY_UNCERTAIN,
                ImageQualityGate().evaluate(scene.image, altered).status,
            )

    def test_roi_preserves_full_frame_coordinates_and_indices(self):
        scene = render_scene(layout=TripleLayout.VERTICAL)
        detector = TargetGeometryDetector()
        full = detector.detect(scene.image, scene.config)
        cropped = detector.detect(scene.image, scene.config, roi=(200, 30, 610, 770))
        self.assertEqual(
            [f.face_index for f in full.faces], [f.face_index for f in cropped.faces]
        )
        for face in cropped.faces:
            np.testing.assert_allclose(
                project([face.center], np.asarray(face.homography))[0],
                [255.5, 255.5],
                atol=0.001,
            )
            np.testing.assert_allclose(
                project([[255.5, 255.5]], np.asarray(face.inverse_homography))[0],
                face.center,
                atol=0.001,
            )
        for roi in (
            (0, 0, 0, 30),
            (-1, 0, 30, 30),
            (0.0, 0, 30, 30),
            (0, 0, 9999, 9999),
        ):
            with self.assertRaises(ValueError):
                detector.detect(scene.image, scene.config, roi=roi)

    def test_foreign_groups_require_explicit_selection(self):
        scene = render_scene(layout=TripleLayout.VERTICAL)
        image = np.concatenate((scene.image, scene.image), axis=1)
        detector = TargetGeometryDetector()
        ambiguous = detector.detect(image, scene.config)
        self.assertFalse(ambiguous.layout_valid)
        self.assertNotEqual(
            QualityStatus.READY, ImageQualityGate().evaluate(image, ambiguous).status
        )
        selected = detector.detect(
            image, scene.config, candidate_ids=("f0", "f2", "f4")
        )
        self.assertTrue(selected.layout_valid)
        self.assertEqual(3, len(selected.faces))
        self.assertTrue(all(f.center[0] < 800 for f in selected.faces))
        with self.assertRaises(ValueError):
            detector.detect(image, scene.config, candidate_ids=("f0",) * 3)

    def test_external_orientation_requires_image_binding(self):
        from neural_network.archery_ml.geometry.benchmark import rgb_pixel_sha256
        from neural_network.archery_ml.geometry.orientation import (
            OrientationCalibration,
            apply_orientation,
            geometry_fingerprint,
        )

        scene = render_scene()
        geometry = TargetGeometryDetector().detect(scene.image, scene.config)
        calibration = OrientationCalibration(
            0.2,
            rgb_pixel_sha256(scene.image),
            "manual_calibration",
            "review-1",
            "reviewer",
            geometry_fingerprint(geometry),
        )
        calibrated = apply_orientation(scene.image, geometry, calibration)
        self.assertEqual("manual_calibration", calibrated.orientation_reference)
        self.assertEqual(geometry.faces[0].center, calibrated.faces[0].center)
        np.testing.assert_allclose(
            project(
                [geometry.faces[0].center], np.asarray(calibrated.faces[0].homography)
            )[0],
            [255.5, 255.5],
            atol=0.001,
        )
        altered = scene.image.copy()
        altered[0, 0] = 0
        with self.assertRaises(ValueError):
            apply_orientation(altered, geometry, calibration)

        with self.assertRaisesRegex(ValueError, "another image"):
            apply_orientation(
                scene.image.reshape((400, 1600, 3)), geometry, calibration
            )
        changed = replace(geometry, fit_error=geometry.fit_error + 1)
        with self.assertRaisesRegex(ValueError, "geometry revision"):
            apply_orientation(scene.image, changed, calibration)
        with self.assertRaisesRegex(ValueError, "already applied"):
            apply_orientation(scene.image, calibrated, calibration)

    def test_occlusion_severity_has_no_false_ready(self):
        for fraction in (0.1, 0.25, 0.4, 0.6):
            scene = render_scene("occlusion", occlusion=fraction, perspective=0.15)
            geometry = TargetGeometryDetector().detect(scene.image, scene.config)
            if (
                ImageQualityGate().evaluate(scene.image, geometry).status
                is QualityStatus.READY
            ):
                for face in geometry.faces:
                    center = project(
                        [face.center], np.linalg.inv(scene.norm_to_image[0])
                    )[0]
                    self.assertLess(np.linalg.norm(center), 0.015)

    def test_nan_image_and_repeated_seeded_scenes(self):
        scene = render_scene()
        with self.assertRaises(ValueError):
            TargetGeometryDetector().detect(
                np.full_like(scene.image, np.nan, dtype=float), scene.config
            )
        np.testing.assert_array_equal(
            render_scene("arrows", seed=7).image, render_scene("arrows", seed=7).image
        )

    def test_rotation_scale_and_sensor_noise_have_independent_truth(self):
        for angle, scale in ((-30.0, 0.6), (20.0, 1.2), (40.0, 1.0)):
            scene = render_scene(
                "sensor_noise", rotation_deg=angle, scale_factor=scale, perspective=0.12
            )
            result = TargetGeometryDetector().detect(scene.image, scene.config)
            self.assertEqual(
                QualityStatus.READY,
                ImageQualityGate().evaluate(scene.image, result).status,
            )
            center = project(
                [result.faces[0].center], np.linalg.inv(scene.norm_to_image[0])
            )[0]
            self.assertLess(np.linalg.norm(center), 0.015)

    def test_selected_geometry_restores_full_frame_evidence_coordinates(self):
        from neural_network.archery_ml.geometry.selected_geometry import (
            SelectedTargetGeometryPipeline,
        )

        scene = render_scene("arrows")
        image = np.full((900, 1000, 3), 118, np.uint8)
        image[50:850, 100:900] = scene.image
        candidate = {
            "id": "f0",
            "center": [500, 450],
            "diameter": 350,
            "bbox": [300, 250, 700, 650],
        }
        group = {
            "candidate_ids": ["f0"],
            "format": "SINGLE",
            "layout": None,
            "minimum_scoring_zone": 5,
        }
        adapter = SelectedTargetGeometryPipeline()
        detect = adapter.detector.detect
        captured = []

        def observe(*args, **kwargs):
            geometry = detect(*args, **kwargs)
            captured.append(geometry)
            return geometry

        with patch.object(adapter.detector, "detect", side_effect=observe):
            result, full = adapter._run(
                image, [candidate], group, target_config=scene.config
            )
        self.assertTrue(result["faces"])
        x0, y0 = result["roi"][:2]
        self.assertGreater(x0, 0)
        self.assertGreater(y0, 0)
        for local, restored in zip(captured[0].faces, result["faces"]):
            self.assertTrue(local.observed_arcs)
            for arc, full_arc in zip(local.observed_arcs, restored["observed_arcs"]):
                np.testing.assert_allclose(
                    full_arc["points"], np.asarray(arc["points"]) + [x0, y0]
                )
            if local.excluded_points:
                np.testing.assert_allclose(
                    restored["excluded_points"],
                    np.asarray(local.excluded_points) + [x0, y0],
                )
        for local, restored in zip(
            captured[0].candidate_proposals, full.candidate_proposals
        ):
            np.testing.assert_allclose(
                restored["center"], np.asarray(local["center"]) + [x0, y0]
            )
