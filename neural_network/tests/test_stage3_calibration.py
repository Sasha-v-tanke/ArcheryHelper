"""Synthetic board tests are calibration regression evidence, never real holdout."""
import copy
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
from PIL import Image

from neural_network.archery_ml.geometry.calibration import BoardTemplate, BoardCalibration, TargetOrientationCalibrator, project
from neural_network.archery_ml.geometry.calibration_tools import render_board, config_for, overlay_axes
from neural_network.archery_ml.geometry.detector import EllipseFit, FaceGeometry, GeometryResult, TargetGeometryDetector
from neural_network.archery_ml.geometry.quality import ImageQualityGate, QualityStatus
from neural_network.archery_ml.geometry.normalize import TargetNormalizer
from neural_network.archery_ml.geometry.target_templates import CANONICAL_CENTER_PX as C, CANONICAL_RADIUS_PX as R
from neural_network.archery_ml.geometry.evaluate import GeometryEvaluator
from neural_network.archery_ml.data.view import DatasetView
from neural_network.archery_ml.contracts import DatasetSample, ImageGeometryAnnotation, save_manifest
from neural_network.archery_ml.geometry.cli import main as geometry_main
from neural_network.archery_ml.data.importers.mobile_geometry import import_mobile_geometry
from neural_network.archery_ml.data.registry import DatasetRegistry
from dataset_tools.mobile.make_example import create


def fixture(name="single", angle=0, scale=1., perspective=False):
    template = BoardTemplate.load(Path(f"dataset_tools/boards/{name}-v1.json"))
    p = template.payload
    image = render_board(template, config_for(template))
    radians = np.deg2rad(angle)
    rotate = np.array([[np.cos(radians), -np.sin(radians),0], [np.sin(radians),np.cos(radians),0], [0,0,1.]])
    transform = np.array([[scale,0,550],[0,scale,550],[0,0,1.]]) @ rotate @ np.array([[1,0,-300],[0,1,-400],[0,0,1.]])
    if perspective:
        transform = transform @ np.array([[1,.02,0],[.01,1,0],[.00008,.00004,1.]])
    image = cv2.warpPerspective(image, transform, (1100,1100), borderValue=(240,240,240))
    faces = []
    pixels = np.array([[R,0,C],[0,R,C],[0,0,1.]])
    for face in p["faces"]:
        x,y = face["center_mm"]
        radius = face["radius_mm"]
        local = np.array([[1/radius,0,-x/radius],[0,1/radius,-y/radius],[0,0,1.]])
        h = pixels @ rotate.T @ local @ np.linalg.inv(transform)
        angles = np.linspace(0,2*np.pi,100,endpoint=False)
        boundary = project(np.column_stack((x+radius*np.cos(angles),y+radius*np.sin(angles))),transform)
        (cx,cy),(dx,dy),a = cv2.fitEllipse(boundary.astype(np.float32))
        ellipse = EllipseFit(cx,cy,dx,dy,a)
        center = tuple(project([[x,y]],transform)[0])
        bounds = (*boundary.min(axis=0), *boundary.max(axis=0))
        # Deliberately give provisional indices in reverse physical order.
        faces.append(FaceGeometry(len(p["faces"])-1-face["face_index"],center,ellipse,tuple(map(tuple,h)),
            tuple(map(tuple,np.linalg.inv(h))),.001,1.,bounds,1.,3))
    geometry = GeometryResult(tuple(faces),1.,.001,None,len(faces),True,1100,1100)
    return template,image,geometry,transform


class OrientationCalibrationTest(unittest.TestCase):
    def test_rotation_projective_scale_and_all_layouts_preserve_physical_axes(self):
        for layout in ("single", "vertical", "triangular"):
            for angle in (0,37,90,180,270):
                for scale,perspective in ((1.,False),(.95,True)):
                    with self.subTest(layout=layout,angle=angle,scale=scale):
                        board,image,geometry,transform = fixture(layout,angle,scale,perspective)
                        self.assertEqual(QualityStatus.READY,ImageQualityGate().evaluate(image,geometry).status)
                        calibrator = TargetOrientationCalibrator(board)
                        observed = calibrator.observe(image)
                        self.assertEqual(observed.to_dict(), BoardCalibration.from_dict(json.loads(json.dumps(observed.to_dict()))).to_dict())
                        result = calibrator.calibrate(image,geometry,observed)
                        self.assertEqual("physical_marker",result.geometry.orientation_reference)
                        for face,physical in zip(result.geometry.faces,board.payload["faces"]):
                            expected = np.array([[0,0],[.3,-.2],[-.2,.3]])
                            points = expected * physical["radius_mm"] + physical["center_mm"]
                            predicted = (project(project(points,transform),face.homography)-C)/R
                            np.testing.assert_allclose(expected,predicted,atol=.012)
                            np.testing.assert_allclose(project(project(points,transform),np.asarray(face.inverse_homography) @ np.asarray(face.homography)),project(points,transform),atol=1e-6)
                        crops = TargetNormalizer().normalize(image,result.geometry)
                        self.assertTrue(all(c.orientation_reference == "physical_marker" for c in crops))

    def test_occluded_marker_allowed_but_two_missing_fail(self):
        board,image,geometry,_ = fixture()
        # Board top-left marker projects to (270,170)..(320,220).
        image[160:230,260:330] = 255
        calibrator = TargetOrientationCalibrator(board)
        self.assertEqual(3,len(calibrator.observe(image).marker_correspondences))
        calibrator.calibrate(image,geometry,calibrator.observe(image))
        image[160:230,770:840] = 255
        with self.assertRaisesRegex(ValueError,"insufficient markers"):
            calibrator.calibrate(image,geometry,calibrator.observe(image))

    def test_mirror_and_wrong_marker_reject(self):
        board,image,geometry,_ = fixture()
        calibrator = TargetOrientationCalibrator(board)
        mirrored = cv2.flip(image,1)
        with self.assertRaises(ValueError):
            calibrator.calibrate(mirrored,geometry,calibrator.observe(mirrored))
        changed = copy.deepcopy(board.payload)
        changed["markers"][0]["id"] = 49
        wrong = TargetOrientationCalibrator(BoardTemplate(changed))
        with self.assertRaisesRegex(ValueError,"unknown"):
            wrong.calibrate(image,geometry,wrong.observe(image))

    def test_record_replay_tampering_revision_and_face_indices_reject(self):
        board,image,geometry,_ = fixture("vertical")
        calibrator = TargetOrientationCalibrator(board)
        observed = calibrator.observe(image)
        for changed in (replace(observed,image_rgb_sha256="wrong"), replace(observed,board_template_sha256="wrong"),
                        replace(observed,calibration_revision="unknown"), replace(observed,marker_correspondences=())):
            with self.assertRaises(ValueError):
                calibrator.calibrate(image,geometry,changed)
        invalid = replace(geometry,faces=tuple(replace(f,face_index=0) for f in geometry.faces))
        with self.assertRaisesRegex(ValueError,"face_index"):
            calibrator.calibrate(image,invalid,observed)

    def test_calibration_preserves_detector_center_scale_and_fit_errors(self):
        board,image,geometry,transform = fixture()
        face = geometry.faces[0]
        n = np.array([[1/R,0,-C/R],[0,1/R,-C/R],[0,0,1.]])
        perturb = np.array([[1.1,0,.1],[0,1.1,0],[0,0,1.]])
        h = np.linalg.inv(n) @ perturb @ n @ np.asarray(face.homography)
        changed = replace(face,homography=tuple(map(tuple,h)),inverse_homography=tuple(map(tuple,np.linalg.inv(h))),fit_error=.02)
        calibrator = TargetOrientationCalibrator(board)
        result = calibrator.calibrate(image,replace(geometry,faces=(changed,)),calibrator.observe(image))
        physical_center = project([board.payload["faces"][0]["center_mm"]],transform)
        error = np.linalg.norm((project(physical_center,result.geometry.faces[0].homography)-C)/R)
        self.assertAlmostEqual(.1,error,places=5)
        self.assertEqual(.02,result.geometry.faces[0].fit_error)

    def test_singular_geometry_and_degenerate_correspondences_reject(self):
        board,image,geometry,_ = fixture()
        calibrator = TargetOrientationCalibrator(board)
        invalid = replace(geometry,faces=(replace(geometry.faces[0],homography=((0,0,0),)*3),))
        with self.assertRaisesRegex(ValueError,"geometry READY"):
            calibrator.calibrate(image,invalid,calibrator.observe(image))
        observed = calibrator.observe(image)
        points = tuple({"marker_id":i,"image_corners_px":[[j,0] for j in range(4)]} for i in range(3))
        bad = replace(observed,marker_correspondences=points)
        with patch.object(calibrator,"observe",return_value=bad), self.assertRaises(ValueError):
            calibrator.calibrate(image,geometry,bad)

    def test_detector_integration_and_axis_overlay(self):
        for layout in ("single","vertical","triangular"):
            for angle in (0,37,90,180,270):
                with self.subTest(layout=layout,angle=angle):
                    board,image,_,_ = fixture(layout,angle)
                    geometry = TargetGeometryDetector().detect(image,config_for(board))
                    calibrator = TargetOrientationCalibrator(board)
                    result = calibrator.calibrate(image,geometry,calibrator.observe(image))
                    self.assertEqual(image.shape,overlay_axes(image,result.geometry).shape)

    def test_evaluator_uses_image_only_calibration_and_keeps_geometry_ready_without_markers(self):
        board,image,geometry,transform = fixture("single",37)
        point = project([[330,380]],transform)[0] / [1100,1100]
        annotations = (ImageGeometryAnnotation("landmark",(tuple(point),),face_index=0,
                       canonical_points=((.3,-.2),),annotation_quality="reviewed",uncertainty_norm=.002),)
        class Detector:
            def detect(self,image,config):
                return geometry
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "image.png"
            Image.fromarray(image).save(path)
            sample = DatasetSample("fixture",path,None,geometry_annotations=annotations)
            calibrator = TargetOrientationCalibrator(board)
            evaluator = GeometryEvaluator(detector=Detector(),orientation_calibrator=calibrator)
            report = evaluator.evaluate(DatasetView.from_samples([sample]),config_for(board)).to_dict()
            self.assertLess(report["point_error_2d"]["max"],.012)
            self.assertEqual("ANCHORED",report["sample_results"][0]["calibration"]["orientation_status"])
            bad_gt = replace(sample,geometry_annotations=(replace(annotations[0],canonical_points=((-1.,-1.),)),))
            bad = evaluator.evaluate(DatasetView.from_samples([bad_gt]),config_for(board)).to_dict()
            self.assertGreater(bad["point_error_2d"]["max"],1.)
            # Lose markers while preserving the target and its valid geometry.
            gray = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
            image[(gray < 10)] = 255  # marker glyphs are black; target black rings use value 20.
            Image.fromarray(image).save(path)
            result = evaluator.evaluate(DatasetView.from_samples([sample]),config_for(board)).to_dict()["sample_results"][0]
            self.assertEqual("READY",result["status"])
            self.assertEqual("UNANCHORED",result["calibration"]["orientation_status"])

    def test_mixed_manifest_cli_selects_configuration_and_repeats_deterministically(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "example"
            create(root)
            registry = DatasetRegistry.load(root / "sources.json")
            samples = import_mobile_geometry(root,registry.sources[0])
            manifest = root / "manifest.json"
            save_manifest(manifest,samples)
            for layout in ("single","vertical","triangular"):
                output = root / f"{layout}.json"
                args = ["--manifest",str(manifest),"--split","mobile_dev","--face-diameter-mm","200",
                        "--board-template",f"dataset_tools/boards/{layout}-v1.json","--output",str(output)]
                if layout != "single":
                    args += ["--format","TRIPLE","--triple-layout",layout.upper()]
                with patch("builtins.print"):
                    self.assertEqual(0,geometry_main(args))
                    before = output.read_bytes()
                    self.assertEqual(0,geometry_main(args))
                self.assertEqual(before,output.read_bytes())
                self.assertEqual(1,json.loads(before)["samples"])


if __name__ == "__main__":
    unittest.main()
