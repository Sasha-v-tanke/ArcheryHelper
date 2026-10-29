"""Independent truth, occlusions, ambiguity and fail-closed integration."""
from dataclasses import replace
from unittest.mock import patch
import unittest

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
    def fixture(self, fraction=0., corrupt=False):
        truth=np.array([[350.,25.,450.],[15.,300.,430.],[.04,-.025,1.]])
        affine=truth.copy()
        affine[2]=[0.,0.,1.]
        initial=np.linalg.inv(affine)
        rings=[]
        for radius,color in ((.2,'yellow'),(.4,'red'),(.6,'blue')):
            angle=np.linspace(fraction*2*np.pi,2*np.pi,360,endpoint=False)
            points=project(radius*np.column_stack((np.cos(angle),np.sin(angle))),truth)
            if corrupt and radius==.4:
                points[:30]+=[8.,-10.]
            rings.append(balance_ring(points,np.ones(len(points)),radius,color,initial))
        return initial,rings,truth

    def test_exact_projective_truth_without_central_pixels(self):
        initial,rings,truth=self.fixture()
        result=estimate(initial,rings)
        self.assertEqual('AUTO_READY',result.diagnostics['solver_status'])
        center=project([[0.,0.]],np.linalg.inv(result.transform))[0]
        self.assertLess(np.linalg.norm(center-project([[0.,0.]],truth)[0]),.1)
        self.assertTrue(safe_transform(initial,result.transform))
        probes=np.array([[.2,.1],[-.3,.1],[0.,-.4]])
        mapped=project(project(probes,truth),result.transform)
        np.testing.assert_allclose(np.linalg.norm(mapped,axis=1),np.linalg.norm(probes,axis=1),atol=.001)

    def test_partial_arcs_and_outliers(self):
        initial,rings,truth=self.fixture(.25,True)
        result=estimate(initial,rings)
        center=project([[0.,0.]],np.linalg.inv(result.transform))[0]
        self.assertLess(np.linalg.norm(center-project([[0.,0.]],truth)[0]),3.)
        self.assertGreater(result.diagnostics['angular_coverage'],.6)
        self.assertTrue(np.isfinite(result.diagnostics['held_sector_error']))

    def test_one_ring_and_insufficient_sectors_never_ready(self):
        initial,rings,_=self.fixture(.7)
        for evidence in (rings[:1],rings):
            self.assertNotEqual('AUTO_READY',estimate(initial,evidence).diagnostics['solver_status'])

    def test_permuted_radii_and_optimizer_failure_never_ready(self):
        initial,rings,_=self.fixture()
        wrong=[replace(r,radius=.4 if r.radius==.2 else .2 if r.radius==.4 else r.radius) for r in rings]
        self.assertNotEqual('AUTO_READY',estimate(initial,wrong).diagnostics['solver_status'])
        with patch('neural_network.archery_ml.geometry.uncertainty.fit',side_effect=np.linalg.LinAlgError('failure')):
            self.assertEqual('REJECTED',estimate(initial,rings).diagnostics['solver_status'])

    def test_singular_nonfinite_and_horizon_rejected(self):
        initial,rings,_=self.fixture()
        for matrix in (np.zeros((3,3)),np.full((3,3),np.nan),np.array([[1.,0.,0.],[0.,1.,0.],[2.,0.,1.]])):
            self.assertFalse(safe_transform(initial,matrix))
        bad=replace(rings[0],points=np.full_like(rings[0].points,np.inf))
        with self.assertRaises(ValueError):
            fit(initial,[bad])

    def test_safety_boundary_cannot_masquerade_as_stable_solution(self):
        _,rings,_=self.fixture()
        poor_seed=np.array([[1/350,0.,-450/350],[0.,1/300,-430/300],[0.,0.,1.]])
        with patch('neural_network.archery_ml.geometry.uncertainty.hypotheses',return_value=[(poor_seed,poor_seed)]):
            result=estimate(poor_seed,rings)
        self.assertNotEqual('AUTO_READY',result.diagnostics['solver_status'])
        self.assertIn('extrapolation_limit_reached',result.diagnostics['rejection_reasons'])

    def test_determinism_and_coordinate_scaling(self):
        initial,rings,_=self.fixture(.1,True)
        first=estimate(initial,rings);second=estimate(initial,rings)
        np.testing.assert_array_equal(first.transform,second.transform)
        scale=np.array([[2.,0.,100.],[0.,2.,-50.],[0.,0.,1.]])
        transformed=[replace(r,points=project(r.points,scale)) for r in rings]
        other=estimate(initial@np.linalg.inv(scale),transformed)
        np.testing.assert_allclose(first.transform,other.transform@scale,atol=1e-7)

    def test_rendered_damage_and_scale(self):
        detector=TargetGeometryDetector();gate=ImageQualityGate()
        for scenario in ('clean','center_destroyed','arrows','damaged_ring','lighting'):
            for zone in (1,5,6):
                with self.subTest(scenario=scenario,zone=zone):
                    scene=render_scene(scenario,zone=zone,perspective=.15)
                    geometry=detector.detect(scene.image,scene.config)
                    self.assertEqual(QualityStatus.READY,gate.evaluate(scene.image,geometry).status)
                    center=project([geometry.faces[0].center],np.linalg.inv(scene.norm_to_image[0]))[0]
                    self.assertLess(np.linalg.norm(center),.015)
                    self.assertEqual('unanchored_image_or_layout_axes',geometry.orientation_reference)

    def test_negative_clipped_and_uncertainty_cannot_normalize(self):
        for scenario in ('blank','false_colors','insufficient','clipped'):
            scene=render_scene(scenario,perspective=.15)
            geometry=TargetGeometryDetector().detect(scene.image,scene.config)
            self.assertNotEqual(QualityStatus.READY,ImageQualityGate().evaluate(scene.image,geometry).status)
            with self.assertRaises(ValueError):TargetNormalizer().normalize(scene.image,geometry)
        scene=render_scene();geometry=TargetGeometryDetector().detect(scene.image,scene.config)
        for face in (replace(geometry.faces[0],solver_status='NEEDS_REVIEW'),
                     replace(geometry.faces[0],center_uncertainty_norm=float('nan')),
                     replace(geometry.faces[0],center_uncertainty_norm=.5),
                     replace(geometry.faces[0],angular_coverage=.1),
                     replace(geometry.faces[0],center=(1.,1.))):
            altered=replace(geometry,faces=(face,))
            self.assertEqual(QualityStatus.GEOMETRY_UNCERTAIN,ImageQualityGate().evaluate(scene.image,altered).status)

    def test_roi_preserves_full_frame_coordinates_and_indices(self):
        scene=render_scene(layout=TripleLayout.VERTICAL)
        detector=TargetGeometryDetector()
        full=detector.detect(scene.image,scene.config)
        cropped=detector.detect(scene.image,scene.config,roi=(200,30,610,770))
        self.assertEqual([f.face_index for f in full.faces],[f.face_index for f in cropped.faces])
        for face in cropped.faces:
            np.testing.assert_allclose(project([face.center],np.asarray(face.homography))[0],[255.5,255.5],atol=.001)
            np.testing.assert_allclose(project([[255.5,255.5]],np.asarray(face.inverse_homography))[0],face.center,atol=.001)
        for roi in ((0,0,0,30),(-1,0,30,30),(0.,0,30,30),(0,0,9999,9999)):
            with self.assertRaises(ValueError):detector.detect(scene.image,scene.config,roi=roi)

    def test_foreign_groups_require_explicit_selection(self):
        scene=render_scene(layout=TripleLayout.VERTICAL)
        image=np.concatenate((scene.image,scene.image),axis=1)
        detector=TargetGeometryDetector()
        ambiguous=detector.detect(image,scene.config)
        self.assertFalse(ambiguous.layout_valid)
        self.assertNotEqual(QualityStatus.READY,ImageQualityGate().evaluate(image,ambiguous).status)
        selected=detector.detect(image,scene.config,candidate_ids=('f0','f2','f4'))
        self.assertTrue(selected.layout_valid)
        self.assertEqual(3,len(selected.faces))
        self.assertTrue(all(f.center[0]<800 for f in selected.faces))
        with self.assertRaises(ValueError):detector.detect(image,scene.config,candidate_ids=('f0',)*3)

    def test_external_orientation_requires_image_binding(self):
        import hashlib
        from neural_network.archery_ml.geometry.orientation import OrientationCalibration,apply_orientation
        scene=render_scene()
        geometry=TargetGeometryDetector().detect(scene.image,scene.config)
        calibration=OrientationCalibration(.2,hashlib.sha256(scene.image.tobytes()).hexdigest(),
                                           'manual_calibration','review-1','reviewer')
        calibrated=apply_orientation(scene.image,geometry,calibration)
        self.assertEqual('manual_calibration',calibrated.orientation_reference)
        self.assertEqual(geometry.faces[0].center,calibrated.faces[0].center)
        np.testing.assert_allclose(project([geometry.faces[0].center],np.asarray(calibrated.faces[0].homography))[0],
                                   [255.5,255.5],atol=.001)
        altered=scene.image.copy();altered[0,0]=0
        with self.assertRaises(ValueError):apply_orientation(altered,geometry,calibration)

    def test_occlusion_severity_has_no_false_ready(self):
        for fraction in (.1,.25,.4,.6):
            scene=render_scene('occlusion',occlusion=fraction,perspective=.15)
            geometry=TargetGeometryDetector().detect(scene.image,scene.config)
            if ImageQualityGate().evaluate(scene.image,geometry).status is QualityStatus.READY:
                for face in geometry.faces:
                    center=project([face.center],np.linalg.inv(scene.norm_to_image[0]))[0]
                    self.assertLess(np.linalg.norm(center),.015)

    def test_nan_image_and_repeated_seeded_scenes(self):
        scene=render_scene()
        with self.assertRaises(ValueError):TargetGeometryDetector().detect(np.full_like(scene.image,np.nan,dtype=float),scene.config)
        np.testing.assert_array_equal(render_scene('arrows',seed=7).image,render_scene('arrows',seed=7).image)
