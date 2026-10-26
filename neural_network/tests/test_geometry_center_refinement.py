"""Image-only center refinement, independent of human feedback files."""
import unittest
import cv2
import numpy as np

from neural_network.archery_ml.geometry.calibration import project
from neural_network.archery_ml.geometry.detector import (
    _Candidate, _ellipse_from_cv, _initial_image_to_norm,
    _fit_projective_transform, _ring_fit_error, _stable_transform,
)


class CenterRefinementTest(unittest.TestCase):
    def fixture(self, skew):
        physical_to_image = np.array([[350.,30.,450.],[10.,310.,430.],[skew,-skew*.6,1.]])
        angles = np.linspace(0,2*np.pi,300,endpoint=False)
        rings = [(project(radius*np.column_stack((np.cos(angles),np.sin(angles))),physical_to_image).astype(np.float32),radius)
                 for radius in (.2,.4,.5)]
        contour = rings[-1][0].reshape(-1,1,2)
        candidate = _Candidate(contour,_ellipse_from_cv(cv2.fitEllipse(contour)),float(cv2.contourArea(contour)))
        initial = _initial_image_to_norm(candidate,.5,(np.array([1.,0.]),np.array([0.,1.])))
        return initial,rings,project([[0.,0.]],physical_to_image)[0]

    def test_subpixel_center_across_opposite_perspectives_and_round_trip(self):
        for skew in (-.04,-.02,.02,.04):
            initial,rings,center = self.fixture(skew)
            result = _fit_projective_transform(initial,rings)
            predicted = project([[0.,0.]],np.linalg.inv(result))[0]
            self.assertLess(np.linalg.norm(predicted-center),.05)
            self.assertLess(_ring_fit_error(result,rings),_ring_fit_error(initial,rings))
            np.testing.assert_array_equal(result,_stable_transform(initial,result,rings))
            probes = np.array([[100.,200.],[450.,430.],[800.,700.]])
            np.testing.assert_allclose(project(project(probes,result),np.linalg.inv(result)),probes,atol=1e-8)

    def test_partial_arcs_keep_safe_finite_transform_and_do_not_increase_residual(self):
        initial,rings,_ = self.fixture(.12)
        partial = [(points[:180],radius) for points,radius in rings]
        result = _fit_projective_transform(initial,partial)
        self.assertTrue(np.isfinite(result).all())
        self.assertLessEqual(_ring_fit_error(result,partial),_ring_fit_error(initial,partial))
        np.testing.assert_array_equal(result,_stable_transform(initial,result,partial))

    def test_empty_ring_evidence_cannot_invent_a_center(self):
        initial,rings,_ = self.fixture(.02)
        np.testing.assert_array_equal(initial,_fit_projective_transform(initial,[]))
