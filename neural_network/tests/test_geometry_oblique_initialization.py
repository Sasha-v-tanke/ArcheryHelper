import unittest

import numpy as np

from neural_network.archery_ml.geometry.calibration import project
from neural_network.archery_ml.geometry.detector import _Candidate,EllipseFit,_initial_image_to_norm


class ObliqueInitializationTest(unittest.TestCase):
    def test_oblique_ellipse_initialization_removes_shear_without_defining_up(self):
        angle=np.deg2rad(43)
        rotation=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
        theta=np.linspace(0,2*np.pi,100,endpoint=False)
        points=np.column_stack((90*np.cos(theta),50*np.sin(theta)))@rotation.T+[130,170]
        candidate=_Candidate(points.astype(np.float32).reshape(-1,1,2),EllipseFit(130,170,180,100,43),100.)
        h=_initial_image_to_norm(candidate,.5,(np.array([1.,0]),np.array([0.,1.])))
        np.testing.assert_allclose(np.linalg.norm(project(points,h),axis=1),.5,atol=1e-7)
        np.testing.assert_allclose(project([[130,170]],h),[[0,0]],atol=1e-7)
        # A circular face has no identifiable principal direction.
        for fitted_angle in (0,43,110):
            circle=_Candidate(candidate.contour,EllipseFit(130,170,180,180,fitted_angle),100.)
            h=_initial_image_to_norm(circle,.5,(np.array([1.,0]),np.array([0.,1.])))
            np.testing.assert_allclose(project([[130,80]],h),[[0,-.5]],atol=1e-7)
