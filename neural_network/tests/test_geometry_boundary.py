import unittest
import io

import cv2
import numpy as np
from PIL import Image

from neural_network.archery_ml.geometry.boundary import canonical_display


class BoundaryTest(unittest.TestCase):
    def test_small_print_boundary_bulge_is_not_trimmed_to_fitted_ellipse(self):
        image=np.full((512,512,3),245,np.uint8)
        theta=np.linspace(-np.pi,np.pi,720,endpoint=False)
        radii=128+2.5*np.exp(-(theta/.1)**2)
        points=np.rint(255.5+np.column_stack((np.cos(theta),np.sin(theta)))*radii[:,None]).astype(np.int32)
        cv2.fillPoly(image,[points],(20,150,230))
        display,info=canonical_display(image,6)
        self.assertEqual("OBSERVED_BOUNDARY",info["status"])
        np.testing.assert_array_equal(display[256,385],image[256,385])

    def test_mask_repeats_after_png_roundtrip_and_other_images(self):
        image=self.target()
        cv2.line(image,(90,230),(240,260),(20,20,20),7)
        first,info=canonical_display(image,6)
        encoded=io.BytesIO();Image.fromarray(image).save(encoded,format="PNG");encoded.seek(0)
        decoded=np.array(Image.open(encoded))
        canonical_display(np.roll(image,9,axis=0),6)
        second,repeated=canonical_display(decoded,6)
        self.assertEqual(info,repeated)
        np.testing.assert_array_equal(first,second)

    def target(self):
        image=np.full((512,512,3),245,np.uint8)
        for radius,color in [(128,(20,150,230)),(102,(230,40,40)),(51,(245,220,30))]:
            cv2.circle(image,(261,252),radius,color,-1)
        return image

    def test_shifted_edge_removes_paper_and_preserves_opposite_colored_edge(self):
        image=self.target();original=image.copy()
        display,info=canonical_display(image,6)
        self.assertEqual("OBSERVED_BOUNDARY",info["status"])
        # Nominal circle keeps this paper pixel and clips the blue pixel opposite.
        np.testing.assert_array_equal(display[252,130],[128]*3)
        np.testing.assert_array_equal(display[252,387],image[252,387])
        np.testing.assert_array_equal(image,original)
        self.assertGreater(info["maximum_model_discrepancy_norm"],.01)
        self.assertFalse(info["independent_gt"])

    def test_holes_and_boundary_occluder_do_not_create_a_notch(self):
        image=self.target()
        cv2.circle(image,(250,260),12,(15,15,15),-1)
        cv2.line(image,(60,252),(210,252),(20,20,20),7)
        display,info=canonical_display(image,6)
        self.assertEqual("OBSERVED_BOUNDARY",info["status"])
        np.testing.assert_array_equal(display[252,138],image[252,138])
        np.testing.assert_array_equal(display[260,250],image[260,250])

    def test_absent_edge_is_uncertain_not_fabricated_observation(self):
        display,info=canonical_display(np.full((512,512,3),245,np.uint8),6)
        self.assertEqual("BOUNDARY_UNCERTAIN",info["status"])
        self.assertNotIn("polygon_canonical_px",info)

    def test_white_scoring_zones_are_not_cut_at_blue_ring(self):
        image=self.target();display,info=canonical_display(image,1)
        self.assertEqual("MODEL_BOUNDARY",info["status"])
        np.testing.assert_array_equal(display[255,450],image[255,450])

    def test_long_occlusion_is_reported_and_never_claimed_observed(self):
        image=self.target();image[:210,:]=128
        display,info=canonical_display(image,6)
        self.assertEqual("BOUNDARY_UNCERTAIN",info["status"])
        self.assertGreater(info["maximum_unobserved_arc_deg"],30)
