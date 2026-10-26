import copy
import json
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np
from PIL import Image

from neural_network.archery_ml.contracts import DatasetSample
from neural_network.archery_ml.data.view import DatasetView
from neural_network.archery_ml.geometry.evaluate import GeometryEvaluator
from neural_network.archery_ml.geometry.calibration import project
from neural_network.archery_ml.geometry.pilot_inventory import raw_to_rgb_matrix,inventory
from neural_network.archery_ml.geometry.selection import TargetCandidateDetector,TargetGroupBuilder,TargetGroupSelector
from neural_network.archery_ml.geometry.selected_geometry import SelectedTargetGeometryPipeline
from neural_network.tests import test_geometry_evaluation as evaluation_fixtures


def scene(columns=3):
    image = np.full((650,600,3),40,np.uint8)
    for x in [110,300,490][:columns]:
        cv2.rectangle(image,(x-85,45),(x+85,605),(245,245,245),-1)
        for y in (130,325,520):
            for radius,color in ((70,(20,150,230)),(56,(230,40,40)),(28,(245,220,30))):
                cv2.circle(image,(x,y),radius,color,-1)
    return image


class PilotTest(unittest.TestCase):
    def test_local_selection_persists_can_be_corrected_and_rejects_stale_ids(self):
        from neural_network.archery_ml.geometry.pilot import batch
        from neural_network.archery_ml.geometry.pilot_viewer import update
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/"input";source.mkdir();output=root/"output"
            Image.fromarray(scene(2)).save(source/"targets.png")
            batch(source,output)
            detections=json.loads((output/"detections.json").read_text())
            key,det=next(iter(detections.items()))
            triples=[g for g in det["groups"] if g["format"]=="TRIPLE"]
            value={**triples[0],"status":"USER_CONFIRMED","detection_fingerprint":det["fingerprint"]}
            saved=update(output,key,"selection",value)
            self.assertEqual("USER_CONFIRMED",saved["selections"][key]["status"])
            self.assertEqual("READY",saved["results"][key]["status"])
            saved=update(output,key,"review",{"rating":"Correct","candidate_ids":triples[0]["candidate_ids"]})
            self.assertTrue(saved["reviews"][key]["geometry_artifact_revision"])
            value={**triples[1],"status":"USER_CONFIRMED","detection_fingerprint":det["fingerprint"]}
            saved=update(output,key,"selection",value)
            self.assertEqual(triples[1]["candidate_ids"],saved["results"][key]["candidate_ids"])
            self.assertTrue(saved["reviews"][key]["stale"])
            self.assertEqual("Correct",saved["reviews"][key]["rating"])
            stored=json.loads((output/"target-selections.json").read_text())
            self.assertEqual(triples[1]["candidate_ids"],stored[key]["candidate_ids"])
            with self.assertRaisesRegex(ValueError,"stale"):
                update(output,key,"selection",{**value,"detection_fingerprint":"stale"})
            with self.assertRaisesRegex(ValueError,"actual candidate"):
                update(output,key,"review",{"rating":"Correct","candidate_ids":["invented"]})

    def test_partial_independent_metrics_reject_pseudo_gt_and_measure_each_face(self):
        from neural_network.archery_ml.geometry.pilot_metrics import independent_metrics
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"gt.json"
            path.write_text(json.dumps({"kind":"pseudo_gt"}))
            with self.assertRaisesRegex(ValueError,"not independent"):
                independent_metrics(path,{})
            points=[[-.2,-.2],[.2,-.2],[.2,.2],[-.2,.2]]
            image=(np.array(points)*256+255.5).tolist()
            payload={"kind":"independent_gt","method":"physical_measurement","revision":"r1","reference":"survey",
                "annotator":"one","reviewer":"two","images":[{"original_sha256":"sample","candidate_ids":["f0"],
                "coordinate_space":"original_raw_px","faces":[{"face_index":0,"image_points_px":image,"canonical_points":points}]}]}
            path.write_text(json.dumps(payload))
            results={"sample":{"status":"READY","candidate_ids":["f0"],"orientation_status":"UNANCHORED",
                "faces":[{"face_index":0,"original_to_canonical":np.eye(3).tolist()}]}}
            report=independent_metrics(path,results)
            self.assertEqual(1,report["independent_gt_images"])
            self.assertLess(report["p95_2d"],1e-10)
            self.assertEqual(["0"],list(report["per_image"][0]["face_errors_2d"]))

    def test_nine_faces_build_three_columns_without_mixing_or_claiming_ownership(self):
        image = scene()
        candidates = TargetCandidateDetector().detect(image)
        self.assertEqual(9,len(candidates))
        groups = TargetGroupBuilder().build(candidates,image)
        triples = [g for g in groups if g["format"]=="TRIPLE"]
        self.assertEqual(3,len(triples))
        for g in triples:
            self.assertIs(type(g["minimum_zone_proposal"]),int)
            self.assertEqual(6,g["minimum_zone_proposal"])
            xs = [c["center"][0] for c in candidates if c["id"] in g["candidate_ids"]]
            self.assertLess(max(xs)-min(xs),2)
        selected = TargetGroupSelector().select(candidates,groups)
        self.assertEqual("AMBIGUOUS",selected["status"])
        self.assertFalse(selected["candidate_ids"])

    def test_broken_annulus_prefers_outer_edge_over_denser_inner_vertices(self):
        from neural_network.archery_ml.geometry.detector import _ellipse_inliers
        outer=np.linspace(0,2*np.pi,96,endpoint=False)
        inner=np.linspace(0,2*np.pi,300,endpoint=False)
        points=np.vstack((np.column_stack((100+90*np.cos(outer),100+90*np.sin(outer))),
                          np.column_stack((100+72*np.cos(inner),100+72*np.sin(inner))))).astype(np.float32)
        fitted=cv2.fitEllipse(_ellipse_inliers(points.reshape(-1,1,2),prefer_outer=True))
        self.assertAlmostEqual(180,max(fitted[1]),delta=2.)

    def test_large_background_and_blue_connection_do_not_absorb_real_faces(self):
        image=scene(2)
        cv2.rectangle(image,(395,0),(599,649),(20,150,230),-1)
        cv2.line(image,(365,520),(410,520),(20,150,230),5)
        candidates=TargetCandidateDetector().detect(image)
        self.assertEqual(6,len(candidates))
        groups=[g for g in TargetGroupBuilder().build(candidates,image) if g["format"]=="TRIPLE"]
        self.assertEqual(2,len(groups))

    def test_initial_scale_uses_fitted_ellipse_not_extrema_of_surviving_arc(self):
        from neural_network.archery_ml.geometry.detector import _Candidate,EllipseFit,_initial_image_to_norm
        angles=np.linspace(-.5,.5,80)
        points=np.column_stack((100+90*np.cos(angles),100+60*np.sin(angles))).astype(np.float32)
        candidate=_Candidate(points.reshape(-1,1,2),EllipseFit(100,100,180,120,0),100.)
        transform=_initial_image_to_norm(candidate,.5,(np.array([1.,0]),np.array([0.,1.])))
        expected=project([[190,100],[100,160],[10,100],[100,40]],transform)
        np.testing.assert_allclose(np.linalg.norm(expected,axis=1),.5,atol=1e-7)

    def test_roi_maps_back_to_full_original_and_only_selected_faces_are_fitted(self):
        image = scene(2)
        candidates = TargetCandidateDetector().detect(image)
        self.assertEqual(6,len(candidates))
        group = next(g for g in TargetGroupBuilder().build(candidates,image) if g["format"]=="TRIPLE")
        group["physical_face_order_confirmed"] = True
        matrix = raw_to_rgb_matrix(6,650,600)
        result = SelectedTargetGeometryPipeline().run(image,candidates,group,matrix)
        self.assertEqual("READY",result["status"],result["diagnostics"])
        self.assertEqual(3,len(result["faces"]))
        self.assertEqual(3,len(result["foreign_candidates_masked"]))
        for face in result["faces"]:
            point = np.array([[100.,200.]])
            np.testing.assert_allclose(point,project(project(point,face["original_to_canonical"]),face["canonical_to_original"]),atol=1e-6)
            self.assertLess(face["round_trip_max_px"],1e-6)
        self.assertEqual("UNANCHORED",result["orientation_status"])

    def test_wrong_face_count_never_normalizes_to_ready(self):
        image = scene(1)
        candidates = TargetCandidateDetector().detect(image)
        group = next(g for g in TargetGroupBuilder().build(candidates,image) if g["format"]=="TRIPLE")
        group["candidate_ids"] = group["candidate_ids"][:2]
        result = SelectedTargetGeometryPipeline().run(image,candidates,group)
        self.assertEqual("REJECTED",result["status"])
        self.assertEqual("WRONG_FACE_COUNT",result["quality_status"])

    def test_inventory_duplicates_exif_and_original_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/"input").mkdir()
            im = Image.new("RGB",(30,20),"red")
            exif = Image.Exif();exif[274]=6
            im.save(root/"input/a.jpg",exif=exif)
            before = (root/"input/a.jpg").read_bytes()
            (root/"input/b.jpg").write_bytes(before)
            result = inventory(root/"input",root/"output")
            self.assertEqual(2,result["files"])
            self.assertEqual(1,result["unique_originals"])
            self.assertEqual([20,30],result["images"][0]["rgb_size"])
            self.assertEqual(before,(root/"input/a.jpg").read_bytes())
            self.assertEqual("unknown",result["images"][0]["capture_session_id"])

    def test_exif_matrices_match_pillow_pixel_transposes_for_all_eight_modes(self):
        from PIL import ImageOps
        pixels = np.arange(5*7,dtype=np.uint8).reshape(5,7)
        for orientation in range(1,9):
            image = Image.fromarray(pixels)
            image.getexif()[274] = orientation
            expected = np.asarray(ImageOps.exif_transpose(image))
            for y,x in ((0,0),(2,4),(4,6)):
                px,py = np.rint(project([[x,y]],raw_to_rgb_matrix(orientation,7,5))[0]).astype(int)
                self.assertEqual(pixels[y,x],expected[py,px])

    def test_false_positive_ready_without_gt_is_retained_in_false_acceptance(self):
        fixture = evaluation_fixtures.GeometryEvaluationTest()
        image,_ = fixture._fixture()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"target.png"
            Image.fromarray(image).save(path)
            sample = DatasetSample("false-positive",path,None,capture_metadata={"suitability":"unusable"})
            report = GeometryEvaluator().evaluate(DatasetView.from_samples([sample]),fixture._config()).to_dict()
        self.assertEqual(1,report["ready_samples"])
        self.assertEqual(1.,report["coverage"]["false_accept_fraction"])
        self.assertIsNone(report["point_error_2d"]["p95"])
        self.assertEqual(0,report["point_error_2d"]["count"])
