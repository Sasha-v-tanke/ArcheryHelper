import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image

from neural_network.archery_ml.contracts import DatasetSample, TargetMetadata, ImageGeometryAnnotation
from neural_network.archery_ml.data.mobile_workflow import capture_plan, register, review, export, coverage, preview, source_for
from neural_network.archery_ml.data.importers.mobile_geometry import import_mobile_geometry
from neural_network.archery_ml.data.geometry_provenance import verify_mobile_gt
from neural_network.archery_ml.data.pipeline import rebuild_dataset
from unittest.mock import patch
from dataset_tools.mobile.validate_all import run
from dataset_tools.mobile.make_example import create
from neural_network.archery_ml.data.registry import DatasetRegistry
from neural_network.archery_ml.contracts import save_manifest


class MobileWorkflowTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name) / "mobile"
        self.photo = Path(self.tmp.name) / "source.png"
        Image.new("RGB",(200,200),(80,80,80)).save(self.photo)
        self.policy = json.loads(Path("dataset_tools/stage3_geometry_policy.json").read_text())
        plan = capture_plan(self.policy)
        plan.update(status="approved-before-capture",revision="fixture-plan-v1",sessions=[
            {"capture_session_id":"dev-1","device_model":"fixture-camera","split":"mobile_dev"},
            {"capture_session_id":"holdout-1","device_model":"fixture-other","split":"mobile_real_test"}])
        self.plan_path = Path(self.tmp.name) / "plan.json"
        self.plan_path.write_text(json.dumps(plan))

    def draft(self, usable=True, photo=None, normalize=False):
        path = register(self.root,photo or self.photo,self.plan_path,"dev-1",TargetMetadata("SINGLE",1,"RECURVE"),"fixture-annotator","fixture-v1",normalize)
        payload = json.loads(path.read_text())
        payload["capture_metadata"].update(suitability="usable" if usable else "unusable",
            orientation_reference="manual_calibration" if usable else "unknown",
            independent_gt_method="physical_measurement" if usable else "independent_suitability_review",
            evaluation_landmark_ids=["a","b","c","d"] if usable else [],synthetic_fixture=True)
        if usable:
            payload["geometry_annotations"] = [
                ImageGeometryAnnotation("target_center",((.5,.5),),face_index=0,uncertainty_norm=.002).to_dict(),
                ImageGeometryAnnotation("landmark",((.2,.2),(.8,.2),(.8,.8),(.2,.8)),face_index=0,
                    canonical_points=((-.6,-.6),(.6,-.6),(.6,.6),(-.6,.6)),uncertainty_norm=.002).to_dict()]
        path.write_text(json.dumps(payload))
        return path

    def test_register_review_export_rebuild_is_deterministic_and_original_unchanged(self):
        original = self.photo.read_bytes()
        path = self.draft()
        reviewed = review(self.root,path,"fixture-reviewer","controlled synthetic fixture review",self.policy)
        self.assertTrue(reviewed.exists())
        export(self.root)
        first = import_mobile_geometry(self.root,source_for(self.root))
        second = import_mobile_geometry(self.root,source_for(self.root))
        self.assertEqual(first,second)
        self.assertEqual("fixture-v1",verify_mobile_gt(first[0])["revision"])
        self.assertEqual(original,self.photo.read_bytes())
        output = self.root / "output"
        options = dict(registry_path=self.root / "sources.json",data_root=output)
        one = rebuild_dataset(**options)
        baseline = {p:p.read_bytes() for p in [one.manifest_path,one.report_path,one.snapshot_path]}
        rebuild_dataset(**options)
        self.assertTrue(all(p.read_bytes() == content for p,content in baseline.items()))
        preview(self.root,reviewed,self.root / "preview.png")
        self.assertTrue((self.root / "preview.png").exists())

    def test_negative_needs_no_invented_geometry(self):
        path = self.draft(usable=False)
        review(self.root,path,"fixture-reviewer","unusable confirmed independently",self.policy)
        export(self.root)
        imported = import_mobile_geometry(self.root,source_for(self.root))[0]
        self.assertFalse(imported.geometry_annotations)
        self.assertEqual("unusable",imported.capture_metadata["suitability"])

    def test_independent_review_and_uncertainty_enforced(self):
        path = self.draft()
        for reviewer,notes in (("fixture-annotator","notes"),("fixture-reviewer","")):
            with self.assertRaisesRegex(ValueError,"independent reviewer"):
                review(self.root,path,reviewer,notes,self.policy)
        data = json.loads(path.read_text())
        data["geometry_annotations"][1]["uncertainty_norm"] = .1
        path.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError,"uncertainty"):
            review(self.root,path,"fixture-reviewer","notes",self.policy)

    def test_shared_control_points_and_detector_gt_rejected(self):
        path = self.draft()
        data = json.loads(path.read_text())
        for meta,reason in (({"calibration_landmark_ids":["a"]},"independent"),
                            ({"calibration_landmark_ids":[],"independent_gt_method":"detector"},"independent GT")):
            data["capture_metadata"].update(meta)
            path.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError,reason):
                review(self.root,path,"fixture-reviewer","notes",self.policy)

    def test_bad_face_and_unknown_suitability_reject(self):
        path = self.draft()
        data = json.loads(path.read_text())
        data["geometry_annotations"][0]["face_index"] = 2
        path.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError,"face_index"):
            review(self.root,path,"fixture-reviewer","notes",self.policy)
        data["geometry_annotations"][0]["face_index"] = 0
        data["capture_metadata"]["suitability"] = "unknown"
        path.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError,"suitability"):
            review(self.root,path,"fixture-reviewer","notes",self.policy)

    def test_split_leakage_and_plan_mutation_reject(self):
        path = self.draft()
        reviewed = review(self.root,path,"fixture-reviewer","notes",self.policy)
        data = json.loads(reviewed.read_text())
        data.update(id="another-frame",split="mobile_real_test")
        other = reviewed.with_name("other.json")
        other.write_text(json.dumps(data))
        (self.root / "audits" / other.name).write_text(json.dumps({"reviewed_record_sha256":hashlib.sha256(other.read_bytes()).hexdigest()}))
        with self.assertRaisesRegex(ValueError,"capture session/split"):
            export(self.root)
        other.unlink()
        (self.root / "capture-plan.json").write_text("changed")
        with self.assertRaisesRegex(ValueError,"capture plan checksum"):
            export(self.root)

    def test_explicit_exif_derivative_preserves_original_and_provenance(self):
        photo = Path(self.tmp.name) / "rotated.jpg"
        exif = Image.Exif()
        exif[274] = 6
        Image.new("RGB",(60,40),(90,90,90)).save(photo,exif=exif)
        original = photo.read_bytes()
        with self.assertRaisesRegex(ValueError,"EXIF"):
            self.draft(usable=False,photo=photo)
        path = self.draft(usable=False,photo=photo,normalize=True)
        review(self.root,path,"fixture-reviewer","EXIF derivative and original checked",self.policy)
        export(self.root)
        imported = import_mobile_geometry(self.root,source_for(self.root))[0]
        self.assertEqual([40,60],imported.capture_metadata["image_size"])
        self.assertEqual(hashlib.sha256(original).hexdigest(),imported.capture_metadata["original_sha256"])
        verify_mobile_gt(imported)
        self.assertEqual(original,photo.read_bytes())

    def test_reviewed_record_mutation_is_detected(self):
        path = self.draft()
        reviewed = review(self.root,path,"fixture-reviewer","notes",self.policy)
        data = json.loads(reviewed.read_text())
        data["capture_metadata"]["reviewer"] = "different-person"
        reviewed.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError,"changed after independent review"):
            export(self.root)

    def test_coverage_has_both_splits_and_all_required_configurations(self):
        empty = coverage(self.root,self.policy)
        self.assertEqual("BLOCKED: EXTERNAL CAPTURE REQUIRED",empty["status"])
        self.draft()
        result = coverage(self.root,self.policy)
        self.assertEqual(1,result["originals"])
        self.assertEqual(18,len(result["splits"]["mobile_real_test"]["configurations"]))
        self.assertEqual(0,result["splits"]["mobile_real_test"]["originals"])
        self.assertEqual(1,result["splits"]["mobile_dev"]["originals"])

    def test_final_runner_blocks_before_evaluation_without_frozen_evidence(self):
        with patch("dataset_tools.mobile.validate_all.evaluate_main") as evaluator:
            result = run(Path(self.tmp.name) / "missing-manifest.json",Path("dataset_tools/stage3_geometry_policy.json"),
                         Path(self.tmp.name) / "blocked",final=True)
        self.assertEqual("BLOCKED",result["status"])
        self.assertFalse(result["final_holdout_executed"])
        evaluator.assert_not_called()

    def test_development_matrix_runner_preserves_failures_and_never_claims_closure(self):
        example = Path(self.tmp.name) / "example"
        create(example)
        samples = import_mobile_geometry(example,DatasetRegistry.load(example / "sources.json").sources[0])
        manifest = example / "manifest.json"
        save_manifest(manifest,samples)
        output = Path(self.tmp.name) / "dev-run"
        status = run(manifest,Path("dataset_tools/stage3_geometry_policy.json"),output)
        self.assertEqual("BLOCKED",status["status"])
        self.assertEqual(3,status["configuration_reports"])
        self.assertFalse(status["final_holdout_executed"])
        self.assertEqual("FAIL",status["validation_status"])
        with self.assertRaises(FileExistsError):
            run(manifest,Path("dataset_tools/stage3_geometry_policy.json"),output)


if __name__ == "__main__":
    unittest.main()
