"""Self-contained regression checks for pilot artifact sealing and replay states."""
from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image

from neural_network.archery_ml.geometry.pilot import batch, result_category
from neural_network.archery_ml.geometry.pilot_repro import (
    artifact_inventory, artifact_differences, require_manifest,
)
from neural_network.tests.test_stage3_pilot import scene


def snapshot(output):
    return {p.relative_to(output).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in output.rglob("*") if p.is_file()}


class ArtifactManifestTest(unittest.TestCase):
    def test_timing_only_normalized_but_result_mutation_detected(self):
        with tempfile.TemporaryDirectory() as root:
            output = Path(root)
            (output/"geometry-results.json").write_text(
                json.dumps({"one":{"status":"READY","processing_seconds":0.11}}))
            before = artifact_inventory(output)
            (output/"geometry-results.json").write_text(
                json.dumps({"one":{"processing_seconds":9.9,"status":"READY"}}))
            self.assertEqual(before,artifact_inventory(output))
            (output/"geometry-results.json").write_text(
                json.dumps({"one":{"processing_seconds":9.9,"status":"REJECTED"}}))
            self.assertEqual("content_changed",
                artifact_differences(before,artifact_inventory(output),"existing")[0]["difference"])

    def test_missing_unexpected_changed_image_and_content_types(self):
        with tempfile.TemporaryDirectory() as root:
            output = Path(root)
            (output/"crops").mkdir()
            crop = output/"crops"/"selected.png"
            crop.write_bytes(b"original image bytes")
            (output/"metrics.json").write_text('{"ready":1}')
            before = artifact_inventory(output)
            crop.write_bytes(b"corrupt png")
            (output/"metrics.json").unlink()
            (output/"unexpected.json").write_text('{"wrong":true}')
            findings = artifact_differences(before,artifact_inventory(output),"existing")
            self.assertEqual({("crops/selected.png","content_changed"),
                              ("metrics.json","missing"),
                              ("unexpected.json","unexpected")},
                             {(f["path"],f["difference"]) for f in findings})
            self.assertEqual(before,require_manifest({"artifact_manifest":before}))
            with self.assertRaisesRegex(ValueError,"legacy or unsupported"):
                require_manifest({"results_sha256":"legacy"})

    def test_physical_orientation_is_not_a_failure_when_anchored(self):
        selection = {"status":"USER_CONFIRMED"}
        self.assertEqual("success",result_category(selection,{"status":"READY","orientation_status":"ANCHORED"}))
        self.assertEqual("orientation",result_category(selection,{"status":"READY","orientation_status":"UNANCHORED"}))
        self.assertEqual("geometry",result_category(selection,{"status":"REJECTED","orientation_status":"UNANCHORED"}))
        self.assertEqual("selection",result_category({"status":"AMBIGUOUS"},{"status":"REJECTED"}))

    def test_real_batch_replay_detects_modified_metrics_without_touching_original(self):
        with tempfile.TemporaryDirectory() as root:
            base = Path(root)
            source = base/"input"
            source.mkdir()
            Image.fromarray(scene(1)).save(source/"one.png")
            output = base/"output"
            with redirect_stdout(io.StringIO()):
                batch(source,output)
                batch(source,output,verify_replay=True)
            self.assertEqual("VERIFIED",json.loads((output/"reproducibility.json").read_text())["status"])
            record = json.loads((output/"run-manifest.json").read_text())
            self.assertEqual(1,record["artifact_manifest"]["version"])
            metrics = output/"metrics.json"
            value = json.loads(metrics.read_text())
            value["ready"] = -999
            metrics.write_text(json.dumps(value))
            prior = snapshot(output)
            with redirect_stdout(io.StringIO()),self.assertRaisesRegex(ValueError,"pilot replay differs"):
                batch(source,output,verify_replay=True)
            self.assertEqual(prior,snapshot(output))
            diagnosis = json.loads((base/"output-replay-mismatch.json").read_text())
            self.assertEqual("MISMATCH",diagnosis["status"])
            self.assertIn("metrics.json",{v["path"] for v in diagnosis["artifact_differences"]})
            self.assertFalse(list(base.glob(".output-replay-*")))
