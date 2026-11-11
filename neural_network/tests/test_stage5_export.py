"""Stage 5 release regression tests. Mocked compiler tests are NOT parity evidence."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch
from PIL import Image

from neural_network.archery_ml.contracts import (
    ModelInputSpec, ModelMetadata, ModelOutputSpec, ModelPostprocessSpec,
)
from neural_network.archery_ml.export import export_model, replay_release, verify_release
from neural_network.archery_ml.stage5_parity import (
    _match_points, compare_outputs, read_golden_manifest, sha256_file,
)
from neural_network.archery_ml.postprocess import ImpactPrediction


def _metadata(size: int = 512) -> ModelMetadata:
    return ModelMetadata(
        contract_version=2, model_version="stage5-test",
        input=ModelInputSpec(width=size, height=size, mean=(0., 0., 0.), std=(1., 1., 1.)),
        output=ModelOutputSpec(type="impact_heatmap_offset", stride=2),
        postprocess=ModelPostprocessSpec(confidence_threshold=0.5, nms_radius=1),
    )


def _build_golden(root: Path, *, count: int = 20, size: int = 512) -> Path:
    records = []
    for index in range(count):
        path = root / f"face-{index:02}.png"
        image = Image.new("RGB", (size, size))
        image.putpixel((0, 0), (index + 1, 22, 33))
        image.save(path)
        records.append({
            "sample_id": f"face-{index:02}", "image_path": path.name,
            "crop_sha256": sha256_file(path),
            "original_sha256": hashlib.sha256(f"original-{index}".encode()).hexdigest(),
            "face_index": 0, "split": "val",
        })
    manifest = root / "golden.json"
    manifest.write_text(json.dumps({"schema_version": 1, "samples": records}), encoding="utf-8")
    return manifest


class TestGoldenManifest(unittest.TestCase):
    def test_strict_golden_set_and_stable_sort(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = _build_golden(root)
            records = read_golden_manifest(path, _metadata())
            self.assertEqual(len(records), 20)
            self.assertEqual(records[0].sample_id, "face-00")

    def test_reject_missing_and_excess_samples(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for count in (19, 51):
                path = _build_golden(root, count=count)
                with self.assertRaisesRegex(ValueError, "20 to 50"):
                    read_golden_manifest(path, _metadata())

    def test_reject_tampered_pixels_and_duplicate_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = _build_golden(root)
            (root / "face-00.png").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "checksum"):
                read_golden_manifest(path, _metadata())
            path = _build_golden(root)
            document = json.loads(path.read_text())
            document["samples"][1]["original_sha256"] = document["samples"][0]["original_sha256"]
            path.write_text(json.dumps(document))
            with self.assertRaisesRegex(ValueError, "duplicate"):
                read_golden_manifest(path, _metadata())

    def test_reject_path_escape_and_unreviewed_split(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = _build_golden(root)
            document = json.loads(path.read_text())
            document["samples"][0]["image_path"] = "../escape.png"
            path.write_text(json.dumps(document))
            with self.assertRaisesRegex(ValueError, "path"):
                read_golden_manifest(path, _metadata())
            path = _build_golden(root)
            document = json.loads(path.read_text())
            document["samples"][0]["split"] = "mobile_real_test"
            path.write_text(json.dumps(document))
            with self.assertRaisesRegex(ValueError, "reviewed"):
                read_golden_manifest(path, _metadata())

    def test_reject_wrong_mode(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = _build_golden(root)
            bad = root / "face-00.png"
            Image.new("L", (512, 512)).save(bad)
            document = json.loads(path.read_text())
            document["samples"][0]["crop_sha256"] = sha256_file(bad)
            path.write_text(json.dumps(document))
            with self.assertRaisesRegex(ValueError, "RGB"):
                read_golden_manifest(path, _metadata())


class TestParity(unittest.TestCase):
    def test_binary_matching_accepts_reordered_points(self):
        p1 = ImpactPrediction(0.1, 0.2, .9, 0)
        p2 = ImpactPrediction(-0.5, 0.4, .8, 1)
        self.assertEqual(_match_points([p1, p2], [p2, p1]), (0., 0.))

    def test_reject_missing_and_displaced_points(self):
        first = ImpactPrediction(.1, .2, .9, 0)
        with self.assertRaisesRegex(ValueError, "count"):
            _match_points([first], [])
        with self.assertRaisesRegex(ValueError, "tolerance"):
            _match_points([first], [ImpactPrediction(.12, .2, .9, 0)])

    def test_compare_exact_and_reject_tensor_drift(self):
        metadata = _metadata(8)
        logits = torch.full((1, 1, 4, 4), -8.0, dtype=torch.float32)
        offsets = torch.zeros((1, 2, 4, 4), dtype=torch.float32)
        logits[0, 0, 2, 2] = 8.
        a = (logits, offsets)
        result = compare_outputs(a, (logits.clone(), offsets.clone()), metadata, 0)
        self.assertEqual(result["point_count"], 1)
        bad = logits.clone()
        bad[0, 0, 1, 1] = 8.
        with self.assertRaisesRegex(ValueError, "heatmap_logits parity"):
            compare_outputs(a, (bad, offsets), metadata, 0)

    def test_reject_nan_and_output_shapes(self):
        metadata = _metadata(8)
        logits = torch.zeros((1, 1, 4, 4))
        offsets = torch.zeros((1, 2, 4, 4))
        nan = logits.clone()
        nan[0, 0, 0, 0] = float("nan")
        with self.assertRaisesRegex(ValueError, "NaN"):
            compare_outputs((logits, offsets), (nan, offsets), metadata, 0)
        with self.assertRaisesRegex(ValueError, "shape"):
            compare_outputs((logits, offsets), (logits[:, :, :3], offsets), metadata, 0)


class _TestModel(torch.nn.Module):
    def forward(self, image: torch.Tensor):
        return (torch.full((1, 1, 256, 256), -9., dtype=torch.float32),
                torch.zeros((1, 2, 256, 256), dtype=torch.float32))


class TestAtomicExport(unittest.TestCase):
    def _setup(self, root):
        golden = _build_golden(root)
        checkpoint = root / "best_model.pt"
        checkpoint.write_bytes(b"fake checkpoint for isolated packaging tests")
        metadata = root / "source-metadata.json"
        dataset_hash = hashlib.sha256(b"stage4-synthetic-fixture").hexdigest()
        metadata.write_text(json.dumps({
            **_metadata().to_dict(), "training_dataset_sha256": dataset_hash,
        }), encoding="utf-8")
        golden_document = json.loads(golden.read_text())
        golden_document["source_manifest_sha256"] = dataset_hash
        golden_document["source_snapshot_sha256"] = hashlib.sha256(b"snapshot-fixture").hexdigest()
        golden_document["source_checkpoint_sha256"] = sha256_file(checkpoint)
        golden.write_text(json.dumps(golden_document), encoding="utf-8")
        return checkpoint, metadata, golden

    def test_successful_mocked_export_has_consistent_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint, metadata, golden = self._setup(root)
            destination = root / "bundle"
            model = _TestModel()
            with (mock.patch("neural_network.archery_ml.export.load_detector",
                             return_value=(model, _metadata())),
                  mock.patch("neural_network.archery_ml.export._compile_xnnpack",
                             return_value=b"XnnpackBackend fake fixture"),
                  mock.patch("neural_network.archery_ml.export._runtime_forward",
                             return_value=lambda tensor: model(tensor)),
                  mock.patch("neural_network.archery_ml.export.importlib.metadata.version",
                             return_value="1.5.1")):
                self.assertEqual(export_model(checkpoint, metadata, golden, destination), destination)
            parity = verify_release(destination)
            self.assertEqual(parity["golden_count"], 20)
            self.assertEqual(parity["status"], "PASS")
            released_metadata = json.loads((destination / "model_metadata.json").read_text())
            self.assertEqual(released_metadata["executorch"]["input_layout"], "NCHW")
            self.assertEqual(released_metadata["executorch"]["output_order"],
                             ["heatmap_logits", "offsets"])
            self.assertEqual(released_metadata["executorch"]["pte_bytes"],
                             (destination / "impact_model.pte").stat().st_size)
            index = json.loads((destination / "golden/index.json").read_text())
            self.assertEqual(len(index["samples"]), 20)
            self.assertEqual(len(list((destination / "golden/images").iterdir())), 20)
            with mock.patch("neural_network.archery_ml.export._runtime_forward",
                            return_value=lambda tensor: model(tensor)):
                replay = replay_release(destination)
            self.assertEqual(replay["status"], "PASS")
            self.assertEqual(replay["golden_count"], 20)

            image = destination / "golden/images/face-00.png"
            image.write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError, "SHA256"):
                verify_release(destination)

    def test_semantically_swapped_outputs_fail_even_with_rehashed_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint, metadata, golden = self._setup(root)
            target = root / "bundle"
            model = _TestModel()
            with (mock.patch("neural_network.archery_ml.export.load_detector",
                             return_value=(model, _metadata())),
                  mock.patch("neural_network.archery_ml.export._compile_xnnpack",
                             return_value=b"XnnpackBackend fake fixture"),
                  mock.patch("neural_network.archery_ml.export._runtime_forward",
                             return_value=lambda tensor: model(tensor)),
                  mock.patch("neural_network.archery_ml.export.importlib.metadata.version",
                             return_value="1.5.1")):
                export_model(checkpoint, metadata, golden, target)
            meta = target / "model_metadata.json"
            document = json.loads(meta.read_text())
            document["executorch"]["output_order"] = ["offsets", "heatmap_logits"]
            meta.write_text(json.dumps(document))
            lines = []
            for path in sorted(p for p in target.rglob("*") if p.is_file()
                               and p.name != "sha256.txt"):
                lines.append(f"{sha256_file(path)}  {path.relative_to(target).as_posix()}")
            (target / "sha256.txt").write_text("\n".join(lines) + "\n")
            with self.assertRaisesRegex(ValueError, "inconsistent Stage 5"):
                verify_release(target)

    def test_runtime_mismatch_cleans_incomplete_release(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint, metadata, golden = self._setup(root)
            destination = root / "bundle"
            def wrong(_):
                return (torch.ones((1, 1, 256, 256)), torch.zeros((1, 2, 256, 256)))
            with (mock.patch("neural_network.archery_ml.export.load_detector",
                             return_value=(_TestModel(), _metadata())),
                  mock.patch("neural_network.archery_ml.export._compile_xnnpack",
                             return_value=b"XnnpackBackend fake fixture"),
                  mock.patch("neural_network.archery_ml.export._runtime_forward",
                             return_value=wrong)):
                with self.assertRaisesRegex(ValueError, "parity"):
                    export_model(checkpoint, metadata, golden, destination)
            self.assertFalse(destination.exists())
            self.assertEqual(list(root.glob(".stage5-*")), [])

    def test_export_rejects_missing_golden_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint, metadata, golden = self._setup(root)
            document = json.loads(golden.read_text())
            del document["source_snapshot_sha256"]
            golden.write_text(json.dumps(document))
            with mock.patch("neural_network.archery_ml.export.load_detector",
                            return_value=(_TestModel(), _metadata())):
                with self.assertRaisesRegex(ValueError, "lacks immutable"):
                    export_model(checkpoint, metadata, golden, root / "release")
            self.assertFalse((root / "release").exists())

    def test_export_rejects_golden_provenance_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint, metadata, golden = self._setup(root)
            document = json.loads(golden.read_text())
            document["source_checkpoint_sha256"] = "a" * 64
            golden.write_text(json.dumps(document))
            with mock.patch("neural_network.archery_ml.export.load_detector",
                            return_value=(_TestModel(), _metadata())):
                with self.assertRaisesRegex(ValueError, "different checkpoint"):
                    export_model(checkpoint, metadata, golden, root / "release")
            self.assertFalse((root / "release").exists())

    def test_export_rejects_mismatched_stage4_dataset_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint, metadata, golden = self._setup(root)
            document = json.loads(golden.read_text())
            document["source_manifest_sha256"] = "a" * 64
            golden.write_text(json.dumps(document))
            with mock.patch("neural_network.archery_ml.export.load_detector",
                            return_value=(_TestModel(), _metadata())):
                with self.assertRaisesRegex(ValueError, "does not match"):
                    export_model(checkpoint, metadata, golden, root / "release")

    def test_missing_real_compiler_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint, metadata, golden = self._setup(root)
            destination = root / "bundle"
            with (mock.patch("neural_network.archery_ml.export.load_detector",
                             return_value=(_TestModel(), _metadata())),
                  mock.patch("neural_network.archery_ml.export._compile_xnnpack",
                             side_effect=ImportError("ExecuTorch unavailable"))):
                with self.assertRaisesRegex(ImportError, "unavailable"):
                    export_model(checkpoint, metadata, golden, destination)
            self.assertFalse(destination.exists())


if __name__ == "__main__":
    unittest.main()
