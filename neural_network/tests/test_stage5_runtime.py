"""Real XNNPACK compiler/runtime smoke on synthetic data, not a quality gate.

This runs on CI with ExecuTorch installed and MUST NOT be replaced by mocks.
Synthetic crops do not validate arrow recognition or a production checkpoint.
"""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image

from neural_network.archery_ml.contracts import (
    ModelInputSpec, ModelMetadata, ModelOutputSpec, ModelPostprocessSpec,
)
from neural_network.archery_ml.export import export_model, verify_release
from neural_network.archery_ml.models import ImpactPointNet
from neural_network.archery_ml.provenance import fingerprint
from neural_network.archery_ml.stage5_parity import sha256_file


class TestRealExecuTorchRuntime(unittest.TestCase):
    def test_compile_and_run_twenty_synthetic_faces(self):
        # The names and provenance below are explicitly synthetic fixture labels.
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            model = ImpactPointNet(fpn_channels=8, pretrained=False).eval()
            metadata = ModelMetadata(
                contract_version=2, model_version="stage5-synthetic-smoke",
                input=ModelInputSpec(
                    width=512, height=512, color_space="RGB",
                    mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225),
                ),
                output=ModelOutputSpec(type="impact_heatmap_offset", stride=2),
                postprocess=ModelPostprocessSpec(confidence_threshold=0.35, nms_radius=4),
            )
            fake_source_hash = hashlib.sha256(b"synthetic-manifest").hexdigest()
            provenance = {"version": 3, "manifest_sha256": fake_source_hash}
            checkpoint = root / "synthetic-checkpoint.pt"
            torch.save({
                "model_state": model.state_dict(),
                "model_config": {"fpn_channels": 8},
                "model_version": metadata.model_version,
                "postprocess": metadata.postprocess.to_dict(),
                "input": metadata.input.to_dict(),
                "output": metadata.output.to_dict(),
                "training_provenance": provenance,
            }, checkpoint)
            metadata_path = root / "model_metadata.json"
            metadata_path.write_text(json.dumps({
                **metadata.to_dict(),
                "training_provenance_sha256": fingerprint(provenance),
                "training_dataset_sha256": fake_source_hash,
            }))
            samples = []
            for index in range(20):
                path = root / f"synthetic-{index:02}.png"
                # Distinct deterministic RGB PNG input hashes with one original per face.
                image = Image.new("RGB", (512, 512), color=(90 + index, 110, 130))
                image.save(path)
                samples.append({
                    "sample_id": f"synthetic-{index:02}", "image_path": path.name,
                    "crop_sha256": sha256_file(path),
                    "original_sha256": hashlib.sha256(f"synthetic-camera-{index}".encode()).hexdigest(),
                    "face_index": 0, "split": "val",
                })
            golden = root / "golden.json"
            golden.write_text(json.dumps({"schema_version": 1, "samples": samples}))
            target = root / "release"
            self.assertEqual(export_model(checkpoint, metadata_path, golden, target), target)
            report = verify_release(target)
            self.assertEqual(report["status"], "PASS")
            self.assertEqual(report["golden_count"], 20)
            self.assertEqual(report["backend"], "XNNPACK")


if __name__ == "__main__":
    unittest.main()
