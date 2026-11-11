"""Generate an explicit SYNTHETIC Stage 4 checkpoint fixture under torch 2.8.

Only used to prove old-checkpoint state_dict load in the separately pinned
ExecuTorch 1.5 / torch 2.13 export environment. It is NOT trained ML evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from PIL import Image

from neural_network.archery_ml.contracts import (
    ModelInputSpec, ModelMetadata, ModelOutputSpec, ModelPostprocessSpec,
)
from neural_network.archery_ml.models import ImpactPointNet
from neural_network.archery_ml.provenance import fingerprint
from neural_network.archery_ml.stage5_parity import sha256_file


def create_fixture(root: Path) -> None:
    if root.exists():
        raise FileExistsError(root)
    root.mkdir(parents=True)
    torch.manual_seed(314159)
    model = ImpactPointNet(fpn_channels=48, pretrained=False).eval()
    with torch.no_grad():
        model.heatmap_head.weight.zero_()
        model.heatmap_head.bias.fill_(2.0)
        model.offset_head.weight.fill_(0.005)
        model.offset_head.bias.fill_(0.25)

    metadata = ModelMetadata(
        contract_version=2, model_version="stage5-checkpoint-cross-version-synthetic",
        input=ModelInputSpec(width=512, height=512, color_space="RGB",
                             mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225)),
        output=ModelOutputSpec(type="impact_heatmap_offset", stride=2),
        postprocess=ModelPostprocessSpec(confidence_threshold=0.35, nms_radius=4),
    )
    synthetic_manifest_hash = hashlib.sha256(b"synthetic-stage4-manifest").hexdigest()
    provenance = {"version": 3, "manifest_sha256": synthetic_manifest_hash}
    checkpoint = root / "best_model.pt"
    torch.save({
        "model_state": model.state_dict(),
        "model_config": {"fpn_channels": 48},
        "model_version": metadata.model_version,
        "postprocess": metadata.postprocess.to_dict(),
        "input": metadata.input.to_dict(),
        "output": metadata.output.to_dict(),
        "training_provenance": provenance,
    }, checkpoint)
    (root / "model_metadata.json").write_text(json.dumps({
        **metadata.to_dict(),
        "training_provenance_sha256": fingerprint(provenance),
        "training_dataset_sha256": synthetic_manifest_hash,
    }, indent=2), encoding="utf-8")

    samples = []
    for index in range(20):
        image_path = root / f"synthetic-{index:02d}.png"
        Image.new("RGB", (512, 512), color=(60 + index, 100, 170)).save(image_path)
        samples.append({
            "sample_id": f"synthetic-{index:02d}",
            "image_path": image_path.name,
            "crop_sha256": sha256_file(image_path),
            "original_sha256": hashlib.sha256(f"synthetic-original-{index}".encode()).hexdigest(),
            "face_index": 0,
            "split": "val",
        })
    (root / "golden.json").write_text(json.dumps({
        "schema_version": 1,
        "source_checkpoint_sha256": sha256_file(checkpoint),
        "source_manifest_sha256": synthetic_manifest_hash,
        "source_snapshot_sha256": hashlib.sha256(b"synthetic-stage4-snapshot").hexdigest(),
        "samples": samples,
    }, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Synthetic Stage 4 checkpoint compatibility fixture")
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args()
    create_fixture(args.output_directory)
    print(json.dumps({"torch": torch.__version__, "checkpoint": str(
        args.output_directory / "best_model.pt")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
