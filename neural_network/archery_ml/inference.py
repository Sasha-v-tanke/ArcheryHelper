"""Python reference inference on one verified, already normalized face."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from neural_network.archery_ml.contracts import ModelMetadata
from neural_network.archery_ml.models import ImpactPointNet
from neural_network.archery_ml.postprocess import ImpactPrediction, decode_impacts


def load_detector(checkpoint: Path, metadata_path: Path, *, device: str = "cpu") -> tuple[ImpactPointNet, ModelMetadata]:
    metadata = ModelMetadata.from_dict(json.loads(metadata_path.read_text(encoding="utf-8")))
    if metadata.output.type != "impact_heatmap_offset" or metadata.output.stride != 2:
        raise ValueError("unsupported model output contract")
    if metadata.input.width != metadata.input.height or metadata.input.width % 32:
        raise ValueError("invalid model input size")
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if state.get("model_version") != metadata.model_version:
        raise ValueError("checkpoint and metadata versions differ")
    model = ImpactPointNet(fpn_channels=int(state["model_config"]["fpn_channels"]))
    model.load_state_dict(state["model_state"])
    return model.to(device).eval(), metadata


def predict_canonical_face(
    model: ImpactPointNet,
    metadata: ModelMetadata,
    image: Image.Image,
    *,
    face_index: int = 0,
) -> list[ImpactPrediction]:
    """Input MUST be a Stage 3 canonical face, not the original camera photo."""
    size = (metadata.input.width, metadata.input.height)
    rgb = image.convert("RGB").resize(size, Image.Resampling.BILINEAR)
    array = np.asarray(rgb, dtype=np.float32) / 255.0
    mean = np.asarray(metadata.input.mean, dtype=np.float32)
    std = np.asarray(metadata.input.std, dtype=np.float32)
    if np.any(std <= 0) or not np.isfinite(std).all():
        raise ValueError("invalid input normalization standard deviation")
    tensor = torch.from_numpy(((array - mean) / std).transpose(2, 0, 1).copy())
    device = next(model.parameters()).device
    with torch.inference_mode():
        heatmap, offsets = model(tensor.unsqueeze(0).to(device))
    return decode_impacts(heatmap[0].cpu(), offsets[0].cpu(),
                          confidence_threshold=metadata.postprocess.confidence_threshold,
                          nms_radius=metadata.postprocess.nms_radius, face_index=face_index)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Stage 4 on a verified canonical face image")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--canonical-face", type=Path, required=True)
    parser.add_argument("--face-index", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    model, metadata = load_detector(args.checkpoint, args.metadata, device=args.device)
    with Image.open(args.canonical_face) as image:
        predictions = predict_canonical_face(model, metadata, image, face_index=args.face_index)
    print(json.dumps([vars(point) for point in predictions], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
