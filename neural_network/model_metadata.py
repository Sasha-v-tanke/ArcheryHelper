import hashlib
import json
from pathlib import Path

from neural_network.config import IMG_SIZE, MAX_SHOTS
from neural_network.contracts import (
    LegacyModelMetadataV1,
    ModelInputSpec,
    ModelMetadata,
    ModelOutputSpec,
    ModelPostprocessSpec,
)


MODEL_METADATA = ModelMetadata(
    contract_version=2,
    model_version="impact-v1.0.0",
    input=ModelInputSpec(width=512, height=512),
    output=ModelOutputSpec(type="impact_heatmap_offset", stride=2),
    postprocess=ModelPostprocessSpec(
        confidence_threshold=0.35,
        nms_radius=4,
    ),
)

LEGACY_MODEL_METADATA = LegacyModelMetadataV1(
    contract_version=1,
    width=IMG_SIZE,
    height=IMG_SIZE,
    output="shot_set_polar",
    max_shots=MAX_SHOTS,
    coordinate_system="target_center_polar",
    confidence_threshold=0.5,
)


def write_model_metadata(output_dir: str) -> None:
    _write_metadata(output_dir, MODEL_METADATA.to_dict())


def load_model_metadata(path: str | Path) -> ModelMetadata:
    return ModelMetadata.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def write_legacy_model_metadata(output_dir: str) -> None:
    _write_metadata(output_dir, LEGACY_MODEL_METADATA.to_dict())


def load_legacy_model_metadata(path: str | Path) -> LegacyModelMetadataV1:
    return LegacyModelMetadataV1.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def write_sha256(path: str, output_dir: str) -> None:
    model_path = Path(path)
    digest = hashlib.sha256(model_path.read_bytes()).hexdigest()
    Path(output_dir, "sha256.txt").write_text(f"{digest}  {model_path.name}\n", encoding="utf-8")


def _write_metadata(output_dir: str, data: dict) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    metadata_path = output_path / "model_metadata.json"
    metadata_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
