"""Manifest-backed per-face canonical crops. Never train directly on raw photos."""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
import hashlib
import math
import re

import numpy as np
import torch
from PIL import Image, ImageEnhance
from torch.utils.data import Dataset

from neural_network.archery_ml.data.view import DatasetView
from neural_network.archery_ml.heatmap import encode_impacts


@dataclass(frozen=True)
class FaceRecord:
    sample_id: str
    image_path: Path
    sha256: str
    source_image_sha256: str
    face_index: int
    points: tuple[tuple[float, float], ...]
    target_metadata: object | None


class CanonicalImpactDataset(Dataset):
    """Uses verified canonical_faces entries in DatasetSample.capture_metadata.

    Each entry contains face_index, image_path, and annotation_status='verified'.
    An empty point list is a supervised negative only when verified explicitly.
    Relative image paths resolve against the manifest directory.
    """

    def __init__(
        self,
        view: DatasetView,
        *,
        manifest_dir: Path,
        input_size: int = 512,
        stride: int = 2,
        sigma: float = 1.5,
        augment: bool = False,
        horizontal_flip_prob: float = 0.5,
        brightness: float = 0.15,
        input_mean: tuple[float, float, float] = (0.0, 0.0, 0.0),
        input_std: tuple[float, float, float] = (1.0, 1.0, 1.0),
    ) -> None:
        if input_size < 64 or input_size % 32 or stride != 2:
            raise ValueError("Stage 4 requires input multiple of 32 and stride 2")
        if not 0 <= horizontal_flip_prob <= 1 or brightness < 0:
            raise ValueError("invalid augmentation parameters")
        self.input_size = input_size
        self.sigma = sigma
        self.augment = augment
        self.horizontal_flip_prob = horizontal_flip_prob
        self.brightness = brightness
        if len(input_mean) != 3 or len(input_std) != 3 or not all(
            math.isfinite(v) for v in (*input_mean, *input_std)
        ) or any(v <= 0 for v in input_std):
            raise ValueError("invalid input normalization")
        self.input_mean = torch.tensor(input_mean, dtype=torch.float32).view(3, 1, 1)
        self.input_std = torch.tensor(input_std, dtype=torch.float32).view(3, 1, 1)
        self.records: list[FaceRecord] = []
        workspace_root = manifest_dir.parent.resolve()
        self.unprepared_samples = 0
        for sample in view:
            entries = (sample.capture_metadata or {}).get("canonical_faces")
            if entries is None:
                if sample.annotations:
                    raise ValueError(f"canonical impacts without Stage 3 crops in {sample.id}")
                self.unprepared_samples += 1
                continue
            if not isinstance(entries, list) or not entries:
                raise ValueError(f"invalid canonical_faces for {sample.id}")
            original = sample.image_path.resolve()
            if not original.is_relative_to(workspace_root) or not original.is_file():
                raise ValueError(f"missing or external original source image for {sample.id}")
            original_sha = hashlib.sha256(original.read_bytes()).hexdigest()
            attested_sha = ((sample.capture_metadata or {}).get("stage4_review") or {}).get("source_image_sha256")
            if attested_sha is not None and attested_sha != original_sha:
                raise ValueError(f"original image SHA256 differs from Stage 4 review in {sample.id}")
            indices = [entry.get("face_index") for entry in entries]
            if any(type(index) is not int or index < 0 for index in indices) or len(set(indices)) != len(indices):
                raise ValueError(f"duplicate or invalid face_index in {sample.id}")
            if {a.face_index for a in sample.annotations} - set(indices):
                raise ValueError(f"canonical impact without a matching crop in {sample.id}")
            if sample.target_metadata is not None:
                expected_count = 3 if sample.target_metadata.format == "TRIPLE" else 1
                if set(indices) != set(range(expected_count)):
                    raise ValueError(f"incomplete canonical face layout in {sample.id}")
            for entry in entries:
                if entry.get("annotation_status") != "verified":
                    raise ValueError(f"unverified canonical labels in {sample.id}")
                name = entry.get("image_path")
                if not isinstance(name, str) or not name.strip():
                    raise ValueError(f"missing canonical crop path in {sample.id}")
                image_path = Path(name)
                if not image_path.is_absolute():
                    image_path = manifest_dir / image_path
                image_path = image_path.resolve()
                if not image_path.is_relative_to(workspace_root):
                    raise ValueError(f"canonical crop escapes data workspace in {sample.id}")
                if not image_path.is_file():
                    raise FileNotFoundError(image_path)
                expected_sha = entry.get("sha256")
                if not isinstance(expected_sha, str) or not re.fullmatch(r"[0-9a-f]{64}", expected_sha):
                    raise ValueError(f"missing canonical crop SHA256 in {sample.id}")
                if hashlib.sha256(image_path.read_bytes()).hexdigest() != expected_sha:
                    raise ValueError(f"canonical crop SHA256 mismatch in {sample.id}")
                with Image.open(image_path) as file:
                    if file.size != (input_size, input_size):
                        raise ValueError(f"canonical crop dimensions mismatch in {sample.id}")
                    if file.mode != "RGB":
                        raise ValueError(f"canonical crop is not RGB in {sample.id}")
                index = entry["face_index"]
                points = tuple((a.x_norm, a.y_norm) for a in sample.annotations if a.face_index == index)
                self.records.append(FaceRecord(sample.id, image_path, expected_sha, original_sha, index, points, sample.target_metadata))
        if not self.records:
            raise ValueError("no verified canonical faces: Stage 3 materialization is required")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        record = self.records[index]
        crop_bytes = record.image_path.read_bytes()
        if hashlib.sha256(crop_bytes).hexdigest() != record.sha256:
            raise ValueError(f"canonical crop SHA256 mismatch during loading: {record.sample_id}")
        with Image.open(BytesIO(crop_bytes)) as file:
            if file.size != (self.input_size, self.input_size) or file.mode != "RGB":
                raise ValueError(f"canonical crop changed format: {record.sample_id}")
            image = file.copy()
        points = record.points
        if self.augment:
            if torch.rand(()).item() < self.horizontal_flip_prob:
                image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                points = tuple((-x, y) for x, y in points)
            if self.brightness:
                delta = (2 * torch.rand(()).item() - 1) * self.brightness
                image = ImageEnhance.Brightness(image).enhance(1 + delta)
        data = np.asarray(image, dtype=np.float32).copy() / 255.0
        tensor = (torch.from_numpy(data).permute(2, 0, 1) - self.input_mean) / self.input_std
        heatmap, offsets, positive = encode_impacts(points, self.input_size // 2,
                                                     self.input_size // 2, sigma=self.sigma)
        return {"image": tensor, "heatmap": heatmap, "offsets": offsets, "positive": positive,
                "points": points, "sample_id": record.sample_id,
                "face_index": record.face_index, "target_metadata": record.target_metadata}


def collate_faces(batch: list[dict]) -> dict:
    return {"image": torch.stack([item["image"] for item in batch]),
            "heatmap": torch.stack([item["heatmap"] for item in batch]),
            "offsets": torch.stack([item["offsets"] for item in batch]),
            "positive": torch.stack([item["positive"] for item in batch]),
            "records": [{key: item[key] for key in ("points", "sample_id", "face_index", "target_metadata")}
                        for item in batch]}
