import json
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from pathlib import Path

from src.ai.config import MAX_SHOTS, MISS, SHOT, IMG_SIZE
from src.ai.contracts import build_manifest_from_dirs


class ArcheryDataset(Dataset):
    def __init__(self, data_dir, json_dir, aug_transform=None, num_aug=2):
        self.samples = build_manifest_from_dirs(Path(data_dir), Path(json_dir))
        self.images = [str(sample.image_path) for sample in self.samples]
        self.jsons = [str(sample.annotation_path) for sample in self.samples]
        self.base_len = len(self.samples)

        self.aug_transform = aug_transform  # твой CustomAugmentation
        self.num_aug = num_aug  # сколько доп. версий на каждый оригинал

    def __len__(self):
        return self.base_len * (1 + self.num_aug)

    def __getitem__(self, idx):
        base_idx = idx % self.base_len
        aug_idx = idx // self.base_len  # 0 = оригинал, >0 = аугментированная версия

        sample = self.samples[base_idx]

        img = Image.open(sample.image_path).convert("RGB")
        with open(sample.annotation_path, "r") as f:
            data = json.load(f)
        shots = data["shots"]

        if img.size != (IMG_SIZE, IMG_SIZE):
            img = self._reshape_image(img)

        coords = []
        for s in shots[:MAX_SHOTS]:
            coords.append([SHOT, s["r_norm"], s["theta_deg"]])
        while len(coords) < MAX_SHOTS:
            coords.append(MISS)
        coords = torch.tensor(coords, dtype=torch.float32).flatten()

        # если это аугментированная версия → применяем aug_transform
        if aug_idx > 0 and self.aug_transform:
            img, coords = self.aug_transform(img, coords)

        img = transforms.ToTensor()(img)
        return img, coords

    def _reshape_image(self, img):
        return img.resize((IMG_SIZE, IMG_SIZE))
