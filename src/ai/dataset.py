import os
import glob
import json
import torch
from torch import Tensor
from torch.utils.data import Dataset
from PIL import Image, ImageFile
from torchvision import transforms
from torchvision.transforms.v2 import Transform

from src.ai.config import MAX_SHOTS, MISS, SHOT, IMG_SIZE


class ArcheryDataset(Dataset):
    def __init__(self, data_dir, json_dir, aug_transform=None, num_aug=2):
        photo_types = ["*.jpeg", "*.jpg", "*.png"]
        self.images = sorted([f for ext in photo_types for f in glob.glob(os.path.join(data_dir, ext))])
        self.jsons = sorted(glob.glob(os.path.join(json_dir, '*.json')))
        self._check()
        self.base_len = len(self.images)

        self.aug_transform = aug_transform  # твой CustomAugmentation
        self.num_aug = num_aug  # сколько доп. версий на каждый оригинал

    def _check(self):
        indexes_1 = [e.split('.')[0].split('/')[-1] for e in self.images]
        indexes_2 = [e.split('.')[0].split('/')[-1] for e in self.jsons]
        indexes = [e for e in indexes_1 if e in indexes_2]
        self.images = [e for e in self.images if e.split('.')[0].split('/')[-1] in indexes]
        self.jsons = [e for e in self.jsons if e.split('.')[0].split('/')[-1] in indexes]

    def __len__(self):
        return self.base_len * (1 + self.num_aug)

    def __getitem__(self, idx):
        base_idx = idx % self.base_len
        aug_idx = idx // self.base_len  # 0 = оригинал, >0 = аугментированная версия

        img_path = self.images[base_idx]
        json_path = self.jsons[base_idx]

        img = Image.open(img_path).convert("RGB")
        with open(json_path, "r") as f:
            data = json.load(f)
        shots = data["shots"]

        if img.size != (IMG_SIZE, IMG_SIZE, 3):
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
