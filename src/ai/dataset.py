import os
import glob
import json
import torch
from torch import Tensor
from torch.utils.data import Dataset
from PIL import Image, ImageFile
from torchvision import transforms
from torchvision.transforms.v2 import Transform

from src.ai.config import MAX_SHOTS, MISS


class ArcheryDataset(Dataset):
    def __init__(self, data_dir: str, json_dir: str, transform: Transform = None):
        photo_types = ["*.jpeg", "*.jpg", "*.png"]
        self.images = sorted([f for ext in photo_types for f in glob.glob(os.path.join(data_dir, ext))])
        self.jsons = sorted(glob.glob(os.path.join(json_dir, '*.json')))
        self.transform = transform

        self._check()

    def _check(self):
        indexes_1 = [e.split('.')[0].split('/')[-1] for e in self.images]
        indexes_2 = [e.split('.')[0].split('/')[-1] for e in self.jsons]
        indexes = [e for e in indexes_1 if e in indexes_2]
        self.images = [e for e in self.images if e.split('.')[0].split('/')[-1] in indexes]
        self.jsons = [e for e in self.jsons if e.split('.')[0].split('/')[-1] in indexes]
        print(self.images)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> (Tensor, Tensor):
        img_path = self.images[idx]
        json_path = self.jsons[idx]

        img = Image.open(img_path).convert("RGB")

        with open(json_path, "r") as f:
            data = json.load(f)
        shots = data["shots"]

        coords = []
        for s in shots[:MAX_SHOTS]:
            coords.append([s["r_norm"], s["theta_deg"]])

        while len(coords) < MAX_SHOTS:
            coords.append(MISS)
        coords = torch.tensor(coords, dtype=torch.float32).flatten()

        if self.transform:
            img, coords = self.transform(img, coords)

        img = transforms.ToTensor()(img)
        return img, coords
