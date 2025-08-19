import os
import glob
import json
import torch
from torch.utils.data import Dataset
from PIL import Image

from src.ai.config import MAX_SHOTS, MISS


class ArcheryDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        photo_types = ["*.jpeg", "*.jpg", "*.png"]
        self.images = sorted([f for ext in photo_types for f in glob.glob(os.path.join(data_dir, ext))])
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        json_path = (img_path.replace(".jpeg", ".json")
                     .replace(".jpg", ".json")
                     .replace(".png", ".json"))

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

        return img, coords
