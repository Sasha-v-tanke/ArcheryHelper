import os
import torch

from path_manager import MODELS
from src.ai.config import MISS


def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.mps.is_available() else 'cpu')


def collate_fn(batch):
    imgs, coords = zip(*batch)
    imgs = torch.stack(imgs, dim=0)
    coords = torch.stack(coords, dim=0)
    return imgs, coords


def save_model(model):
    number = 0
    while os.path.exists(os.path.join(MODELS, f"archery_resnet_{number}.pth")):
        number += 1
    torch.save(model.state_dict(), os.path.join(MODELS, f"archery_resnet_{number}.pth"))
    print(f"Model saved to archery_resnet_{number}.pth")


def load_model(model, device):
    number = 0
    while os.path.exists(os.path.join(MODELS, f"archery_resnet_{number}.pth")):
        number += 1
    model.load_state_dict(torch.load(os.path.join(MODELS, f"archery_resnet_{number - 1}.pth"), map_location=device))


def zip_shots(shots):
    return [[shots[i], shots[i + 1]] for i in range(0, len(shots), 2)]


def unzip_shots(shots):
    return [e for shot in shots for e in shot]


def filter_shots(shots):
    return [shot for shot in shots if shot != MISS]
