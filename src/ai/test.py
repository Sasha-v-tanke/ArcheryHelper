import os.path
import matplotlib.pyplot as plt
import numpy as np
import torch

from path_manager import NEW_NORMALIZED_DATASET, MODELS
from src.ai.model import ArcheryResNet
from src.ai.config import *
from src.ai.dataset import ArcheryDataset
from src.ai.transform import CustomAugmentation
from src.ai.ui import draw_target
from src.ai.utils import get_device, load_model


def visualize_model(model, dataset, device, n_samples=3):
    model.eval()
    fig, axes = plt.subplots(n_samples, 3, figsize=(12, 4 * n_samples))

    with torch.no_grad():
        for i in range(n_samples):
            img, coords = dataset[i]
            img_tensor = img.unsqueeze(0).to(device)
            pred = model(img_tensor).cpu().numpy().flatten()

            img_np = (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            target_real = draw_target(coords.numpy())
            target_pred = draw_target(pred)

            axes[i, 0].imshow(img_np)
            axes[i, 0].set_title("Original Image")
            axes[i, 1].imshow(target_real)
            axes[i, 1].set_title("Target with real points")
            axes[i, 2].imshow(target_pred)
            axes[i, 2].set_title("Target predicted")
            for ax in axes[i]:
                ax.axis("off")
    plt.tight_layout()
    plt.show()


# ==== После тренировки ====
if __name__ == "__main__":
    device = get_device()
    dataset = ArcheryDataset(NEW_NORMALIZED_DATASET, transform=CustomAugmentation())
    model = ArcheryResNet(MAX_SHOTS * 2).to(device)
    load_model(model, device)

    visualize_model(model, dataset, device, n_samples=3)
