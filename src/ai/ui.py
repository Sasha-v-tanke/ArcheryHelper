import math
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from src.ai.config import SHOT
from src.ai.utils import zip_shots, filter_shots


def draw_target(shots: list):
    """
    shots: список [r_norm, theta_deg, ...] по парам
    """
    img = Image.open("/Users/alex/Projects/Python/archery/src/ai/target.png")
    img = np.array(img, dtype=np.uint8)
    size = img.shape[0]
    cx, cy = size // 2, size // 2
    max_r = size // 2

    shots = filter_shots(zip_shots(shots))

    for p, r_n, theta in shots:
        if p < 0.5 * SHOT:
            continue
        r_pix = r_n * max_r
        x = int(cx + r_pix * math.cos(theta / 180 * math.pi))
        y = int(cy + r_pix * math.sin(theta / 180 * math.pi))
        POINT_SIZE = 8
        BORDER_SIZE = 10
        if 0 <= x < size and 0 <= y < size:
            img[y - BORDER_SIZE:y + BORDER_SIZE + 1, x - BORDER_SIZE:x + BORDER_SIZE + 1] = [0, 0, 0]
            img[y - POINT_SIZE:y + POINT_SIZE + 1, x - POINT_SIZE:x + POINT_SIZE + 1] = [0, 255, 0]
    return img


def show_history(train_loss, val_loss):
    """
    Отображает историю потерь обучения и валидации на одном графике.
    """
    plt.figure(figsize=(8, 5))
    plt.plot(train_loss, label='Train Loss', color='blue')
    plt.plot(val_loss, label='Validation Loss', color='orange')
    plt.title('Training and Validation Loss History')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()
    plt.show()
