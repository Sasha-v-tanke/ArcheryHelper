import math
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from src.ai.config import *
from src.ai.utils import zip_shots, filter_shots


def draw_target(shots, size=TARGET_SIZE, border_width=2, flag=False):
    """
    shots: список [r_norm, theta_deg, ...] по парам
    """
    if flag:
        img = Image.open("/Users/alex/Projects/Python/archery/src/ai/target.png")
        img = np.array(img, dtype=np.uint8)
        size = img.shape[0]
        cx, cy = size // 2, size // 2
        max_r = size // 2
    else:
        img = np.zeros((size, size, 3), dtype=np.uint8)
        img[:, :] = (255, 255, 255)
        cx, cy = size // 2, size // 2
        max_r = size // 2

        ring_width = max_r / len(RING_COLORS)
        rings = [(i, color) for i, color in enumerate(RING_COLORS[::-1])]

        for i, color in rings[::-1]:
            inner_r = int(ring_width * i)
            outer_r = int(ring_width * (i + 1))

            y, x = np.ogrid[:size, :size]
            distance_sq = (x - cx) ** 2 + (y - cy) ** 2
            zone_mask = (distance_sq <= outer_r ** 2) & (distance_sq > inner_r ** 2)

            img[zone_mask] = np.array(Image.new("RGB", (size, size), color))[zone_mask]

            if i > 0:
                border_inner = int(inner_r - border_width / 2)
                border_outer = int(inner_r + border_width / 2)

                border_mask = (distance_sq <= border_outer ** 2) & (distance_sq > border_inner ** 2)

                if i != 7 and i != 6:
                    img[border_mask] = (0, 0, 0)
                elif i == 7:
                    img[border_mask] = (255, 255, 255)

        y, x = np.ogrid[:size, :size]
        distance_sq = (x - cx) ** 2 + (y - cy) ** 2
        border_inner = int(ring_width / 2 - border_width / 4)
        border_outer = int(ring_width / 2 + border_width / 4)
        border_mask = (distance_sq <= border_outer ** 2) & (distance_sq > border_inner ** 2)
        img[cx - 1:cx + 2, cy - 1:cy + 2] = (0, 0, 0)
        img[border_mask] = (0, 0, 0)

        y, x = np.ogrid[:size, :size]
        distance_sq = (x - cx) ** 2 + (y - cy) ** 2
        outer_r = int(max_r)
        inner_r = int(max_r - border_width)
        border_mask = (distance_sq <= outer_r ** 2) & (distance_sq >= inner_r ** 2)
        img[border_mask] = [0, 0, 0]

        # image = Image.fromarray(img)
        # img_cropped = image.crop((1, 1, 1000, 1000))
        # img_cropped.save("target.png", format="PNG", compress_level=0)

    shots = filter_shots(zip_shots(shots))

    for r_n, theta in shots:
        r_pix = r_n * max_r
        x = int(cx + r_pix * math.cos(theta / 180 * math.pi))
        y = int(cy + r_pix * math.sin(theta / 180 * math.pi))
        if 0 <= x < size and 0 <= y < size:
            img[y - 3:y + 4, x - 3:x + 4] = [0, 255, 0]
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
