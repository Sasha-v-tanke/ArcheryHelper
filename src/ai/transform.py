import random
from PIL import ImageEnhance

from src.ai.config import IMG_SIZE, SHOT


class CustomAugmentation:

    def __init__(self, size=IMG_SIZE):
        self.size = size

    def __call__(self, img, coords):
        img = img.resize((self.size, self.size))

        factor = random.uniform(0.7, 1.3)
        img = ImageEnhance.Brightness(img).enhance(factor)

        angle = random.uniform(-30, 30)
        img = img.rotate(angle)

        coords = coords.clone()
        for i in range(2, len(coords), 3):
            if coords[i - 2] < 0.5 * SHOT:
                continue
            coords[i] = (coords[i] - angle + 360.0) % 360.0

        return img, coords
