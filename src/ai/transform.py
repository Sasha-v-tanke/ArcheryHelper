import random
from torchvision import transforms
from PIL import ImageEnhance

from src.ai.config import IMG_SIZE


class CustomAugmentation:
    def __init__(self, size=IMG_SIZE):
        self.size = size

    def __call__(self, img, coords):
        img = img.resize((self.size, self.size))

        if random.random() < 0.5:
            factor = random.uniform(0.7, 1.3)
            img = ImageEnhance.Brightness(img).enhance(factor)

        angle = 0
        if random.random() < 0.5:
            angle = random.uniform(-30, 30)
            img = img.rotate(angle)

        coords = coords.clone()
        for i in range(1, len(coords), 2):
            coords[i] = (coords[i] - angle + 360.0) % 360.0

        img = transforms.ToTensor()(img)
        return img, coords
