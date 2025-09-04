import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms

from path_manager import CONVERTED_DATASET_PATH, NEW_NORMALIZED_DATASET, MODELS
from src.ai.model import ArcheryResNet
from src.ai.config import OUTPUT_DIM
from src.ai.utils import get_device, load_model
from src.ai.transform import CustomAugmentation
from src.ai.ui import draw_target


# === Проверка одной картинки ===
def predict_single(image_path: str):
    device = torch.device('cpu')

    # Загружаем модель
    ptl_model_path = MODELS + "/model.ptl"
    model = torch.jit.load(ptl_model_path)
    model.eval()

    model2 = ArcheryResNet()
    load_model(model2, device)
    model2.eval()

    ptl_model_path = MODELS + "/model.pt"
    model3 = torch.jit.load(ptl_model_path)
    model3.eval()

    # Загружаем изображение
    img = Image.open(image_path).convert("RGB")

    # Аугментацию для одной картинки можно отключить, чтобы не искажать данные
    # transform = CustomAugmentation()
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    img_tensor = transform(img).unsqueeze(0).to(device)

    # Предсказание
    with torch.no_grad():
        pred = model(img_tensor).cpu().numpy().flatten()
        pred2 = model2(img_tensor).cpu().numpy().flatten()
        pred3 = model3(img_tensor).cpu().numpy().flatten()

    print("Предсказанные координаты (ptl):", pred)
    print("Предсказанные координаты (pth):", pred2)
    print("Предсказанные координаты (pt ):", pred3)

    # Визуализация
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(draw_target(pred))
    axes[0].set_title("Predicted Targetby ptl")
    axes[0].axis("off")
    axes[1].imshow(draw_target(pred2))
    axes[1].set_title("Predicted Targetby pth")
    axes[1].axis("off")
    axes[2].imshow(draw_target(pred3))
    axes[2].set_title("Predicted Targetby pt")
    axes[2].axis("off")

    plt.show()


if __name__ == "__main__":
    img_path = CONVERTED_DATASET_PATH + "/10.jpeg"
    predict_single(img_path)
