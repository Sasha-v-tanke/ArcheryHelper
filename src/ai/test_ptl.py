import cv2
import numpy as np
import torch
import os
from PIL import Image
from matplotlib import pyplot as plt
from torchvision import transforms

from path_manager import MODELS
from src.ai.config import IMG_SIZE

PTL_PATH = os.path.join(MODELS, "model.ptl")

# 1️⃣ Проверяем существование файла
if not os.path.isfile(PTL_PATH):
    raise FileNotFoundError(f"Файл {PTL_PATH} не найден!")

print(f"[INFO] Файл {PTL_PATH} найден.")

# 2️⃣ Загружаем модель
try:
    model = torch.jit.load(PTL_PATH, map_location="cpu")
    print("[INFO] Модель успешно загружена через torch.jit.load")
except Exception as e:
    raise RuntimeError(f"[ERROR] Ошибка загрузки модели: {e}")

# 3️⃣ Загружаем картинку и готовим input
try:
    model.eval()

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    img = Image.open("/Users/alex/Projects/Python/archery/data/converted/10.jpeg").convert("RGB")
    w, h = img.size
    print(w, h)
    # img_cropped = img.crop((left, top, right, bottom))

    # 3️⃣ Resize 256x256 (BILINEAR)
    # img_resized = img_cropped.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)

    # 4️⃣ В numpy и нормализация
    img_np = np.array(img).astype(np.float32) / 255.0
    img_norm = (img_np - mean) / std

    # 5️⃣ HWC → CHW + batch
    tensor = np.transpose(img_norm, (2, 0, 1))[None, ...]
    input_tensor = torch.from_numpy(tensor).float()
    first_channel = tensor[0, 0].flatten()[:5]
    print("Первые 5 значений канала R:", first_channel)
    # Покажем картинку
    # img_display = np.transpose(input_tensor[0].numpy(), (1, 2, 0)) * std + mean
    # img_display = np.clip(img_display, 0, 1)
    # plt.imshow(img_display)
    # plt.title("Преобразованное изображение")
    # plt.axis("off")
    # plt.show()

    # 4️⃣ Прогоняем через модель
    with torch.no_grad():
        output = model(input_tensor)

    if isinstance(output, torch.Tensor):
        print(f"[INFO] Модель успешно выполнена. Output shape: {output.shape}")
        print(output)
    else:
        print(f"[INFO] Модель вернула результат не в виде Tensor: {output}")

except Exception as e:
    raise RuntimeError(f"[ERROR] Ошибка выполнения модели: {e}")

print("[INFO] Проверка модели PTL завершена успешно!")

# Первые 30 значений канала R: [1.2042983, 1.2042983, 1.2042983, 1.2042983, 1.1871736, 1.1871736, 1.1871736, 1.1700488, 1.1700488, 1.1529241, 1.1357993, 1.1186745, 1.1186745, 1.1186745, 1.1186745, 1.1186745, 1.1357993, 1.1529241, 1.1700488, 1.1871736, 1.2042983, 1.221423, 1.221423, 1.221423, 1.221423, 1.2042983, 1.2042983, 1.2042983, 1.1871736, 1.1700488]
#  Выход модели: -0.096842706, 67.32509, 0.10071109, 31.681873, 0.18095663, 26.051353, -12.432594, 32.80282, -12.586867, 32.899834, -12.415337, 32.91286, -12.377387, 32.989983, -12.466174, 32.88127, -12.441792, 32.90059, -12.437213, 32.81659
# 0.044039357, 4.277527, -0.017455619, 4.74417, 0.05127637, 4.2936716, -1.5891137, 4.1659822, -1.6371455, 4.0639434, -1.5937941, 4.0980835, -1.6229973, 4.0814967, -1.570068, 4.1106324, -1.621376, 4.161525, -1.624518, 4.098395
