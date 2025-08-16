import os
import random
import shutil

import cv2
import numpy as np
import matplotlib.pyplot as plt


def adjust_gamma(image_path, save_as, gamma_factor=1.1):
    """
    Корректирует гамму изображения, показывает результат и сохраняет его.

    :param image_path: путь к изображению
    :param save_as: путь, куда сохранить результат
    :param gamma_factor: коэффициент гаммы (>1 — темнее, <1 — светлее)
    """
    # Загружаем изображение в BGR
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        raise ValueError(f"Не удалось загрузить изображение: {image_path}")

    # Конвертируем в RGB для отображения
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    # Строим таблицу преобразования для коррекции гаммы
    inv_gamma = 1.0 / gamma_factor
    table = np.array([((i / 255.0) ** inv_gamma) * 255
                      for i in np.arange(256)]).astype("uint8")

    # Применяем LUT (Look-Up Table)
    adjusted_rgb = cv2.LUT(image_rgb, table)

    # Сохраняем в BGR (OpenCV формат)
    adjusted_bgr = cv2.cvtColor(adjusted_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(save_as, adjusted_bgr)


def change_brightness(folder_path, output_path):
    files = os.listdir(folder_path)
    files.sort(key=lambda x: int(x.split(".")[0]))
    os.makedirs(output_path, exist_ok=True)
    for file in files:
        if file.endswith(".jpg") or file.endswith(".png"):
            for i in range(3):
                new_filename = file.split(".")[0] + "_" + str(i) + ".png"
                new_filename = output_path + '/' + new_filename

                if i == 0:
                    koef = random.randrange(int(0.5 * 1000), int(0.7 * 1000)) / 1000.0
                elif i == 1:
                    koef = random.randrange(int(0.7 * 1000), int(1.3 * 1000)) / 1000.0
                else:
                    koef = random.randrange(int(1.3 * 1000), int(1.5 * 1000)) / 1000.0
                adjust_gamma(folder_path + '/' + file, new_filename, koef)
                if not os.path.exists(new_filename.replace('.png', '.json')):
                    shutil.copy2(folder_path + '/' + file.replace('.png', '.json'),
                                 new_filename.replace('.png', '.json'))


def change_all(folder_path, output_path):
    dirs = os.listdir(folder_path)
    for entry in dirs:
        change_brightness(folder_path + '/' + entry, output_path + '/' + entry)


# Пример вызова:
if __name__ == '__main__':
    print('Test')
    change_all('../../data/normalized', '../../data/brightness')
    # adjust_gamma("/Users/alex/Projects/Python/archery/data/normalized/50/6.png", gamma_factor=0.5)
    # adjust_gamma("/Users/alex/Projects/Python/archery/data/normalized/50/6.png", gamma_factor=1.5)
