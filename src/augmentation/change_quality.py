import os
import shutil

import cv2
from matplotlib import pyplot as plt

from src.extract_arrows import extract_arrow_from_custom_json, save_json
from src.show_photo_with_arrows import show_points


def degrade_image(image_path, save_as, scale_factor=0.5, quality=30):
    """
    Ухудшает изображение: уменьшает разрешение и сохраняет с низким качеством JPEG.

    :param image_path: путь к исходному изображению
    :param save_as: путь для сохранения результата
    :param scale_factor: во сколько раз уменьшить размеры (0.5 = в 2 раза меньше)
    :param quality: качество JPEG (0–100, меньше = хуже)
    """
    # Загружаем изображение
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Не удалось загрузить изображение: {image_path}")

    # Уменьшаем разрешение
    new_w = max(1, int(image.shape[1] * scale_factor))
    new_h = max(1, int(image.shape[0] * scale_factor))
    small_img = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Сохраняем с низким качеством
    cv2.imwrite(save_as, small_img, [cv2.IMWRITE_JPEG_QUALITY, quality])


def degrade_json(filename, output, scale_factor=0.5):
    arrows = extract_arrow_from_custom_json(filename)
    arrows = [(e[0] * scale_factor, e[1] * scale_factor) for e in arrows]
    save_json(output, {'arrows': arrows})
    # print(arrows)


def degrade_dir(folder_path, output_path):
    files = os.listdir(folder_path)
    files.sort(key=lambda x: int(x.split(".")[0]))
    os.makedirs(output_path, exist_ok=True)
    for file in files:
        if file.endswith(".png"):
            filename = folder_path + '/' + file
            output = output_path + '/' + file
            degrade_image(filename, output)
            degrade_json(filename.replace('.png', '.json'), output.replace('.png', '.json'))


def degrade_all(folder_path, output_path):
    dirs = os.listdir(folder_path)
    os.makedirs(output_path, exist_ok=True)
    for entry in dirs:
        degrade_dir(folder_path + '/' + entry, output_path + '/' + entry)


if __name__ == '__main__':
    print('Test')
    # degrade_all('../../data/brightness',
    #             '../../data/quality')
    show_points('/Users/alex/Projects/Python/archery/data/normalized/61/5.png',
                '/Users/alex/Projects/Python/archery/data/normalized/61/5.json',
                True)
    show_points('/Users/alex/Projects/Python/archery/data/brightness/61/5_0.png',
                '/Users/alex/Projects/Python/archery/data/brightness/61/5_0.json',
                True)
    show_points('/Users/alex/Projects/Python/archery/data/quality/61/5_0.png',
                '/Users/alex/Projects/Python/archery/data/quality/61/5_0.json',
                True, (0, 255, 0), 1)
