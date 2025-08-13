import json
import os
import shutil

import cv2
import numpy as np
from PIL import Image

from src.extract_arrows import extract_points_from_original_json
from src.find_all_photos import find_all_photos, get_similar_json


def normalize(filename, output_directory, points, scale, index):
    try:
        original_image = Image.open(filename).convert("RGB")
    except:
        return False
    os.makedirs(output, exist_ok=True)

    # Получаем координаты точек в оригинальном масштабе
    (scale_x, scale_y) = scale
    # Сортируем точки
    points_sorted = sorted(points, key=lambda p: p[1])  # Сначала по Y
    top = min(points_sorted[:2], key=lambda p: p[0])
    bottom = max(points_sorted[2:], key=lambda p: p[0])

    points_sorted = sorted(points, key=lambda p: p[0])  # Теперь по X
    left = min(points_sorted[:2], key=lambda p: p[1])
    right = max(points_sorted[2:], key=lambda p: p[1])

    # Масштабируем координаты
    top = (top[0] * scale_x, top[1] * scale_y)
    bottom = (bottom[0] * scale_x, bottom[1] * scale_y)
    left = (left[0] * scale_x, left[1] * scale_y)
    right = (right[0] * scale_x, right[1] * scale_y)

    # Преобразуем в numpy массивы
    top = np.array(top)
    bottom = np.array(bottom)
    left = np.array(left)
    right = np.array(right)

    # Рассчитываем размер
    size = max(np.linalg.norm(right - left), np.linalg.norm(bottom - top))

    # Точки исходного изображения
    src_points = np.float32([top, bottom, left, right])

    # Точки для выравнивания
    dst_points = np.float32([
        [size / 2, 0],  # Верх
        [size / 2, size],  # Низ
        [0, size / 2],  # Лево
        [size, size / 2]  # Право
    ])

    # Вычисляем матрицу преобразования
    M, _ = cv2.findHomography(src_points, dst_points)

    # Применяем преобразование
    img_np = np.array(original_image)
    if img_np.ndim == 2:  # Если изображение в градациях серого
        img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
    elif img_np.shape[2] == 4:  # Если изображение с альфа-каналом
        img_np = img_np[:, :, :3]

    img_aligned = cv2.warpPerspective(img_np, M, (int(size), int(size)))

    # --- Обработка координат попаданий ---
    try:
        hits = extract_points_from_original_json(get_similar_json(filename))
    except:
        return False
    if not hits:
        return False

    # Преобразуем координаты попаданий (x, y) с помощью матрицы M
    points = np.array(hits, dtype=np.float32).reshape(-1, 1, 2)  # формат для cv2.perspectiveTransform

    aligned_points = cv2.perspectiveTransform(points, M)
    aligned_points = aligned_points.reshape(-1, 2).tolist()  # в обычный список

    # Сохраняем результат
    result_image = Image.fromarray(img_aligned.astype('uint8'), 'RGB')
    out_path = os.path.join(output_directory, f"{index}.png")
    result_image.save(out_path)

    save_path = os.path.join(output_directory, f"{index}.json")

    data = {'arrows': aligned_points}
    with open(save_path, 'w') as f:
        json.dump(data, f, indent=2)
    return True


def run(dataset, output, keypoints, scale, count=-1):
    if os.path.exists(output):
        shutil.rmtree(output)

    print('Looking for photos...')
    images = find_all_photos(dataset)

    print('Find', len(images), 'photos')
    print('Normalizing photos...')
    previous_percent = 0
    for index, image in enumerate(images):
        try:
            result = normalize(image, output, keypoints, scale, index)
            if count > 0 and result:
                count -= 1
                if count == 0:
                    break
        except:
            pass
        if index >= (previous_percent + 5) * 0.01 * len(images):
            previous_percent += 5
            print(previous_percent, end='%\n')
    print('Done!')


def test(dataset, output, keypoints, scale):
    print('Run test')

    run(dataset, output, keypoints, scale, 1)

    from src.show_photo_with_arrows import show_points

    show_points(output + '/1.png', output + '/1.json', custom_json=True)


if __name__ == '__main__':
    # use app to find parameters
    keypoints = [(237, 166), (194, 566), (420, 334), (47, 364)]
    scale = (6.64, 6.64)
    dataset = '../../data/original_dataset'
    output = '../../data/normalized'
    # test(dataset, output, keypoints, scale)
    run(dataset, output, keypoints, scale)
