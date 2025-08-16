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
        image = Image.open(filename).convert("RGB")
        arrows_scale = (image.width / 600, image.height / 600)
        image = image.resize((600, 600), Image.LANCZOS)  # todo: make parameter for (600, 600)
    except:
        return False
    os.makedirs(output_directory, exist_ok=True)

    # Получаем координаты точек в оригинальном масштабе
    (scale_x, scale_y) = scale
    top, bottom, left, right = points
    # print(scale, points)
    for point in points:
        if point[0] < left[0]:
            left = point
        if point[0] > right[0]:
            right = point
        if point[1] < top[1]:
            top = point
        if point[1] > bottom[1]:
            bottom = point

    # Scale to original coords
    top = (top[0] * scale_x, top[1] * scale_y)
    bottom = (bottom[0] * scale_x, bottom[1] * scale_y)
    left = (left[0] * scale_x, left[1] * scale_y)
    right = (right[0] * scale_x, right[1] * scale_y)

    src_points = np.float32([top, bottom, left, right])
    dst_points = np.float32([
        [0.5, 0], [0.5, 1], [0, 0.5], [1, 0.5]
    ]) * max(np.linalg.norm(np.array(right) - np.array(left)),
             np.linalg.norm(np.array(bottom) - np.array(top)))

    # Ensure no extra white space by calculating bounding size
    output_size = int(dst_points[:, 0].max()), int(dst_points[:, 1].max())

    M, _ = cv2.findHomography(src_points, dst_points)
    # print(M)
    img_np = np.array(image)
    if img_np.ndim == 2:
        img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
    elif img_np.shape[2] == 4:
        img_np = img_np[:, :, :3]

    aligned_image = cv2.warpPerspective(img_np, M, output_size)

    # --- Обработка координат попаданий ---
    try:
        hits = extract_points_from_original_json(get_similar_json(filename))
    except:
        return False
    if not hits:
        return False

    # Преобразуем координаты попаданий (x, y) с помощью матрицы M
    hits = [(e[0] / arrows_scale[0], e[1] / arrows_scale[1]) for e in hits]
    points = np.array(hits, dtype=np.float32).reshape(-1, 1, 2)  # формат для cv2.perspectiveTransform

    aligned_points = cv2.perspectiveTransform(points, M)
    aligned_points = aligned_points.reshape(-1, 2).tolist()  # в обычный список

    # Сохраняем результат
    result_image = Image.fromarray(aligned_image.astype('uint8'), 'RGB')
    out_path = os.path.join(output_directory, f"{index}.png")
    result_image.save(out_path)

    save_path = os.path.join(output_directory, f"{index}.json")

    data = {'arrows': aligned_points}
    with open(save_path, 'w') as f:
        json.dump(data, f, indent=2)
    return True


def run(dataset, output, keypoints, scale, count=-1):
    # if os.path.exists(output):
    #     shutil.rmtree(output)

    print('Looking for photos...')
    images = find_all_photos(dataset)

    print('Find', len(images), 'photos')
    print('Normalizing photos...')
    previous_percent = 0
    for index, image in enumerate(images):
        result = normalize(image, output, keypoints, scale, index)
        if count > 0 and result:
            count -= 1
            if count == 0:
                break

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
