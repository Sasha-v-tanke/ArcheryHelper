import cv2
import numpy as np
import matplotlib.pyplot as plt


def rotate_image_and_points(image_path, save_as, angle, points):
    """
    Поворачивает изображение и заданные точки на угол.

    :param image_path: путь к изображению
    :param save_as: путь для сохранения результата
    :param angle: угол (градусы, по часовой стрелке)
    :param points: список точек [(x1, y1), (x2, y2), ...]
    :return: список новых точек
    """
    # Загружаем изображение
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Не удалось загрузить изображение: {image_path}")

    h, w = image.shape[:2]
    center = (w // 2, h // 2)

    # Матрица поворота
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)

    # Вычисляем новый размер, чтобы ничего не обрезалось
    cos = abs(rot_mat[0, 0])
    sin = abs(rot_mat[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))

    # Корректируем матрицу для смещения центра
    rot_mat[0, 2] += (new_w / 2) - center[0]
    rot_mat[1, 2] += (new_h / 2) - center[1]

    # Поворачиваем изображение
    rotated = cv2.warpAffine(image, rot_mat, (new_w, new_h))

    # Преобразуем точки
    points_np = np.array(points, dtype=np.float32)
    ones = np.ones((points_np.shape[0], 1), dtype=np.float32)
    points_hom = np.hstack([points_np, ones])
    rotated_points = (rot_mat @ points_hom.T).T

    # Сохраняем
    cv2.imwrite(save_as, rotated)

    # Визуализация
    rotated_rgb = cv2.cvtColor(rotated, cv2.COLOR_BGR2RGB)
    plt.imshow(rotated_rgb)
    plt.scatter(rotated_points[:, 0], rotated_points[:, 1], c='red', s=40)
    plt.title(f"Поворот {angle}°")
    plt.axis("off")
    plt.show()

    return rotated_points.tolist()

# Пример:
# new_points = rotate_image_and_points(
#     "input.jpg", "rotated.jpg", 30,
#     points=[(100, 150), (200, 250)]
# )
# print("Новые координаты:", new_points)
