import cv2
import numpy as np
import matplotlib.pyplot as plt


def normalize_target_by_yellow_zone(image_path, output_size=500):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Ошибка: не удалось загрузить изображение {image_path}")
        return
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([35, 255, 255])
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print("Жёлтая зона не найдена")
        plt.imshow(img_rgb)
        plt.axis('off')
        plt.show()
        return

    # Берем только самый большой контур
    largest_contour = max(contours, key=cv2.contourArea)

    # Создаем маску для этого контура
    combined_mask = np.zeros_like(mask)
    cv2.drawContours(combined_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)

    # Находим минимальный повернутый прямоугольник для самого большого контура
    rect = cv2.minAreaRect(largest_contour)
    (cx, cy), (w, h), angle = rect
    cx, cy = float(cx), float(cy)

    # Подгоняем угол и размеры так, чтобы w — ширина по ориентации
    if w < h:
        w, h = h, w
        angle += 90

    # Масштабируем так, чтобы ширина 10 колец поместилась в output_size
    # Если ширина желтой зоны соответствует 2 кольцам,
    # ширина одного кольца = w / 2,
    # значит ширина 10 колец = 5 * w
    scale = output_size / (5 * w)

    # Матрицы преобразований (3x3)
    M_trans1 = np.array([[1, 0, -cx],
                         [0, 1, -cy],
                         [0, 0, 1]], dtype=np.float32)

    angle_rad = -angle * np.pi / 180
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    M_rot = np.array([[cos_a, -sin_a, 0],
                      [sin_a, cos_a, 0],
                      [0, 0, 1]], dtype=np.float32)

    M_scale = np.array([[scale, 0, 0],
                        [0, scale, 0],
                        [0, 0, 1]], dtype=np.float32)

    M_trans2 = np.array([[1, 0, output_size / 2],
                         [0, 1, output_size / 2],
                         [0, 0, 1]], dtype=np.float32)

    M_full_3x3 = M_trans2 @ M_scale @ M_rot @ M_trans1
    M_full = M_full_3x3[:2, :]

    warped = cv2.warpAffine(img_rgb, M_full, (output_size, output_size), flags=cv2.INTER_LINEAR)

    # Для визуализации желтой зоны применяем то же преобразование к маске
    warped_mask = cv2.warpAffine(combined_mask, M_full, (output_size, output_size))
    _, warped_mask_thresh = cv2.threshold(warped_mask, 128, 255, cv2.THRESH_BINARY)

    # Обводим жёлтую зону жёлтым цветом
    contours_warped, _ = cv2.findContours(warped_mask_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours_warped:
        cv2.drawContours(warped, contours_warped, -1, (255, 255, 0), 3)

    plt.figure(figsize=(8, 8))
    plt.imshow(warped)
    plt.axis('off')
    plt.title("Нормализованное изображение с желтой зоной в центре")
    plt.show()
