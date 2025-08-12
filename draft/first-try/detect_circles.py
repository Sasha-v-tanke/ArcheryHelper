import cv2
import numpy as np
import matplotlib.pyplot as plt


def detect_target_rings(image_path):
    # Загрузка изображения
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Конвертация в RGB для matplotlib
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    output = img_rgb.copy()
    h, w = img.shape[:2]

    # Диапазоны цветов в HSV с улучшенными параметрами
    color_ranges = [
        ([20, 100, 100], [30, 255, 255], 'yellow'),  # Желтый
        ([0, 120, 100], [10, 255, 255], 'red'),  # Красный
        ([100, 100, 50], [140, 255, 255], 'blue'),  # Синий
        ([0, 0, 0], [180, 255, 50], 'black'),  # Черный (чувствительнее)
        ([0, 0, 180], [180, 30, 255], 'white')  # Белый (чувствительнее)
    ]

    all_contours = []

    for lower, upper, color_name in color_ranges:
        # Создание маски с улучшенными параметрами
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array(lower), np.array(upper))

        # Улучшение маски
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)

        # Поиск контуров
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 0.01 * h * w:  # Минимальная площадь 1% от изображения
                continue

            # Улучшенная проверка на круглость
            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0:
                continue

            circularity = 4 * np.pi * area / (perimeter ** 2)
            if circularity < 0.5:
                continue

            all_contours.append((cnt, color_name))

    if not all_contours:
        print("Кольца мишени не обнаружены")
        return

    # Находим точный центр (медиана центров всех контуров)
    centers = []
    for cnt, _ in all_contours:
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            centers.append((M["m10"] / M["m00"], M["m01"] / M["m00"]))

    if not centers:
        print("Не удалось определить центр")
        return

    center = np.median(centers, axis=0)
    center = (int(center[0]), int(center[1]))

    # Сортируем кольца по расстоянию от центра (внутрь -> наружу)
    all_contours.sort(key=lambda x: cv2.pointPolygonTest(x[0], center, True), reverse=True)

    # Рисуем все кольца с разделительными линиями
    for i, (cnt, color_name) in enumerate(all_contours):
        color_map = {
            'yellow': (255, 255, 0),
            'red': (255, 0, 0),
            'blue': (0, 0, 255),
            'black': (30, 30, 30),
            'white': (255, 255, 255)
        }
        color = color_map.get(color_name, (0, 255, 0))
        color = (0, 255, 0)

        # Рисуем внешнюю границу
        cv2.drawContours(output, [cnt], -1, color, 3)

        # Рисуем внутреннюю границу (разделительную линию)
        if i < len(all_contours) - 1:
            inner_cnt = all_contours[i + 1][0]
            epsilon = 0.001 * cv2.arcLength(inner_cnt, True)
            approx = cv2.approxPolyDP(inner_cnt, epsilon, True)
            cv2.drawContours(output, [approx], -1, (255, 255, 255), 2)

    # Рисуем точный центр
    cv2.circle(output, center, 7, (0, 255, 0), -1)
    cv2.circle(output, center, 15, (0, 255, 0), 2)

    # Вывод информации
    print(f"Найдено {len(all_contours)} колец мишени")
    print(f"Точный центр: {center}")

    # Отображение
    plt.figure(figsize=(12, 8))
    plt.imshow(output)
    plt.title("Точное обнаружение колец мишени", fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.show()


# Обработка изображения
detect_target_rings("09.png")
