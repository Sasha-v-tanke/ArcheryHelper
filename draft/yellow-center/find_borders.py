import cv2
import numpy as np
import matplotlib.pyplot as plt


def show_yellow_zone_outline(image_path):
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([35, 255, 255])
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    # Найдем все контуры желтых зон
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        print("Жёлтая зона не найдена")
        plt.imshow(img_rgb)
        plt.axis('off')
        plt.show()
        return

    # Выбираем контур с максимальной площадью
    largest_contour = max(contours, key=cv2.contourArea)

    # Создаем пустую маску и рисуем только самый большой контур
    combined_mask = np.zeros_like(mask)
    cv2.drawContours(combined_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)

    # Далее — ищем контур объединённой маски (уже один)
    combined_contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Копия для рисования
    img_draw = img_rgb.copy()

    if combined_contours:
        # Обводим внешний контур жёлтым цветом
        cv2.drawContours(img_draw, combined_contours, -1, (255, 255, 0), 3)

        # Находим минимальный повернутый прямоугольник вокруг объединённых контуров
        all_points = np.vstack(combined_contours)
        rect = cv2.minAreaRect(all_points)
        box = cv2.boxPoints(rect)
        box = box.astype(np.int32)

        # Рисуем прямоугольник (bounding box) красным цветом
        cv2.drawContours(img_draw, [box], 0, (255, 0, 0), 3)

    plt.figure(figsize=(8, 8))
    plt.imshow(img_draw)
    plt.axis('off')
    plt.title('Обводка жёлтой зоны и bounding box')
    plt.show()
