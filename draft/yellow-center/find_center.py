import cv2
import numpy as np
import matplotlib.pyplot as plt


def show_yellow_zone_filled(image_path):
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

    # Создаем пустую маску, куда будем объединять все желтые области
    combined_mask = np.zeros_like(mask)

    # Объединим все контуры в маску
    cv2.drawContours(combined_mask, contours, -1, 255, thickness=cv2.FILLED)

    # Создаем копию для рисования
    img_draw = img_rgb.copy()

    # Найдем контур объединённой маски для обводки
    combined_contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if combined_contours:
        # Обводим внешний контур желтым цветом
        cv2.drawContours(img_draw, combined_contours, -1, (255, 255, 0), 3)

    # Закрашиваем всю объединённую область зелёным цветом
    img_draw[combined_mask == 255] = (0, 255, 0)

    plt.figure(figsize=(8, 8))
    plt.imshow(img_draw)
    plt.axis('off')
    plt.title('Объединённая и залитая зелёным жёлтая зона')
    plt.show()
