import cv2
import numpy as np
import matplotlib.pyplot as plt


def process_target_image(path, square_size=1000):
    image = cv2.imread(path)
    if image is None:
        print(f"[ОШИБКА] Не удалось открыть файл: {path}")
        return
    orig = image.copy()
    h, w = image.shape[:2]
    image_area = h * w

    # ---- HSV фильтр для белого (как было изначально) ----
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_white = np.array([0, 0, 160])
    upper_white = np.array([180, 60, 255])
    mask = cv2.inRange(hsv, lower_white, upper_white)

    # ---- Морфология ----
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)

    # ---- Контуры ----
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best_contour = None
    best_area = 0

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < image_area * 0.3:  # минимум 30% кадра
            continue
        if area > best_area:
            best_area = area
            best_contour = contour

    if best_contour is None:
        print(f"[ОШИБКА] Не найден подходящий контур: {path}")
        return

    # ---- Прямоугольник для перспективного преобразования ----
    rect = cv2.minAreaRect(best_contour)  # возвращает центр, размеры, угол
    box = cv2.boxPoints(rect)  # 4 точки прямоугольника
    box = np.array(box, dtype="float32")

    # ---- Упорядочиваем точки ----
    def order_points(pts):
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect

    rect_pts = order_points(box)
    (tl, tr, br, bl) = rect_pts

    # ---- Перспективное преобразование ----
    width = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))
    height = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))
    dst = np.array([[0, 0], [width - 1, 0],
                    [width - 1, height - 1], [0, height - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect_pts, dst)
    warped = cv2.warpPerspective(orig, M, (width, height))

    # ---- Вписываем в квадрат ----
    max_side = max(width, height)
    scale = square_size / max_side
    new_w, new_h = int(width * scale), int(height * scale)
    resized = cv2.resize(warped, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    square_img = np.ones((square_size, square_size, 3), dtype=np.uint8) * 255
    offset_x = (square_size - new_w) // 2
    offset_y = (square_size - new_h) // 2
    square_img[offset_y:offset_y + new_h, offset_x:offset_x + new_w] = resized

    # ---- Контур для отладки ----
    debug_img = orig.copy()
    cv2.drawContours(debug_img, [best_contour], -1, (0, 0, 255), 3)

    def cv2_to_rgb(img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # ---- Вывод ----
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 2, 1)
    plt.imshow(cv2_to_rgb(orig))
    plt.title("Оригинал")
    plt.axis("off")

    plt.subplot(2, 2, 2)
    plt.imshow(mask, cmap="gray")
    plt.title("Маска белого листа")
    plt.axis("off")

    plt.subplot(2, 2, 3)
    plt.imshow(cv2_to_rgb(debug_img))
    plt.title("Найденный контур")
    plt.axis("off")

    plt.subplot(2, 2, 4)
    plt.imshow(cv2_to_rgb(square_img))
    plt.title("Мишень вписана в квадрат")
    plt.axis("off")

    plt.tight_layout()
    plt.show()


# ---- Пример запуска ----
files = [
    "/Users/alex/Projects/Python/archery/draft/09.png",
    "/Users/alex/Projects/Python/archery/data/new_dataset/158.jpeg",
    "/Users/alex/Projects/Python/archery/draft/example.jpg"
]
if __name__ == "__main__":
    for f in files:
        process_target_image(f)
