import tkinter as tk
from tkinter import messagebox
import cv2
from PIL import Image, ImageTk
import glob
import os
import numpy as np

from path_manager import NEW_DATASET_PATH, NEW_NORMALIZED_DATASET

# === Настройки директорий ===
INPUT_DIR = NEW_DATASET_PATH  # папка с исходными фото
OUTPUT_DIR = NEW_NORMALIZED_DATASET  # папка для сохранённых
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === Глобальные переменные ===
image_files = glob.glob(os.path.join(INPUT_DIR, "*.jpeg"))
image_files = [img for img in image_files if not os.path.exists(OUTPUT_DIR + '/' + img.split('/')[-1])]
current_index = 0
points = []
point_handles = []
current_img = None
current_img_name = ""
normalized_img = None

# === tkinter окно ===
root = tk.Tk()
root.title("Нормализация по четырём точкам")

frame = tk.Frame(root)
frame.pack()

canvas_src = tk.Canvas(frame, width=600, height=600, bg="gray")
canvas_src.pack(side=tk.LEFT, padx=5, pady=5)

canvas_dst = tk.Canvas(frame, width=600, height=600, bg="gray")
canvas_dst.pack(side=tk.LEFT, padx=5, pady=5)


def load_image():
    global current_img, current_img_name, tk_img_src, points, point_handles, normalized_img
    points = []
    point_handles = []
    normalized_img = None
    canvas_src.delete("all")
    canvas_dst.delete("all")

    if current_index >= len(image_files):
        messagebox.showinfo("Готово", "Больше изображений нет")
        return

    current_img_name = os.path.basename(image_files[current_index])
    cv_img = cv2.imread(image_files[current_index])
    cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)

    # уменьшаем исходное фото до 600x600
    cv_img = cv2.resize(cv_img, (600, 600), interpolation=cv2.INTER_AREA)

    current_img = cv_img
    pil_img = Image.fromarray(current_img)
    tk_img_src = ImageTk.PhotoImage(pil_img)
    canvas_src.create_image(0, 0, anchor="nw", image=tk_img_src)


def draw_points():
    global point_handles
    canvas_src.delete("point")
    canvas_src.delete("line")
    n = len(points)
    for (x, y) in points:
        h = canvas_src.create_oval(x - 5, y - 5, x + 5, y + 5, fill="red", tags="point")
        point_handles.append(h)
    # рисуем линии между точками
    if n >= 2:
        for i in range(n - 1):
            canvas_src.create_line(points[i][0], points[i][1], points[i + 1][0], points[i + 1][1], fill="yellow",
                                   width=2, tags="line")
        if n == 4:
            canvas_src.create_line(points[3][0], points[3][1], points[0][0], points[0][1], fill="yellow", width=2,
                                   tags="line")


def find_nearest_point(x, y):
    for i, (px, py) in enumerate(points):
        if abs(px - x) < 10 and abs(py - y) < 10:
            return i
    return None


selected_point = None


def mouse_down(event):
    global selected_point, points
    idx = find_nearest_point(event.x, event.y)
    if idx is not None:
        selected_point = idx
    elif len(points) < 4:
        points.append((event.x, event.y))
        draw_points()


def mouse_move(event):
    global points, selected_point
    if selected_point is not None:
        points[selected_point] = (event.x, event.y)
        draw_points()


def mouse_up(event):
    global selected_point
    selected_point = None
    normalize_image()


def order_points(pts):
    # преобразуем в numpy
    pts = np.array(pts, dtype="float32")

    # сумма координат -> левый верхний (min), правый нижний (max)
    s = pts.sum(axis=1)
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]

    # разность координат -> правый верхний (min), левый нижний (max)
    diff = np.diff(pts, axis=1)
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]

    return np.array([tl, tr, br, bl], dtype="float32")


def normalize_image():
    global normalized_img, tk_img_dst
    if len(points) != 4:
        return

    # упорядочиваем точки
    pts1 = order_points(points)

    # целевой квадрат 600x600
    w, h = 600, 600
    pts2 = np.float32([[0, 0], [w, 0], [w, h], [0, h]])

    # матрица трансформации
    M = cv2.getPerspectiveTransform(pts1, pts2)
    warped = cv2.warpPerspective(current_img, M, (w, h))

    normalized_img = warped
    pil_img = Image.fromarray(warped)
    tk_img_dst = ImageTk.PhotoImage(pil_img)
    canvas_dst.delete("all")
    canvas_dst.create_image(0, 0, anchor="nw", image=tk_img_dst)


def save_image():
    if normalized_img is None:
        # messagebox.showwarning("Ошибка", "Нет нормализованного изображения")
        return
    save_path = os.path.join(OUTPUT_DIR, current_img_name)
    cv2.imwrite(save_path, cv2.cvtColor(normalized_img, cv2.COLOR_RGB2BGR))
    next_image()
    # messagebox.showinfo("Сохранено", f"Файл сохранён: {save_path}")


def next_image():
    global current_index
    current_index += 1
    load_image()


# === кнопки ===
btn_frame = tk.Frame(root)
btn_frame.pack()

tk.Button(btn_frame, text="Нормализовать", command=normalize_image).pack(side=tk.LEFT, padx=5)
tk.Button(btn_frame, text="Сохранить", command=save_image).pack(side=tk.LEFT, padx=5)
tk.Button(btn_frame, text="Далее", command=next_image).pack(side=tk.LEFT, padx=5)

canvas_src.bind("<Button-1>", mouse_down)
canvas_src.bind("<B1-Motion>", mouse_move)
canvas_src.bind("<ButtonRelease-1>", mouse_up)

load_image()

root.mainloop()
