import tkinter as tk
from tkinter import filedialog, Canvas, Button, Frame, Label
from PIL import Image, ImageTk, ImageDraw
import numpy as np
import cv2


class TargetAlignmentApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Выравнивание мишени")

        # Переменные
        self.image_path = ""
        self.points = []
        self.dragging_point = None
        self.rect_lines = []
        self.guide_lines = []

        # Создание интерфейса
        self.create_widgets()

    def create_widgets(self):
        # Фрейм для изображения
        self.image_frame = Frame(self.root)
        self.image_frame.pack(side=tk.LEFT, padx=10, pady=10)

        # Холст для изображения
        self.canvas = Canvas(self.image_frame, width=600, height=600, bg='white')
        self.canvas.pack()
        self.canvas.bind("<Button-1>", self.canvas_click)
        self.canvas.bind("<B1-Motion>", self.canvas_drag)
        self.canvas.bind("<ButtonRelease-1>", self.canvas_release)

        # Фрейм для управления
        self.control_frame = Frame(self.root)
        self.control_frame.pack(side=tk.RIGHT, padx=10, pady=10)

        # Кнопки
        self.load_btn = Button(self.control_frame, text="Загрузить изображение", command=self.load_image)
        self.load_btn.pack(pady=5)

        self.next_btn = Button(self.control_frame, text="Далее", command=self.process_image, state=tk.DISABLED)
        self.next_btn.pack(pady=5)

        self.status_label = Label(self.control_frame, text="Выберите 4 точки: верх, низ, лево, право")
        self.status_label.pack(pady=10)

        # Холст для результата
        self.result_canvas = Canvas(self.control_frame, width=300, height=300, bg='white')
        self.result_canvas.pack(pady=10)

    def load_image(self):
        self.image_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
        if self.image_path:
            self.reset_state()
            self.display_image()

    def reset_state(self):
        self.points = []
        self.rect_lines = []
        self.guide_lines = []
        self.dragging_point = None
        self.canvas.delete("all")
        self.result_canvas.delete("all")
        self.next_btn.config(state=tk.DISABLED)
        self.status_label.config(text="Выберите 4 точки: верх, низ, лево, право")

    def display_image(self):
        self.original_image = Image.open(self.image_path)
        self.tk_image = ImageTk.PhotoImage(self.original_image)

        # Масштабирование изображения для отображения
        width, height = self.original_image.size
        ratio = min(600 / width, 600 / height)
        self.display_width = int(width * ratio)
        self.display_height = int(height * ratio)

        self.display_image_scaled = self.original_image.resize((self.display_width, self.display_height), Image.LANCZOS)
        self.tk_image_scaled = ImageTk.PhotoImage(self.display_image_scaled)

        self.canvas.config(width=self.display_width, height=self.display_height)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image_scaled)

    def canvas_click(self, event):
        if len(self.points) < 4:
            # Добавляем новую точку
            x, y = event.x, event.y
            self.points.append((x, y))
            self.draw_points()

            if len(self.points) == 4:
                self.draw_rectangle()
                self.next_btn.config(state=tk.NORMAL)
                self.status_label.config(text="Точки выбраны. Можете перемещать их или нажать 'Далее'")
        else:
            # Проверяем, кликнули ли по существующей точке
            for i, (px, py) in enumerate(self.points):
                if (event.x - px) ** 2 + (event.y - py) ** 2 <= 36:  # 6px радиус
                    self.dragging_point = i
                    break

    def canvas_drag(self, event):
        if self.dragging_point is not None and len(self.points) == 4:
            # Перемещаем точку
            self.points[self.dragging_point] = (event.x, event.y)
            self.draw_points()
            self.draw_rectangle()

    def canvas_release(self, event):
        self.dragging_point = None

    def draw_points(self):
        self.canvas.delete("point")
        for x, y in self.points:
            self.canvas.create_oval(x - 5, y - 5, x + 5, y + 5, fill='red', tags="point")

    def draw_rectangle(self):
        # Удаляем предыдущие линии
        for line in self.rect_lines + self.guide_lines:
            self.canvas.delete(line)
        self.rect_lines = []
        self.guide_lines = []

        if len(self.points) != 4:
            return

        # Сортируем точки: верх, низ, лево, право
        points_sorted = sorted(self.points, key=lambda p: p[1])  # Сначала по Y
        top = min(points_sorted[:2], key=lambda p: p[0])
        bottom = max(points_sorted[2:], key=lambda p: p[0])

        points_sorted = sorted(self.points, key=lambda p: p[0])  # Теперь по X
        left = min(points_sorted[:2], key=lambda p: p[1])
        right = max(points_sorted[2:], key=lambda p: p[1])

        # Рисуем прямоугольник
        x1 = left[0]
        y1 = top[1]
        x2 = right[0]
        y2 = bottom[1]

        self.rect_lines.append(self.canvas.create_line(x1, y1, x2, y1, fill='blue', width=2))  # Верх
        self.rect_lines.append(self.canvas.create_line(x2, y1, x2, y2, fill='blue', width=2))  # Право
        self.rect_lines.append(self.canvas.create_line(x2, y2, x1, y2, fill='blue', width=2))  # Низ
        self.rect_lines.append(self.canvas.create_line(x1, y2, x1, y1, fill='blue', width=2))  # Лево

        # Рисуем направляющие линии
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        self.guide_lines.append(
            self.canvas.create_line(x1, center_y, x2, center_y, fill='green', width=1, dash=(4, 2)))  # Горизонталь
        self.guide_lines.append(
            self.canvas.create_line(center_x, y1, center_x, y2, fill='green', width=1, dash=(4, 2)))  # Вертикаль

    def process_image(self):
        if len(self.points) != 4:
            return

        # Получаем координаты точек в оригинальном масштабе
        width, height = self.original_image.size
        scale_x = width / self.display_width
        scale_y = height / self.display_height

        # Сортируем точки
        points_sorted = sorted(self.points, key=lambda p: p[1])  # Сначала по Y
        top = min(points_sorted[:2], key=lambda p: p[0])
        bottom = max(points_sorted[2:], key=lambda p: p[0])

        points_sorted = sorted(self.points, key=lambda p: p[0])  # Теперь по X
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
        img_np = np.array(self.original_image)
        if img_np.ndim == 2:  # Если изображение в градациях серого
            img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
        elif img_np.shape[2] == 4:  # Если изображение с альфа-каналом
            img_np = img_np[:, :, :3]

        img_aligned = cv2.warpPerspective(img_np, M, (int(size), int(size)))

        # Отображаем результат
        self.show_result(img_aligned)

    def show_result(self, image_np):
        # Конвертируем numpy array в изображение PIL
        result_image = Image.fromarray(image_np)

        # Масштабируем для отображения
        width, height = result_image.size
        ratio = min(300 / width, 300 / height)
        display_width = int(width * ratio)
        display_height = int(height * ratio)

        result_image_scaled = result_image.resize((display_width, display_height), Image.LANCZOS)
        self.tk_result_image = ImageTk.PhotoImage(result_image_scaled)

        self.result_canvas.config(width=display_width, height=display_height)
        self.result_canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_result_image)

        # Сохраняем выровненное изображение
        result_image.save("aligned_target.png")
        self.status_label.config(text="Результат сохранен как aligned_target.png")


if __name__ == "__main__":
    root = tk.Tk()
    app = TargetAlignmentApp(root)
    root.mainloop()
