import tkinter as tk
from PIL import Image, ImageTk
import cv2
import numpy as np

# Список файлов
files = [
    "/Users/alex/Projects/Python/archery/draft/09.png",
    "/Users/alex/Projects/Python/archery/data/new_dataset/158.jpeg",
    "/Users/alex/Projects/Python/archery/draft/example.jpg"
]

MAX_DISPLAY_SIZE = 300  # макс ширина/высота для отображения


class HSVClickViewer:
    def __init__(self, root, file_paths):
        self.root = root
        self.root.title("HSV Pixel Viewer - Click Mode")
        self.file_paths = file_paths

        self.orig_images = []  # оригинальные изображения
        self.images_tk = []  # уменьшенные для Tkinter
        self.scales = []  # коэффициенты масштабирования

        self.label_hsv = tk.Label(root, text="HSV: ", font=("Arial", 14))
        self.label_hsv.pack(side="top")

        self.canvas_frame = tk.Frame(root)
        self.canvas_frame.pack()

        self.load_and_prepare_images()
        self.display_images()

    def load_and_prepare_images(self):
        for path in self.file_paths:
            img_cv = cv2.imread(path)
            if img_cv is None:
                print(f"[Ошибка] Не удалось открыть: {path}")
                continue
            img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
            h, w = img_cv.shape[:2]
            scale = min(MAX_DISPLAY_SIZE / w, MAX_DISPLAY_SIZE / h, 1.0)
            new_w, new_h = int(w * scale), int(h * scale)
            img_resized = cv2.resize(img_cv, (new_w, new_h), interpolation=cv2.INTER_AREA)

            self.orig_images.append(img_cv)
            self.images_tk.append(ImageTk.PhotoImage(Image.fromarray(img_resized)))
            self.scales.append(scale)

    def display_images(self):
        for i, img_tk in enumerate(self.images_tk):
            label = tk.Label(self.canvas_frame, image=img_tk, borderwidth=2, relief="solid")
            label.grid(row=0, column=i, padx=5)
            label.bind("<Button-1>", lambda e, idx=i: self.on_click(e, idx))

    def on_click(self, event, idx):
        scale = self.scales[idx]
        x_disp, y_disp = event.x, event.y
        x_orig = min(int(x_disp / scale + 0.5), self.orig_images[idx].shape[1] - 1)
        y_orig = min(int(y_disp / scale + 0.5), self.orig_images[idx].shape[0] - 1)

        pixel = self.orig_images[idx][y_orig, x_orig]
        pixel_hsv = cv2.cvtColor(np.uint8([[pixel]]), cv2.COLOR_RGB2HSV)[0][0]
        h, s, v = pixel_hsv
        self.label_hsv.config(text=f"HSV: ({h}, {s}, {v})")
        print(h, s, v)


if __name__ == "__main__":
    root = tk.Tk()
    app = HSVClickViewer(root, files)
    root.mainloop()
