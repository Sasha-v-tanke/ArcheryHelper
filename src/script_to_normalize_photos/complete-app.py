import os
import tkinter as tk
from tkinter import filedialog, Canvas, Button, Frame, Label
from PIL import Image, ImageTk
import numpy as np
import cv2

from normalize_all import run


class App:
    def __init__(self, root, dataset_folder):
        self.root = root
        self.dataset_folder = dataset_folder

        # UI
        self.image_frame = None
        self.main_canvas = None
        self.result_canvas = None
        self.control_frame = None
        self.save_btn = None
        self.status_label = None
        self.display_width = 600
        self.display_height = 600
        self.result_width = 300
        self.result_height = 300

        # data
        self.photo_index = 0
        self.photos = []
        self.image = None
        self.tk_image = None
        self.tk_result_image = None
        self.parameters = dict()

        # points
        self.points = []
        self.dragging_point = None
        self.rect_lines = []

        # start
        self.create_widgets()
        self.load_photos()
        self.load_next_photo()

    def create_widgets(self):
        # Image frame
        self.image_frame = Frame(self.root)
        self.image_frame.pack(side=tk.LEFT, padx=10, pady=10)

        self.main_canvas = Canvas(self.image_frame, width=self.display_width, height=self.display_height, bg='white')
        self.main_canvas.pack()
        self.main_canvas.bind("<Button-1>", self.canvas_click)
        self.main_canvas.bind("<B1-Motion>", self.canvas_drag)
        self.main_canvas.bind("<ButtonRelease-1>", self.canvas_release)

        # Controls frame
        self.control_frame = Frame(self.root)
        self.control_frame.pack(side=tk.RIGHT, padx=10, pady=10)

        self.save_btn = Button(self.control_frame, text="Save", command=self.save, state=tk.NORMAL)
        self.save_btn.pack(pady=5)

        self.status_label = Label(self.control_frame, text="Choose 4 points: top, bottom, left, right")
        self.status_label.pack(pady=10)

        self.result_canvas = Canvas(self.control_frame, width=self.result_width, height=self.result_height, bg='white')
        self.result_canvas.pack(pady=10)

    def canvas_click(self, event):
        if len(self.points) < 4:
            self.points.append((event.x, event.y))
            self.draw_points()
            if len(self.points) == 4:
                self.draw_rectangle()
                self.status_label.config(text="Points chosen. Adjust if needed, then click Save")
        else:
            for i, (px, py) in enumerate(self.points):
                if (event.x - px) ** 2 + (event.y - py) ** 2 <= 36:
                    self.dragging_point = i
                    break

    def canvas_drag(self, event):
        if self.dragging_point is not None and len(self.points) == 4:
            self.points[self.dragging_point] = (event.x, event.y)
            self.draw_points()
            self.draw_rectangle()

    def process_image(self):
        if len(self.points) != 4:
            return

        w, h = self.image.size
        scale_x = w / self.display_width
        scale_y = h / self.display_height

        self.parameters = {
            'points': self.points,
            'scale': (scale_x, scale_x)
        }

        top, bottom, left, right = self.points
        for point in self.points:
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
        img_np = np.array(self.image)
        if img_np.ndim == 2:
            img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
        elif img_np.shape[2] == 4:
            img_np = img_np[:, :, :3]

        aligned_image = cv2.warpPerspective(img_np, M, output_size)

        # Show preview
        self.update(aligned_image)

    def canvas_release(self, event):
        self.dragging_point = None
        self.process_image()

    def draw_points(self):
        self.main_canvas.delete("point")
        for x, y in self.points:
            self.main_canvas.create_oval(x - 5, y - 5, x + 5, y + 5, fill='red', tags="point")

    def draw_rectangle(self):
        for line in self.rect_lines:
            self.main_canvas.delete(line)
        self.rect_lines = []

        if len(self.points) != 4:
            return

        top, bottom, left, right = self.points
        for point in self.points:
            if point[0] < left[0]:
                left = point
            if point[0] > right[0]:
                right = point
            if point[1] < top[1]:
                top = point
            if point[1] > bottom[1]:
                bottom = point

        x1, y1 = left[0], top[1]
        x2, y2 = right[0], bottom[1]

        self.rect_lines.append(self.main_canvas.create_line(x1, y1, x2, y1, fill='blue', width=2))
        self.rect_lines.append(self.main_canvas.create_line(x2, y1, x2, y2, fill='blue', width=2))
        self.rect_lines.append(self.main_canvas.create_line(x2, y2, x1, y2, fill='blue', width=2))
        self.rect_lines.append(self.main_canvas.create_line(x1, y2, x1, y1, fill='blue', width=2))

        self.rect_lines.append(self.main_canvas.create_line(*top, *bottom, fill='green', width=1, dash=(4, 2)))
        self.rect_lines.append(self.main_canvas.create_line(*left, *right, fill='green', width=1, dash=(4, 2)))

    def load_photos(self):
        # path = self.dataset_folder + '/arrow-sequences/versions/1/'
        # folders = [entry for entry in os.scandir(path) if entry.is_dir()]
        # self.photos = [folder.path + '/00.png' for folder in folders]
        self.photos = ["/Users/alex/Projects/Python/archery/data/original_dataset/arrow-sequences/versions/1/11/00.png"]

    def update(self, image_np):
        result_image = Image.fromarray(image_np)
        result_image_scaled = result_image.resize((self.result_width, self.result_height), Image.LANCZOS)
        self.tk_result_image = ImageTk.PhotoImage(result_image_scaled)
        self.result_canvas.config(width=self.result_width, height=self.result_height)
        self.result_canvas.create_image(2.5, 2.5, anchor=tk.NW, image=self.tk_result_image)
        self.result_canvas.create_oval(2.5, 2.5, self.result_width, self.result_height, outline="black",
                                       tags="center_circle")

    def save(self):
        print(self.photos[self.photo_index - 1], self.parameters)
        folder = '/'.join(self.photos[self.photo_index - 1].split('/')[:-1])
        package = self.photos[self.photo_index - 1].split('/')[-2]
        output = '../../data/normalize/' + package
        run(folder, output, self.parameters['points'], self.parameters['scale'])
        self.load_next_photo()

    def load_next_photo(self):
        self.points = []
        self.main_canvas.delete("all")
        self.result_canvas.delete("all")
        self.image = None
        self.tk_image = None
        self.tk_result_image = None
        self.rect_lines = []
        self.parameters = dict()

        if self.photo_index >= len(self.photos):
            print('All photos prepared')
            exit(0)

        self.image = Image.open(self.photos[self.photo_index])
        self.photo_index += 1

        self.image = self.image.resize((self.display_width, self.display_height), Image.LANCZOS)
        self.tk_image = ImageTk.PhotoImage(self.image)

        self.main_canvas.config(width=self.display_width, height=self.display_height)
        self.main_canvas.create_image(2.5, 2.5, anchor=tk.NW, image=self.tk_image)


if __name__ == "__main__":
    dataset_folder = '../../data/original_dataset'

    root = tk.Tk()
    app = App(root, dataset_folder)
    root.mainloop()
