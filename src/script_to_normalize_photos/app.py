import tkinter as tk
from tkinter import filedialog, Canvas, Button, Frame, Label
from PIL import Image, ImageTk
import numpy as np
import cv2


# all the photos in the dataset are similar
# so you can find parameters for normalizing for only one photo
# and use them for all

# remember

class TargetAlignmentApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Target Alignment")

        self.image_path = ""
        self.points = []
        self.dragging_point = None
        self.rect_lines = []
        self.guide_lines = []
        self.aligned_image = None  # store last aligned image for saving

        self.saved_parameters = None

        self.create_widgets()

    def create_widgets(self):
        # Image frame
        self.image_frame = Frame(self.root)
        self.image_frame.pack(side=tk.LEFT, padx=10, pady=10)

        self.canvas = Canvas(self.image_frame, width=600, height=600, bg='white')
        self.canvas.pack()
        self.canvas.bind("<Button-1>", self.canvas_click)
        self.canvas.bind("<B1-Motion>", self.canvas_drag)
        self.canvas.bind("<ButtonRelease-1>", self.canvas_release)

        # Controls frame
        self.control_frame = Frame(self.root)
        self.control_frame.pack(side=tk.RIGHT, padx=10, pady=10)

        self.load_btn = Button(self.control_frame, text="Load Image", command=self.load_image)
        self.load_btn.pack(pady=5)

        self.update_btn = Button(self.control_frame, text="Update", command=self.process_image, state=tk.DISABLED)
        self.update_btn.pack(pady=5)

        self.save_btn = Button(self.control_frame, text="Save", command=self.save_image, state=tk.DISABLED)
        self.save_btn.pack(pady=5)

        self.status_label = Label(self.control_frame, text="Choose 4 points: top, bottom, left, right")
        self.status_label.pack(pady=10)

        self.result_canvas = Canvas(self.control_frame, width=300, height=300, bg='white')
        self.result_canvas.pack(pady=10)

    def load_image(self):
        self.image_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
        if self.image_path:
            self.reset_state()
            self.display_image()

    def reset_state(self):
        self.points.clear()
        self.rect_lines.clear()
        self.guide_lines.clear()
        self.dragging_point = None
        self.canvas.delete("all")
        self.result_canvas.delete("all")
        self.update_btn.config(state=tk.DISABLED)
        self.save_btn.config(state=tk.DISABLED)
        self.status_label.config(text="Choose 4 points: top, bottom, left, right")
        self.aligned_image = None

    def display_image(self):
        self.original_image = Image.open(self.image_path)
        width, height = self.original_image.size
        ratio = min(600 / width, 600 / height)
        self.display_width = round(width * ratio)
        self.display_height = round(height * ratio)

        self.display_image_scaled = self.original_image.resize((self.display_width, self.display_height), Image.LANCZOS)
        self.tk_image_scaled = ImageTk.PhotoImage(self.display_image_scaled)

        self.canvas.config(width=self.display_width, height=self.display_height)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image_scaled)

    def canvas_click(self, event):
        if len(self.points) < 4:
            self.points.append((event.x, event.y))
            self.draw_points()
            if len(self.points) == 4:
                self.draw_rectangle()
                self.update_btn.config(state=tk.NORMAL)
                self.status_label.config(text="Points chosen. Adjust if needed, then click Update")
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

    def canvas_release(self, event):
        self.dragging_point = None

    def draw_points(self):
        self.canvas.delete("point")
        for x, y in self.points:
            self.canvas.create_oval(x - 5, y - 5, x + 5, y + 5, fill='red', tags="point")

    def draw_rectangle(self):
        for line in self.rect_lines + self.guide_lines:
            self.canvas.delete(line)
        self.rect_lines.clear()
        self.guide_lines.clear()

        if len(self.points) != 4:
            return

        points_sorted = sorted(self.points, key=lambda p: p[1])
        top = min(points_sorted[:2], key=lambda p: p[0])
        bottom = max(points_sorted[2:], key=lambda p: p[0])
        points_sorted = sorted(self.points, key=lambda p: p[0])
        left = min(points_sorted[:2], key=lambda p: p[1])
        right = max(points_sorted[2:], key=lambda p: p[1])

        x1, y1 = left[0], top[1]
        x2, y2 = right[0], bottom[1]

        self.rect_lines.append(self.canvas.create_line(x1, y1, x2, y1, fill='blue', width=2))
        self.rect_lines.append(self.canvas.create_line(x2, y1, x2, y2, fill='blue', width=2))
        self.rect_lines.append(self.canvas.create_line(x2, y2, x1, y2, fill='blue', width=2))
        self.rect_lines.append(self.canvas.create_line(x1, y2, x1, y1, fill='blue', width=2))

        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        self.guide_lines.append(self.canvas.create_line(x1, cy, x2, cy, fill='green', width=1, dash=(4, 2)))
        self.guide_lines.append(self.canvas.create_line(cx, y1, cx, y2, fill='green', width=1, dash=(4, 2)))

    def process_image(self):
        if len(self.points) != 4:
            return

        w, h = self.original_image.size
        scale_x = w / self.display_width
        scale_y = h / self.display_height

        self.saved_parameters = {
            'image_path': self.image_path,
            'points': self.points,
            'scale': (scale_x, scale_x)
        }

        points_sorted = sorted(self.points, key=lambda p: p[1])
        top = min(points_sorted[:2], key=lambda p: p[0])
        bottom = max(points_sorted[2:], key=lambda p: p[0])
        points_sorted = sorted(self.points, key=lambda p: p[0])
        left = min(points_sorted[:2], key=lambda p: p[1])
        right = max(points_sorted[2:], key=lambda p: p[1])

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
        img_np = np.array(self.original_image)
        if img_np.ndim == 2:
            img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
        elif img_np.shape[2] == 4:
            img_np = img_np[:, :, :3]

        self.aligned_image = cv2.warpPerspective(img_np, M, output_size)

        # Show preview
        self.show_result(self.aligned_image)
        self.save_btn.config(state=tk.NORMAL)

    def show_result(self, image_np):
        result_image = Image.fromarray(image_np)
        w, h = result_image.size
        ratio = min(300 / w, 300 / h)
        dw, dh = round(w * ratio), round(h * ratio)

        result_image_scaled = result_image.resize((dw, dh), Image.LANCZOS)
        self.tk_result_image = ImageTk.PhotoImage(result_image_scaled)
        self.result_canvas.config(width=dw, height=dh)
        self.result_canvas.create_image(2, 2, anchor=tk.NW, image=self.tk_result_image)

    def save_image(self):
        if self.saved_parameters is not None:
            scale_x, scale_y = self.saved_parameters['scale']
            points = self.saved_parameters['points']
            image_path = self.saved_parameters['image_path']
            print(f"File: {image_path}")
            print(f"Chosen points (display coords): {points}")
            print(f"Scale factors: scale_x={scale_x}, scale_y={scale_y}")
            print('=' * 20)
            self.reset_state()


if __name__ == "__main__":
    root = tk.Tk()
    app = TargetAlignmentApp(root)
    root.mainloop()
