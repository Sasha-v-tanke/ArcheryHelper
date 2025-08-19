#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Версия с нижней панелью кнопок, растянутой по горизонтали:
- Кнопки распределены равномерно в один ряд на всю ширину окна.
- Панель находится снизу.
- Сохраняется функционал зоны X, направляющих и перетаскивания точек.
"""

import json
import math
import os
import glob
from datetime import datetime
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk, ImageDraw

# === НАСТРОЙКИ ===
IMAGE_DIR = r"../../data/normalized-new"
OUTPUT_DIR = IMAGE_DIR
TARGET_SIZE = 600
POINT_RADIUS = 6
PHOTO_MAX_SIZE = (600, 600)
SUPPORTED_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")

RING_COLORS = [
    "#FFFFFF", "#FFFFFF",  # белые
    "#000000", "#000000",  # чёрные
    "#0000FF", "#0000FF",  # синие
    "#FF0000", "#FF0000",  # красные
    "#FFFF00", "#FFFF00",  # жёлтые
]


class TargetApp:
    def __init__(self, root):
        self.root = root
        root.title("Фото + Мишень (Tkinter)")
        root.geometry("1280x720")

        os.makedirs(OUTPUT_DIR, exist_ok=True)
        self.files = self._collect_files(IMAGE_DIR)
        self.idx = 0

        # Основная область для фото и мишени
        main_frame = tk.Frame(root)
        main_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        left = tk.Frame(main_frame)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.photo_label = tk.Label(left, text="—")
        self.photo_label.pack(anchor=tk.W, padx=10, pady=(10, 0))

        self.photo_canvas = tk.Canvas(left, width=PHOTO_MAX_SIZE[0], height=PHOTO_MAX_SIZE[1], bg="#222")
        self.photo_canvas.pack(padx=10, pady=10)

        right = tk.Frame(main_frame)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.target_label = tk.Label(right, text="Мишень: кликайте для отметок")
        self.target_label.pack(anchor=tk.W, padx=10, pady=(10, 0))

        self.target_canvas = tk.Canvas(right, width=TARGET_SIZE, height=TARGET_SIZE, bg="#ddd")
        self.target_canvas.pack(padx=10, pady=10)
        self.target_canvas.bind("<Button-1>", self.on_target_click)
        self.target_canvas.bind("<B1-Motion>", self.on_drag)
        self.target_canvas.bind("<ButtonRelease-1>", self.on_release)

        self.info_var = tk.StringVar(value="Снимок: 0/0 | Выстрелов: 0")
        self.info_label = tk.Label(right, textvariable=self.info_var)
        self.info_label.pack(anchor=tk.W, padx=10, pady=(0, 10))

        # Нижняя панель кнопок под всем контентом
        btns = tk.Frame(root, bg="#eee", relief=tk.RAISED, bd=2)
        btns.pack(side=tk.BOTTOM, fill=tk.X)

        self.reset_btn = tk.Button(btns, text="🔄 Сбросить", command=self.reset_shots)
        self.save_btn = tk.Button(btns, text="💾 Сохранить", command=self.save_and_next)
        self.next_btn = tk.Button(btns, text="➡ Следующее фото", command=self.next_photo)

        self.reset_btn.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=5, pady=5)
        self.save_btn.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=5, pady=5)
        self.next_btn.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=5, pady=5)

        self.shots = []
        self.shot_items = []
        self.dragging = None

        self.draw_target()
        self.load_current_photo()
        self.update_info()

    def _collect_files(self, directory):
        files = []
        if os.path.isdir(directory):
            for ext in SUPPORTED_EXTS:
                files.extend(glob.glob(os.path.join(directory, f"*{ext}")))
        files = [file for file in files if not os.path.exists(file.replace('.jpeg', '.json'))]
        return sorted(files)

    def current_file(self):
        if not self.files:
            return None
        return self.files[self.idx % len(self.files)]

    def load_current_photo(self):
        path = self.current_file()
        if not path:
            self.photo_canvas.delete("all")
            self.photo_label.config(text="Нет изображений")
            return
        img = Image.open(path).convert("RGB")
        img.thumbnail(PHOTO_MAX_SIZE, Image.LANCZOS)
        draw = ImageDraw.Draw(img, "RGBA")
        cx, cy = img.size[0] // 2, img.size[1] // 2
        r = min(cx, cy) - 2
        for angle in range(0, 360, 30):
            dx = r * math.cos(math.radians(angle))
            dy = r * math.sin(math.radians(angle))
            draw.line((cx, cy, cx + dx, cy + dy), fill=(0, 128, 0, 128), width=1)

        self._photo_img = ImageTk.PhotoImage(img)
        self.photo_canvas.delete("all")
        cx, cy = PHOTO_MAX_SIZE[0] // 2, PHOTO_MAX_SIZE[1] // 2
        self.photo_canvas.create_image(cx, cy, image=self._photo_img)
        self.photo_label.config(text=f"Фото: {os.path.basename(path)}")

    def next_photo(self):
        if not self.files:
            return
        if self.idx + 1 == len(self.files):
            exit(0)
        self.idx = (self.idx + 1) % len(self.files)
        self.reset_shots(hard=True)
        self.load_current_photo()
        self.update_info()

    def draw_target(self):
        c = self.target_canvas
        c.delete("all")
        size = TARGET_SIZE
        cx, cy = size // 2, size // 2
        radius = size // 2 - 2

        ring_width = radius / len(RING_COLORS)
        for i, color in enumerate(RING_COLORS):
            r_outer = radius - i * ring_width
            c.create_oval(cx - r_outer, cy - r_outer, cx + r_outer, cy + r_outer,
                          fill=color, outline="green")

        # Зона X
        x_radius = ring_width / 2
        c.create_oval(cx - x_radius, cy - x_radius, cx + x_radius, cy + x_radius,
                      fill="#FFFF00", outline="green")

        # Часовые направляющие
        for angle in range(0, 360, 30):
            dx = radius * math.cos(math.radians(angle))
            dy = radius * math.sin(math.radians(angle))
            c.create_line(cx, cy, cx + dx, cy + dy, fill="#008000", dash=(4, 4))

        c.create_oval(cx - 2, cy - 2, cx + 2, cy + 2, fill="#111", outline="")

    def reset_shots(self, hard=False):
        for item in self.shot_items:
            self.target_canvas.delete(item)
        self.shot_items.clear()
        self.shots.clear()
        self.update_info()

    def on_target_click(self, event):
        for i, item in enumerate(self.shot_items):
            coords = self.target_canvas.coords(item)
            if coords[0] <= event.x <= coords[2] and coords[1] <= event.y <= coords[3]:
                self.dragging = i
                return
        x, y = event.x, event.y
        item = self.target_canvas.create_oval(
            x - POINT_RADIUS, y - POINT_RADIUS, x + POINT_RADIUS, y + POINT_RADIUS,
            fill="#222", outline="#fff"
        )
        self.shot_items.append(item)
        self.shots.append(self._make_shot(x, y))
        self.update_info()

    def on_drag(self, event):
        if self.dragging is not None:
            idx = self.dragging
            x, y = event.x, event.y
            self.target_canvas.coords(self.shot_items[idx],
                                      x - POINT_RADIUS, y - POINT_RADIUS,
                                      x + POINT_RADIUS, y + POINT_RADIUS)
            self.shots[idx] = self._make_shot(x, y)

    def on_release(self, event):
        self.dragging = None

    def _make_shot(self, x, y):
        size = TARGET_SIZE
        cx, cy = size // 2, size // 2
        dx, dy = x - cx, y - cy
        r_pix = math.hypot(dx, dy)
        max_r = size // 2 - 2
        r_norm = r_pix / max_r
        theta = math.degrees(math.atan2(dy, dx))
        if theta < 0:
            theta += 360.0
        return {
            "image": os.path.basename(self.current_file()) if self.current_file() else None,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "x": float(x),
            "y": float(y),
            "dx": float(dx),
            "dy": float(dy),
            "r_norm": float(r_norm),
            "theta_deg": float(theta),
        }

    def update_info(self):
        total = len(self.files)
        cur = (self.idx + 1) if total else 0
        self.info_var.set(f"Снимок: {cur}/{total} | Выстрелов: {len(self.shots)}")

    def save_and_next(self):
        if not self.files:
            return
        img_name = os.path.basename(self.current_file())
        base, _ = os.path.splitext(img_name)
        out_path = os.path.join(OUTPUT_DIR, f"{base}.json")
        payload = {
            "image": img_name,
            "created": datetime.now().isoformat(timespec="seconds"),
            "target": {"size": TARGET_SIZE, "rings": len(RING_COLORS), "ring_colors": RING_COLORS},
            "shots": self.shots,
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        self.next_photo()
        # messagebox.showinfo("Сохранено", f"Серия сохранена в: {out_path}")


def main():
    root = tk.Tk()
    app = TargetApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
