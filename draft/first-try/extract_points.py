import json
from extract_only_points import extract_points
import cv2
import numpy as np
from matplotlib import pyplot as plt


def show_points(image_path):
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    json_path = image_path.replace('.jpg', '.json').replace('.png', '.json')

    for point in extract_points(json_path):
        (x, y) = point
        point_color = (0, 255, 0)
        cv2.circle(img_rgb, (x, y), 5, point_color, -1)

    plt.figure(figsize=(15, 10))
    plt.imshow(img_rgb)
    plt.axis('off')
    plt.tight_layout()

    plt.show()


show_points("/data/original_dataset/arrow-sequences/versions/1/11/05.png")
