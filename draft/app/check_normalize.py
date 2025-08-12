import json
import cv2
import numpy as np
from matplotlib import pyplot as plt


def show_points(image_path, json_path):
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    with open(json_path) as f:
        data = json.load(f)
    if 'aligned_hits' in data.keys():
        arrows = [shape for shape in data['aligned_hits']]
    else:
        arrows = [shape for shape in data['points']]
    # print(arrows)
    for idx, arrow in enumerate(arrows):
        (x, y) = arrow
        point_color = (0, 255, 0)
        x, y = int(x), int(y)
        cv2.circle(img_rgb, (x, y), 5, point_color, -1)

    plt.figure(figsize=(15, 10))
    plt.imshow(img_rgb)
    plt.axis('off')
    plt.tight_layout()

    plt.show()

# for i in range(9, 17):
#     show_points(f"/Users/alex/Projects/Python/archery/aligned/{i}.png",
#                 f"/Users/alex/Projects/Python/archery/aligned/{i}.json")
