import cv2
import numpy as np
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

import cv2

img = cv2.imread('example.jpg', 0)  # 0 = grayscale
edges = cv2.Canny(img, 50, 150)

plt.figure(figsize=(12, 8))
plt.imshow(edges, cmap='gray')
plt.axis('off')
plt.show()
