import numpy as np
from matplotlib import pyplot as plt

from src.ai.ui import draw_target

if __name__ == '__main__':
    output = [0.02080202, 1.3568408, 0.003408637, 1.5882763, -0.01461004, 1.4385384, -0.48691177, 1.3041178,
              -0.51011443, 1.3501085, -0.55772465, 1.3779, -0.48537567, 1.2842315, -0.5015256, 1.3170123, -0.52694255,
              1.3278692, -0.48004967, 1.2762485]

    out = draw_target(output)
    # out2 = draw_target(output)
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(out)
    ax.set_title("Original Image")
    ax.axis('off')
    plt.show()
