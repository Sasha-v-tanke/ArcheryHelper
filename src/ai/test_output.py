import numpy as np
from matplotlib import pyplot as plt

from src.ai.ui import draw_target

if __name__ == '__main__':
    output = [0.044039357, 4.277527, -0.017455619, 4.74417, 0.05127637, 4.2936716, -1.5891137, 4.1659822, -1.6371455,
              4.0639434, -1.5937941, 4.0980835, -1.6229973, 4.0814967, -1.570068, 4.1106324, -1.621376, 4.161525,
              -1.624518, 4.098395
              ]
    output = np.array(output)
    out = draw_target(output)
    out2 = draw_target(output, flag=True)
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(out)
    ax.set_title("Original Image")
    ax.axis('off')
    plt.show()
