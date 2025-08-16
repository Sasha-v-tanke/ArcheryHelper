import json

from matplotlib import image as mpimg, pyplot as plt


def show_image_with_points(image_path, points, point_color="green", point_size=1):
    img = mpimg.imread(image_path)

    # Создаём фигуру
    plt.imshow(img)

    # Извлекаем x и y отдельно
    xs, ys = zip(*points)

    # Отображаем точки поверх изображения
    plt.scatter(xs, ys, c=point_color, s=point_size)

    # Отключаем оси для красоты
    plt.axis("off")

    plt.show()


with open('../../data/new_dataset/annotation.json') as f:
    data = json.load(f)[0]
    # for key, value in data.items():
    #     print(key, value, sep=': ')

    # for e in (data['annotations'][0]['result']):
    #     print(e)
    arrows = [(e['value']['x'] * e['original_width'] / 100.0, e['value']['y'] * e['original_height'] / 100.0) for e in
              data['annotations'][0]['result']]
    print(arrows)
    show_image_with_points('/Users/alex/Projects/Python/archery/data/new_dataset/1.jpeg', arrows)
