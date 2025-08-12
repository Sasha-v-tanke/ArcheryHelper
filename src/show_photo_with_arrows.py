import cv2
from matplotlib import pyplot as plt
from src.extract_arrows import extract_arrow_from_custom_json, extract_points_from_original_json


def show_points(image_path, json_path, custom_json, hit_color=(0, 255, 0), hit_radius=5):
    """
    show image with hits
    :param image_path: path to image
    :param json_path: path to json file
    :param custom_json: false -> image from original dataset, true -> from custom
    :param hit_radius: radius of hits
    :param hit_color: color of hits
    """
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # switch to correct extractor
    extract_points = extract_arrow_from_custom_json if custom_json else extract_points_from_original_json

    for point in extract_points(json_path):
        (x, y) = point
        point_color = hit_color
        cv2.circle(img_rgb, (x, y), hit_radius, point_color, -1)

    plt.figure(figsize=(10, 10))
    plt.imshow(img_rgb)
    plt.axis('off')
    plt.tight_layout()

    plt.show()


if __name__ == '__main__':
    print('Test')
    image_path = "../data/original_dataset/arrow-sequences/versions/1/8/05.png"
    json_path = "../data/original_dataset/arrow-sequences/versions/1/8/05.json"
    show_points(image_path, json_path, custom_json=False)
