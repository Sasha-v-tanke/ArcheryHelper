import json
import os.path

import numpy as np


def extract_points_from_original_json(filename):
    """
    extract arrows' coordinates from json from original dataset
    :param filename: path to the photo
    :return: list of coordinates for each arrow on the photo (type: int)
    """
    if not os.path.exists(filename):
        raise Exception("File not found: " + filename)

    result = []
    photo_number = int(filename.split('/')[-1].split('.')[0])
    # example: path/to/json/in/dataset/10.json -> 10
    for index in range(1, photo_number + 1):
        # find previous json filename with other arrows
        index_str = str(index).rjust(2, '0')
        jsonfile = '/'.join(filename.split('/')[:-1]) + f'/{index_str}.json'
        if not os.path.exists(jsonfile):
            continue

        with open(jsonfile) as f:
            data = json.load(f)

        arrows = [shape for shape in data['shapes'] if shape['label'].lower() == 'impact']
        for idx, arrow in enumerate(arrows):
            points = np.array(arrow['points'], dtype=np.int32)
            for x, y in points:
                result.append((round(x), round(y)))

    return result


def extract_arrow_from_custom_json(filename, keyword='arrows'):
    """
    extract arrows' coordinates from json from custom dataset
    :param filename: path to the photo
    :param keyword: key for arrows in json
    :return: list of coordinates for each arrow on the photo
    """
    if not os.path.exists(filename):
        raise "File not found"

    with open(filename) as f:
        data = json.load(f)
    arrows = [(round(arrow[0]), round(arrow[1])) for arrow in data[keyword]]

    return arrows


def save_json(filename, arrows):
    with open(filename, 'w') as f:
        json.dump(arrows, f)


if __name__ == '__main__':
    print("Test")
    original_json = "../data/original_dataset/arrow-sequences/versions/1/8/06.json"
    print(extract_points_from_original_json(original_json))
    custom_json = "../draft/aligned/1.json"
    print(extract_arrow_from_custom_json(custom_json, 'aligned_hits'))
