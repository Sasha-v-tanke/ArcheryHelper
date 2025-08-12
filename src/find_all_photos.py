import os


def find_all_photos(folder):
    image_files = []
    # Expand ~ and convert to absolute
    base = os.path.abspath(os.path.expanduser(folder))
    for root, _, files in os.walk(base):
        for f in files:
            if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                full = os.path.join(root, f)
                image_files.append(full)
    return image_files


def get_similar_json(image_path):
    """
    :param image_path: path to image, which json is needed
    :return: path to json for this image
    """
    return image_path.replace('.png', '.json').replace('.jpg', '.json')


if __name__ == '__main__':
    print('Test')
    assert len(find_all_photos('../data/original_dataset')) == 1085
    print('Success')
