import os

from draft.find_white.find_background import process_target_image


def run():
    files = os.listdir('../../data/new_dataset')
    process_target_image(files[0])


if __name__ == '__main__':
    run()
