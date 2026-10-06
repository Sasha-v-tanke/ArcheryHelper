import glob
import json
import os
import shutil

from path_manager import NEW_NORMALIZED_DATASET, YOLO_DATASET_PATH


def prepare_dataset(from_path, to_path, json_path, split_coef=0.9):
    files = glob.glob(os.path.join(from_path, f"*jpeg"))
    os.makedirs(os.path.join(to_path, 'train', 'images'), exist_ok=True)
    os.makedirs(os.path.join(to_path, 'train', 'labels'), exist_ok=True)
    os.makedirs(os.path.join(to_path, 'test', 'images'), exist_ok=True)
    os.makedirs(os.path.join(to_path, 'test', 'labels'), exist_ok=True)

    for ind, file in enumerate(files):
        filename = file.split('/')[-1]
        folder = 'train' if ind < len(files) * split_coef else 'test'
        if os.path.exists(os.path.join(to_path, folder, 'images', filename)):
            continue
        shutil.copy(file, os.path.join(to_path, folder, 'images', filename))
        with open(os.path.join(json_path, filename.split('.')[0] + '.json'), 'r') as f:
            data = json.load(f)
            arrows = [(e['r_norm'], e['theta_deg']) for e in data['shots']]
            with open(os.path.join(to_path, folder, 'labels', filename.split('.')[0] + '.txt'), 'w') as out:
                for arrow in arrows:
                    out.write(f'{arrow[0]} {arrow[1]}\n')


if __name__ == '__main__':
    prepare_dataset(NEW_NORMALIZED_DATASET, YOLO_DATASET_PATH, NEW_NORMALIZED_DATASET)
