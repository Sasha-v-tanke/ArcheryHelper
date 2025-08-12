import json

import numpy as np


def extract_points(filename):
    cur_number = int(filename.split('/')[-1].split('.')[0])
    result = []
    for number in range(1, cur_number + 1):
        number_str = str(number).rjust(2, '0')
        jsonfile = '/'.join(filename.split('/')[:-1]) + f'/{number_str}.json'
        try:
            with open(jsonfile) as f:
                data = json.load(f)
            arrows = [shape for shape in data['shapes'] if shape['label'].lower() == 'impact']

            for idx, arrow in enumerate(arrows):
                points = np.array(arrow['points'], dtype=np.int32)
                for k, point in enumerate(points):
                    result.append(point)
        except:
            pass
    return result
