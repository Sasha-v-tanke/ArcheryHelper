import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(BASE_DIR, 'data')
NEW_DATASET_PATH = os.path.join(DATA_PATH, 'new_dataset')
ORIGINAL_DATASET_PATH = os.path.join(DATA_PATH, 'original_dataset')

NORMALIZED_DATASET = os.path.join(DATA_PATH, 'normalized')
NEW_NORMALIZED_DATASET = os.path.join(DATA_PATH, 'normalized-new')

MODELS = os.path.join(BASE_DIR, 'models')
