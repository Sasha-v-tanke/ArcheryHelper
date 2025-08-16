from src.find_all_photos import find_all_photos, get_similar_json
from src.show_photo_with_arrows import show_points

files = find_all_photos('../data/original_dataset')
for file in files:
    try:
        show_points(file, get_similar_json(file), False)
    except:
        continue
