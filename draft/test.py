from find_borders import show_yellow_zone_outline
from find_center import show_yellow_zone_filled
from normilize_center import normalize_target_by_yellow_zone

test_images = [
    "/Users/alex/Projects/Python/archery/09.png",
    "/Users/alex/Projects/Python/archery/example.jpg",
    "/Users/alex/Projects/Python/archery/aligned_target.png"
]

for img in test_images:
    normalize_target_by_yellow_zone(img)
