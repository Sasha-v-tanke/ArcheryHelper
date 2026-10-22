import re
from pathlib import Path


def original_family(image_path: Path) -> str:
    """Strip Roboflow augmentation hashes while retaining distinct originals."""
    name = re.split(r"_(?:jpg|jpeg|png)\.rf\.", image_path.name, flags=re.IGNORECASE)[0]
    return Path(name).stem.casefold()
