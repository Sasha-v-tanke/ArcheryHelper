import re
from pathlib import Path


def original_family(image_path: Path) -> str:
    """Strip Roboflow augmentation hashes while retaining distinct originals."""
    parts = re.split(r"_(?:jpg|jpeg|png)\.rf\.", image_path.name, flags=re.IGNORECASE)
    name = parts[0] if len(parts) > 1 else image_path.stem
    return name.casefold()
