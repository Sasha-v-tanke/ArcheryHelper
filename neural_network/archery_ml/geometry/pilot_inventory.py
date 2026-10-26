"""Read-only inventory of historical photographs; no invented capture provenance."""
import hashlib
import html
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps, ExifTags


EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def raw_to_rgb_matrix(orientation, width, height):
    return np.asarray({
        2: [[-1,0,width-1],[0,1,0],[0,0,1]],
        3: [[-1,0,width-1],[0,-1,height-1],[0,0,1]],
        4: [[1,0,0],[0,-1,height-1],[0,0,1]],
        5: [[0,1,0],[1,0,0],[0,0,1]],
        6: [[0,-1,height-1],[1,0,0],[0,0,1]],
        7: [[0,-1,height-1],[-1,0,width-1],[0,0,1]],
        8: [[0,1,0],[-1,0,width-1],[0,0,1]],
    }.get(orientation, [[1,0,0],[0,1,0],[0,0,1]]), dtype=float)


def _value(value):
    if isinstance(value,bytes):
        return {"bytes":len(value),"sha256":hashlib.sha256(value).hexdigest()}
    if isinstance(value,(list,tuple)):
        return [_value(v) for v in value]
    if isinstance(value,(str,int,float)) or value is None:
        return value
    try:
        return float(value)
    except (ValueError,TypeError,ZeroDivisionError):
        return str(value)


def inventory(input_dir, output):
    source, output = Path(input_dir).resolve(), Path(output).resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"input photographs unavailable: {source}")
    if output == source or output.is_relative_to(source):
        raise ValueError("output must be outside the original photograph tree")
    (output / "rgb").mkdir(parents=True,exist_ok=True)
    (output / "previews").mkdir(exist_ok=True)
    rows, seen = [], {}
    for number,path in enumerate(sorted(p for p in source.rglob("*") if p.is_file() and p.suffix.lower() in EXTENSIONS),1):
        row = {"number":number,"source_path":str(path),"filename":path.relative_to(source).as_posix(),
               "read_status":"ERROR","split":"mobile_dev","dataset_role":"legacy_exploratory",
               "capture_session_id":"unknown","target_config":"unknown","capture_plan":"not_pre_registered"}
        try:
            raw = path.read_bytes()
            sha = hashlib.sha256(raw).hexdigest()
            row.update(original_id=sha,original_sha256=sha,bytes=len(raw),duplicate_of=seen.get(sha))
            seen.setdefault(sha,number)
            with Image.open(path) as image:
                exif = image.getexif()
                fields = {ExifTags.TAGS.get(k,str(k)):_value(v) for k,v in exif.items()}
                try:
                    fields.update({ExifTags.TAGS.get(k,str(k)):_value(v) for k,v in exif.get_ifd(34665).items()})
                except (ValueError,KeyError,TypeError):
                    pass
                orientation = exif.get(274,1)
                upright = ImageOps.exif_transpose(image).convert("RGB")
                matrix = raw_to_rgb_matrix(orientation,*image.size)
                row.update(original_size=list(image.size),rgb_size=list(upright.size),exif=fields,
                    exif_orientation=orientation,exif_orientation_valid=orientation in range(1,9),
                    device_model=fields.get("Model","unknown"),capture_datetime=fields.get("DateTimeOriginal",fields.get("DateTime","unknown")),
                    raw_to_rgb=matrix.tolist(),rgb_to_raw=np.linalg.inv(matrix).tolist(),
                    orientation_operation="EXIF transpose" if orientation in range(2,9) else "identity",
                    orientation_warning=None if orientation in range(1,9) else "invalid EXIF; storage axes retained, display orientation unverified")
                rgb_path = output / "rgb" / f"{sha}.png"
                if not rgb_path.exists():
                    upright.save(rgb_path,compress_level=1)
                else:
                    with Image.open(rgb_path) as cached:
                        if cached.size!=upright.size or cached.convert("RGB").tobytes()!=upright.tobytes():
                            raise ValueError("cached RGB differs from EXIF-transposed original; use a new output revision")
                row.update(rgb_path=rgb_path.relative_to(output).as_posix(),rgb_sha256=hashlib.sha256(rgb_path.read_bytes()).hexdigest())
                preview = upright.copy()
                preview.thumbnail((1100,1100))
                preview_path = output / "previews" / f"{sha}.jpg"
                preview.save(preview_path,quality=90)
                row.update(preview_path=preview_path.relative_to(output).as_posix(),read_status="OK")
        except (OSError,ValueError,TypeError) as error:
            row["error"] = str(error)
        rows.append(row)
    result = {"input":str(source),"role":"legacy_exploratory / mobile_dev; NOT independent holdout", "images":rows,
              "files":len(rows),"unique_originals":len(seen),"exact_duplicates":sum(r.get("duplicate_of") is not None for r in rows)}
    save_json(output / "inventory.json",result)
    cards = []
    for row in rows:
        preview = f'<img loading="lazy" src="{row["preview_path"]}" alt="photo {row["number"]}">' if row["read_status"] == "OK" else html.escape(row.get("error","unreadable"))
        cards.append(f'<article>{preview}<b>#{row["number"]:02}</b> {html.escape(row["filename"])}<br>{row["read_status"]}; session unknown</article>')
    (output / "contact-sheet.html").write_text('<!doctype html><meta charset="utf-8"><title>Historical target photographs</title><style>body{font:14px system-ui;background:#eee}main{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:12px}article{background:white;padding:8px;overflow-wrap:anywhere}img{width:100%;height:270px;object-fit:contain}</style><h1>Legacy exploratory photos — not holdout</h1><main>'+''.join(cards)+'</main>')
    return result
