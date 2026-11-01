"""Print a measured board layout or inspect image-only orientation calibration."""
import argparse
import base64
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from neural_network.archery_ml.geometry.calibration import BoardTemplate, TargetOrientationCalibrator, project
from neural_network.archery_ml.geometry.detector import TargetGeometryDetector
from neural_network.archery_ml.geometry.normalize import TargetNormalizer
from neural_network.archery_ml.geometry.target_templates import CANONICAL_CENTER_PX, CANONICAL_RADIUS_PX
from neural_network.archery_ml.targets import TargetConfig, TargetFormat, TenRingMode, TripleLayout, TargetTemplate


def config_for(template, minimum_zone=1, mode="RECURVE"):
    p = template.payload
    return TargetConfig(TargetFormat(p["format"]), minimum_zone, TenRingMode(mode),
                        round(p["faces"][0]["radius_mm"] * 2),
                        None if p["triple_layout"] is None else TripleLayout(p["triple_layout"]))


def render_board(template, config, *, pixels_per_mm=1.):
    p = template.payload
    image = np.full((round(p["height_mm"] * pixels_per_mm), round(p["width_mm"] * pixels_per_mm), 3), 255, np.uint8)
    colors = [(255,255,255)]*2 + [(20,20,20)]*2 + [(20,150,230)]*2 + [(230,40,40)]*2 + [(245,220,30)]*2
    for face in p["faces"]:
        center = tuple(np.rint(np.asarray(face["center_mm"]) * pixels_per_mm).astype(int))
        for ring in reversed(TargetTemplate.from_config(config).ring_boundaries):
            radius = round(face["radius_mm"] * ring.radius_norm * pixels_per_mm)
            cv2.circle(image, center, radius, colors[ring.score - 1], -1)
            cv2.circle(image, center, radius, (100,100,100), 1)
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    for marker in p["markers"]:
        corners = np.rint(np.asarray(marker["corners_mm"]) * pixels_per_mm).astype(int)
        x, y = corners[0]
        size = int(corners[1,0] - x)
        if (size < 12 or not np.array_equal(corners,[[x,y],[x+size,y],[x+size,y+size],[x,y+size]])
                or x < 0 or y < 0 or x+size > image.shape[1] or y+size > image.shape[0]):
            raise ValueError("printing requires in-bounds axis-aligned square markers of at least 12 pixels")
        image[y:y+size,x:x+size] = cv2.aruco.generateImageMarker(dictionary, marker["id"], size)[:,:,None]
    return image


def overlay_axes(image, geometry):
    canvas = image.copy()
    c, r = CANONICAL_CENTER_PX, CANONICAL_RADIUS_PX
    for face in geometry.faces:
        origin, right, up = np.rint(project([[c,c],[c+r*.5,c],[c,c-r*.5]], face.inverse_homography)).astype(int)
        cv2.arrowedLine(canvas, tuple(origin), tuple(right), (230,20,20), 2)
        cv2.arrowedLine(canvas, tuple(origin), tuple(up), (20,180,20), 2)
        cv2.putText(canvas, f"face {face.face_index}: UP", tuple(up), cv2.FONT_HERSHEY_SIMPLEX, .5, (20,180,20), 1)
    return canvas


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["print-board", "calibrate"])
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image", type=Path)
    parser.add_argument("--minimum-scoring-zone", type=int, choices=[1,5,6], default=1)
    parser.add_argument("--ten-ring-mode", choices=["RECURVE", "COMPOUND"], default="RECURVE")
    args = parser.parse_args(argv)
    board = BoardTemplate.load(args.template)
    config = config_for(board, args.minimum_scoring_zone, args.ten_ring_mode)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.command == "print-board":
        image = render_board(board, config, pixels_per_mm=4.)
        _, encoded = cv2.imencode(".png", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        p = board.payload
        args.output.write_text(f'<svg xmlns="http://www.w3.org/2000/svg" width="{p["width_mm"]}mm" height="{p["height_mm"]}mm" viewBox="0 0 {image.shape[1]} {image.shape[0]}">'
                               f'<image width="{image.shape[1]}" height="{image.shape[0]}" href="data:image/png;base64,{base64.b64encode(encoded).decode()}"/></svg>')
        return 0
    if args.image is None:
        parser.error("calibrate requires --image")
    with Image.open(args.image) as raw:
        if raw.getexif().get(274, 1) != 1:
            raise ValueError("EXIF must be explicitly normalized to a tracked derivative first")
        image = np.asarray(raw.convert("RGB"))
    calibrator = TargetOrientationCalibrator(board)
    geometry = TargetGeometryDetector().detect(image, config)
    calibrated = calibrator.calibrate(image, geometry, calibrator.observe(image))
    args.output.write_text(json.dumps(calibrated.record, indent=2, sort_keys=True) + "\n")
    Image.fromarray(overlay_axes(image, calibrated.geometry)).save(args.output.with_suffix(".axes.png"))
    for face in TargetNormalizer().normalize(image, calibrated.geometry):
        Image.fromarray(face.image).save(args.output.with_name(f"{args.output.stem}.face-{face.face_index}.png"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
