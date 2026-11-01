"""Existing geometry pipeline on an explicitly selected group, with full-frame transforms."""
from dataclasses import asdict, replace
import math
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from neural_network.archery_ml.geometry.calibration import project
from neural_network.archery_ml.geometry.detector import TargetGeometryDetector, _checked_inverse, _matrix_tuple
from neural_network.archery_ml.geometry.normalize import TargetNormalizer
from neural_network.archery_ml.geometry.boundary import canonical_display
from neural_network.archery_ml.geometry.quality import ImageQualityGate, QualityStatus
from neural_network.archery_ml.geometry.selection import config_for_group


def draw_ellipse(image, candidate, color, thickness=-1, expand=1.):
    e = candidate["ellipse"]
    cv2.ellipse(image,((e["center_x"],e["center_y"]),(e["diameter_x"]*expand,e["diameter_y"]*expand),e["angle_deg"]),color,thickness)


class SelectedTargetGeometryPipeline:
    def __init__(self, orientation_calibrator=None):
        self.detector = TargetGeometryDetector()
        self.gate = ImageQualityGate()
        self.orientation_calibrator = orientation_calibrator

    def run(self, image, candidates, group, raw_to_rgb=None, output_dir=None):
        return self._run(image,candidates,group,raw_to_rgb,output_dir)[0]

    def evaluate(self, image, candidates, group, target_config, output_dir=None):
        """Return the same pipeline's typed geometry for independent measurement."""
        return self._run(image,candidates,group,output_dir=output_dir,target_config=target_config)

    def _run(self, image, candidates, group, raw_to_rgb=None, output_dir=None, target_config=None):
        selected = [next(c for c in candidates if c["id"]==cid) for cid in group["candidate_ids"]]
        config = target_config or config_for_group(group,group.get("minimum_scoring_zone"))
        height,width = image.shape[:2]
        colored_radius = min(.6,(11-config.minimum_scoring_zone)*.1)
        visible_radius = (11-config.minimum_scoring_zone)*.1
        factor = visible_radius/colored_radius
        bounds = []
        for c in selected:
            x0,y0,x1,y1 = c["bbox"]
            cx,cy = c["center"]
            bounds.append([cx+(x0-cx)*factor,cy+(y0-cy)*factor,cx+(x1-cx)*factor,cy+(y1-cy)*factor])
        margin = max(c["diameter"] for c in selected)*.08
        x0 = max(0,math.floor(min(b[0] for b in bounds)-margin))
        y0 = max(0,math.floor(min(b[1] for b in bounds)-margin))
        x1 = min(width,math.ceil(max(b[2] for b in bounds)+margin))
        y1 = min(height,math.ceil(max(b[3] for b in bounds)+margin))
        result = {"status":"REJECTED","quality_status":"GEOMETRY_UNCERTAIN","roi":[x0,y0,x1,y1],
            "candidate_ids":group["candidate_ids"],"orientation_status":"UNANCHORED",
            "orientation_reference":"unanchored_image_or_layout_axes",
            "face_index_reference":"USER_DECLARED_ORDER" if group.get("physical_face_order_confirmed") else "PROVISIONAL_IMAGE_LAYOUT_ORDER",
            "physical_face_order_verified":bool(group.get("physical_face_order_confirmed")),
            "processing_config":{**asdict(config),"format":config.format.value,"ten_ring_mode":config.ten_ring_mode.value,
                "triple_layout":None if config.triple_layout is None else config.triple_layout.value},
            "actual_config":{"ten_ring_mode":"unknown","face_diameter_mm":"unknown","minimum_scoring_zone":"inferred; requires confirmation"},
            "config_assumptions":["RECURVE used for geometry only; scoring mode unknown","face diameter 1 is a dimensionless placeholder, not a physical measurement"],
            "faces":[],"diagnostics":[],"foreign_candidates_masked":[]}
        if target_config is not None:
            result["actual_config"] = result["processing_config"].copy()
            result["config_assumptions"] = []
        full = None
        try:
            if x1<=x0 or y1<=y0:
                raise ValueError("invalid selected ROI")
            isolated = image.copy()
            for c in candidates:
                if c["id"] not in group["candidate_ids"]:
                    for target in selected:
                        if math.dist(c["center"],target["center"]) < .42*(c["diameter"]+target["diameter"]):
                            raise ValueError("selected and foreign candidate regions overlap; selection uncertain")
                    draw_ellipse(isolated,c,(128,128,128),expand=1.1)
                    result["foreign_candidates_masked"].append(c["id"])
            roi_geometry = self.detector.detect(isolated[y0:y1,x0:x1],config)
            roi_quality = self.gate.evaluate(image[y0:y1,x0:x1],roi_geometry)
            translation = np.array([[1,0,-x0],[0,1,-y0],[0,0,1.]])
            faces = []
            assigned = set()
            for face in roi_geometry.faces:
                center = (face.center[0]+x0,face.center[1]+y0)
                nearest = min(selected,key=lambda c:math.dist(center,c["center"]))
                index = group["candidate_ids"].index(nearest["id"])
                if index in assigned or math.dist(center,nearest["center"]) > .4*nearest["diameter"]:
                    raise ValueError("fitted face does not match the selected candidate identities")
                assigned.add(index)
                h = np.asarray(face.homography)@translation
                inverse = _checked_inverse(h)
                faces.append(replace(face,face_index=index,center=center,ellipse=replace(face.ellipse,
                    center_x=face.ellipse.center_x+x0,center_y=face.ellipse.center_y+y0),homography=_matrix_tuple(h),
                    inverse_homography=_matrix_tuple(inverse),visible_bounds=tuple(v+(x0 if i%2==0 else y0) for i,v in enumerate(face.visible_bounds))))
            full = replace(roi_geometry,faces=tuple(sorted(faces,key=lambda f:f.face_index)),image_width=width,image_height=height,
                visible_bounds=None if roi_geometry.visible_bounds is None else tuple(v+(x0 if i%2==0 else y0) for i,v in enumerate(roi_geometry.visible_bounds)))
            full_quality = self.gate.evaluate(image,full)
            quality = roi_quality if roi_quality.status is not QualityStatus.READY else full_quality
            result.update(quality_status=quality.status.value,geometry_confidence=full.geometry_confidence,fit_error=full.fit_error,
                quality_roi=asdict(roi_quality),quality_full_frame=asdict(full_quality))
            if quality.status is QualityStatus.READY and self.orientation_calibrator is not None:
                try:
                    board = self.orientation_calibrator.template.payload
                    if board["format"] != config.format.value or board["triple_layout"] != group["layout"]:
                        raise ValueError("board template differs from selected configuration")
                    calibrated = self.orientation_calibrator.calibrate(
                        image, full, self.orientation_calibrator.observe(image))
                    full = calibrated.geometry
                    result.update(calibration=calibrated.record, orientation_status="ANCHORED",
                        orientation_reference=full.orientation_reference, physical_face_order_verified=True,
                        face_index_reference="PHYSICAL_BOARD_MARKERS")
                except (ValueError, np.linalg.LinAlgError, cv2.error) as error:
                    result["calibration"] = {"orientation_status":"UNANCHORED", "reason":str(error)}
            raw_to_rgb = np.eye(3) if raw_to_rgb is None else np.asarray(raw_to_rgb,dtype=float)
            for face in full.faces:
                original_h = np.asarray(face.homography)@raw_to_rgb
                original_inverse = _checked_inverse(original_h)
                probes = np.array([[0,0],[width*.5,height*.5],[width-1,height-1]],dtype=float)
                error = float(np.max(np.linalg.norm(project(project(probes,original_h),original_inverse)-probes,axis=1)))
                if not np.isfinite(error) or error>1e-4:
                    raise ValueError("inconsistent full-original forward/inverse transforms")
                result["faces"].append({**asdict(face),"center_rgb":list(face.center),
                    "center_original":project([face.center],np.linalg.inv(raw_to_rgb))[0].tolist(),
                    "original_to_canonical":original_h.tolist(),"canonical_to_original":original_inverse.tolist(),
                    "rgb_to_canonical":[list(r) for r in face.homography],"canonical_to_rgb":[list(r) for r in face.inverse_homography],
                    "round_trip_max_px":error,"physical_face_index_verified":result["physical_face_order_verified"]})
            if quality.status is QualityStatus.READY:
                # Existing normalizer is unchanged. The second copy masks foreign
                # targets; original RGB and original quality evidence remain intact.
                crops = TargetNormalizer().normalize(isolated,full)
                if output_dir:
                    output_dir = Path(output_dir)
                    output_dir.mkdir(parents=True,exist_ok=True)
                for crop in crops:
                    display,boundary=canonical_display(crop.image,config.minimum_scoring_zone)
                    result["faces"][crop.face_index]["display_boundary"]=boundary
                    if output_dir:
                        path = output_dir/f"face-{crop.face_index}.png"
                        Image.fromarray(crop.image).save(path)
                        result["faces"][crop.face_index]["crop_path"] = str(path)
                        display_path=output_dir/f"face-{crop.face_index}-display.png"
                        Image.fromarray(display).save(display_path)
                        result["faces"][crop.face_index]["display_crop_path"] = str(display_path)
                        result["faces"][crop.face_index]["display_mask"] = "observed silhouette when supported; raw canonical crop and coordinate transform retained"
                result["status"] = "READY"
            else:
                result["diagnostics"].append(quality.status.value)
        except (ValueError,np.linalg.LinAlgError,cv2.error) as error:
            if result["quality_status"] == "READY":
                result["quality_status"] = "GEOMETRY_UNCERTAIN"
            result["diagnostics"].append(str(error))
        result["artifact_revision"]=hashlib.sha256(json.dumps({"config":result["processing_config"],
            "candidates":group["candidate_ids"],"transforms":[f["rgb_to_canonical"] for f in result["faces"]],
            "display_masks":[f.get("display_boundary",{}).get("mask_sha256") for f in result["faces"]]},sort_keys=True).encode()).hexdigest()[:16]
        return result, full
