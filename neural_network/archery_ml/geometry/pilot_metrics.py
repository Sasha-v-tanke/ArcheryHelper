"""Optional partial independent GT, distinct from detector proposals and visual ratings."""
import hashlib
import json

import numpy as np

from neural_network.archery_ml.geometry.calibration import project
from neural_network.archery_ml.geometry.evaluate import _metrics,_equal_group_metrics


def independent_metrics(path,results):
    empty = {"status":"QUALITATIVE ONLY / QUANTITATIVE NOT VERIFIED","independent_gt_images":0,
             "median_2d":None,"p95_2d":None,"per_image":[],"issues":[]}
    if not path.is_file():
        return empty
    raw=path.read_bytes();payload=json.loads(raw)
    if not isinstance(payload,dict) or payload.get("kind")!="independent_gt" or payload.get("method") not in {"physical_measurement","surveyed_print_template","independent_calibration"}:
        raise ValueError("pseudo_gt and visual_review are not independent quantitative GT")
    if (not payload.get("revision") or not payload.get("reference") or not payload.get("annotator")
            or not payload.get("reviewer") or payload["annotator"]==payload["reviewer"]):
        raise ValueError("independent GT requires version, reference and independent reviewer")
    per_image,groups,issues = [],[],[]
    for item in payload["images"]:
        original=item["original_sha256"]
        result=results.get(original)
        if result is None or result["status"]!="READY":
            issues.append({"original_id":original,"reason":"selected geometry not READY"});continue
        if item.get("candidate_ids") != result.get("candidate_ids"):
            issues.append({"original_id":original,"reason":"GT refers to another or unconfirmed group"});continue
        face_results={f["face_index"]:f for f in result["faces"]}
        errors,by_face = [],{}
        for face in item["faces"]:
            index=face["face_index"]
            if index not in face_results or str(index) in by_face:
                raise ValueError("invalid or duplicate independent GT face_index")
            image=np.asarray(face["image_points_px"],dtype=float)
            expected=np.asarray(face["canonical_points"],dtype=float)
            if image.shape!=expected.shape or image.ndim!=2 or image.shape[1]!=2 or len(image)<4:
                raise ValueError("each measured face requires four matched independent points")
            for values in (image,expected):
                if not np.isfinite(values).all() or len(np.unique(values,axis=0))<4 or np.linalg.matrix_rank(values-values.mean(axis=0))<2:
                    raise ValueError("independent GT points must be finite, unique and non-collinear")
            if item.get("coordinate_space") not in {"original_raw_px","upright_rgb_px"}:
                raise ValueError("declare independent GT coordinate space")
            key="original_to_canonical" if item["coordinate_space"]=="original_raw_px" else "rgb_to_canonical"
            predicted=(project(image,face_results[index][key])-255.5)/256.
            values=np.linalg.norm(predicted-expected,axis=1).tolist()
            by_face[str(index)]=_metrics(values).to_dict();errors.extend(values)
        if errors:
            groups.append(errors)
            per_image.append({"original_id":original,"face_errors_2d":by_face,"point_error_2d":_metrics(errors).to_dict(),
                "missing_faces":sorted(set(face_results)-{int(k) for k in by_face}),"orientation_status":result["orientation_status"]})
    if not groups:
        return {**empty,"issues":issues,"gt_sha256":hashlib.sha256(raw).hexdigest()}
    weighted=_equal_group_metrics(groups)
    return {"status":"PARTIAL INDEPENDENT GT — NOT FINAL HOLDOUT","independent_gt_images":len(groups),"per_image":per_image,
        "median_2d":weighted["median"],"p95_2d":weighted["p95"],"equal_image_weighted":weighted,"issues":issues,
        "gt_revision":payload["revision"],"gt_sha256":hashlib.sha256(raw).hexdigest(),"reference":payload["reference"]}
