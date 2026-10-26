"""Development feedback benchmark and image-only proposals for remaining photos."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from neural_network.archery_ml.geometry.calibration import project
from neural_network.archery_ml.geometry.cli import _git_state
from neural_network.archery_ml.geometry.pilot_inventory import save_json
from neural_network.archery_ml.geometry.selected_geometry import SelectedTargetGeometryPipeline


REVISION = "radial-gn-guarded-v1"


def metrics(values):
    return {"count":len(values),"median":float(np.median(values)),
        "p95":float(np.percentile(values,95)),"max":float(np.max(values))} if values else {}


def run(pilot, reviews_path, output, publish=None):
    pilot, reviews_path, output = Path(pilot).resolve(), Path(reviews_path), Path(output).resolve()
    if output==pilot or output.is_relative_to(pilot):
        raise ValueError("adaptation output must be separate from original pilot")
    raw = reviews_path.read_bytes()
    review_sha = hashlib.sha256(raw).hexdigest()
    reviews = json.loads(raw)["images"]
    reviewed = {key:value for key,value in reviews.items() if value.get("status")=="CONFIRMED"}
    if not reviewed:
        raise ValueError("no confirmed centers available")
    output.mkdir(parents=True,exist_ok=True)
    snapshot = output/("input-reviews-"+review_sha+".json")
    if not snapshot.exists():
        snapshot.write_bytes(raw)
    inventory = json.loads((pilot/"inventory.json").read_text())
    detections = json.loads((pilot/"detections.json").read_text())
    choices = json.loads((pilot/"target-selections.json").read_text())
    baseline = json.loads((pilot/"geometry-results.json").read_text())
    sha, dirty = _git_state()
    source_sha = hashlib.sha256(Path(__file__).with_name("detector.py").read_bytes()).hexdigest()
    pipeline, results, proposals, comparisons, issues = SelectedTargetGeometryPipeline(), {}, {}, [], []
    seen = set()
    for row in inventory["images"]:
        key = row.get("original_id")
        if row["read_status"]!="OK" or key in seen:
            continue
        seen.add(key)
        path = pilot/row["rgb_path"]
        if hashlib.sha256(path.read_bytes()).hexdigest()!=row["rgb_sha256"]:
            raise ValueError("source image changed after inventory")
        choice = choices.get(key,{})
        if not choice.get("candidate_ids"):
            issues.append({"original_id":key,"reason":"target group not selected"})
            continue
        image = np.asarray(Image.open(path).convert("RGB"))
        result = pipeline.run(image,detections[key]["candidates"],choice,row["raw_to_rgb"],output/"crops"/key)
        result.update(original_id=key,number=row["number"])
        results[key] = result
        old_faces = {choice["candidate_ids"][f["face_index"]]:f for f in baseline[key].get("faces",[])}
        new_faces = {choice["candidate_ids"][f["face_index"]]:f for f in result.get("faces",[])}
        if key in reviewed:
            reference = reviewed[key]
            if (reference.get("image_sha256")!=row["rgb_sha256"]
                    or reference.get("coordinate_space")!="upright_rgb_px"
                    or reference.get("image_size")!=row["rgb_size"]):
                raise ValueError("review/image binding mismatch")
            for point in reference["points"]:
                cid = point["id"]
                if cid not in old_faces or cid not in new_faces:
                    issues.append({"original_id":key,"point_id":cid,"reason":"no matched geometry face"})
                    continue
                expected = np.array([point["x"],point["y"]])
                before, after = old_faces[cid], new_faces[cid]
                comparisons.append({"original_id":key,"number":row["number"],"point_id":cid,
                    "reviewed_center":expected.tolist(),"before":before["center_rgb"],"after":after["center_rgb"],
                    "before_error_px":float(np.linalg.norm(expected-before["center_rgb"])),
                    "after_error_px":float(np.linalg.norm(expected-after["center_rgb"])),
                    "before_error_norm":float(np.linalg.norm((project([expected],before["rgb_to_canonical"])[0]-255.5)/256)),
                    "after_error_norm":float(np.linalg.norm((project([expected],after["rgb_to_canonical"])[0]-255.5)/256))})
        else:
            points = [{"id":cid,"x":face["center_rgb"][0],"y":face["center_rgb"][1]} for cid,face in new_faces.items()]
            if not points:
                issues.append({"original_id":key,"reason":"no refined centers; baseline retained"})
                continue
            proposals[key] = {"original_sha256":key,"image_sha256":row["rgb_sha256"],
                "status":"PROPOSED","algorithm_revision":REVISION,"quality_status":result["quality_status"],
                "points":points,"human_confirmed":False}
        print(f"#{row['number']:02}: {result['quality_status']} / {'review benchmark' if key in reviewed else 'automatic proposal'}",flush=True)
    expected_points = sum(len(value["points"]) for value in reviewed.values())
    before = metrics([r["before_error_px"] for r in comparisons])
    after = metrics([r["after_error_px"] for r in comparisons])
    acceptable = bool(before and after and len(comparisons)==expected_points
        and after["median"]<before["median"] and after["p95"]<=before["p95"])
    revision = hashlib.sha256((REVISION+source_sha+review_sha).encode()).hexdigest()[:20]
    report = {"algorithm_revision":REVISION,"algorithm_sha":sha,"working_tree_dirty":dirty,
        "detector_source_sha256":source_sha,"review_sha256":review_sha,"reviewed_images":len(reviewed),
        "expected_reviewed_points":expected_points,"measured_points":len(comparisons),
        "before_error_px":before,"after_error_px":after,
        "before_error_norm":metrics([r["before_error_norm"] for r in comparisons]),
        "after_error_norm":metrics([r["after_error_norm"] for r in comparisons]),
        "improved_points":sum(r["after_error_px"]<r["before_error_px"] for r in comparisons),
        "quality_statuses":dict(Counter(r["quality_status"] for r in results.values())),
        "proposal_images":len(proposals),"proposal_points":sum(len(r["points"]) for r in proposals.values()),
        "comparisons":comparisons,"issues":issues,"proposal_revision":revision,
        "development_feedback_only":True,"independent_holdout":False,
        "publish_allowed":acceptable,
        "note":"Human-reviewed centers are development references; target scale, orientation and independent holdout accuracy are not attested."}
    if _git_state()!=(sha,dirty) or hashlib.sha256(Path(__file__).with_name("detector.py").read_bytes()).hexdigest()!=source_sha:
        raise ValueError("algorithm changed during feedback evaluation")
    save_json(output/"geometry-results.json",results)
    save_json(output/"report.json",report)
    if publish is not None:
        if not acceptable:
            raise ValueError("feedback benchmark did not improve; proposals were not published")
        if Path(publish).resolve()==reviews_path.resolve():
            raise ValueError("proposals must never overwrite human reviews")
        save_json(publish,{"schema_version":1,"revision":revision,"algorithm_revision":REVISION,
            "detector_source_sha256":source_sha,"development_review_sha256":review_sha,"images":proposals})
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot",type=Path,default=Path("data/reports/stage3-finalization/pilot"))
    parser.add_argument("--reviews",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--publish",type=Path)
    args = parser.parse_args(argv)
    result = run(args.pilot,args.reviews,args.output,args.publish)
    print(json.dumps({key:value for key,value in result.items() if key!="comparisons"},indent=2))


if __name__ == "__main__":
    main()
