"""Generate an explicitly synthetic development example; never real validation GT."""
import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image

from neural_network.archery_ml.contracts import DatasetSample, ImageGeometryAnnotation, TargetMetadata
from neural_network.archery_ml.geometry.calibration import BoardTemplate
from neural_network.archery_ml.geometry.calibration_tools import config_for, render_board
from neural_network.archery_ml.data.geometry_provenance import annotation_digest


def create(output):
    output.mkdir(parents=True, exist_ok=False)
    samples = []
    for name in ("single", "vertical", "triangular"):
        board = BoardTemplate.load(Path(f"dataset_tools/boards/{name}-v1.json"))
        config = config_for(board)
        image = render_board(board,config)
        path = output / f"synthetic-{name}.png"
        Image.fromarray(image).save(path)
        checksum = hashlib.sha256(path.read_bytes()).hexdigest()
        annotations = []
        for face in board.payload["faces"]:
            x,y = face["center_mm"]
            annotations.append(ImageGeometryAnnotation("target_center",((x/600,y/800),),face_index=face["face_index"],
                annotation_quality="reviewed",uncertainty_norm=0.))
            points = ((-.3,-.3),(.3,-.3),(.3,.3),(-.3,.3))
            annotations.append(ImageGeometryAnnotation("landmark",tuple(((x+a*face["radius_mm"])/600,(y+b*face["radius_mm"])/800) for a,b in points),
                face_index=face["face_index"], canonical_points=points,annotation_quality="reviewed",uncertainty_norm=0.))
        metadata = {"capture_session_id":f"synthetic-fixture-{name}","original_id":checksum,"original_sha256":checksum,
            "device_model":"synthetic-renderer-NOT-A-CAMERA","image_size":[600,800],"gt_revision":"synthetic-fixture-v1",
            "annotator":"synthetic-fixture-generator-NOT-A-PERSON","reviewer":"synthetic-fixture-test-NOT-HUMAN-REVIEW",
            "orientation_reference":"board","suitability":"usable","synthetic_fixture":True,
            "independent_gt_method":"synthetic_print_template","annotation_state":"synthetic-fixture"}
        sample = DatasetSample(f"synthetic-{name}",Path(path.name),None,split="mobile_dev",
            geometry_annotations=tuple(annotations),capture_metadata=metadata,
            target_metadata=TargetMetadata(config.format.value,1,config.ten_ring_mode.value,
                None if config.triple_layout is None else config.triple_layout.value,config.face_diameter_mm))
        metadata["gt_annotations_sha256"] = annotation_digest(sample)
        samples.append(sample.to_dict())
    (output / "geometry.json").write_text(json.dumps({"notice":"SYNTHETIC EXAMPLE ONLY: not independent real GT or human review", "samples":samples},indent=2) + "\n")
    source = {"id":"synthetic_mobile_example","importer":"mobile_geometry","url":str(output.resolve()),"version":"fixture-v1",
        "license":"private","author":"synthetic fixture generator","allowed_tasks":["geometry_evaluation"],"annotation_space":"image"}
    (output / "sources.json").write_text(json.dumps({"registry_version":1,"sources":[source]},indent=2)+"\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    create(parser.parse_args().output)
