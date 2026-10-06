import json
import tempfile
import unittest
from pathlib import Path

from neural_network.contracts import (
    DatasetSample,
    ImpactAnnotation,
    ModelInputSpec,
    ModelMetadata,
    ModelOutputSpec,
    ModelPostprocessSpec,
    SCHEMA_VERSION,
    ShotAnnotation,
    TargetMetadata,
    build_manifest_from_dirs,
    load_manifest,
    save_manifest,
    validate_manifest,
)


class ContractsTest(unittest.TestCase):
    def test_contract_objects_roundtrip(self):
        target = TargetMetadata(
            format="TRIPLE",
            minimum_scoring_zone=6,
            ten_ring_mode="RECURVE",
            triple_layout="VERTICAL",
            face_diameter_mm=400,
        )
        impact = ImpactAnnotation(0.1, -0.2, face_index=2, confidence=0.9)
        sample = DatasetSample(
            id="1",
            image_path=Path("images/1.jpg"),
            annotation_path=None,
            split="train",
            source_id="camera",
            group_id="session-1",
            annotations=(impact,),
            target_metadata=target,
        )
        input_spec = ModelInputSpec(512, 512)
        metadata = ModelMetadata(
            contract_version=2,
            model_version="impact-v1.0.0",
            input=input_spec,
            output=ModelOutputSpec("impact_heatmap_offset", 2),
            postprocess=ModelPostprocessSpec(0.35, 4),
        )

        self.assertEqual(sample, DatasetSample.from_dict(sample.to_dict()))
        self.assertEqual(impact, ImpactAnnotation.from_dict(impact.to_dict()))
        self.assertEqual(input_spec, ModelInputSpec.from_dict(input_spec.to_dict()))
        self.assertEqual(metadata, ModelMetadata.from_dict(metadata.to_dict()))

    def test_legacy_shot_annotation_roundtrip(self):
        annotation = ShotAnnotation(0.1, -0.2)

        self.assertEqual(annotation, ShotAnnotation.from_dict(annotation.to_dict()))

    def test_manifest_serialization_roundtrip_uses_schema_v2(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            image = tmp_path / "1.jpg"
            image.write_bytes(b"image")
            manifest = tmp_path / "manifest.json"

            samples = [
                DatasetSample(
                    id="1",
                    image_path=image,
                    annotation_path=None,
                    source_id="camera",
                    group_id="session-1",
                    annotations=(ImpactAnnotation(0.1, 0.2),),
                    target_metadata=TargetMetadata(
                        format="SINGLE",
                        minimum_scoring_zone=1,
                        ten_ring_mode="RECURVE",
                        face_diameter_mm=400,
                    ),
                )
            ]
            save_manifest(manifest, samples)

            payload = json.loads(manifest.read_text(encoding="utf-8"))
            self.assertEqual(SCHEMA_VERSION, payload["schema_version"])
            self.assertEqual(samples, load_manifest(manifest))
            self.assertTrue(validate_manifest(samples).is_valid)

    def test_load_manifest_accepts_legacy_source_field(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest = Path(tmp) / "manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "samples": [
                            {
                                "id": "1",
                                "image_path": "images/1.jpg",
                                "annotation_path": "annotations/1.json",
                                "split": "train",
                                "source": "legacy",
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            sample = load_manifest(manifest)[0]

            self.assertEqual("legacy", sample.source_id)

    def test_build_manifest_rejects_missing_annotation(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp) / "images"
            json_dir = Path(tmp) / "json"
            data_dir.mkdir()
            json_dir.mkdir()
            (data_dir / "1.jpg").write_bytes(b"image")

            with self.assertRaisesRegex(ValueError, "missing annotations: 1"):
                build_manifest_from_dirs(data_dir, json_dir)

    def test_build_manifest_pairs_by_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp) / "images"
            json_dir = Path(tmp) / "json"
            data_dir.mkdir()
            json_dir.mkdir()
            (data_dir / "2.jpg").write_bytes(b"image")
            (data_dir / "1.jpg").write_bytes(b"image")
            (json_dir / "1.json").write_text(json.dumps({"shots": []}), encoding="utf-8")
            (json_dir / "2.json").write_text(json.dumps({"shots": []}), encoding="utf-8")

            samples = build_manifest_from_dirs(data_dir, json_dir)

            self.assertEqual(["1", "2"], [sample.id for sample in samples])
            self.assertEqual(["1", "2"], [sample.group_id for sample in samples])


if __name__ == "__main__":
    unittest.main()
