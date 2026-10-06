import json
import tempfile
import unittest
from pathlib import Path

from neural_network.contracts import (
    DatasetSample,
    ModelInputSpec,
    ModelMetadata,
    ShotAnnotation,
    build_manifest_from_dirs,
    load_manifest,
    save_manifest,
    validate_manifest,
)


class ContractsTest(unittest.TestCase):
    def test_contract_objects_roundtrip(self):
        sample = DatasetSample("1", Path("images/1.jpg"), Path("annotations/1.json"), "train", "camera")
        annotation = ShotAnnotation(0.1, -0.2)
        input_spec = ModelInputSpec(256, 256)
        metadata = ModelMetadata(1, 256, 256, "shot_set_polar", 6, "target_center_polar", 0.5)

        self.assertEqual(sample, DatasetSample.from_dict(sample.to_dict()))
        self.assertEqual(annotation, ShotAnnotation.from_dict(annotation.to_dict()))
        self.assertEqual(input_spec, ModelInputSpec.from_dict(input_spec.to_dict()))
        self.assertEqual(metadata, ModelMetadata.from_dict(metadata.to_dict()))

    def test_manifest_serialization_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            image = tmp_path / "1.jpg"
            annotation = tmp_path / "1.json"
            image.write_bytes(b"image")
            annotation.write_text(json.dumps({"shots": []}), encoding="utf-8")
            manifest = tmp_path / "manifest.json"

            samples = [DatasetSample("1", image, annotation)]
            save_manifest(manifest, samples)

            self.assertEqual(samples, load_manifest(manifest))
            self.assertTrue(validate_manifest(samples).is_valid)

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


if __name__ == "__main__":
    unittest.main()
