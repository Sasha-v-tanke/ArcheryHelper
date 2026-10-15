import json
import math
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from neural_network.archery_ml.data.importers.coco import import_coco
from neural_network.archery_ml.data.importers.legacy import import_legacy
from neural_network.archery_ml.data.importers.roboflow import import_roboflow
from neural_network.archery_ml.data.importers.yolo import import_yolo
from neural_network.archery_ml.data.registry import DatasetSource


class DatasetImportersTest(unittest.TestCase):
    def test_legacy_importer_converts_polar_annotations(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = root / "series-1-frame-1.jpg"
            Image.new("RGB", (32, 32), (128, 128, 128)).save(image)
            image.with_suffix(".json").write_text(
                json.dumps({"shots": [{"r_norm": 0.5, "theta_deg": 90.0}]}),
                encoding="utf-8",
            )
            source = self._source("legacy", root)

            samples = import_legacy(root, source)

            self.assertEqual(1, len(samples))
            self.assertAlmostEqual(0.0, samples[0].annotations[0].x_norm, places=6)
            self.assertAlmostEqual(0.5, samples[0].annotations[0].y_norm, places=6)

    def test_yolo_importer_converts_bbox_center_to_canonical_coordinates(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_dir = root / "train" / "images"
            label_dir = root / "train" / "labels"
            image_dir.mkdir(parents=True)
            label_dir.mkdir(parents=True)
            image = image_dir / "sample.jpg"
            Image.new("RGB", (32, 32), (128, 128, 128)).save(image)
            (label_dir / "sample.txt").write_text("0 0.75 0.25 0.1 0.1\n", encoding="utf-8")
            source = self._source("yolo", root, class_mapping={"0": "impact"})

            samples = import_yolo(root, source)

            self.assertEqual("train", samples[0].split)
            self.assertAlmostEqual(0.5, samples[0].annotations[0].x_norm)
            self.assertAlmostEqual(-0.5, samples[0].annotations[0].y_norm)

    def test_coco_importer_accepts_bbox_annotations(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_dir = root / "images"
            image_dir.mkdir()
            image = image_dir / "sample.jpg"
            Image.new("RGB", (100, 100), (128, 128, 128)).save(image)
            (root / "_annotations.coco.json").write_text(
                json.dumps(
                    {
                        "images": [{"id": 1, "file_name": "images/sample.jpg", "width": 100, "height": 100}],
                        "categories": [{"id": 7, "name": "arrow"}],
                        "annotations": [{"id": 1, "image_id": 1, "category_id": 7, "bbox": [70, 20, 10, 10]}],
                    }
                ),
                encoding="utf-8",
            )
            source = self._source("coco", root, class_mapping={"arrow": "impact"})

            samples = import_coco(root, source)

            self.assertAlmostEqual(0.5, samples[0].annotations[0].x_norm)
            self.assertAlmostEqual(-0.5, samples[0].annotations[0].y_norm)

    def test_coco_importer_reads_roboflow_split_exports(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for split, red in (("train", 64), ("valid", 128), ("test", 192)):
                split_dir = root / split
                split_dir.mkdir()
                Image.new("RGB", (100, 100), (red, 0, 0)).save(split_dir / "same.jpg")
                (split_dir / "_annotations.coco.json").write_text(
                    json.dumps(
                        {
                            "images": [{"id": 1, "file_name": "same.jpg", "width": 100, "height": 100}],
                            "categories": [{"id": 1, "name": "arrow"}],
                            "annotations": [
                                {
                                    "id": 1,
                                    "image_id": 1,
                                    "category_id": 1,
                                    "bbox": [1, 2, 3, 4],
                                    "keypoints": [10, 20, 2, 30, 40, 2],
                                }
                            ],
                        }
                    ),
                    encoding="utf-8",
                )
            source = self._source(
                "coco",
                root,
                class_mapping={"arrow": "impact"},
                options={"impact_point": "keypoint", "keypoint_index": 0},
            )

            samples = import_coco(root, source)

            self.assertEqual(3, len(samples))
            self.assertEqual(["test", "train", "val"], sorted(sample.split for sample in samples))
            self.assertEqual(3, len({sample.id for sample in samples}))
            self.assertEqual(3, len({sample.image_path for sample in samples}))
            for sample in samples:
                self.assertEqual(1, len(sample.raw_annotations) + len(sample.annotations))

    def test_coco_keypoint_mode_rejects_invisible_keypoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = root / "sample.jpg"
            Image.new("RGB", (100, 100), (128, 128, 128)).save(image)
            (root / "_annotations.coco.json").write_text(
                json.dumps(
                    {
                        "images": [{"id": 1, "file_name": "sample.jpg", "width": 100, "height": 100}],
                        "categories": [{"id": 1, "name": "arrow"}],
                        "annotations": [
                            {
                                "id": 1,
                                "image_id": 1,
                                "category_id": 1,
                                "bbox": [70, 20, 10, 10],
                                "keypoints": [10, 20, 0, 30, 40, 2],
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            source = self._source(
                "coco",
                root,
                class_mapping={"arrow": "impact"},
                options={"impact_point": "keypoint", "keypoint_index": 0},
            )

            with self.assertRaisesRegex(ValueError, "missing visible keypoint 0"):
                import_coco(root, source)

    def test_roboflow_importer_detects_yolo_export(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_dir = root / "valid" / "images"
            label_dir = root / "valid" / "labels"
            image_dir.mkdir(parents=True)
            label_dir.mkdir(parents=True)
            Image.new("RGB", (32, 32), (128, 128, 128)).save(image_dir / "sample.jpg")
            (label_dir / "sample.txt").write_text("0 0.5 0.5 0.1 0.1\n", encoding="utf-8")
            (root / "data.yaml").write_text("names: [arrow]\n", encoding="utf-8")
            source = self._source(
                "roboflow",
                root,
                class_mapping={"0": "impact"},
                options={"export_format": "yolo"},
            )

            samples = import_roboflow(root, source)

            self.assertEqual("val", samples[0].split)
            self.assertTrue(math.isclose(0.0, samples[0].annotations[0].x_norm))

    def _source(self, importer, root, class_mapping=None, options=None):
        return DatasetSource.from_dict(
            {
                "id": f"source-{importer}",
                "importer": importer,
                "url": str(root),
                "version": "1",
                "license": "CC-BY-4.0",
                "author": "author",
                "allowed_tasks": ["impact_detection"],
                "class_mapping": class_mapping or {},
                "options": options or {},
            }
        )


if __name__ == "__main__":
    unittest.main()
