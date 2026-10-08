import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from neural_network.archery_ml.contracts import (
    DatasetSample,
    ImageGeometryAnnotation,
    ImagePointAnnotation,
    ImpactAnnotation,
    save_manifest,
)
from neural_network.archery_ml.data.cli import main
from neural_network.archery_ml.data.importers.coco import import_coco
from neural_network.archery_ml.data.importers.yolo import import_yolo
from neural_network.archery_ml.data.pipeline import rebuild_dataset
from neural_network.archery_ml.data.registry import DatasetSource
from neural_network.archery_ml.data.snapshot import load_snapshot
from neural_network.archery_ml.data.view import DatasetView
from neural_network.archery_ml.data.workspace import DatasetWorkspace


class DatasetStage25Test(unittest.TestCase):
    def test_coco_image_space_preserves_raw_impact_and_geometry(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            images = root / "images"
            images.mkdir()
            Image.new("RGB", (100, 100), (80, 80, 80)).save(images / "sample.jpg")
            (root / "_annotations.coco.json").write_text(
                json.dumps(
                    {
                        "images": [{"id": 1, "file_name": "images/sample.jpg", "width": 100, "height": 100}],
                        "categories": [
                            {"id": 1, "name": "arrow_tip"},
                            {"id": 2, "name": "target_face"},
                        ],
                        "annotations": [
                            {"id": 1, "image_id": 1, "category_id": 1, "bbox": [70, 20, 10, 10]},
                            {
                                "id": 2,
                                "image_id": 1,
                                "category_id": 2,
                                "bbox": [10, 10, 80, 80],
                                "segmentation": [[10, 10, 90, 10, 90, 90, 10, 90]],
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )
            source = self._source(
                "coco",
                root,
                annotation_space="image",
                allowed_tasks=["impact_detection", "geometry_evaluation"],
                class_mapping={"arrow_tip": "impact", "target_face": "target_face"},
            )

            sample = import_coco(root, source)[0]

            self.assertEqual((), sample.annotations)
            self.assertEqual(1, len(sample.raw_annotations))
            self.assertAlmostEqual(0.75, sample.raw_annotations[0].x_fraction)
            self.assertAlmostEqual(0.25, sample.raw_annotations[0].y_fraction)
            self.assertEqual((0.7, 0.2, 0.1, 0.1), sample.raw_annotations[0].bbox)
            self.assertEqual(1, len(sample.geometry_annotations))
            self.assertEqual("target_face", sample.geometry_annotations[0].kind)
            self.assertEqual(
                ((0.1, 0.1), (0.9, 0.1), (0.9, 0.9), (0.1, 0.9)),
                sample.geometry_annotations[0].points,
            )

    def test_yolo_keypoint_mode_uses_keypoint_not_bbox_center(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            images = root / "train" / "images"
            labels = root / "train" / "labels"
            images.mkdir(parents=True)
            labels.mkdir(parents=True)
            Image.new("RGB", (100, 100), (100, 100, 100)).save(images / "sample.jpg")
            (labels / "sample.txt").write_text(
                "0 0.5 0.5 0.2 0.2 0.75 0.25 2\n",
                encoding="utf-8",
            )
            source = self._source(
                "yolo",
                root,
                annotation_space="image",
                class_mapping={"0": "impact"},
                options={"impact_point": "keypoint", "keypoint_index": 0, "keypoint_dimensions": 3},
            )

            sample = import_yolo(root, source)[0]

            self.assertEqual((), sample.annotations)
            self.assertEqual(1, len(sample.raw_annotations))
            self.assertAlmostEqual(0.75, sample.raw_annotations[0].x_fraction)
            self.assertAlmostEqual(0.25, sample.raw_annotations[0].y_fraction)

    def test_registry_requires_image_space_for_geometry_roles(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "annotation_space=image"):
                self._source(
                    "coco",
                    Path(tmp),
                    annotation_space="canonical",
                    allowed_tasks=["geometry_evaluation"],
                    class_mapping={"target_face": "target_face"},
                )

    def test_rebuild_creates_deterministic_manifest_report_and_snapshot(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_dir = root / "source"
            source_dir.mkdir()
            image = source_dir / "sample.jpg"
            Image.new("RGB", (32, 32), (120, 80, 40)).save(image)
            image.with_suffix(".json").write_text(
                json.dumps({"shots": [{"r_norm": 0.25, "theta_deg": 90.0}]}),
                encoding="utf-8",
            )
            registry_path = root / "registry.json"
            registry_path.write_text(
                json.dumps(
                    {
                        "registry_version": 1,
                        "sources": [
                            {
                                "id": "local",
                                "importer": "legacy",
                                "url": str(source_dir),
                                "version": "1",
                                "license": "USER_OWNED",
                                "author": "local",
                                "allowed_tasks": ["impact_detection"],
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            data_root = root / "data"

            first = rebuild_dataset(registry_path=registry_path, data_root=data_root, seed=7)
            manifest_bytes = first.manifest_path.read_bytes()
            report_bytes = first.report_path.read_bytes()
            snapshot_bytes = first.snapshot_path.read_bytes()

            second = rebuild_dataset(registry_path=registry_path, data_root=data_root, seed=7)

            self.assertEqual(manifest_bytes, second.manifest_path.read_bytes())
            self.assertEqual(report_bytes, second.report_path.read_bytes())
            self.assertEqual(snapshot_bytes, second.snapshot_path.read_bytes())
            snapshot = load_snapshot(second.snapshot_path)
            self.assertEqual(
                hashlib.sha256(second.manifest_path.read_bytes()).hexdigest(),
                snapshot.manifest_sha256,
            )
            self.assertEqual(1, snapshot.report.images)
            self.assertEqual(1, snapshot.report.canonical_impacts)
            self.assertEqual(0, snapshot.report.raw_impacts)
            self.assertEqual(("local",), tuple(source.id for source in snapshot.sources))

    def test_dataset_view_filters_raw_geometry_and_split(self):
        samples = (
            DatasetSample(
                id="canonical",
                image_path=Path("canonical.jpg"),
                annotation_path=None,
                split="train",
                source_id="a",
                group_id="a",
                annotations=(ImpactAnnotation(0.0, 0.0),),
            ),
            DatasetSample(
                id="raw",
                image_path=Path("raw.jpg"),
                annotation_path=None,
                split="test",
                source_id="b",
                group_id="b",
                raw_annotations=(ImagePointAnnotation(0.5, 0.5),),
                geometry_annotations=(ImageGeometryAnnotation("target_center", ((0.5, 0.5),)),),
            ),
        )

        view = DatasetView.from_samples(samples).select(
            splits=["test"],
            source_ids=["b"],
            require_raw_impacts=True,
            require_geometry=True,
        )

        self.assertEqual(("raw",), view.sample_ids())

    def test_preview_cli_renders_selected_annotations(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = root / "sample.jpg"
            Image.new("RGB", (64, 64), (20, 20, 20)).save(image)
            manifest = root / "manifest.json"
            save_manifest(
                manifest,
                [
                    DatasetSample(
                        id="raw",
                        image_path=image,
                        annotation_path=None,
                        source_id="source",
                        group_id="group",
                        raw_annotations=(ImagePointAnnotation(0.5, 0.5),),
                        geometry_annotations=(
                            ImageGeometryAnnotation(
                                "target_face",
                                ((0.1, 0.1), (0.9, 0.1), (0.9, 0.9), (0.1, 0.9)),
                            ),
                        ),
                    )
                ],
            )
            output = root / "preview"

            result = main(
                [
                    "preview",
                    "--manifest",
                    str(manifest),
                    "--output",
                    str(output),
                    "--raw-only",
                    "--with-geometry",
                    "--limit",
                    "1",
                ]
            )

            self.assertEqual(0, result)
            self.assertEqual(1, len(list(output.glob("*.jpg"))))
            index = json.loads((output / "index.json").read_text(encoding="utf-8"))
            self.assertEqual("raw", index[0]["sample_id"])

    def test_workspace_uses_separate_materialized_and_derived_directories(self):
        workspace = DatasetWorkspace(Path("data"))

        self.assertEqual(Path("data/sources"), workspace.sources_dir)
        self.assertEqual(Path("data/dataset_pipeline/cache"), workspace.cache_dir)
        self.assertEqual(Path("data/manifests/dataset_manifest.json"), workspace.manifest_path)
        self.assertEqual(Path("data/snapshots/dataset_snapshot.json"), workspace.snapshot_path)
        self.assertEqual(Path("data/previews"), workspace.previews_dir)

    def _source(
        self,
        importer,
        root,
        *,
        annotation_space="canonical",
        allowed_tasks=None,
        class_mapping=None,
        options=None,
    ):
        return DatasetSource.from_dict(
            {
                "id": f"source-{importer}",
                "importer": importer,
                "url": str(root),
                "version": "1",
                "license": "CC-BY-4.0",
                "author": "author",
                "allowed_tasks": allowed_tasks or ["impact_detection"],
                "annotation_space": annotation_space,
                "class_mapping": class_mapping or {},
                "options": options or {},
            }
        )


if __name__ == "__main__":
    unittest.main()
