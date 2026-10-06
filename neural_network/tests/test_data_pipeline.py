import tempfile
import unittest
from pathlib import Path

from PIL import Image

from neural_network.archery_ml.contracts import DatasetSample, ImpactAnnotation, TargetMetadata
from neural_network.archery_ml.data.deduplicate import deduplicate_samples
from neural_network.archery_ml.data.split import assign_splits
from neural_network.archery_ml.data.validate import validate_dataset


class DatasetPipelineTest(unittest.TestCase):
    def test_validation_rejects_invalid_coordinate_and_face_index(self):
        with tempfile.TemporaryDirectory() as tmp:
            image = Path(tmp) / "sample.jpg"
            Image.new("RGB", (16, 16), (0, 0, 0)).save(image)
            sample = DatasetSample(
                id="sample",
                image_path=image,
                annotation_path=None,
                source_id="source",
                group_id="group",
                annotations=(ImpactAnnotation(1.2, 0.0, face_index=1),),
                target_metadata=TargetMetadata(
                    format="SINGLE",
                    minimum_scoring_zone=1,
                    ten_ring_mode="RECURVE",
                ),
            )

            report = validate_dataset([sample])

            self.assertFalse(report.is_valid)
            self.assertEqual(
                {"invalid_coordinate", "invalid_face_index"},
                {issue.code for issue in report.issues},
            )

    def test_dedup_removes_exact_images_and_groups_perceptual_duplicates(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            black = root / "a.jpg"
            exact = root / "b.jpg"
            gray = root / "c.jpg"
            Image.new("RGB", (16, 16), (0, 0, 0)).save(black, quality=100, subsampling=0)
            exact.write_bytes(black.read_bytes())
            Image.new("RGB", (16, 16), (32, 32, 32)).save(gray, quality=100, subsampling=0)
            annotation = (ImpactAnnotation(0.0, 0.0),)
            samples = [
                self._sample("a", black, "a", annotation),
                self._sample("b", exact, "b", annotation),
                self._sample("c", gray, "c", annotation),
            ]

            result = deduplicate_samples(samples)

            self.assertEqual(("b",), result.exact_duplicates_removed)
            self.assertEqual(2, len(result.samples))
            self.assertEqual(result.samples[0].group_id, result.samples[1].group_id)
            self.assertEqual(1, len(result.near_duplicate_groups))

    def test_split_is_deterministic_group_aware_and_preserves_mobile_real_test(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            samples = []
            for index in range(8):
                image = root / f"{index}.jpg"
                Image.new("RGB", (8, 8), (index, index, index)).save(image)
                group = f"group-{index // 2}"
                samples.append(self._sample(str(index), image, group, ()))
            mobile_image = root / "mobile.jpg"
            Image.new("RGB", (8, 8), (255, 255, 255)).save(mobile_image)
            samples.append(
                DatasetSample(
                    id="mobile",
                    image_path=mobile_image,
                    annotation_path=None,
                    split="mobile_real_test",
                    source_id="mobile",
                    group_id="mobile-group",
                )
            )

            first = assign_splits(samples, seed=123, train_ratio=0.5, val_ratio=0.25, test_ratio=0.25)
            second = assign_splits(samples, seed=123, train_ratio=0.5, val_ratio=0.25, test_ratio=0.25)

            self.assertEqual([sample.split for sample in first], [sample.split for sample in second])
            self.assertEqual("mobile_real_test", first[-1].split)
            group_splits = {}
            for sample in first[:-1]:
                group_splits.setdefault(sample.group_id, set()).add(sample.split)
            self.assertTrue(all(len(splits) == 1 for splits in group_splits.values()))

    def _sample(self, sample_id, image, group_id, annotations):
        return DatasetSample(
            id=sample_id,
            image_path=image,
            annotation_path=None,
            source_id="source",
            group_id=group_id,
            annotations=annotations,
        )


if __name__ == "__main__":
    unittest.main()
