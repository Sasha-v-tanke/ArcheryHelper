"""Stage 4 regression coverage; uses tiny synthetic data, never downloads weights."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from PIL import Image

from neural_network.archery_ml.contracts import DatasetSample, ImpactAnnotation, TargetMetadata, save_manifest
from neural_network.archery_ml.data.view import DatasetView
from neural_network.archery_ml.evaluate import evaluate_faces
from neural_network.archery_ml.heatmap import canonical_to_grid, encode_impacts, grid_to_canonical, impact_loss
from neural_network.archery_ml.impact_dataset import CanonicalImpactDataset, collate_faces
from neural_network.archery_ml.models import ImpactPointNet
from neural_network.archery_ml.postprocess import ImpactPrediction, decode_impacts
from neural_network.archery_ml.train import _verify_dataset, train, evaluate_checkpoint


class ImpactTargetsTest(unittest.TestCase):
    def test_roundtrip_and_negative_target_gradient(self):
        for x, y in [(0.0, 0.0), (0.35, -0.49), (-1.0, 1.0)]:
            gx, gy = canonical_to_grid(x, y, 32, 32)
            rx, ry = grid_to_canonical(gx, gy, 32, 32)
            self.assertAlmostEqual(rx, x)
            self.assertAlmostEqual(ry, y)
        empty = encode_impacts([], 32, 32)
        self.assertFalse(empty[2].any())
        logits = torch.zeros((1, 1, 32, 32), requires_grad=True)
        offsets = torch.zeros((1, 2, 32, 32), requires_grad=True)
        loss, _ = impact_loss((logits, offsets), tuple(t.unsqueeze(0) for t in empty))
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(logits.grad)

    def test_offsets_localize_subpixel_points_and_collision_rejected(self):
        point = (0.21, -0.13)
        heatmap, offsets, positive = encode_impacts([point], 32, 32)
        self.assertEqual(1, positive.sum())
        logits = torch.full_like(heatmap, -15)
        logits[positive] = 15
        predictions = decode_impacts(logits, offsets, confidence_threshold=0.5, nms_radius=2)
        self.assertEqual(1, len(predictions))
        self.assertAlmostEqual(point[0], predictions[0].x_norm, places=5)
        self.assertAlmostEqual(point[1], predictions[0].y_norm, places=5)
        with self.assertRaisesRegex(ValueError, "same output cell"):
            encode_impacts([(0.0, 0.0), (0.001, 0.001)], 32, 32)

    def test_nms_ties_and_invalid_offsets(self):
        logits = torch.zeros((1, 16, 16))
        offsets = torch.zeros((2, 16, 16))
        self.assertEqual(1, len(decode_impacts(logits, offsets, confidence_threshold=0.5, nms_radius=2)))
        offsets[0, 0, 0] = float("nan")
        self.assertEqual([], decode_impacts(logits, offsets, confidence_threshold=0.5, nms_radius=2))
        with self.assertRaises(ValueError):
            decode_impacts(logits, offsets, confidence_threshold=-0.1, nms_radius=2)

    def test_model_shapes_and_gradient(self):
        model = ImpactPointNet(fpn_channels=16)
        model.train()
        heatmap, offsets = model(torch.rand((2, 3, 64, 64)))
        self.assertEqual((2, 1, 32, 32), tuple(heatmap.shape))
        self.assertEqual((2, 2, 32, 32), tuple(offsets.shape))
        labels = encode_impacts([(0.0, 0.2)], 32, 32)
        total, _ = impact_loss((heatmap, offsets), tuple(t.unsqueeze(0).expand(2, *t.shape)
                                                            for t in labels))
        total.backward()
        self.assertTrue(torch.isfinite(total))
        self.assertIsNotNone(model.heatmap_head.weight.grad)
        with self.assertRaisesRegex(ValueError, "multiples of 32"):
            model(torch.rand((1, 3, 66, 64)))


class MetricsTest(unittest.TestCase):
    def test_independent_matching_and_empty_false_positives(self):
        meta = TargetMetadata(format="SINGLE", minimum_scoring_zone=1,
                              ten_ring_mode="RECURVE", face_diameter_mm=400)
        first = ([ImpactPrediction(0.01, 0.01, 0.99), ImpactPrediction(0.7, 0.7, 0.9)],
                 [(0.01, 0.01), (-0.2, 0.0)], meta)
        second = ([ImpactPrediction(0.0, 0.0, 0.9)], [], meta)
        metrics = evaluate_faces([first, second])
        self.assertEqual((1, 2, 1), (metrics.true_positives,
                                      metrics.false_positives, metrics.false_negatives))
        self.assertAlmostEqual(1 / 3, metrics.precision)
        self.assertAlmostEqual(0.5, metrics.recall)
        self.assertEqual(1.0, metrics.false_positives_per_empty_face)

    def test_matching_maximum_cardinality(self):
        # First prediction may take either GT; second can only take first GT.
        a = ImpactPrediction(0.00, 0.0, 0.9)
        b = ImpactPrediction(-0.04, 0.0, 0.8)
        metrics = evaluate_faces([([a, b], [(-0.02, 0.0), (0.04, 0.0)], None)],
                                 match_radius_ring=0.5)
        self.assertEqual(2, metrics.true_positives)


class CanonicalDataTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        Image.new("RGB", (64, 64), color=(50, 100, 150)).save(self.root / "canonical.png")

    def sample(self, *, split="train", annotations=(ImpactAnnotation(0.1, 0.2),),
               capture=None, group_id="g1"):
        if capture is None:
            capture = {"canonical_faces": [{"face_index": 0, "image_path": "canonical.png",
                                              "annotation_status": "verified"}]}
        return DatasetSample(id=split + "-sample", image_path=self.root / "raw-photo.png",
                             annotation_path=None, split=split, source_id="fixture", group_id=group_id,
                             annotations=annotations, capture_metadata=capture)

    def test_fails_closed_on_raw_only_and_unverified(self):
        with self.assertRaisesRegex(ValueError, "no verified canonical faces"):
            CanonicalImpactDataset(DatasetView.from_samples([self.sample(capture={})]),
                                   manifest_dir=self.root, input_size=64)
        with self.assertRaisesRegex(ValueError, "unverified"):
            CanonicalImpactDataset(DatasetView.from_samples([
                self.sample(capture={"canonical_faces": [{"face_index": 0,
                    "image_path": "canonical.png", "annotation_status": "unverified"}]})]),
                manifest_dir=self.root, input_size=64)

    def test_verified_empty_face_and_augmentation(self):
        data = CanonicalImpactDataset(DatasetView.from_samples([
            self.sample(annotations=(), group_id="empty")]), manifest_dir=self.root, input_size=64,
            augment=True, horizontal_flip_prob=1, brightness=0)
        self.assertEqual(1, len(data))
        self.assertEqual(0, int(data[0]["positive"].sum()))
        batch = collate_faces([data[0]])
        self.assertEqual((1, 3, 64, 64), tuple(batch["image"].shape))

    def test_split_leakage_and_snapshot_integrity(self):
        samples = [self.sample(split="train"), self.sample(split="val")]
        manifest = self.root / "manifest.json"
        save_manifest(manifest, samples)
        snapshot = type("Snapshot", (), {"manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest()})()
        config = {"dataset": {"manifest": str(manifest), "snapshot": str(self.root / "snapshot.json")}}
        with patch("neural_network.archery_ml.train.load_snapshot", return_value=snapshot):
            with self.assertRaisesRegex(ValueError, "group leakage"):
                _verify_dataset(config)
        snapshot.manifest_sha256 = "incorrect"
        with patch("neural_network.archery_ml.train.load_snapshot", return_value=snapshot):
            with self.assertRaisesRegex(ValueError, "SHA256"):
                _verify_dataset(config)

    def test_training_one_epoch_and_checkpoint_inference(self):
        samples = [self.sample(split="train", group_id="tr"), self.sample(split="val", group_id="vl")]
        manifest = self.root / "manifest.json"
        save_manifest(manifest, samples)
        snapshot = self.root / "snapshot.json"
        snapshot.write_text(json.dumps({"manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest()}))
        config = {"seed": 42, "model": {"version": "test-1", "input_size": 64,
                 "output_stride": 2, "fpn_channels": 8, "pretrained": False},
                  "training": {"epochs": 1, "batch_size": 1, "learning_rate": 0.001,
                               "weight_decay": 0.0, "horizontal_flip_prob": 0.0, "brightness": 0.0},
                  "postprocess": {"confidence_threshold": 0.9, "nms_radius": 2},
                  "dataset": {"manifest": str(manifest), "snapshot": str(snapshot)}}
        config_path = self.root / "config.json"
        config_path.write_text(json.dumps(config))
        # Snapshot implementation requires its full schema; here mock just the binding.
        fake = type("Snapshot", (), {"manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest()})()
        with patch("neural_network.archery_ml.train.load_snapshot", return_value=fake):
            run = train(config_path, output_root=self.root / "runs")
        for file in ("best_model.pt", "last_model.pt", "model_metadata.json", "metrics.csv",
                     "metrics.json", "dataset_info.json", "visualizations/epoch_001.png"):
            self.assertTrue((run / file).is_file(), file)
        with patch("neural_network.archery_ml.train.load_snapshot", return_value=fake):
            result = evaluate_checkpoint(config_path, run / "best_model.pt",
                                         run / "model_metadata.json", split="val")
        self.assertEqual(1, result["faces"])


if __name__ == "__main__":
    unittest.main()
