"""Stage 4 regression coverage; uses tiny synthetic data, never downloads weights."""

import hashlib
import io
from contextlib import redirect_stdout
import json
import tempfile
import unittest
from pathlib import Path
from dataclasses import replace
from unittest.mock import patch

import torch
from PIL import Image

from neural_network.archery_ml.contracts import DatasetSample, ImpactAnnotation, TargetMetadata, save_manifest
from neural_network.archery_ml.data.view import DatasetView
from neural_network.archery_ml.evaluate import evaluate_faces, _match
from neural_network.archery_ml.heatmap import canonical_to_grid, encode_impacts, grid_to_canonical, impact_loss
from neural_network.archery_ml.impact_dataset import CanonicalImpactDataset, collate_faces
from neural_network.archery_ml.inference import load_detector, predict_canonical_face
from neural_network.archery_ml.contracts import ModelInputSpec, ModelMetadata, ModelOutputSpec, ModelPostprocessSpec
from neural_network.archery_ml.models import ImpactPointNet
from neural_network.archery_ml.postprocess import ImpactPrediction, decode_impacts
from neural_network.archery_ml.train import _verify_dataset, train, evaluate_checkpoint, mobile_quality_gate
from neural_network.archery_ml.provenance import (
    fingerprint, training_provenance, verify_checkpoint_holdout, verify_dataset,
)


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

    def test_real_input_output_grid_and_model_size(self):
        model = ImpactPointNet(fpn_channels=48).eval()
        with torch.no_grad():
            heatmap, offsets = model(torch.zeros(1, 3, 512, 512))
        self.assertEqual((1, 1, 256, 256), tuple(heatmap.shape))
        self.assertEqual((1, 2, 256, 256), tuple(offsets.shape))
        self.assertLess(sum(p.numel() for p in model.parameters()) * 4, 30 * 1024 * 1024)

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

    def test_scoring_is_excluded_only_for_real_boundary_uncertainty(self):
        meta = TargetMetadata(format="SINGLE", minimum_scoring_zone=1,
                              ten_ring_mode="RECURVE", face_diameter_mm=400)
        clear = evaluate_faces([([ImpactPrediction(0.0, 0.0, 0.99)], [(0.0, 0.0)], meta)])
        self.assertEqual(1, clear.score_eligible_count)
        self.assertEqual(1.0, clear.exact_score_accuracy_non_line_call)
        ambiguous = evaluate_faces([([ImpactPrediction(0.1, 0.0, 0.99)], [(0.1, 0.0)], meta)])
        self.assertEqual(0, ambiguous.score_eligible_count)
        self.assertIsNone(ambiguous.exact_score_accuracy_non_line_call)

    def test_matching_is_minimum_cost_for_maximum_cardinality(self):
        truths = [(0.08118804397, 0), (0.02866554588, 0)]
        predictions = [ImpactPrediction(0.01458217017, 0, 0.9),
                       ImpactPrediction(0.0352494738, 0, 0.9)]
        matches = _match(predictions, truths, 0.07)
        self.assertEqual(2, len(matches))
        self.assertEqual({(0, 1), (1, 0)}, {(p, t) for p, t, _ in matches})
        self.assertLess(sum(m[2] for m in matches), 0.061)

    def test_matching_maximum_cardinality(self):
        # First prediction may take either GT; second can only take first GT.
        a = ImpactPrediction(0.00, 0.0, 0.9)
        b = ImpactPrediction(-0.04, 0.0, 0.8)
        metrics = evaluate_faces([([a, b], [(-0.02, 0.0), (0.04, 0.0)], None)],
                                 match_radius_ring=0.5)
        self.assertEqual(2, metrics.true_positives)


class CanonicalFixture:
    """Reusable small dataset setup without inheriting parent tests."""
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        Image.new("RGB", (64, 64), color=(50, 100, 150)).save(self.root / "canonical.png")
        Image.new("RGB", (64, 64), color=(95, 35, 140)).save(self.root / "canonical-val.png")
        Image.new("RGB", (64, 64), color=(10, 25, 35)).save(self.root / "raw-train.png")
        Image.new("RGB", (64, 64), color=(15, 55, 75)).save(self.root / "raw-val.png")

    def sample(self, *, split="train", annotations=(ImpactAnnotation(0.1, 0.2),),
               capture=None, group_id="g1"):
        if capture is None:
            crop_name = "canonical-val.png" if split == "val" else "canonical.png"
            capture = {"canonical_faces": [{"face_index": 0, "image_path": crop_name,
                                              "sha256": hashlib.sha256((self.root / crop_name).read_bytes()).hexdigest(),
                                              "annotation_status": "verified"}]}
            if not annotations:
                capture["stage4_review"] = {"verified_empty_faces": [0]}
        return DatasetSample(id=split + "-sample", image_path=self.root / f"raw-{split}.png",
                             annotation_path=None, split=split, source_id="fixture", group_id=group_id,
                             annotations=annotations, capture_metadata=capture)



class CanonicalDataTest(CanonicalFixture, unittest.TestCase):
    def test_fails_closed_on_raw_only_and_unverified(self):
        with self.assertRaisesRegex(ValueError, "canonical impacts without Stage 3 crops"):
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

    def test_negative_faces_require_an_independent_empty_attestation(self):
        approved = self.sample(annotations=(), group_id="empty")
        unreviewed = replace(approved, capture_metadata={
            "canonical_faces": approved.capture_metadata["canonical_faces"],
        })
        inconsistent = replace(approved, capture_metadata={
            **approved.capture_metadata,
            "stage4_review": {"verified_empty_faces": []},
        })
        for sample in (unreviewed, inconsistent):
            with self.subTest(sample=sample.capture_metadata):
                with self.assertRaisesRegex(ValueError, "verified_empty_faces"):
                    CanonicalImpactDataset(DatasetView.from_samples([sample]),
                                           manifest_dir=self.root, input_size=64)
                manifest = self.root / "manifest.json"
                save_manifest(manifest, [sample])
                config = {"dataset": {"manifest": str(manifest),
                                      "snapshot": str(self.root / "snapshot.json")}}
                snapshot = type("Snapshot", (), {
                    "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest()
                })()
                with self.assertRaisesRegex(ValueError, "verified_empty_faces"):
                    verify_dataset(config, snapshot=snapshot)

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

    def test_different_paths_with_identical_canonical_bytes_are_rejected(self):
        # Both raw originals and group ids differ: only model-input bytes overlap.
        (self.root / "canonical-val.png").write_bytes((self.root / "canonical.png").read_bytes())
        samples = [self.sample(split="train", group_id="tr"),
                   self.sample(split="val", group_id="va")]
        manifest = self.root / "manifest.json"
        save_manifest(manifest, samples)
        snapshot = type("Snapshot", (), {
            "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest()
        })()
        cfg = {"dataset": {"manifest": str(manifest), "snapshot": str(self.root / "snapshot.json")}}
        with patch("neural_network.archery_ml.train.load_snapshot", return_value=snapshot):
            with self.assertRaisesRegex(ValueError, "canonical crop SHA256 leakage"):
                _verify_dataset(cfg)

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
        best = run / "best_model.pt"
        metadata_path = run / "model_metadata.json"
        state = torch.load(best, map_location="cpu", weights_only=True)
        self.assertEqual(config["postprocess"], state["postprocess"])
        last = torch.load(run / "last_model.pt", map_location="cpu", weights_only=True)
        self.assertEqual(config["postprocess"], last["postprocess"])
        original_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        for field, replacement in (("confidence_threshold", 0.25), ("nms_radius", 5)):
            changed_metadata = json.loads(json.dumps(original_metadata))
            changed_metadata["postprocess"][field] = replacement
            altered = self.root / f"tampered-{field}.json"
            altered.write_text(json.dumps(changed_metadata), encoding="utf-8")
            with self.subTest(tamper=field):
                with self.assertRaisesRegex(ValueError, "postprocess"):
                    load_detector(best, altered)
                with patch("neural_network.archery_ml.train.load_snapshot", return_value=fake):
                    with self.assertRaisesRegex(ValueError, "postprocess"):
                        evaluate_checkpoint(config_path, best, altered, split="val")
        # Input normalization and output geometry must be immutable as well:
        # jointly editing config and metadata previously bypassed the gate.
        for field in ("mean", "std"):
            changed_metadata = json.loads(json.dumps(original_metadata))
            changed_metadata["input"][field] = [0.3, 0.4, 0.5]
            altered = self.root / f"tampered-input-{field}.json"
            altered.write_text(json.dumps(changed_metadata), encoding="utf-8")
            with self.subTest(input_field=field):
                with self.assertRaisesRegex(ValueError, "input/output"):
                    load_detector(best, altered)
                changed_config = json.loads(json.dumps(config))
                changed_config["model"][f"input_{field}"] = [0.3, 0.4, 0.5]
                changed_config_path = self.root / f"tampered-config-{field}.json"
                changed_config_path.write_text(json.dumps(changed_config), encoding="utf-8")
                with patch("neural_network.archery_ml.train.load_snapshot", return_value=fake):
                    with self.assertRaisesRegex(ValueError, "input/output"):
                        evaluate_checkpoint(changed_config_path, best, altered, split="val")
        inconsistent = json.loads(json.dumps(original_metadata))
        inconsistent["training_dataset_sha256"] = "0" * 64
        bad_provenance_path = self.root / "tampered-training-dataset.json"
        bad_provenance_path.write_text(json.dumps(inconsistent), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "training provenance"):
            load_detector(best, bad_provenance_path)
        legacy_input = dict(state)
        legacy_input.pop("input")
        legacy_input_path = self.root / "legacy-without-input-contract.pt"
        torch.save(legacy_input, legacy_input_path)
        with self.assertRaisesRegex(ValueError, "input/output"):
            load_detector(legacy_input_path, metadata_path)

        legacy = dict(state)
        legacy.pop("postprocess")
        legacy_path = self.root / "legacy-without-postprocess.pt"
        torch.save(legacy, legacy_path)
        with self.assertRaisesRegex(ValueError, "postprocess"):
            load_detector(legacy_path, metadata_path)

        # Changing the approved validation answer without changing the photo
        # or crop must invalidate the original checkpoint's val metrics.
        changed_val = replace(samples[1], annotations=(ImpactAnnotation(0.30, 0.20),))
        save_manifest(manifest, [samples[0], changed_val])
        changed_snapshot = type("Snapshot", (), {
            "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest()
        })()
        with patch("neural_network.archery_ml.train.load_snapshot", return_value=changed_snapshot):
            with self.assertRaisesRegex(ValueError, "val data/annotations differ"):
                evaluate_checkpoint(config_path, best, metadata_path, split="val")
            with self.assertRaisesRegex(ValueError, "val data/annotations differ"):
                evaluate_checkpoint(config_path, best, metadata_path, split="mobile_real_test")
        save_manifest(manifest, samples)

        # A new, internally valid manifest is not an independent holdout if it
        # reuses a photograph that influenced this checkpoint.
        holdout = replace(samples[0], split="mobile_real_test", id="new-holdout")
        save_manifest(manifest, [holdout])
        new_snapshot = type("Snapshot", (), {
            "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest()
        })()
        with patch("neural_network.archery_ml.train.load_snapshot", return_value=new_snapshot):
            with self.assertRaisesRegex(ValueError, "checkpoint train overlaps mobile_real_test"):
                evaluate_checkpoint(config_path, run / "best_model.pt",
                                    run / "model_metadata.json", split="mobile_real_test")
        save_manifest(manifest, samples)
        # No prepared holdout must produce a structured BLOCKED and CLI code 2.
        with patch("neural_network.archery_ml.train.load_snapshot", return_value=fake):
            blocked = evaluate_checkpoint(config_path, run / "best_model.pt",
                                          run / "model_metadata.json", split="mobile_real_test")
            self.assertEqual("BLOCKED", blocked["quality_gate"]["status"])
            from neural_network.archery_ml.train import main
            from unittest.mock import patch as mock_patch
            with mock_patch("sys.argv", ["stage4", "--config", str(config_path),
                                         "--evaluate", str(run / "best_model.pt"),
                                         "--metadata", str(run / "model_metadata.json"),
                                         "--split", "mobile_real_test"]):
                self.assertEqual(2, main())
        with patch("neural_network.archery_ml.train.load_snapshot", return_value=fake):
            second = train(config_path, output_root=self.root / "runs")
        first_weights = torch.load(run / "best_model.pt", map_location="cpu", weights_only=True)["model_state"]
        second_weights = torch.load(second / "best_model.pt", map_location="cpu", weights_only=True)["model_state"]
        self.assertEqual(first_weights.keys(), second_weights.keys())
        self.assertTrue(all(torch.equal(first_weights[k], second_weights[k]) for k in first_weights))


class ReviewRegressions(CanonicalFixture, unittest.TestCase):
    def test_changed_crop_bytes_and_missing_hash_rejected(self):
        valid = self.sample()
        image = self.root / "canonical.png"
        image.write_bytes(image.read_bytes() + b"tampered")
        with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
            CanonicalImpactDataset(DatasetView.from_samples([valid]),
                                   manifest_dir=self.root, input_size=64)
        entry = {"face_index": 0, "image_path": "canonical.png", "annotation_status": "verified"}
        with self.assertRaisesRegex(ValueError, "missing canonical crop SHA256"):
            CanonicalImpactDataset(DatasetView.from_samples([
                self.sample(capture={"canonical_faces": [entry]})]),
                manifest_dir=self.root, input_size=64)

    def test_crop_tampering_during_epoch_rejected(self):
        dataset = CanonicalImpactDataset(DatasetView.from_samples([self.sample()]),
                                         manifest_dir=self.root, input_size=64)
        (self.root / "canonical.png").write_bytes((self.root / "canonical.png").read_bytes() + b"t")
        with self.assertRaisesRegex(ValueError, "SHA256 mismatch during loading"):
            dataset[0]

    def test_non_rgb_canonical_crop_rejected(self):
        image = self.root / "canonical.png"
        Image.new("L", (64, 64), color=100).save(image)
        record = self.sample()
        with self.assertRaisesRegex(ValueError, "not RGB"):
            CanonicalImpactDataset(DatasetView.from_samples([record]),
                                   manifest_dir=self.root, input_size=64)

    def test_wrong_crop_resolution_and_path_escape_rejected(self):
        record = self.sample()
        with self.assertRaisesRegex(ValueError, "dimensions mismatch"):
            CanonicalImpactDataset(DatasetView.from_samples([record]),
                                   manifest_dir=self.root, input_size=128)
        entry = {"face_index": 0, "image_path": "/etc/hosts", "sha256": "0" * 64,
                 "annotation_status": "verified"}
        with self.assertRaisesRegex(ValueError, "escapes data workspace"):
            CanonicalImpactDataset(DatasetView.from_samples([
                self.sample(capture={"canonical_faces": [entry]})]),
                manifest_dir=self.root, input_size=64)

    def test_triple_missing_face_and_unmaterialized_canonical_labels_rejected(self):
        one = self.sample()
        triple = TargetMetadata(format="TRIPLE", minimum_scoring_zone=6,
                                ten_ring_mode="RECURVE", triple_layout="VERTICAL")
        one = DatasetSample(id=one.id, image_path=one.image_path,
                            annotation_path=one.annotation_path, split=one.split,
                            source_id=one.source_id, group_id=one.group_id,
                            annotations=one.annotations, capture_metadata=one.capture_metadata,
                            target_metadata=triple)
        with self.assertRaisesRegex(ValueError, "incomplete canonical face layout"):
            CanonicalImpactDataset(DatasetView.from_samples([one]),
                                   manifest_dir=self.root, input_size=64)
        with self.assertRaisesRegex(ValueError, "canonical impacts without Stage 3 crops"):
            CanonicalImpactDataset(DatasetView.from_samples([
                self.sample(capture={})]), manifest_dir=self.root, input_size=64)

    def test_normalization_and_inference_must_use_exact_size(self):
        data = CanonicalImpactDataset(DatasetView.from_samples([self.sample()]),
                    manifest_dir=self.root, input_size=64, augment=False,
                    input_mean=(0.5, 0.5, 0.5), input_std=(0.25, 0.25, 0.25))
        self.assertAlmostEqual((50 / 255 - 0.5) / 0.25,
                               float(data[0]["image"][0, 0, 0]), places=5)
        with self.assertRaisesRegex(ValueError, "normalization"):
            CanonicalImpactDataset(DatasetView.from_samples([self.sample()]),
                                   manifest_dir=self.root, input_size=64,
                                   input_std=(0, 1, 1))
        metadata = ModelMetadata(contract_version=2, model_version="v1",
                 input=ModelInputSpec(64, 64),
                 output=ModelOutputSpec("impact_heatmap_offset", 2),
                 postprocess=ModelPostprocessSpec(0.4, 2))
        with self.assertRaisesRegex(ValueError, "exact model input size"):
            predict_canonical_face(ImpactPointNet(fpn_channels=8), metadata,
                                   Image.new("RGB", (120, 120)))

    def test_same_image_bytes_under_different_filenames_cannot_cross_splits(self):
        (self.root / "raw-val.png").write_bytes((self.root / "raw-train.png").read_bytes())
        samples = [self.sample(split="train", group_id="train"),
                   self.sample(split="val", group_id="val")]
        manifest = self.root / "manifest.json"
        save_manifest(manifest, samples)
        snap = type("Snapshot", (), {"manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest()})()
        config = {"dataset": {"manifest": str(manifest), "snapshot": str(self.root / "snapshot.json")}}
        with patch("neural_network.archery_ml.train.load_snapshot", return_value=snap):
            with self.assertRaisesRegex(ValueError, "SHA256 leakage"):
                _verify_dataset(config)

    def test_duplicate_samples_and_image_leakage(self):
        samples = [self.sample(split="train"), replace(self.sample(split="val", group_id="different"),
                                                        image_path=self.root / "raw-train.png")]
        manifest = self.root / "manifest.json"
        save_manifest(manifest, samples)
        snap = type("Snapshot", (), {"manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest()})()
        cfg = {"dataset": {"manifest": str(manifest), "snapshot": str(self.root / "snapshot.json")}}
        with patch("neural_network.archery_ml.train.load_snapshot", return_value=snap):
            with self.assertRaisesRegex(ValueError, "raw image leakage"):
                _verify_dataset(cfg)
        save_manifest(manifest, [samples[0], samples[0]])
        snap.manifest_sha256 = hashlib.sha256(manifest.read_bytes()).hexdigest()
        with patch("neural_network.archery_ml.train.load_snapshot", return_value=snap):
            with self.assertRaisesRegex(ValueError, "duplicate manifest sample ID"):
                _verify_dataset(cfg)


class CheckpointProvenanceTest(CanonicalFixture, unittest.TestCase):
    def test_mobile_holdout_cannot_count_reused_crop_as_independent_original(self):
        sample_a = replace(self.sample(split="train", group_id="holdout-a"),
                           id="holdout-a", split="mobile_real_test")
        sample_b = replace(self.sample(split="train", group_id="holdout-b"),
                           id="holdout-b", split="mobile_real_test",
                           image_path=self.root / "raw-val.png")
        manifest = self.root / "holdout-manifest.json"
        save_manifest(manifest, [sample_a, sample_b])
        cfg = {"dataset": {"manifest": str(manifest),
                           "snapshot": str(self.root / "holdout-snapshot.json")}}
        snapshot = type("Snapshot", (), {
            "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest()
        })()
        with self.assertRaisesRegex(ValueError, "identical canonical crop reused"):
            verify_dataset(cfg, snapshot=snapshot)

    def test_separate_holdout_manifest_reusing_canonical_bytes_is_rejected(self):
        manifest = self.root / "manifest.json"
        snapshots = self.root / "snapshot.json"
        training = [self.sample(split="train", group_id="training"),
                    self.sample(split="val", group_id="validation")]
        save_manifest(manifest, training)
        cfg = {"dataset": {"manifest": str(manifest), "snapshot": str(snapshots)}}
        snap = type("Snapshot", (), {
            "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest()
        })()
        saved = training_provenance(verify_dataset(cfg, snapshot=snap))
        checkpoint = {"training_provenance": saved}
        metadata = {"training_provenance_sha256": fingerprint(saved)}
        # The original photograph is different, while the canonical model
        # input is byte-identical to one used to fit the checkpoint.
        new_original = self.root / "raw-new.png"
        Image.new("RGB", (64, 64), (7, 77, 107)).save(new_original)
        holdout = replace(training[0], split="mobile_real_test", id="holdout",
                          group_id="independent", image_path=new_original)
        save_manifest(manifest, [holdout])
        snap.manifest_sha256 = hashlib.sha256(manifest.read_bytes()).hexdigest()
        evaluation = verify_dataset(cfg, snapshot=snap)
        with self.assertRaisesRegex(ValueError, "canonical_crop_sha256"):
            verify_checkpoint_holdout(checkpoint, metadata, evaluation, split="mobile_real_test")
        with self.assertRaisesRegex(ValueError, "lacks versioned training/validation annotation provenance"):
            verify_checkpoint_holdout({}, metadata, evaluation, split="mobile_real_test")
        with self.assertRaisesRegex(ValueError, "metadata training provenance differ"):
            verify_checkpoint_holdout(checkpoint, {"training_provenance_sha256": "0" * 64},
                                      evaluation, split="mobile_real_test")


class ValidationAnnotationProvenanceTest(CanonicalFixture, unittest.TestCase):
    def test_one_validation_coordinate_changes_only_label_digest(self):
        manifest = self.root / "manifest.json"
        original = [self.sample(split="train", group_id="tr"),
                    self.sample(split="val", group_id="va")]
        save_manifest(manifest, original)
        config = {"dataset": {"manifest": str(manifest),
                              "snapshot": str(self.root / "snapshot.json")}}
        snapshot = type("Snapshot", (), {
            "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest()
        })()
        initial = verify_dataset(config, snapshot=snapshot)
        saved = training_provenance(initial)
        checkpoint = {"training_provenance": saved}
        metadata = {"training_provenance_sha256": fingerprint(saved)}
        verify_checkpoint_holdout(checkpoint, metadata, initial, split="val")
        revised = replace(original[1], annotations=(ImpactAnnotation(0.30, 0.20),))
        save_manifest(manifest, [original[0], revised])
        snapshot.manifest_sha256 = hashlib.sha256(manifest.read_bytes()).hexdigest()
        modified = verify_dataset(config, snapshot=snapshot)
        for key in ("original_sha256", "canonical_crop_sha256", "groups"):
            self.assertEqual(initial["splits"]["val"][key],
                             modified["splits"]["val"][key])
        self.assertNotEqual(initial["splits"]["val"]["annotation_sha256"],
                            modified["splits"]["val"]["annotation_sha256"])
        with self.assertRaisesRegex(ValueError, "val data/annotations differ"):
            verify_checkpoint_holdout(checkpoint, metadata, modified, split="val")
        with self.assertRaisesRegex(ValueError, "val data/annotations differ"):
            verify_checkpoint_holdout(checkpoint, metadata, modified, split="mobile_real_test")
        previous = {**saved, "version": 1}
        with self.assertRaisesRegex(ValueError, "versioned training/validation annotation provenance"):
            verify_checkpoint_holdout({"training_provenance": previous},
                                      {"training_provenance_sha256": fingerprint(previous)},
                                      initial, split="val")


class ScoringMetadataProvenanceTest(CanonicalFixture, unittest.TestCase):
    def test_changed_scoring_contract_invalidates_validation_metrics(self):
        original_val = replace(self.sample(split="val", group_id="va"),
                               target_metadata=TargetMetadata(
                                   format="SINGLE", minimum_scoring_zone=1,
                                   ten_ring_mode="RECURVE", face_diameter_mm=400))
        originals = [self.sample(split="train", group_id="tr"), original_val]
        manifest = self.root / "manifest.json"
        save_manifest(manifest, originals)
        config = {"dataset": {"manifest": str(manifest),
                              "snapshot": str(self.root / "snapshot.json")}}
        snapshot = type("Snapshot", (), {
            "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest()
        })()
        baseline = verify_dataset(config, snapshot=snapshot)
        saved = training_provenance(baseline)
        checkpoint = {"training_provenance": saved}
        metadata = {"training_provenance_sha256": fingerprint(saved)}
        verify_checkpoint_holdout(checkpoint, metadata, baseline, split="val")

        # Both the canonical target image and GT coordinates stay unchanged,
        # but scoring mode changes the validation metric and its provenance.
        adjusted_val = replace(original_val, target_metadata=TargetMetadata(
            format="SINGLE", minimum_scoring_zone=1,
            ten_ring_mode="COMPOUND", face_diameter_mm=400))
        save_manifest(manifest, [originals[0], adjusted_val])
        snapshot.manifest_sha256 = hashlib.sha256(manifest.read_bytes()).hexdigest()
        current = verify_dataset(config, snapshot=snapshot)
        for field in ("original_sha256", "canonical_crop_sha256", "groups"):
            self.assertEqual(baseline["splits"]["val"][field],
                             current["splits"]["val"][field])
        self.assertNotEqual(baseline["splits"]["val"]["annotation_sha256"],
                            current["splits"]["val"]["annotation_sha256"])
        with self.assertRaisesRegex(ValueError, "val data/annotations differ"):
            verify_checkpoint_holdout(checkpoint, metadata, current, split="val")


class QualityGateTest(unittest.TestCase):
    def _metrics(self):
        return {"precision": 0.98, "recall": 0.98, "true_positives": 98,
                "false_negatives": 2, "localization_median_ring": 0.1,
                "localization_p95_ring": 0.2, "exact_score_accuracy_non_line_call": 0.99,
                "false_positives_per_empty_face": 0.0, "empty_faces": 10,
                "score_eligible_count": 50}

    def test_holdout_requires_sufficient_independent_evidence(self):
        metrics = self._metrics()
        self.assertEqual("BLOCKED", mobile_quality_gate(metrics, unique_images=299)["status"])
        self.assertEqual("PASS", mobile_quality_gate(metrics, unique_images=300)["status"])
        metrics["localization_median_ring"] = None
        self.assertEqual("BLOCKED", mobile_quality_gate(metrics, unique_images=300)["status"])

    def test_holdout_quality_fails_on_accuracy_or_empty_false_positives(self):
        metrics = self._metrics()
        metrics["recall"] = 0.95
        self.assertEqual("FAIL", mobile_quality_gate(metrics, unique_images=300)["status"])
        metrics["recall"] = 0.98
        metrics["false_positives_per_empty_face"] = 0.1
        self.assertEqual("FAIL", mobile_quality_gate(metrics, unique_images=300)["status"])


class EvaluationCLIStatusTest(unittest.TestCase):
    def test_holdout_nonpassing_status_exits_nonzero(self):
        from neural_network.archery_ml.train import main
        from unittest.mock import patch as mock_patch
        args = ["archery_ml.train", "--config", "config.json", "--evaluate", "model.pt",
                "--metadata", "model_metadata.json", "--split", "mobile_real_test"]
        for status, expected_exit in (("PASS", 0), ("FAIL", 1), ("BLOCKED", 2)):
            with self.subTest(status=status):
                evaluation = {"split": "mobile_real_test", "quality_gate": {"status": status}}
                with mock_patch("sys.argv", args), mock_patch(
                    "neural_network.archery_ml.train.evaluate_checkpoint", return_value=evaluation
                ), redirect_stdout(io.StringIO()) as output:
                    self.assertEqual(expected_exit, main())
                self.assertIn(f'"status": "{status}"', output.getvalue())

    def test_validation_not_treated_as_mobile_gate(self):
        from neural_network.archery_ml.train import main
        args = ["archery_ml.train", "--config", "config.json", "--evaluate", "model.pt",
                "--metadata", "model_metadata.json", "--split", "val"]
        with patch("sys.argv", args), patch(
            "neural_network.archery_ml.train.evaluate_checkpoint", return_value={"split": "val"}
        ), redirect_stdout(io.StringIO()):
            self.assertEqual(0, main())


if __name__ == "__main__":
    unittest.main()
