# Stage 4 — ImpactPointNet (Python)

## Purpose and scope

This stage adds an unordered impact-point detector, without modifying Android, ExecuTorch export, legacy ResNet18 consumers or Room. The old implementation remains available until the later parity/migration gate.

Reference architecture: 512×512 RGB canonical face → TorchVision MobileNetV3-Large feature extractor → top-down depthwise FPN → 256×256 heatmap logits and 2-channel subpixel offset grid. The model has **no fixed shot count**. Its heatmap predicts candidate locations, and the offset head refines each cell; NMS and thresholds come from the versioned model metadata. The score itself is never predicted by the network.

Implementation: `neural_network/archery_ml/{models/impact_net.py,heatmap.py,postprocess.py,impact_dataset.py,train.py,inference.py,evaluate.py}`. Entry points from repository root: `python -m archery_ml.train` and `python -m archery_ml.inference`.

## Dataset contract and critical prerequisites

Stage 2.5's canonical **manifest and snapshot** are required. The Stage 2.5 registry's `annotation_space=image` impact labels are **not canonical impacts**. A raw camera image must first pass Stage 3 quality/ownership checks and be transformed into a *separate canonical crop for every target face*, with correctly transformed impact coordinates. A directory of original photographs is not accepted by the Stage 4 loader.

For each sample with verified canonical impacts the Stage 3 materialization step must append `capture_metadata.canonical_faces` to its manifest entry:

```json
{
  "canonical_faces": [
    {
      "face_index": 0,
      "image_path": "../canonical_faces/example-face0.png",
      "sha256": "<sha256 hex digest of this exact crop>",
      "annotation_status": "verified"
    }
  ]
}
```

- Paths are relative to the **manifest's directory** or absolute within its data workspace. Crop filenames must be stored under `data/`, not in Git. Absolute paths outside the `data/` workspace, symlinks escaping it, or missing/mismatched hashes are rejected.
- Images **must already** be canonical 512×512 RGB faces, not resized by the trainer. `face_index` must identify the same local face in canonical `ImpactAnnotation`; triples require three distinct crop records with indices `0,1,2`. An empty face is a negative sample **only** if it is explicitly present and verified, not because an image lacks annotations.
- `source_id` and `group_id` are mandatory for crop-bearing samples. The manifest/snapshot SHA256 must agree, sample IDs are unique, and source+group and original image paths cannot cross splits. `mobile_real_test` is independent and is not used by training/validation/checkpoint selection.
- One heatmap cell can encode at most one impact offset. Collisions are **rejected** at target creation rather than silently hiding a shot. Such dense impacts require a different supervision strategy or higher resolution.
- Captured `annotation_status="verified"` is a **machine-readable assertion**, not proof that someone has independently checked the photograph and its GT. The external Stage 3 provenance/independent review gate remains necessary.

If Stage 3 only has raw images or geometry without independently verified impact coordinates, **Stage 4 real training and acceptance are BLOCKED**. Do not fabricate annotation status, fill missing faces, or tune from holdout.

## Reproducible workflow

Install dependencies from `requirements.txt` in a Python 3.11 environment (not every baseline CI job installs PyTorch). Verify a real Stage 3 canonical export, rebuild the manifest and snapshot together so `snapshot.manifest_sha256` binds the updated sample metadata, and inspect a representative sample and every split before running:

```bash
python -m archery_ml.train --config neural_network/configs/impact_mobilenetv3.json
```

All parameters are config-driven: seed, input size, FPN width, pretrained backbone flag, batch size, epochs, optimizer, Gaussian sigma, horizontal reflection, brightness and postprocess threshold/NMS. No geometric warps that would invalidate impact labels are applied. Optional per-channel `model.input_mean` and `model.input_std` are shared by the dataset, visualization, saved contract and inference. If using pretrained ImageNet V2 weights, configure mean `[0.485, 0.456, 0.406]`, std `[0.229, 0.224, 0.225]` and allow the weights download explicitly.

Each run writes `runs/impact-<UTC timestamp>/` with `config.json`, `dataset_info.json` (Git commit, manifest and snapshot hashes, face/sample counts), `model_metadata.json` (contract v2, normalization, model version, source manifest hash), `metrics.csv`, `metrics.json`, `last_model.pt`, `best_model.pt` and per-epoch validation overlays. Only `train` samples affect gradients, only `val` selects the best checkpoint.

After training, **without retuning** the checkpoint's threshold and NMS:

```bash
python -m archery_ml.train \
  --config neural_network/configs/impact_mobilenetv3.json \
  --evaluate runs/impact-<id>/best_model.pt \
  --metadata runs/impact-<id>/model_metadata.json \
  --split mobile_real_test

python -m archery_ml.inference \
  --checkpoint runs/impact-<id>/best_model.pt \
  --metadata runs/impact-<id>/model_metadata.json \
  --canonical-face data/canonical_faces/example-face0.png \
  --face-index 0
```

### Metrics and acceptance

Evaluation computes one-to-one, maximum-cardinality point matching within **0.5 ring** (one scoring ring = 0.1 canonical radial units). Reports precision, recall, F1, FP per verified empty face, median and P95 radial localization error in ring units, exact scoring accuracy excluding uncertain `line_call` GT, and a correction-rate proxy. Scoring uses deterministic Stage 1 scoring fixtures when target diameter/metadata is available; lacking scoring metadata **does not imply** score accuracy passed.

The `mobile_real_test` quality gate returns `BLOCKED` if there are fewer than **300 unique verified original image IDs**, zero impact GT, no verified empty faces, zero score-eligible non-line-call shots, or any missing metric. With sufficient data, it returns `PASS` only if precision/recall ≥ 97%, median ≤ 0.15 ring, P95 ≤ 0.35 ring, exact non-line-call score ≥ 98%, and no false accepts on verified empty faces; otherwise `FAIL`. The zero-empty-FP policy is conservative; adjust by documented product requirements rather than hiding failures.

The 58-image Stage 3 exploratory set is *not* a holdout and cannot establish this gate. Real stage acceptance also requires an independently reviewed mobile dataset, verified capture sessions, face ownership and reproducible provenance. Synthetic tests only establish engineering behavior, **not detection accuracy on real archery photographs**.

## Verification and deliberate exclusions

The focused regression test is `OMP_NUM_THREADS=2 python -m unittest discover -s neural_network/tests -p test_stage4_impact.py -v`; the dedicated PR workflow runs it with locked Torch/TorchVision on CPU. It checks network shapes and gradients, negative faces, offsets, NMS ties, matching cardinality, split leakage, manifest/crop tampering, triplets, inference dimensions, a single deterministic training epoch and checkpoint reuse. Existing Stage 0–3 tests and Android remain authoritative for legacy behavior.

Excluded: automatic Stage 3 crop materialization (needs approved, independently reviewed GT and ownership), actual multi-epoch training against real sources, Android deployment, ExecuTorch parity and claiming quality-gate success. These are separate prerequisites/next stages, not silently completed by this PR.

Sources: [TorchVision MobileNetV3](https://docs.pytorch.org/vision/0.23/models/generated/torchvision.models.mobilenet_v3_large.html), [PyTorch serialization](https://docs.pytorch.org/docs/2.8/notes/serialization.html), [Stage 2.5 dataset](stage-2.5-dataset-infrastructure.md), [Stage 3 finalization](stage-3-finalization-2026-10-26.md), [Notion implementation plan](https://app.notion.com/p/3f099ff957ed8108b097f02f508c7ffd).
