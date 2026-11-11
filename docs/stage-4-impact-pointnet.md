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
- Images **must already** be canonical 512×512 RGB faces, not resized by the trainer. `face_index` must identify the same local face in canonical `ImpactAnnotation`; triples require three distinct crop records with indices `0,1,2`. An empty face is a negative sample **only** if it is explicitly present and independently reviewed: `capture_metadata.stage4_review.verified_empty_faces` must list exactly those face indices having no canonical impact GT. The bridge persists this review evidence, and both the loader and split verifier reject omitted or inconsistent assertions. A zero-annotation face without this evidence cannot improve the false-accept metric.
- `source_id` and `group_id` are mandatory for crop-bearing samples. The manifest/snapshot SHA256 must agree, sample IDs are unique, and source+group, original-image paths/SHA256 **and exact SHA256 of the 512×512 canonical crop bytes** cannot cross splits, including `mobile_real_test`. `mobile_real_test` is independent and is not used by training/validation/checkpoint selection.
- One heatmap cell can encode at most one impact offset. Collisions are **rejected** at target creation rather than silently hiding a shot. Such dense impacts require a different supervision strategy or higher resolution.
- Captured `annotation_status="verified"` is a **machine-readable assertion**, not proof that someone has independently checked the photograph and its GT. The external Stage 3 provenance/independent review gate remains necessary.

If Stage 3 only has raw images or geometry without independently verified impact coordinates, **Stage 4 real training and acceptance are BLOCKED**. Do not fabricate annotation status, fill missing faces, or tune from holdout.

## Stage 3 → 4 reviewed materialization

The included bridge can transform *already reviewed* raw-image impact coordinates using Stage 3's selected-geometry `original_to_canonical` homographies and bind Stage 3's **existing raw (unmasked) canonical crop artifacts** by checksum. This is not a model-driven pseudo-labeling or ownership classifier. Its inputs must be independently reviewed.

Generate a **draft** review queue from existing raw annotations to avoid manually calculating SHA-256 or re-entering arrow coordinates:

```bash
python -m archery_ml.prepare --draft-review-map data/stage4-review-draft.json
```

This produces an **unapproved** map: ownership is false, face IDs are unset, and reviewer identities are empty. Inspect the Stage 3 target selection and each raw arrow; assign the selected face ID or `null` for a foreign board, record the approved geometry JSON path and hash, then obtain independent annotation/ownership review. You may use `--include-unlabeled` to add candidate negatives, but only a reviewer can confirm that a face is actually empty. No model prediction is used to fabricate labels.

Stage 3's native `geometry-results.json` is a **dictionary keyed by original camera-file SHA256**, not a single result. The materializer can read that index directly: use it as `geometry_path`, provide its full-file SHA256, and supply the exactly matching `selected_candidate_ids` in the independently signed-off review map. The chosen entry must have `original_id == original_sha256`, `selection_status=USER_CONFIRMED`, `ownership_confirmed=true` and matching selected candidate IDs. Unselected group previews, AUTO_CONFIDENT proposals and foreign results are rejected even if their transforms are geometrically READY.

To export one Stage 3 result to a self-contained, relocatable JSON record (with crop paths recalculated relative to the exported file), use:

```bash
python -m archery_ml.prepare \
  --export-geometry-results data/stage3/geometry-results.json \
  --original-sha256 <64-hex-camera-file-sha256> \
  --output-geometry data/stage4-selected/<sha256>.json
```

This export does **not** approve the impact/ownership review. Record the output artifact path and returned hash in the separate review map; the subsequent materialization performs the same identity and candidate checks. The original image must not be silently renamed or converted before computing its SHA256.

The resulting JSON map is keyed by `DatasetSample.id`, for example:

```json
{
  "sample-id": {
    "annotator": "reviewer-A",
    "reviewer": "reviewer-B",
    "face_ownership_reviewed": true,
    "original_sha256": "<64-char SHA256 of source photo>",
    "raw_annotations_sha256": "<SHA256 of sorted-key compact JSON of raw_annotations array>",
    "geometry_path": "stage3/selected/result.json",
    "geometry_sha256": "<SHA256 of the selected Stage 3 result JSON>",
    "selected_candidate_ids": ["candidate-1"],
    "impact_face_indices": [0, 0, null],
    "verified_empty_faces": []
  }
}
```

`geometry_path` is resolved relative to the review-map directory; geometry JSON must have `status=READY`, `quality_status=READY`, an `artifact_revision`, and `faces[i].crop_path` (relative to the geometry JSON directory) plus a `faces[i].original_to_canonical` 3x3 homography. `impact_face_indices` must independently assign **every** raw annotation, in source order, to a selected face index; `null` explicitly marks an arrow belonging to a foreign board, which is not included in Stage 4 training. This is necessary when photos contain multiple targets or the source has `face_index=null`. The raw source annotations remain unchanged and hash-bound. Triples require confirmed physical face order. `verified_empty_faces` is the list of independently checked empty face indices. A missing impact label is **not** proof of an empty target. Both review identities must be distinct nonblank strings. Reviewers should not enter invented verification values.

```bash
python -m archery_ml.prepare \
  --review-map data/stage4-review.json \
  --manifest data/manifests/dataset_manifest.json \
  --snapshot data/snapshots/dataset_snapshot.json \
  --output-manifest data/manifests/stage4_manifest.json \
  --output-snapshot data/snapshots/stage4_snapshot.json
```

The original Stage 2.5 manifest/snapshot remain unchanged. The bridge validates source photo, review digest, selected geometry and crop geometry; transforms review-bound original-image impact points to canonical face points (center 255.5 px, radius 256 px); then writes a **new** manifest/snapshot with crop hashes and parent source hashes. Training re-verifies crop bytes on every read. It refuses missing reviewed ownership, missing or ambiguous per-impact face assignments, invalid homographies, unconfirmed empty faces, non-RGB/incorrect crop sizes, or reusing output files. It cannot retroactively produce true impact annotations from the 58 unlabelled historical photos.

## Reproducible workflow

Install dependencies from `requirements.txt` in a Python 3.11 environment (not every baseline CI job installs PyTorch). Verify a real Stage 3 canonical export, produce derived Stage 4 manifest and snapshot with the bridge above, and inspect a representative sample and every split before running:

```bash
python -m archery_ml.train --config neural_network/configs/impact_mobilenetv3.json
```

Deterministic training pins the PyTorch CPU thread pool to one thread, preventing variation in multithreaded reductions; this is a deliberate throughput trade-off. Record hardware and versions before interpreting speed.

All parameters are config-driven: seed, input size, FPN width, pretrained backbone flag, batch size, epochs, optimizer, Gaussian sigma, horizontal reflection, brightness and postprocess threshold/NMS. No geometric warps that would invalidate impact labels are applied. Per-channel `model.input_mean` and `model.input_std` are shared by the dataset, visualization, saved contract and inference. The production config enables **ImageNet V2 pretrained weights** with mean `[0.485, 0.456, 0.406]` and std `[0.229, 0.224, 0.225]` (validated as a pair); the first training run may download these public weights. Set `pretrained=false` only for an intentionally scratch-trained/offline experiment, not as an assumed production-quality baseline.

Each checkpoint embeds the original-image SHA256, canonical-crop SHA256 and source/group IDs from **both training and validation**, plus a versioned SHA256 of canonical ground-truth annotations for every prepared face (including verified empty faces). The annotation digest includes `sample_id`, `source_id`, `group_id`, original/crop SHA256, `face_index`, sorted impact annotations with their exact coordinates **and the face scoring/target metadata** (`format`, `minimum_scoring_zone`, `ten_ring_mode`, `triple_layout`, `face_diameter_mm`). This prevents changing the scoring context of the validation set without invalidating checkpoint provenance. The model metadata binds this provenance with a SHA256 of its canonical JSON record. Re-evaluating `val` fails if any annotation changes, even when photo/crop bytes and group IDs are unchanged. For `test` and `mobile_real_test`, the checker rejects overlap with train/val and also rejects changed train/val annotations whenever those splits are present in the evaluation manifest. A holdout-only manifest is still permitted; independent holdout GT must be separately versioned and audited. The new provenance schema is version 2: older checkpoints with missing/legacy GT provenance are rejected for independent evaluation rather than silently accepted.

Each run writes `runs/impact-<UTC timestamp>/` with `config.json`, `dataset_info.json` (Git commit, manifest and snapshot hashes, face/sample counts), `model_metadata.json` (contract v2, normalization, model version, source manifest hash), `metrics.csv`, `metrics.json`, `last_model.pt`, `best_model.pt` and per-epoch validation overlays. Only `train` samples affect gradients, only `val` selects the best checkpoint.

Both `confidence_threshold` and `nms_radius` are stored in **every** best/last checkpoint. The same checkpoint now binds the **input contract** (image size, RGB color space, per-channel mean/std) and **output contract** (head type and stride). Detector loading for both reference inference and evaluation compares these saved contracts with `model_metadata.json` and refuses missing or changed values. It additionally verifies the metadata's training-provenance digest and dataset SHA256 against the checkpoint. Editing the evaluation config alongside metadata therefore cannot silently change image normalization. Existing checkpoints without these bindings must be retrained or explicitly migrated with independently verified original settings; no silent fallback is allowed.

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

Evaluation computes one-to-one, maximum-cardinality, minimum-total-distance point matching within **0.5 ring** (one scoring ring = 0.1 canonical radial units). Reports precision, recall, F1, FP per verified empty face, median and P95 radial localization error in ring units, exact scoring accuracy excluding uncertain `line_call` GT, and a correction-rate proxy. Scoring uses deterministic Stage 1 scoring fixtures when target diameter/metadata is available; lacking scoring metadata **does not imply** score accuracy passed.

The `mobile_real_test` quality gate returns `BLOCKED` if there are fewer than **300 unique verified original-image SHA256 hashes**, zero impact GT, no verified empty faces, zero score-eligible non-line-call shots, or any missing metric. With sufficient data, it returns `PASS` only if precision/recall ≥ 97%, median ≤ 0.15 ring, P95 ≤ 0.35 ring, exact non-line-call score ≥ 98%, and no false accepts on verified empty faces; otherwise `FAIL`. The zero-empty-FP policy is conservative; adjust by documented product requirements rather than hiding failures. A missing prepared `mobile_real_test` split returns a structured `BLOCKED` report (with zero faces and a reason), instead of throwing the dataset-loader exception. The CLI returns exit code 0 for `PASS`, 1 for `FAIL`, and 2 for `BLOCKED` on `mobile_real_test`, so an automated release job cannot mistake missing evidence for success. Normal `val`/`test` metrics remain informational. Derived Stage 4 snapshots recompute total impacts using the Stage 2 convention `sum(max(canonical_count, raw_count))`, so the same independently reviewed arrow is not counted twice.

The 58-image Stage 3 exploratory set is *not* a holdout and cannot establish this gate. Real stage acceptance also requires an independently reviewed mobile dataset, verified capture sessions, face ownership and reproducible provenance. Synthetic tests only establish engineering behavior, **not detection accuracy on real archery photographs**.

## Verification and deliberate exclusions

The focused regression test is `OMP_NUM_THREADS=1 python -m unittest discover -s neural_network/tests -p 'test_stage4_*.py' -v`; the dedicated PR workflow runs it with locked Torch/TorchVision on CPU. It checks network shapes and gradients, negative faces, offsets, NMS ties, matching cardinality, split leakage, manifest/crop tampering, triplets, inference dimensions, a single deterministic training epoch and checkpoint reuse. Existing Stage 0–3 tests and Android remain authoritative for legacy behavior.

Excluded: automatic unreviewed Stage 3 crop/impact-label creation, actual multi-epoch training against real sources, Android deployment, ExecuTorch parity and claiming quality-gate success. These are separate prerequisites/next stages, not silently completed by this PR.

Sources: [TorchVision MobileNetV3](https://docs.pytorch.org/vision/0.23/models/generated/torchvision.models.mobilenet_v3_large.html), [PyTorch serialization](https://docs.pytorch.org/docs/2.8/notes/serialization.html), [Stage 2.5 dataset](stage-2.5-dataset-infrastructure.md), [Stage 3 finalization](stage-3-finalization-2026-10-26.md), [Notion implementation plan](https://app.notion.com/p/3f099ff957ed8108b097f02f508c7ffd).
