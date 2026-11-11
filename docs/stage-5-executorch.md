# Stage 5 — ExecuTorch/XNNPACK export and cross-runtime parity

## Contract and scope

Export a **trained Stage 4** `best_model.pt` without altering weights, network topology, normalization, output stride, or postprocessing. The frozen source `model_metadata.json` is contract v2 and **must** match the checkpoint's version-3 training provenance, input/output contract, and postprocessing settings. Export refuses inconsistent artifacts before writing a release.

The exported program consumes a **pre-normalized** `float32[1,3,512,512]` tensor. It returns `heatmap_logits[1,1,256,256]` and `offsets[1,2,256,256]`, **not** final scores or a fixed number of shots. Android is responsible for the exact RGB/255→(value−mean)/std transformation specified in the metadata and for applying the same deterministic `decode_impacts` semantics. The runtime must not silently resize, change channel order, or adjust the postprocessing threshold/NMS radius.

This stage does not implement Android inference, alter Room, or remove PyTorch Mobile (`model.ptl`).

## Isolated export environment

The Stage 4 training environment currently pins PyTorch 2.8 / TorchVision 0.23. The Stage 5 export toolchain is isolated and pinned to PyTorch 2.10, TorchVision 0.25 and ExecuTorch 1.5.1; the Stage 4 checkpoint is loaded strictly in the export environment. The project can continue training with its own pinned packages. Actual runtime compatibility must be established by CI and by the final golden parity run, not presumed from package versions.

```bash
python3.11 -m venv .venv-export
. .venv-export/bin/activate
python -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.10.0 torchvision==0.25.0
python -m pip install numpy==2.2.6 pillow==11.3.0 executorch==1.5.1
python -m unittest discover -s neural_network/tests -p 'test_stage5_*.py' -v
```

If installation, compilation, delegation, or Python ExecuTorch runtime is missing, export **fails**, and no release directory is published. The synthetic smoke test in CI uses random initialized network weights and **does not** establish trained-model quality.

## Real golden manifest

Create 20–50 **already verified** Stage 3 canonical face RGB PNGs from `val`/`test`, independent of training samples and **not from `mobile_real_test`**. For each, preserve original camera-photo SHA256, canonical crop file SHA256, face ownership/index, and the reviewed selection. Never use synthetic crops as substitute for the final real parity gate. The manifest is a JSON object with `schema_version: 1` and `samples`:

```json
{
  "schema_version": 1,
  "samples": [{
    "sample_id": "reviewed-face-001",
    "image_path": "images/reviewed-face-001.png",
    "crop_sha256": "<64 hexadecimal lowercase characters>",
    "original_sha256": "<64 hexadecimal lowercase characters>",
    "face_index": 0,
    "split": "val"
  }]
}
```

`image_path` resolves **inside** the golden manifest directory; symlink traversal, hash mismatch, invalid PNG/RGB, wrong 512×512 dimensions, duplicate identifiers/crop hashes/original-face identities, and unexpected splits are rejected. The example shows one record for readability; an actual export requires 20–50. Review and provenance remain obligations outside this manifest: its field names cannot themselves certify independent annotation/ownership review.

## Export and verify

```bash
python -m archery_ml.export \
  --checkpoint runs/<run>/best_model.pt \
  --metadata runs/<run>/model_metadata.json \
  --golden-manifest data/stage5-golden/golden.json \
  --output-directory runs/<run>/stage5-release

python -m archery_ml.export --verify-release runs/<run>/stage5-release
```

The output directory **must not already exist**. A failed check leaves no release directory. The successful release contains:

```text
stage5-release/
  impact_model.pte
  model_metadata.json        # Stage 4 metadata + bound XNNPACK PTE SHA256 and parity result
  parity.json                # actual eager vs ExecuTorch error metrics, versions and hashes
  sha256.txt                 # SHA256 of every other package file
  golden/
    index.json               # images, original identities, expected points and tensor hashes
    images/*.png             # byte-exact input copies
    expected/*.npz           # eager heatmap logits and offset tensors
```

Release maxima are bounded by the fixed code-reviewable tolerances: logits max absolute `0.01`, offsets max absolute `0.01`, decoded XY distance `0.005` in normalized target units, decoded confidence absolute `0.01`. Matched point count must agree exactly for each face. Non-finite tensors and shape/type changes fail the gate. Release size limit for the `.pte` alone is **30,000,000 bytes**; any separate Android package-size policy must be checked later.

The `sha256.txt` manifest protects files **from accidental or unsigned changes**, but SHA256 alone is not a digital signature or an authenticity mechanism. A trusted build source/distribution is still required. Raw camera photos and private training datasets must not be committed; the bundle should be treated as a release artifact, not a source-code change.

## Acceptance boundaries

- **Engineering smoke**: CI must compile and run the real XNNPACK delegate on 20 deterministic synthetic 512×512 images (this does not prove useful detection quality).
- **Stage 5 golden parity**: actual trained, immutable Stage 4 checkpoint + independently reviewed 20–50 canonical crops must pass real eager/ExecuTorch tensor and point checks. Record exact checkpoint SHA256, PTE SHA256, environment and per-output maxima.
- **Stage 4 detection quality gate**: independent original-photo holdout and production localization metrics must pass separately. A green Stage 5 smoke is **not** approval to deploy the model.
- **Android parity and removal of PyTorch Mobile**: belongs to Stage 6, not this PR.

Reference documentation: [ExecuTorch getting started](https://docs.pytorch.org/executorch/stable/getting-started-setup.html), [Python runtime](https://docs.pytorch.org/executorch/1.5/runtime-python-api-reference.html).
