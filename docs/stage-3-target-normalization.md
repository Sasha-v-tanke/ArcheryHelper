# Stage 3: target geometry and canonical normalization

## Goal

Stage 3 introduces a Python reference pipeline that turns a raw target photograph into deterministic target geometry and one canonical 512x512 image per detected face before any impact-point model is involved.

The stage separates geometry error from future ML localization error. A frame is accepted only when the target geometry and image quality are good enough to preserve the canonical coordinate contract introduced in Stage 1.

## Scope

This stage adds:

- color-based target-face detection for single and triple targets;
- concentric-ring fitting and a per-face projective transform;
- deterministic face ordering for vertical and triangular triples;
- canonical 512x512 rectification;
- an explicit image quality gate with finite rejection statuses;
- synthetic regression tests for rotation, scale, perspective, face layouts, blur, exposure, clipping, and target scale;
- CI coverage for the geometry package.

This stage does not add ImpactPointNet, model training, ExecuTorch export, Android OpenCV, or Android inference.

## Invariants

### Input image

The Python geometry API accepts an RGB `numpy.ndarray` with shape `H x W x 3`.

### Canonical coordinates

The Stage 1 coordinate contract remains authoritative:

- target center is `(0, 0)`;
- `x_norm` grows to the right;
- `y_norm` grows down;
- nominal full-face radius is `1.0`;
- canonical output size is 512x512;
- normalized coordinate `(0, 0)` maps to the canonical image center;
- one normalized radius maps to 256 canonical pixels.

The visible radius may be smaller than `1.0` for cropped target faces. The transform still preserves the full-face normalized coordinate system rather than rescaling a five-zone or six-zone face to look like a full face.

### Geometry ownership

`TargetGeometryDetector` owns detection and fitting. It does not modify the image.

`TargetNormalizer` only applies geometry returned by the detector. It does not re-detect a target or silently repair bad geometry.

`ImageQualityGate` owns acceptance/rejection. It receives the image and the already computed geometry result and returns exactly one finite status.

### ML isolation

No geometry class imports or calls a neural-network model. Future impact inference consumes canonical faces produced by this package.

## Public API

```python
TargetGeometryDetector.detect(image, target_config) -> GeometryResult
TargetNormalizer.normalize(image, geometry_result) -> list[CanonicalFace]
ImageQualityGate.evaluate(image, geometry_result) -> QualityResult
```

`GeometryResult` contains:

- detected face centers;
- per-face visible ellipse fit;
- per-face original-to-canonical homography and its inverse;
- geometry confidence;
- normalized ring-fit error;
- visible bounds;
- expected face count;
- triple-layout validity;
- source image dimensions.

`QualityResult.status` is one of:

- `READY`;
- `TARGET_NOT_FOUND`;
- `WRONG_FACE_COUNT`;
- `TARGET_CLIPPED`;
- `EXCESSIVE_PERSPECTIVE`;
- `TOO_BLURRY`;
- `BAD_EXPOSURE`;
- `TARGET_TOO_SMALL`.

## Detection algorithm

1. Convert the RGB image to HSV and Lab.
2. Build robust yellow, red, and blue masks using HSV hue/saturation ranges with Lab chroma hints.
3. Apply small morphology operations to suppress isolated noise and reconnect ring fragments.
4. Find plausible outer blue-ring contours. Fall back to the union of target-color masks when blue segmentation is incomplete.
5. Fit ellipses and reject candidates with implausible area or axis ratio.
6. Associate yellow, red, and blue ring contours with each face candidate.
7. Initialize an image-to-normalized transform from the detected outer ellipse.
8. Iteratively fit a projective transform against known ring radii:
   - yellow outer boundary: 0.2;
   - red outer boundary: 0.4;
   - blue outer boundary: `min(0.6, visible_radius_norm)`.
9. Measure normalized radial residuals after projection.
10. Extrapolate the visible-face ellipse from the detected colored radius.
11. Validate and deterministically order triple-face centers according to `TripleLayout`.
12. Build the original-to-canonical 512x512 homography for each face.

## Geometry confidence

Per-face confidence combines:

- normalized ring-fit residual;
- number of independently fitted color rings;
- visible-bound coverage;
- apparent target scale;
- perspective axis ratio.

The aggregate result also penalizes wrong face count and invalid triple layout. Confidence is diagnostic; the quality gate remains the only authority for READY/reject decisions.

## Quality gate

The gate evaluates rejection conditions in deterministic order:

1. no usable geometry or invalid expected layout -> `TARGET_NOT_FOUND`;
2. detected count differs from expected -> `WRONG_FACE_COUNT`;
3. visible target bounds leave the image -> `TARGET_CLIPPED`;
4. ellipse axis ratio is below the perspective threshold -> `EXCESSIVE_PERSPECTIVE`;
5. target diameter is below the minimum frame fraction -> `TARGET_TOO_SMALL`;
6. Laplacian variance is below the blur threshold -> `TOO_BLURRY`;
7. mean luminance or clipped-pixel fraction is outside exposure limits -> `BAD_EXPOSURE`;
8. otherwise -> `READY`.

Thresholds are centralized in `QualityThresholds`, not duplicated across detector and tests.

## Failure policy

- Invalid image shape raises `ValueError`; this is a programmer/input-contract error.
- Failure to find a target is represented by an empty/low-confidence `GeometryResult`, not an exception.
- Missing ring colors reduce confidence and fitting precision but do not crash detection if a valid face contour remains.
- The normalizer returns one canonical image per detected face and never invents missing faces.
- A failed quality gate must prevent downstream impact inference.

## Files

Core:

- `neural_network/archery_ml/geometry/target_templates.py`
- `neural_network/archery_ml/geometry/detector.py`
- `neural_network/archery_ml/geometry/normalize.py`
- `neural_network/archery_ml/geometry/quality.py`
- `neural_network/archery_ml/geometry/__init__.py`
- `archery_ml/geometry/__init__.py`

Tests and build:

- `neural_network/tests/test_geometry_pipeline.py`
- `requirements.txt`
- `.github/workflows/baseline.yml`

The existing manual `dataset_tools/prepare_new_dataset/normalize-app.py` remains until the automatic reference pipeline is validated on the user's real photographs. Removing the manual fallback before that parity gate would violate the migration strategy.

## Verification

Synthetic tests cover:

- single-face detection;
- vertical and triangular triple ordering;
- scale and rotation;
- projective distortion;
- canonical point mapping;
- all quality statuses;
- blur and exposure metrics.

The geometric mapping assertion is expressed in normalized-coordinate error so it can be compared directly with later impact-localization tolerances.

Real-photo validation is a separate data gate. The checked-in dataset registry currently contains no verified real source, so this change must not fabricate a fixture or provenance record. Once `mobile_real_test` contains verified photographs, geometry error must be measured on that split separately from model error.

## Agent implementation order

1. Add geometry templates and canonical transform constants derived from the existing `TargetConfig`.
2. Implement immutable geometry result types and face candidate detection.
3. Add ring association and iterative projective fitting.
4. Add triple-layout validation and deterministic face indices.
5. Implement canonical rectification.
6. Implement the finite-status quality gate.
7. Add synthetic target rendering tests and projective mapping assertions.
8. Add OpenCV to the Python runtime/test dependencies and CI.
9. Keep legacy manual normalization until a real-photo parity gate is available.

## Definition of done

Stage 3 code is complete when:

- the three public APIs are available;
- single and triple targets produce deterministic per-face geometry;
- original-to-canonical transforms preserve Stage 1 normalized coordinates under synthetic rotation/scale/perspective tests;
- every documented quality status is covered by a regression test;
- geometry tests run in CI alongside the existing Stage 1/2 and Android baselines;
- no ML model is required for target normalization;
- the remaining real-photo validation dependency is explicit and isolated as a data-quality gate rather than hidden in ML accuracy.
