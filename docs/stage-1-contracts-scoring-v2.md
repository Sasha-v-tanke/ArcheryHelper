# Stage 1: canonical contracts, TargetConfig, and ScoringEngine v2

## Goal

Stage 1 defines the stable contracts shared by dataset preparation, model work, and the Android application before the geometry, model, and persistence migrations begin.

The stage establishes:

- Cartesian canonical impact coordinates;
- target configuration and target templates;
- deterministic scoring independent from ML;
- dataset schema v2;
- model metadata contract v2 without a public fixed shot-count limit;
- compatibility boundaries for the current PyTorch Lite runtime and Room v1 data.

## Scope

### Python contracts

The reusable contract package lives in `neural_network/archery_ml/`.

It defines:

- `ImpactAnnotation(confidence?, x_norm, y_norm, face_index)`;
- dataset schema v2 fields `source_id`, `group_id`, `split`, `annotations`, and `target_metadata`;
- `TargetConfig`, `TargetTemplate`, and ring boundaries;
- model metadata v2 with input, output, and postprocess sections;
- deterministic scoring types and functions.

`neural_network/contracts.py` remains a compatibility re-export while existing code migrates to the package.

### Android domain

The Android domain defines:

- `TargetFormat`, `TripleLayout`, `TenRingMode`;
- `TargetConfig` and `TargetTemplate`;
- `ShotSource`;
- `TrainingSession` as the root training-domain name, with a temporary `Distance` type alias;
- `Shot` with primary `xNorm/yNorm` coordinates and computed radius/angle;
- `ShotPoint` with `faceIndex` and polar conversion helpers.

### Scoring

ML is not responsible for scoring.

`ScoringEngine.score(targetTemplate, impact, arrowDiameterMm, localizationError)` returns:

- `finalCandidate`;
- `isX`;
- `lineCall`.

Ring boundaries come from `TargetTemplate`. Shaft radius is included for line-cut scoring. Localization uncertainty only marks a line call when it crosses a scoring boundary; it does not silently choose a different final score.

### Compatibility

This stage intentionally does not migrate persistence or replace the on-device runtime.

- Room remains schema v1. The repository converts stored radius/angle values to canonical Cartesian coordinates at the domain boundary.
- SharedPreferences migration converts legacy polar coordinates before creating domain shots.
- The current `model.ptl` exporter continues to emit legacy model metadata v1.
- Public model metadata v2 is separate and contains no `MAX_SHOTS`.
- The legacy `ScoreCalculator` remains as a compatibility adapter for UI code that encodes X as `11`.

## Canonical coordinate invariants

- The target center is `(0, 0)`.
- The full nominal face radius is `1.0`.
- `xNorm` grows to the right.
- `yNorm` grows down in image/application coordinates.
- Triple faces use coordinates local to the selected face.
- `faceIndex` identifies the face and does not change scoring mathematics.
- Radius and angle are derived representations, not source-of-truth fields.

## Target-template invariants

For a classic recurve face:

- 10 outer radius: `0.1`;
- 9 outer radius: `0.2`;
- ...
- 1 outer radius: `1.0`.

For compound:

- the 10 boundary is supplied by the template and is `0.05` for the current 40 cm indoor template;
- the remaining ring boundaries retain their nominal radii.

`minimumScoringZone` supports `1`, `5`, and `6`.

## Shared verification fixtures

Python and Android scoring tests read the same fixture:

`app/app/src/test/resources/scoring_v2.csv`

It covers:

- recurve X and 10;
- compound 10 and 9;
- exact ring boundaries;
- misses;
- reduced-zone triple faces;
- shaft line cuts;
- localization uncertainty and `lineCall`;
- triple-face local coordinates.

## Files changed

Python:

- `neural_network/archery_ml/contracts.py`
- `neural_network/archery_ml/targets.py`
- `neural_network/archery_ml/scoring.py`
- `neural_network/model_metadata.py`
- `neural_network/contracts.py`
- `model_converter/convert_pth.py`
- Python contract/scoring tests

Android:

- `domain/ArcheryModels.kt`
- `stats/ScoringEngine.kt`
- Room v1 compatibility mapping
- SharedPreferences compatibility mapping
- edit/statistics coordinate consumers
- Android scoring tests and shared fixtures

CI:

- baseline workflow includes the scoring v2 Python tests.

## Verification

Required checks:

```bash
for test_file in \
  test_contracts.py \
  test_evaluation.py \
  test_predictions.py \
  test_scoring_v2.py \
  test_splits.py \
  test_training_config.py
do
  python -m unittest discover -s neural_network/tests -p "$test_file"
done
```

```bash
cd app
./gradlew testDebugUnitTest assembleDebug
```

Room schema cleanliness must remain unchanged.

## Deferred work

The following items belong to later stages and are deliberately excluded here:

- OpenCV target-face geometry and homography;
- unified dataset converters;
- new impact-localization model architecture;
- ExecuTorch runtime integration;
- Room schema v2 and explicit migration;
- target-configuration UI;
- removal of legacy PyTorch Lite and compatibility aliases.
