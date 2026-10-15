# Stage 2: unified dataset pipeline

## Goal

Stage 2 replaces the ad-hoc dataset ingestion, conversion, validation, deduplication, and splitting scripts with one reproducible pipeline built on the canonical dataset contract introduced in Stage 1.

The pipeline must make provenance explicit, preserve canonical impact coordinates, prevent train/evaluation leakage, and produce the same manifest for the same source data, registry, and split seed.

This stage does not implement target-face geometry or homography. Raw-camera target normalization remains Stage 3. Stage 2.5 extends this pipeline with explicit raw-image annotations and geometry ground truth so real camera datasets can be imported without pretending that image coordinates are already target-canonical.

## Scope

The stage adds:

- a declarative source registry;
- safe source materialization for local sources and verified archives;
- importers for legacy ArcheryHelper JSON, YOLO, COCO, and Roboflow exports;
- schema and content validation;
- exact and perceptual deduplication;
- group-aware deterministic splitting;
- a dataset report;
- a command-line interface;
- unit and integration tests for the complete local pipeline.

The stage retires the old Roboflow, YOLO, and Kotlin DataConverter scripts after the replacement importers are covered by tests.

The manual tools in `dataset_tools/prepare_new_dataset` are deliberately retained. They are interactive annotation/normalization producers, not the canonical ingestion pipeline, and removing them before the Stage 3 geometry/annotation workflow exists would remove functionality without a replacement.

## Invariants

### Canonical annotations

The Stage 1 contract remains authoritative:

- target center is `(0, 0)`;
- `x_norm` grows to the right;
- `y_norm` grows down;
- nominal face radius is `1.0`;
- triple faces use face-local coordinates and `face_index`;
- `DatasetSample.annotations` contains `ImpactAnnotation` values.

Legacy polar annotations are converted to Cartesian coordinates during import.

YOLO and COCO object annotations are reduced to object-center points. This is only valid when the imported dataset image/label frame is already suitable for the canonical target pipeline; perspective target normalization belongs to Stage 3.

### Provenance

Every registered source has:

- stable source id;
- importer type;
- location or archive URL;
- source version;
- license;
- author;
- allowed task list;
- optional class mapping;
- optional importer-specific settings.

HTTP/HTTPS archives require a SHA-256 checksum. Local source directories are treated as explicit original-source inputs and do not require an archive checksum.

Source metadata must be verified before a real source is added to `dataset_tools/sources.json`. The repository does not fabricate licenses or checksums.

### Deduplication

Deduplication has two layers:

1. exact SHA-256 image equality;
2. perceptual dHash similarity with a BK-tree Hamming-distance lookup.

Exact duplicate images are removed only when their annotations and target metadata agree. Conflicting annotations for the same exact image fail the pipeline.

Near duplicates are not silently removed. They are placed into one effective `group_id`, so the splitter cannot distribute them across train/validation/test.

### Splitting

Splitting is group-aware and deterministic:

- all samples with the same `group_id` remain in one split;
- perceptual duplicate groups remain in one split;
- ordering is derived from `SHA256(seed:group_id)`, not filesystem iteration order;
- `mobile_real_test` is preserved as a separate immutable split;
- `mobile_real_test` samples are excluded from train/validation/test ratio calculations;
- a group that mixes `mobile_real_test` and tuning samples is rejected.

### Deterministic manifest

`save_manifest` writes schema v2 samples in stable `(source_id, id)` order and creates the manifest directory when required.

For identical source data, registry, deduplication threshold, and split seed, the resulting manifest is deterministic.

## Architecture

The canonical implementation lives under:

`neural_network/archery_ml/data/`

A top-level `archery_ml.data` compatibility entry point is provided because the project plan defines the public CLI as `python -m archery_ml.data`, while the Stage 1 Python package physically lives under `neural_network/archery_ml`.

### Registry

`data/registry.py`

Public types:

- `DatasetSource`
- `DatasetRegistry`

Registry file:

`dataset_tools/sources.json`

The checked-in registry starts empty until source provenance, license, version, and checksum/location are verified. A dataset source must not be committed with guessed metadata.

### Source materialization

`data/download.py`

Supports:

- an existing local directory;
- a local archive;
- an HTTP/HTTPS archive with mandatory SHA-256;
- ZIP and TAR extraction;
- extraction path validation;
- a workspace cache keyed by source id/version and archive digest.

### Importers

`data/importers/legacy.py`

Reads the existing ArcheryHelper JSON format and converts either Cartesian or `r_norm/theta_deg` annotations to `ImpactAnnotation`.

`data/importers/yolo.py`

Reads standard YOLO detection labels and converts bounding-box centers to canonical point coordinates.

`data/importers/coco.py`

Reads COCO annotations, preferring a visible keypoint and otherwise using the bounding-box center.

`data/importers/roboflow.py`

Consumes an exported Roboflow dataset by delegating to the COCO or YOLO importer. Dataset download credentials are not embedded in source code.

### Validation

`data/validate.py`

Rejects:

- duplicate sample ids;
- missing images;
- missing annotation files;
- unsupported annotation-file types;
- missing `source_id`;
- unknown registered sources;
- missing `group_id`;
- invalid split values;
- canonical coordinates outside the accepted image bounds;
- invalid `face_index` values for single/triple target metadata.

### Deduplication

`data/deduplicate.py`

Returns `DeduplicationResult` with:

- retained samples;
- exact duplicate ids removed;
- perceptual duplicate groups;
- duplicate rate.

The implementation uses union-find to merge pre-existing groups with perceptual duplicate relationships before splitting.

### Split

`data/split.py`

`assign_splits(...)` accepts a seed and train/validation/test ratios.

`assert_no_group_leakage(...)` is the final invariant check after assignment.

### Report

`data/report.py`

Reports:

- image count;
- impact count;
- source distribution;
- target configuration distribution;
- split sizes;
- exact duplicate count;
- perceptual duplicate group count;
- duplicate rate.

### CLI

Run from the repository root:

```bash
python -m archery_ml.data import --source <id>
python -m archery_ml.data validate
python -m archery_ml.data deduplicate
python -m archery_ml.data split --seed 42
python -m archery_ml.data report
```

Defaults:

- registry: `dataset_tools/sources.json`;
- manifest: `data/manifests/dataset_manifest.json`;
- workspace root: `data/`;
- source materialization: `data/sources/`;
- download cache: `data/dataset_pipeline/cache/`.

The `data/` directory remains excluded from Git.

## Execution flow

Normal flow:

1. Select a verified source from the registry.
2. Materialize its local directory/archive or download and verify its remote archive.
3. Run the configured importer.
4. Replace the previous manifest records for the same `source_id`.
5. Validate the combined manifest.
6. Deduplicate exact images and merge near-duplicate groups.
7. Assign deterministic group-aware splits.
8. Generate a report.
9. Use the resulting schema-v2 manifest as the input contract for later ML work.

Failure policy:

- invalid registry metadata fails before import;
- checksum mismatch fails before extraction;
- unsafe archive members fail extraction;
- malformed source annotations fail import;
- invalid canonical data fails validation;
- conflicting exact duplicates fail deduplication;
- mixed evaluation/tuning groups fail splitting.

No stage silently repairs ambiguous source data.

## Files changed

Core:

- `neural_network/archery_ml/contracts.py`
- `neural_network/archery_ml/data/registry.py`
- `neural_network/archery_ml/data/download.py`
- `neural_network/archery_ml/data/importers/*`
- `neural_network/archery_ml/data/validate.py`
- `neural_network/archery_ml/data/deduplicate.py`
- `neural_network/archery_ml/data/split.py`
- `neural_network/archery_ml/data/report.py`
- `neural_network/archery_ml/data/cli.py`
- `archery_ml/data/*`
- `dataset_tools/sources.json`

Tests/CI:

- `neural_network/tests/test_data_registry.py`
- `neural_network/tests/test_data_importers.py`
- `neural_network/tests/test_data_pipeline.py`
- `neural_network/tests/test_data_cli.py`
- `.github/workflows/baseline.yml`

Retired:

- `dataset_tools/roboflow/*`
- `dataset_tools/yolo/*`
- `dataset_tools/data_converter/*`
- direct `roboflow` Python dependency

Retained until replacement exists:

- `dataset_tools/prepare_new_dataset/*`

## Agent implementation order

1. Preserve the Stage 1 schema and extend only the split name contract with `mobile_real_test`.
2. Introduce and validate the source registry.
3. Implement safe source materialization with checksum verification.
4. Implement legacy, YOLO, COCO, and Roboflow-export importers.
5. Implement dataset validation.
6. Implement exact/perceptual deduplication and group merging.
7. Implement deterministic group-aware splitting.
8. Implement reporting and the public CLI.
9. Add unit/integration coverage for every importer and pipeline invariant.
10. Wire the tests into CI and verify Android baseline remains unchanged.
11. Remove only the legacy scripts whose functionality is covered by the replacement.
12. Keep manual annotation/normalization tooling until its functional replacement is delivered.

## Verification

Python CI covers the pre-existing Stage 1 tests plus all `test_data_*.py` tests.

The dataset tests cover:

- registry validation;
- checksum-verified archive materialization;
- legacy polar conversion;
- YOLO import;
- COCO import;
- Roboflow export import;
- invalid coordinates and face indices;
- exact duplicate removal;
- perceptual duplicate grouping;
- deterministic group-aware split;
- `mobile_real_test` preservation;
- complete CLI flow from local source to report.

The Android baseline remains:

```bash
cd app
./gradlew testDebugUnitTest assembleDebug
```

Room schema cleanliness must remain unchanged.

## Definition of done

Stage 2 implementation is complete when:

- all dataset preparation operations are available through `python -m archery_ml.data`;
- a verified registry plus the corresponding original source data can rebuild the manifest without legacy converters;
- repeated runs with the same seed produce the same split assignment and manifest ordering;
- exact and perceptual duplicates cannot leak across splits;
- `mobile_real_test` cannot enter model tuning splits;
- source provenance is explicit and no credentials are embedded in repository scripts;
- Python dataset tests and the existing Android baseline pass.

Actual external source entries are a data-governance input, not something to guess in code: each entry must be added only after its license, version, author, source location, and checksum (for remote archives) are known.


## Stage 2.5 extension

Real-source onboarding, raw-image annotation preservation, deterministic dataset snapshots, `DatasetView`, and visual inspection tooling are documented in `docs/stage-2.5-dataset-infrastructure.md`.
