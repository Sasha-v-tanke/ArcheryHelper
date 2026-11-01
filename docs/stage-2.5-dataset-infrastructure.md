# Stage 2.5: real dataset infrastructure

## Goal

Stage 2.5 turns the Stage 2 dataset components into the local data environment used by geometry evaluation and future model training.

Raw-camera annotations are no longer treated as target-canonical coordinates. External datasets may describe impact points, target centers, target faces, or rings in the coordinate system of the original image. Those labels survive import unchanged until Stage 3 supplies target geometry and a homography.

A public source enters the runtime registry only after its exact exported version, license, annotation meaning, and archive checksum have been verified.

## Local layout

The local workspace is rooted at data/ and remains ignored by Git:

~~~text
data/
  sources/<source_id>/<version>/
  dataset_pipeline/cache/
  manifests/dataset_manifest.json
  reports/dataset_report.json
  snapshots/dataset_snapshot.json
  previews/
~~~

Checked-in code, registry metadata, tests, and documentation stay in Git. Dataset images and derived local artifacts do not.

## Annotation spaces

DatasetSample separates canonical impacts, raw image impacts, and geometry ground truth.

Canonical annotations use ImpactAnnotation and the target-local coordinate system: center (0, 0), face radius 1.0, +x right, +y down.

Raw annotations use ImagePointAnnotation with x_fraction and y_fraction in [0, 1]. They may also preserve the source label, source bbox, and face index. A raw image point is never converted to target xNorm/yNorm by simply remapping the whole photo to [-1, 1].

Geometry annotations use ImageGeometryAnnotation with roles target_center, target_face, and ring. COCO polygons are preserved; a geometry bbox falls back to its four corners.

## Registry contract

A real raw-camera source declares annotation_space=image. Geometry roles are valid only in that space.

YOLO impact sources support bbox_center and keypoint extraction. Keypoint mode accepts keypoint_index and keypoint_dimensions so YOLO pose exports can preserve the actual impact point.

dataset_tools/source_candidates.json contains public datasets worth evaluating. It is not consumed by the pipeline. dataset_tools/sources.json remains the approved runtime registry.

A candidate moves to sources.json only after:

1. exact source/export version is known;
2. author and canonical source page are verified;
3. license is verified;
4. export can be reproduced or downloaded;
5. the archive SHA-256 is known;
6. annotation semantics are visually checked on a representative batch;
7. class mapping is confirmed;
8. overlap with approved sources is inspected.

## Full rebuild

The local full-data smoke path is:

~~~bash
python -m archery_ml.data rebuild
~~~

Selected sources can be rebuilt with repeated --source arguments and a fixed --seed.

The command performs:

~~~text
registry
  -> source materialization
  -> import
  -> validation
  -> exact/perceptual deduplication
  -> group-aware split
  -> manifest
  -> report
  -> snapshot
~~~

Unlike repeated single-source import calls, rebuild starts from the selected registry sources and does not depend on a stale previous manifest.

## Dataset snapshot

The deterministic snapshot records:

- snapshot, pipeline, and dataset schema versions;
- manifest SHA-256;
- registry SHA-256;
- split seed and ratios;
- perceptual dedup threshold;
- selected source ids, versions, licenses, authors, checksums, annotation spaces, and tasks;
- report counts.

No timestamp is stored, so identical inputs and configuration produce identical snapshot JSON.

Future training runs should record the manifest/snapshot hash rather than only a filesystem path.

## Model-agnostic consumer

DatasetView is the common read API for Stage 3 and Stage 4. It filters by split, source, target format, canonical impacts, raw impacts, and geometry ground truth.

Stage 3 uses raw_annotations and geometry_annotations to evaluate normalization. Stage 4 should train only from canonical impacts.

## Inspect and preview

Inspect metadata:

~~~bash
python -m archery_ml.data inspect --raw-only --with-geometry --limit 20
~~~

Render annotation overlays:

~~~bash
python -m archery_ml.data preview --raw-only --with-geometry --limit 20
~~~

Preview outputs annotated JPEGs plus data/previews/index.json. Raw impacts, canonical impacts, target centers, and target/ring geometry use distinct overlays so source mappings can be checked before approval.

## CI

Normal CI stays network-independent. test_data_*.py uses synthetic and minimal local fixtures.

Full external datasets are intentionally not downloaded on each pull request. The rebuild command is the opt-in local full-data smoke path.

## Stage 3 integration

The intended transformation is:

~~~text
raw image point
  -> detected or ground-truth target geometry
  -> Stage 3 homography
  -> canonical face point
  -> ImpactAnnotation(x_norm, y_norm, face_index)
~~~

This keeps geometry error separately measurable from future impact-model error.

## Stage 4 integration

Stage 4 training should accept a manifest/snapshot instead of a hardcoded dataset directory. A raw-image impact is useful for Stage 3 evaluation, but is not silently treated as an ImpactPointNet target.

## Current gate

The reusable Stage 2.5 infrastructure is ready when tests and CI pass. Real-source onboarding remains a data-governance gate: sources.json stays empty until at least two candidates have verified export artifacts, checksums, and annotation semantics.
