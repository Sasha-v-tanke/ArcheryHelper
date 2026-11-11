# Stage 3 geometry validation — 2026-10-22

**Status: implementation improved; Stage 3 data gate remains BLOCKED.** Independent mobile geometry GT and real triple/partial-face configuration coverage are unavailable. Do not treat this audit as authorization to pass the geometry prerequisite for Stage 4 training.

Baseline: `52c418d412cbac5a020301a5dde2336fa752c88b`. Geometry implementation: [`c092af7`](https://github.com/Sasha-v-tanke/ArcheryHelper/commit/c092af7eea59f9b6dbb76376d44daf265bf8212c). Evaluation/audit: [`ee41102`](https://github.com/Sasha-v-tanke/ArcheryHelper/commit/ee41102). PR: [#8](https://github.com/Sasha-v-tanke/ArcheryHelper/pull/8), targeting `feature/stage-3-target-normalization`.

Requirements reviewed: [detailed specification](https://app.notion.com/p/3f099ff957ed80dfa7fee0ff509b70bf), [implementation plan](https://app.notion.com/p/3f099ff957ed8108b097f02f508c7ffd).

## Confirmed causes and corrections

1. Cool black rings entered the blue HSV/Lab mask. A measured boundary at radius 0.8 was interpreted as 0.6: the resulting scale factor around 0.75 explains the approximately 0.25 outer-ring radial errors. The Fig9/Fig11/Fig12/Fig13/Fig14 originals, masks, COCO boundaries, and projected rings were inspected. Warm black could similarly contaminate red. Require Lab chroma of at least 25 for red/blue, and blue greater than red for blue. The unchanged 0.97 clipping coverage threshold was not relaxed.
2. Arrow cuts exposed inner boundaries of color annuli, displaced fragment centers, and corrupted raw ellipse fits. Fit deterministic ellipse consensus from measured points; associate blue fragments through the red interior; collect disconnected red/yellow outer arcs near their expected radius. The 0.035 ellipse-relative inlier band is distinct from the normalized quality residual limit.
3. The perspective guard was bypassed for residuals below 0.05. Small errors on inner rings did not constrain outer extrapolation. Apply the guard unconditionally and reject a projective horizon through the nominal face.
4. Visible bounds and center used the original candidate even after refinement/fallback changed the homography. Derive the center, visible ellipse and bounds from the final inverse transform.
5. A weighted confidence could still accept inconsistent rings. Add `GEOMETRY_UNCERTAIN` for per-face RMS error above 0.03, non-finite error, or fewer than two associated rings. It precedes clipping. Public normalization now raises for every rejected status; diagnostic geometry remains accessible.
6. Global focus/exposure depended on surrounding background. Measure these inside visible target ellipses. Synthetic small triple faces exposed the previous false blur rejection.
7. COCO has a separate annotation defect: all three `10ring-fig12` variants swap Ring-1 and Ring-2. Ring-1 ellipse diameters are approximately 0.849/0.854 image fractions; Ring-2 approximately 0.945/0.947. Colored boundaries match the predicted geometry. Flag contradictory cumulative areas, retain every sample and original label in metrics. This explains the train READY radial maximum above 0.1; it is not evidence for changing detector scale.

## Data integrity and split audit

Manifest SHA-256: `448b42cec5163da27e6ebbe5395cb8926d80a3c22070c620bad46270f222405e`. Reimport from all three COCO files exactly matches the manifest raw impacts, geometry points, and target metadata. All image dimensions match COCO. There are 153 images and 1530 ring polygons, ten per image. `target` is ignored by the registry; there are no `target_face` or explicit `target_center` annotations. Ring-N polygons represent cumulative outer boundaries, not annular strips. Radii remain `(11-N)/10`; compound changes the scoring ten-ring radius, not normalization scale.

The manifest intentionally reallocates deduplicated groups rather than preserving export folders. Its 122/15/16 train/val/test samples represent 65 filename originals in 18 conservative dedup groups. There are no cross-split manifest groups, Roboflow original families, exact SHA-256 duplicates, or dHash-distance-at-most-4 pairs. This does not establish capture-session independence or detect every transformed duplicate.

| Export folder → manifest split | Images |
| --- | ---: |
| test->test | 1 |
| test->train | 7 |
| train->test | 15 |
| train->train | 105 |
| train->val | 12 |
| valid->train | 10 |
| valid->val | 3 |

The previous `test` was diagnosed and is development/regression data. No new detector choices were made from the final all-train, test or supplemental measurements. After synthetic/validation development, two complete non-Fig train groups were reserved before inspecting train metrics/images: `BG5Put18tjEUXD3-z-po-_KYCNNI7e` and `VFkjihU71vnZDPBXcbEPo_2uVKvSbh`, six augmented images from only two originals. Selection used SHA-256 order with seed text `stage3-audit-20261022:` and excluded groups containing Fig names. This is supplemental evidence; prior historical exposure and capture provenance are unknown. It is not a new independent holdout.

The other local source, `archery_scoring_keypoints/v4`, has 438 images with arrow tip/nock keypoints, but no ring or homography GT. The two supplied ZIPs contain already known validation images/diagnostics, not new independent data. Inventing GT from the detector itself would make the check circular.

## Baseline and final metrics

All values are full-radius normalized units; one ordinary ring width is 0.1. Radial metrics pool annotation vertices, center metrics pool images. Every selected image remains in the all-image metrics, including rejected images and contradictory GT. Baseline detector and quality gate were loaded unchanged from 52c418d into the same evaluator for partition comparisons; the original baseline val/test reports were also preserved locally.

| Split | Images | READY before → after | Radial mean before → after | Median before → after | P95 before → after | Max before → after |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| train | 122 | 97 → 112 | 0.042334 → 0.011703 | 0.011623 → 0.007305 | 0.232863 → 0.031147 | 0.454724 → 0.237487 |
| val | 15 | 15 → 15 | 0.011795 → 0.011262 | 0.008017 → 0.008569 | 0.035115 → 0.031602 | 0.096385 → 0.070356 |
| test | 16 | 3 → 14 | 0.132629 → 0.010895 | 0.132801 → 0.008002 | 0.293448 → 0.032095 | 0.394233 → 0.087445 |
| supplemental | 6 | 6 → 6 | 0.012344 → 0.011137 | 0.008640 → 0.008388 | 0.035461 → 0.030794 | 0.131884 → 0.061424 |

| Split | Center P95 before → after | READY radial P95 before → after | Rejected radial P95 before → after | Final Ring-1 P95 | Final Ring-10 P95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| train | 0.037574 → 0.035116 | 0.041063 → 0.025532 | 0.410309 → 0.146956 | 0.057581 | 0.012893 |
| val | 0.040419 → 0.038608 | 0.035115 → 0.031602 | — → — | 0.053962 | 0.013414 |
| test | 0.044047 → 0.046212 | 0.039285 → 0.022652 | 0.306806 → 0.059515 | 0.052421 | 0.014775 |
| supplemental | 0.023867 → 0.025205 | 0.035461 → 0.030794 | — → — | 0.043421 | 0.009051 |

Center error is deliberately computed against the same outer-GT-ellipse center proxy as the baseline. In a projective photograph that proxy differs from the projected physical bullseye; therefore the small test/supplemental center-P95 increases do not by themselves establish worse physical-center localization. Exact center/point mapping is checked with known synthetic homographies. Radial accuracy alone also does not bound angular or full 2D impact error.

All per-original-family final metrics and full ring-specific statistics are in the [metric snapshot](reports/stage-3-validation-2026-10-22.json). For the diagnosed test families:

| Original family | Images | READY | Baseline radial P95 | Final radial P95 |
| --- | ---: | ---: | ---: | ---: |
| 10ring-fig11 | 3 | 3 | 0.039285 | 0.021723 |
| fig11 | 3 | 3 | 0.254311 | 0.031168 |
| fig12 | 3 | 3 | 0.257607 | 0.024024 |
| fig13 | 3 | 2 | 0.356783 | 0.040881 |
| fig14 | 1 | 1 | 0.251098 | 0.018437 |
| fig9 | 3 | 2 | 0.255686 | 0.049542 |

## Rejections and clipping

Final v9 results have 12 `GEOMETRY_UNCERTAIN` frames (10 train, two diagnosed test), all with residuals above 0.03. They are conservatively rejected for inconsistent recovered rings, not declared physically clipped. Several have reasonable GT radial error but insufficient internal consistency. No v9 frame exceeds the unchanged clipping coverage rule after uncertainty rejection. This is not a claim that every paper margin is intact: the rule measures model bounding-box coverage and tolerates up to 3% loss. The previous 13 test `TARGET_CLIPPED` statuses were not reliable evidence of physical clipping because the scale and bounds were wrong.

Physical clipping is tested separately on synthetic images whose full visible target extends beyond the frame. Such a frame returns `TARGET_CLIPPED` and public normalization raises before producing inference input. Tests also verify uncertainty precedes clipping and prevents normalization.

| Split | Rejected file | RMS residual |
| --- | --- | ---: |
| train | `10ring-fig5_png.rf.7ffe43aea5bbab1c4cc4eb10b9edf4a3.jpg` | 0.068491 |
| train | `10ring-fig5_png.rf.f981b9916eefef6999c7057aa413d7ec.jpg` | 0.068787 |
| train | `Fig10_jpg.rf.20e7c5350eed681b4a360b3130a7a575.jpg` | 0.067762 |
| train | `Fig10_jpg.rf.4a21ab4c20ed0c2169b0f431f5d40bf5.jpg` | 0.067947 |
| train | `Fig10_jpg.rf.e3c2355ed63521611ec8a09cdf25e039.jpg` | 0.067411 |
| train | `Fig7_jpg.rf.57ca582743e20fbb64f8a9172be3a6b7.jpg` | 0.068511 |
| train | `Fig8_jpg.rf.96f70576de65e16e78ffd5952dbb1568.jpg` | 0.069370 |
| train | `NaggRo3W659n_vTeZ7Lwf_ZHkg63G9_png.rf.9a320cc0c6f4dddeff72ba935b64d643.jpg` | 0.044309 |
| train | `NaggRo3W659n_vTeZ7Lwf_ZHkg63G9_png.rf.ada1612eb920b5ff33de9ddfc4017305.jpg` | 0.053017 |
| train | `NaggRo3W659n_vTeZ7Lwf_ZHkg63G9_png.rf.daa251746711f37a2b45527964701d7a.jpg` | 0.044594 |
| test | `Fig13_jpg.rf.3d138e88e787d89547d891004c5631d7.jpg` | 0.068469 |
| test | `Fig9_jpg.rf.c0ccba7e8d1d84ff700b33a790b0388b.jpg` | 0.068948 |

## Checks and reproduction

- Python: `python -m unittest discover -s neural_network/tests` — 68 tests passed; 25 geometry/evaluation/audit tests. The legacy interactive model scripts are imported by discovery but their manual prediction routines are not executed.
- Geometry tests cover tint/cyan/pink, arrow cuts, obstructed blue, low-residual unstable extrapolation, final-transform bounds/center, actual clipping, uncertainty/normalization rejection, and single/triple layouts. All triple zone 1/5/6 and RECURVE/COMPOUND combinations are checked under a known projective transform; existing synthetic scale/rotation/noise/blur/exposure checks remain.
- Android: `cd app && ./gradlew testDebugUnitTest assembleDebug --stacktrace` — BUILD SUCCESSFUL. Existing tasks were up-to-date locally; exact-PR-head GitHub Actions provides the fresh hosted build. Check runs are linked from PR #8.
- `git diff --check` — passed. Dataset images and large preview files are not committed. Input COCO hashes, group/sample selection, metrics and rejection filenames are recorded in the JSON snapshot.

Run the read-only split audit:

```bash
python -m neural_network.archery_ml.geometry.audit \
  --manifest data/manifests/dataset_manifest.json \
  --source archery_aiscore_ounar \
  --output data/reports/geometry_split_audit.json
```

Run geometry evaluation separately for train/val/test, keeping each output:

```bash
python -m archery_ml.geometry \
  --manifest data/manifests/dataset_manifest.json \
  --source archery_aiscore_ounar --split val \
  --minimum-scoring-zone 1 --ten-ring-mode RECURVE \
  --output data/reports/geometry_final_val.json \
  --preview-dir data/previews/geometry_final_val
```

Local raw evidence is under `data/reports/stage3-audit/`: original `baseline/{train,val,test}.json`, enriched baseline/final reports, supplemental selection/results, split audit, and COCO integrity. Previews are under `data/previews/stage3-audit/`. Reports are diagnostic, never inputs to inference.

## Remaining gate before Stage 4

1. Obtain a new immutable mobile set with independently annotated target geometry/landmarks, session/original provenance, real singles and both triple layouts, all supported visible-zone configurations and adverse cases. Do not reuse the diagnosed v9 test as holdout.
2. Correct the swapped Ring-1/Ring-2 GT through a versioned annotation change, regenerate the manifest/snapshot, and repeat comparisons while retaining the present raw-label results.
3. Assign and verify an explicit full 2D geometry-error budget relative to scoring/localization tolerance. Current v9 READY P95 values and outer Ring-1 errors are not proof of a production scoring bound or false-accept rate.
4. Freeze the detector and quality policy before that independent evaluation. Calibrate acceptance on development data, confirm rejection behavior and accepted-frame accuracy, and only then close Stage 3 and start the Stage 4 training gate.
