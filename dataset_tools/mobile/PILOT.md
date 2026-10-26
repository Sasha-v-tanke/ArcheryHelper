# Historical mobile-photo pilot

Originals stay read-only. Reports and photographs belong in gitignored `data/`
or outside the repository. These historical data are `mobile_dev /
legacy_exploratory`, never a preregistered independent holdout.

```sh
python -m archery_ml.geometry.pilot --input /path/to/photos \
  --output data/reports/mobile-pilot-58 --selection-mode interactive --serve
```

Open <http://127.0.0.1:8765/review.html>. To resume an existing report without
reprocessing every image:

```sh
python -m archery_ml.geometry.pilot --input /path/to/photos \
  --output data/reports/mobile-pilot-58 --serve-only
```

`Select` previews a group; `Confirm` records ownership and recomputes its
geometry. Alternatively click one or three candidate faces in the desired
index order. `Uncertain` and `No target` explicitly withhold selection.
Automatic proposals never become user confirmations. Unknown camera/session,
scoring mode and physical size remain unknown. Indices follow image layout
until the user separately confirms their physical identity.

Remembered choices require a confirmed anchor, matching actual EXIF device/time
and reliable background feature correspondence. The transfer is only an
automatic suggestion and does not create session provenance. Ambiguous
transfers are withheld. No ownership inference uses arrow counts.

The local server saves `target-selections.json` and `visual-reviews.json` and
regenerates results. A standalone `review.html` also works offline: export JSON
or connect the output directory, then rerun with `--selections saved.json` for
new crops. Keep the entire output folder together. The server only binds to
loopback; personal photos are not uploaded.

`original_to_canonical` operates on original stored image pixels, including the
EXIF conversion. `rgb_to_canonical` operates on upright RGB pixels. Canonical
pixels use center 255.5 and radius 256. Their inverse matrices and a numerical
round-trip check are recorded. Raw canonical crops are retained; display crops
mask pixels beyond the inferred visible target radius for convenient review.
For zone 5/6 faces, the display mask follows supported blue-to-paper transitions
instead of clipping at an ideal circle. Short occluded arcs are interpolated;
insufficient angular support is explicitly `BOUNDARY_UNCERTAIN` and falls back
to the nominal radius. Zone 1 retains white scoring areas. This segmentation
does not change the raw crop, canonical coordinates, homography or READY gate,
and must not be interpreted as improved independent geometry accuracy.
Foreign detected faces are masked before normalization, never in the original.
Unseen foreign faces cannot be guaranteed absent: visually review each group.

If a physically measured board is visible, add `--board-template path/to/board.json`.
The template is pinned in the output directory and reused by interactive updates.
Only observed, verified markers establish ANCHORED axes and physical face IDs;
missing or incompatible markers leave UNANCHORED with an explicit calibration reason.
Evaluation landmarks are never passed into this calibration step.

Run the same command again with `--verify-replay` to verify the saved selection,
original hashes, algorithm source digest, dependencies and all result artifacts.
`run-manifest.json` records the binding; `reproducibility.json` reports VERIFIED
or MISMATCH (nonzero exit). Only processing durations are excluded from result
comparison. The first run records NOT_CHECKED. Algorithm changes during a run
fail explicitly. Offline selections receive the same identity/configuration checks
as interactive confirmations; stale detections require confirmation again.
Replay recomputes in a temporary sibling directory. On MISMATCH or a processing
exception the original output files remain byte-for-byte unchanged; mismatch
details go to `<output-name>-replay-mismatch.json` beside the output directory.
Successful verification updates only `reproducibility.json`. CLI-supplied board
and independent-GT files are staged there as well, never inserted into the
original output before replay verification.

Saved VISUAL_TRACKING_FROM_USER proposals survive a repeated batch when their
fingerprint, IDs and configuration are still valid. They remain AUTO_CONFIDENT,
with ownership and physical face order unconfirmed. A changed fingerprint
requires confirmation again.

`Correct` is a qualitative visual rating, not independent GT. Face order and
ownership require their own confirmation. No marker calibration is inferred
from concentric rings; these results remain `UNANCHORED`.

`representative-sampling.json` selects up to 15 actual frames.
`pseudo-gt-proposals.json` contains detector-generated suggestions only. For
independently measured points, pass `--independent-gt independent.json` with:

```json
{
  "kind": "independent_gt",
  "method": "physical_measurement",
  "reference": "independent survey record identifier",
  "revision": "survey revision",
  "annotator": "person A",
  "reviewer": "person B",
  "images": [{
    "original_sha256": "SHA256 from inventory.json",
    "candidate_ids": ["f0"],
    "coordinate_space": "original_raw_px",
    "faces": [{
      "face_index": 0,
      "image_points_px": [[100,100],[200,100],[200,200],[100,200]],
      "canonical_points": [[-0.2,-0.2],[0.2,-0.2],[0.2,0.2],[-0.2,0.2]]
    }]
  }]
}
```

The example numbers are a schema illustration, not measurements. Supply at
least four independently measured, distinct noncollinear points for each
measured face. Calibration control points alone are not evaluation GT. Partial
coverage and missing faces remain explicit. No quantitative accuracy is reported
without such data. Final Stage 3 still requires independent real-photo coverage,
GT audit, physical calibration and an approved frozen policy.
Repeated original records or repeated landmark coordinates are rejected, so
duplicating good observations cannot improve pilot metrics. Partial coverage,
READY images without measured GT and rejected images remain explicit.
With measured landmarks, each face also reports separate affine scale bias,
center offset, rotation error, anisotropy and residual. This diagnostic fit never
corrects detector predictions; residual is not a direct measurement of camera
perspective. The evaluator separately retains physical-center error, observed
ellipse perspective ratio and visible coverage. Without GT these errors stay
unmeasured; detector fit residual and round-trip checks are not substitutes.

Use `--baseline` only for a new output revision to preserve a first-run snapshot;
later ordinary runs generate `comparison.json`. Existing baseline artifacts
should not be overwritten. Reuse of RGB copies verifies decoded pixels against
freshly read originals. Original SHA256 and EXIF transforms are in the inventory.
