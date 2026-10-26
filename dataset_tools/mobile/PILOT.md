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

Use `--baseline` only for a new output revision to preserve a first-run snapshot;
later ordinary runs generate `comparison.json`. Existing baseline artifacts
should not be overwritten. Reuse of RGB copies verifies decoded pixels against
freshly read originals. Original SHA256 and EXIF transforms are in the inventory.
