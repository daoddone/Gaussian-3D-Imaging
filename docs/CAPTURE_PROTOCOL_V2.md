# CAPTURE PROTOCOL v2 — clinical anatomy capture for the splat→mesh pipeline

Verified against the four Tepole/Gosain papers (docs/Adrian Tepole Gosain Publications/) via a
multi-agent extraction + adversarial verification pass (2026-07-09). Claims are labeled by whether
they are paper-supported or our-own hypotheses. Companion: docs/MESH_EXPORT_SPEC.md (downstream mesh
requirements), docs/PIPELINE_RECOMMENDATION.md (the reconstruction pipeline).

## 1. What their pipeline actually is (corrected facts)
- **Vectra H2 = a single-trigger 2-image stereo device** ("captures two images at different angles"),
  factory-metric, standoff set by two LED beams. NOT a multi-view photogrammetry sweep. One 3D surface
  per trigger; the breast study used 2 visits × 2 states = 4 surfaces (breast2025).
- **Their older porcine/scalp work is MVS from many photos** (porcine2015: 15 photos from 15 angles per
  session) with a **physical ruler for scale** (0.6–2% reconstruction error) and, in pigs, a **tattooed
  10×10 cm grid for dense correspondence** (jbiomech2018, porcine2015, iso2020).
- **Registration:** MeshLab (area) + CloudCompare (surface-to-surface distance). The ALGORITHM (ICP?
  rigid/non-rigid? landmark-seeded?) is NOT stated in any paper — do not assume.
- **Downstream mesh:** open skin-surface PATCH → extruded to a thin solid FE mesh (breast: 35,319 hex
  C3D8, 1.55 mm; scalp: 141,497 tets, 3.6 mm), mm units. They remesh, so cleanliness+scale ≫ density.
- **Accuracy stated:** breast surface-distance error ~1.6 mm (<2 mm); porcine MVS 0.6–2% vs ruler.
- **Their #1 clinical gap (verbatim):** no grid in humans → "limited to overall distance error and
  overall area growth" (no per-point strain). This is the opening for us.

## 2. Our capture protocol (single-timepoint geometry — near-term goal)
1. **Subject fills the frame**; matte, diffuse lighting; minimal background clutter.
2. **Slow continuous orbit**, cover top + sides + recesses (under-chin / inframammary-type folds).
   ≥25–30 cm standoff (iPhone LiDAR near-field bias grows closer than this).
3. **Capture mode (see the app work):** ARKit-4K + LiDAR concurrently on a Pro device if verified;
   else ARKit-1080p+LiDAR and/or HQ 12 MP stills. Record LONG enough for 150–400 sharp selected frames
   (the "strong-capture" branch); the shipped app's 20 s / 60-frame cap is being raised.
4. **Metric scale — MARKERLESS via sensors is the goal:** LiDAR ray-median lock and/or ARKit VIO set
   absolute scale WITHOUT any physical marker. (This is the improvement over their ruler-based MVS.)
5. **Ruler = VALIDATION ONLY (recommended, not required for scale):** place a rigid, matte, ≥10–15 cm
   ruler (or an L of two, or a matte checkerboard of known square) rigidly fixed relative to the subject,
   in-plane near the ROI, visible across many views. It cross-checks the sensor-derived scale; it does
   NOT provide it. If the subject moves relative to the ruler mid-orbit, the ruler reading is invalid.

## 3. Longitudinal / strain capture (future)
- Their bulk tier (breast paper) needs only clean metric surfaces at each timepoint + their own
  registration → our mesh drops in; give a stable common origin to ease their alignment.
- Their dense-strain tier needs persistent material-point correspondence (the grid). **Our candidate
  replacement: texture/splat feature correspondence across timepoints — UNPROVEN.** Validate against a
  physical inked/tattooed grid (phantom first) before claiming it.

## 4. "Exceed Vectra" — HYPOTHESES to demonstrate on capture day (not yet established)
- Many-view coverage vs single-shot stereo → fewer holes in recesses. *Hypothesis.*
- Dense texture correspondence → full-field human strain without tattooing. *Flagship, unproven.*
- View-dependent splat appearance for clinicians. *Real, but not a mesh advantage.*
- Automation (analysis-ready OBJ) + cost/access. *Plausible.*
None of these is a paper-cited Vectra weakness — they are ours to prove.

## 5. Rented-Vectra capture day — plan
- **First accuracy test on a RIGID PHANTOM** (not a deformable breast) to remove skin-motion confounds:
  capture the identical static surface with (a) our iPhone modes {ARKit-4K, ARKit-1080, HQ-stills},
  (b) the Vectra, (c) a ruler in-frame for both. → our-mesh-vs-Vectra surface-to-surface distance in
  CloudCompare; sensor-scale vs ruler-scale check; 4K-vs-modes comparison; all in one session.
- Confirm the Vectra standoff/lighting from the Canfield manual on the day so the reference is valid.
- Sequence Vectra and iPhone of the SAME state with no subject movement between (rigid phantom makes
  this trivial; deformable tissue would need breath-hold).

## 6. Open questions for the collaborator (confirm before freezing anything)
- Exact on-disk mesh format they ingest (OBJ/PLY/STL — inferred, not stated).
- Do they want our decimated Cubit-ready surface, or will they remesh our full-res mesh?
- Would they accept texture/splat correspondence in place of the physical grid for strain, and what
  validation threshold would convince them?
- Purpose of the white strip in breast2025 Fig 1(a) (unstated — scale? registration? landmark?).
- Their required absolute-scale acceptance at capture/reconstruction (papers give only downstream FE
  tolerances + the porcine 0.6–2% ruler error; no capture-side spec).
- Pre-alignment/common-origin convention that would make our mesh co-register cleanly on their side.
