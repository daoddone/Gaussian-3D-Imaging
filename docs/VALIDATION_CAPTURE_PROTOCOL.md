# Validation capture protocol (owner, ~45–60 min total) — metric-accuracy dataset + T16 dry-runs

Scale validation needs NO reconstruction: each capture costs ~10 CPU-minutes server-side
(SfM -> anchor -> marker check). Send everything via Transmit (app) or any file drop (videos).

## 0. One-time print check (30 seconds, no capture)
- Ruler-measure the sheet's 100 mm bar (horizontal) AND one marker's side VERTICALLY.
  The bar alone only proves horizontal print scale; one vertical measurement rules out
  anisotropic "fit to page" scaling for this printout, permanently. Note both numbers.

## 1. App captures with the ArUco sheet (the n=10–15 dataset) — arkit4K, LiDAR on
Vary ONE thing at a time; slow orbit unless the variable IS motion; sheet FLAT near the object:
- Objects (x3–4): different sizes/textures; include something organic/skin-toned if handy.
- Standoff: close (~0.3 m) / mid (~0.6 m) / far (~1 m) on the same object.
- Motion: one brisk/excited orbit (the still-failure + blur regime, on purpose).
- Lighting: one bright, one dim, one mixed/backlit.
- Optional failure probe: one capture with the sheet deliberately tilted ~30° (documents the
  failure signature; flat is the rule).
- Add the RULER in frame for 2–3 of these (in-scene reference independent of printing).

## 2. Native-video PAIRS (T16 dry-runs + referral accuracy) — 3–4 scenes
Right after an app capture of a scene, WITHOUT moving anything, take a 30–60 s native
camera-app video of the same scene (sheet in view). One of these deliberately casual/sloppy.
=> paired comparison: sensor-anchored scale vs marker-primary scale on identical ground truth.

## 3. Mini mode-matrix (feeds T5) — 1 scene
Same scene, same lighting, back-to-back: arkit1080 session + arkit4K session + native video.

## The exact manifest (15 items, one zip: validation_batch_<date>.zip)
Naming: `<ID>_<type>_<object>_<dist>_<light>[_flag]`, types = app4k / app1080 / apphq / vid.
- PRINT CHECK first (no capture): bar = 100.0 mm horizontal AND either marker square = 50.0 mm
  measured VERTICALLY (the markers are the vertical reference; there is no vertical bar).
- P1–P4 pairs (8 items): app4k capture then native VIDEO of the same untouched scene.
  P1 objA ~0.5 m bright (+ruler if easy), video = 4K/30, HDR off, landscape (best case).
  P2 objB ~0.3 m bright, video = FACTORY DEFAULTS untouched (1080p/30, HDR on, PORTRAIT —
  the realistic clinician case). P3 dim, default settings. P4 default settings + SLOPPY.
- P2_RESENT (+1 item, ~1 min): send P2's video to yourself through a REAL channel (email or
  WhatsApp) and include the received copy — tests transit recompression clinicians will hit.
- S1–S5 solo app4k (5): S1 far ~1 m · S2 FAST orbit · S3 mixed/backlit · S4 sheet TILTED ~30°
  (failure probe) · S5 ruler in frame.
- T5a/T5b (2): P1's scene again as app1080 and apphq, back-to-back, same lighting.
App settings: arkit4K + LiDAR ON, 20–40 s slow orbit, ~120–180°, sheet flat + visible throughout.
Native camera: 4K/30 fps, HDR Video OFF, 1x main lens, Action Mode OFF, 30–60 s.
Server-side: everything batches through SfM -> 04_metric_anchor (--marker-mm 50) -> validate_scale;
report = agreement distribution (VIO vs marker vs LiDAR), per-axis anisotropy, abs-mm errors,
confidence outcomes across conditions. NO reconstruction needed; ~10 CPU-min per item.
