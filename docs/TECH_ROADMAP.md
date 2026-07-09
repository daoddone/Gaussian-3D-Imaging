# TECH ROADMAP — phone-capture → metric 3D reconstruction & measurement

**Scope of this document: TECHNOLOGY ONLY.** A general-purpose pipeline that turns short smartphone
captures of a **target subject** (any physical object placed in front of the camera) into a
**metrically-scaled 3D reconstruction** — a Gaussian-splat model (photorealistic, view-dependent) and a
triangulated **surface mesh** (for downstream engineering / finite-element analysis) — plus quantitative
**measurements** (distances, areas, surface deviation) in millimeters.

This is the entry document for engineering work. It is self-contained and domain-neutral: it discusses
sensors, geometry, optimization, file formats, and accuracy — nothing else. Component-level I/O is in
COMPONENT_IO_REFERENCE.md; day-to-day history in DAILY_NOTES.md; the settled reconstruction recipe in
PIPELINE_RECOMMENDATION.md; the downstream-consumer mesh spec in MESH_EXPORT_SPEC.md.

> A companion document, DOMAIN_LAYER.md, records the specific application context and is maintained
> separately. This roadmap does not depend on it. Engineering tasks below are fully specified here.

---

## 1. System overview (the stages)

```
S1  CAPTURE      smartphone (LiDAR-class device): color stream + LiDAR depth + visual-inertial
                 odometry (VIO) poses; on-demand high-res stills; sharpness-based frame selection.
S2  FRONTEND     optional feed-forward geometry (metric depth + relative poses) as a prior/cross-check.
S3  POSES+SCALE  Structure-from-Motion (SfM) camera poses; ABSOLUTE METRIC SCALE from device sensors.
S5  RECONSTRUCT  Gaussian-splat optimization with an in-loop differentiable mesh (MILo); capacity
                 matched to input richness; subject isolation; scalable mesh rasterization.
S6  EXPORT       appearance bake (orthogonality-consensus texture) → OBJ+MTL+PNG (mm) + PLY + review
                 renders + a metric/scale provenance sidecar.
EVAL            subject-centered geometry metrics; mesh roughness; photorealism (PSNR/SSIM/LPIPS);
                 absolute-scale validation against a known-size reference object.
```

## 2. The core principle (settled by experiment — see SWEEP_RESULTS.md)

**Capacity must match input quality.** Optimizer capacity (training schedule, gaussian budget) only
pays off when the capture supplies enough signal (dense, sharp, frame-filling views). On weak captures,
extra capacity manufactures spurious detail; on strong captures it resolves real fine structure. →
**Capture quality is the dominant lever**; no reconstruction knob substitutes for it. The pipeline
therefore has an input-quality branch (weak → fast schedule; strong → quality schedule), auto-selected.

Corollaries proven in the campaign: (a) per-pixel supervision against a noisy depth sensor stamps sensor
noise onto the surface → **use the depth sensor for SCALE, not surface supervision**; (b) camera poses
are a solved problem (SfM registers 100% at sub-1.5 px); (c) subject isolation removes background/floaters
at no quality cost; (d) frame **selection** (sharpest-per-window) improves surface completeness.

## 3. Metric scale — the plan (this is the current priority)

SfM reconstruction is inherently scale-ambiguous (a 7-DoF gauge freedom); absolute scale must come from a
metric reference. On a LiDAR-class phone we have **two independent sensor references and no need for a
physical marker**:

1. **VIO camera-path baseline (PRIMARY).** The device's visual-inertial odometry produces a
   gravity-anchored, metric camera trajectory. Scale = ratio of SfM camera baseline to VIO baseline.
   Immune to the depth sensor's near-field bias. Conditioning: prescribe **excited capture motion**
   (varied-curvature / figure-eight orbit, not a static hover or straight slide) — this materially
   improves VIO scale observability.
2. **LiDAR ray-median lock (SECONDARY / cross-check).** For each SfM point observed in a frame, the ratio
   of sensor depth to SfM camera-depth at the observed pixel; robust median over all observations is the
   scale (gauge-immune). Proven ~1% MAD at ≥250 mm standoff. Use as an independent cross-check on VIO;
   disagreement flags a bad capture.
3. **Known-size reference object (VALIDATION ONLY).** A rigid graduated scale bar in-frame gives an
   independent absolute measurement to QUANTIFY the sensor-derived scale error. NOT the scale source —
   the pipeline is markerless by design; the reference bar only measures how good the markerless scale is.

**Realistic accuracy floor (from literature review): ~2–5% absolute markerless**, tightening toward the
low end by fusing VIO + LiDAR + temporal averaging + excited motion. This is a RELATIVE-vs-ABSOLUTE
distinction: SfM/splat geometry is self-consistent to ~1–2 px reprojection, but absolute SURFACE accuracy
is unvalidated until measured against a reference object. Provenance: every export records the scale
source, cross-check agreement, and a confidence flag.

**Prior art already integrated:** a feed-forward metric-geometry frontend (DA3) and an independent
feed-forward metric cross-check (MapAnything) are wired in. The campaign found their learned metric is
few-% off — no better than the sensor anchors — so they are optional, not the scale source. (A
literature review of newer feed-forward geometry methods is in flight; the working hypothesis is that
none beats a well-locked LiDAR+VIO anchor ON A DEVICE THAT HAS LiDAR — they matter mainly for the
no-depth-sensor case.)

## 4. Component stack (what runs, why)

| Stage | Component | Role | Status |
|---|---|---|---|
| S1 | ARKit (LiDAR stream + VIO) / AVFoundation depth camera | capture; sensors | shipped; frame-cap raised; 12MP-stills mode pending |
| S1 | sharpness frame selection (Laplacian-per-window) | pick sharp, well-spaced frames | shipped (server-side) |
| S2 | Depth-Anything-3 (DA3) | optional feed-forward metric geometry | integrated, optional/fallback |
| S2 | MapAnything | independent metric cross-check | integrated |
| S3 | pycolmap SfM (SuperPoint+LightGlue+hloc) | camera poses | shipped, 100% registration |
| S3 | VIO / LiDAR ray-lock | absolute metric scale | LiDAR shipped (`03b_relock`); VIO-primary = next |
| S5 | MILo (Gaussian splat + in-loop mesh, RaDe-GS raster) | reconstruction | shipped; scalable renderer; capacity auto-select |
| S5 | subject isolation (geometric mask + opacity penalty) | remove background/floaters | shipped |
| S6 | orthogonality-consensus appearance bake | sharp mesh texture | shipped |
| S6 | OBJ/PLY exporter (`export_mesh_obj.py`) | downstream-analysis mesh, mm | shipped; wire into driver = next |
| EVAL | subject-centered metrics + review renders | human + script review | shipped (`export_review.py`) |
| EVAL | photoreal metric (radegs PSNR/SSIM/LPIPS) | appearance quality number | NOT built |
| EVAL | absolute-scale validation harness | measure scale error vs reference | NOT built |

## 5. Prioritized task backlog (engineering, buildable)

**P0 — metric scale (the current priority).**
- **T1. VIO-primary metric anchor — ✅ SHIPPED 2026-07-09** (`scripts/pose_ba/04_metric_anchor.py`).
  Unifies both sensor anchors on any gauge-free SfM/BA model: VIO camera-path (Umeyama on camera centers,
  reusing `stage3_metric.align`; pairwise-baseline internal cross-validation) as PRIMARY when available;
  LiDAR ray-median (03b's estimator, generalized) as cross-check / no-VIO primary. Writes the scaled
  metric model + **`scale_sidecar.json`** {primary, scale, agreement %, confidence high/medium/review,
  notes}. Confidence logic distinguishes ATTRIBUTABLE disagreement (near-field LiDAR bias, auto-flagged
  via median sample depth <0.25 m) from UNEXPLAINED disagreement (forces "review"). Consumers wired:
  stage-5 `provenance_stage5.json` (`metric_scale_anchor`) + OBJ `export_meta.json` auto-discover the
  sidecar. VALIDATED: LiDAR-only path regression-matches 03b exactly (S=0.078135, 1.0% MAD, 32,048
  samples); dual-anchor path on a real close-range capture correctly detects 29.2% anchor disagreement,
  attributes it to near-field bias (0.20 m median), and selects VIO (1.7 mm residual, 48 frames) —
  numerically vindicating VIO-primary.
- **T2. Absolute-scale validation harness — ✅ SHIPPED 2026-07-09** (`scripts/validate_scale.py`).
  Auto path: ArUco/ChArUco corners detected across source images → multi-view DLT triangulation in the
  METRIC model (3 px outlier rejection) → measured vs known side lengths → **end-to-end scale error %**,
  per-axis anisotropy (Cho & Woo), absolute-mm errors alongside % (thin-feature habit), residual/view-count
  quality gates. Appends `reference_validation` into the T1 sidecar (anchor → validation loop closed).
  Manual mode (`--known-mm/--measured-mm`) for viewer measurements. **Math validated by synthetic
  self-test** (`--selftest`: +2.30% injected → +2.38% recovered, anisotropy 0.009%); real-capture
  validation scheduled for the reference-object capture day. Reference spec: rigid matte printed ArUco
  (DICT_4X4_50) of exactly known side, flat near the subject, visible across many views.
- **T3. Excited-motion capture guidance — ✅ SHIPPED 2026-07-09** (`scripts/capture_quality.py`).
  Pre-reconstruction capture score: frames-vs-strong-target, per-frame sharpness (blurry-frame flags),
  **VIO motion conditioning** (direction-diversity eigen-spread, total turning, curvature variation —
  the excited-motion signal), and angular coverage around the optical-axis subject center. Emits
  `capture/capture_quality.json` + concrete guidance strings + the expected capacity branch.
  VALIDATED on both historical feet captures — independently reproduces the campaign's verdicts
  (ARKit feet: WEAK, 57 frames, near-line path diversity 0.102 = the poorly-conditioned-VIO signature,
  78° coverage; HQ feet: FAIR, no VIO, 191° coverage via SfM). Capture-time (in-app) guidance remains
  a future nicety; the server-side score + protocol prescription cover the need.

**P1 — capture modes + benchmarking.**
- **T4. Capture-mode matrix in the app — ✅ IMPLEMENTED 2026-07-09, awaiting owner on-device test.**
  Full matrix per docs/TASK_T4_HIGHRES_CAPTURE.md: `CaptureMode {arkit1080, arkit4K, hqStills}` ×
  LiDAR toggle; arkit4K delivers the real 12 MP still via `captureHighResolutionFrame` per keyframe
  (streamed frame written as FALLBACK on still failure/busy — a failing still path can never yield an
  empty capture); HQ photo-output added NON-fatally (`hq_stills_fallback` metadata when coexistence
  fails); LiDAR-off keeps VIO pose + writes NaN-placeholder depth; defaults reproduce pre-T4 behavior
  exactly. All five files parse clean under Swift 6 (`swiftc -parse`, independently re-verified —
  syntax only; full type-check happens in Xcode). On-device checklist in the task file + agent report:
  the A/B-validity gate is rgb PNGs ≈ 4032×3024 in arkit4K mode.
- **T5. Capture-mode benchmark.** Sweep {frame count} × {resolution} × {sensor combo} on a fixed subject;
  report reconstruction quality (subject roughness, completeness, PSNR) AND end-to-end wall-clock (a
  real-world usability metric). Depends on T4 for the resolution axis to be real.

**P2 — reconstruction + export polish.**
- **T6. Wire OBJ export into the driver — ✅ SHIPPED 2026-07-09.** `milo_supervised` now auto-emits
  `export/mesh.obj` (+MTL+PNG+metric PLY+export_meta) after every mesh reconstruction (flag
  `export_obj: false` to disable; best-effort, never fails the run). Bake + review hooks were already
  auto-wired; this completes the full automatic artifact set. Sidecar discovery hardened for the
  swapped-model layout (`metric/colmap` model + `metric_sfm/` sidecar).
- **T7. Photoreal metric harness — ✅ SHIPPED 2026-07-09** (`scripts/eval_photoreal.py`, subagent-built,
  validated on real data). MILo-convention PSNR/SSIM/LPIPS on rendered views vs GT, honest label
  "train-view reconstruction fidelity" (historical runs have no held-out split). First numbers (face
  flagship, iter 22k): **PSNR 32.97 / SSIM 0.928 / LPIPS 0.305(vgg-bundled)** + best/median/worst
  comparison panels under `review/photoreal/`.
- **T8. Retention-percentile knob — ✅ CODE-COMPLETE 2026-07-09.** `--simp_retention` (train.py) →
  module `SIMP_RETENTION` in `gaussian_model.py`; all 6 distillation call sites routed through it
  (0.999 refinement passes track at the built-in 10× ratio). Default 0.99 = byte-identical stock.
  Driver option `simp_retention`. Behavior validation lands with the next training run.

> **SCALE-TRANSFER CAVEAT (owner point, 2026-07-09) — applies to ALL measurement methods below.**
> Reported accuracy is scale-dependent: numbers from large structures (aerial/street/room scale) do NOT
> transfer to close-range small subjects. A 1–2 cm RMSE that is "sub-percent" on a 10 m building is a
> *gross* error on a 10 cm subject. **Any measurement method adopted here must be re-validated at OUR
> subject scale (~cm–dm, close range) before use** — never trust a paper's headline number across a scale gap.

**P3 — BACKLOGGED (deferred; decided 2026-07-09 to keep the active list lean — the LiDAR+VIO anchor is
the settled metric strategy and needs none of these). Kept as notes for future reference.**
- **T9. Learned foreground segmentation** (SAM2, server-side) to refine the geometric subject mask.
  Deferred: geometric mask is adequate; validate capture+metric first.
- **T10. Edge-aware flatness prior A/B** (built, unproven). Deferred: needs an A/B proving it never
  touches genuine fine surface detail.
- **T11. Measure-on-splat point tool + uncertainty** (Deng & Qin, arXiv:2603.24716). *Most promising of
  the measurement additions* — measures on splat renderings (beats mesh readout on thin/sharp features)
  and adds a per-measurement error ellipsoid we lack. REVISIT candidate. BUT its evidence is drone-scale
  → must pass the scale-transfer caveat (close-range pilot) before adoption.
- **T12. AMB3R metric cross-check** (arXiv:2511.20343). Deferred: only valuable as a NO-LiDAR fallback
  (e.g. a clinician has an old video without depth). Since LiDAR is expected available, not needed now.
- **T13. Surface-aware metrics + reference-scan validation protocol** (Wound3DAssist, arXiv:2508.17635):
  geodesic length/width, area-by-triangle, depth-as-surface-deviation; Chamfer/Hausdorff/Normal-Consistency
  vs a reference scan. Adopt only if scope grows beyond linear distances. (The validation protocol is
  worth reusing on the reference-scan day even sooner.)
- **T14. Validation habits** (Cho & Woo, fcomp.2026.1755361): per-axis anisotropic-scale sanity check;
  report ABSOLUTE median error (not %) for thin features. Cheap; fold into T2 when built.
- **Feed-forward geometry notes (future reference, not priority):** VGGT (2503.11651), MASt3R
  (2406.09756), π³ (2507.13347), MoGe-2 (2507.02546) — all up-to-scale or in-domain-metric; would need
  our sensor anchor anyway. See MEASUREMENT_LITERATURE.md. Only revisit if the no-depth-sensor case
  becomes real. **Decided: LiDAR+VIO sensor anchor is the metric strategy; do not re-plumb for these.**

## 6. Evaluation standard (how any change is judged)
Every reconstruction change is A/B'd against the current best with all other knobs held fixed, judged on
**subject-cropped** renders + metrics (never whole-scene, which is floater-dominated). Metrics: mesh
dihedral roughness (with renders — it's resolution-sensitive), subject-cluster dimensions (metric-scale
sanity), completeness, and — once built — held-out PSNR/SSIM/LPIPS and absolute-scale error vs a
reference object. Artifacts land in `sessions/_sweep_eval/<name>/` and each output's `review/` folder.

## 7. Compute / platform facts (A6000, 48 GB)
Quality-schedule runs are serialized through their densification peaks (measured up to ~47 GB solo; two
concurrent = OOM). Mesh rasterization is chunked at 2^22 triangles. Fast runs ~35–45 min; quality_mid on
a modest subject ~2.5–3 h. Long/dense captures increase SfM + reconstruction wall-clock — benchmark it
(T5), as processing time is a real usability constraint.

---
*Maintain this as the engineering source of truth. Domain-specific rationale lives in DOMAIN_LAYER.md.*
