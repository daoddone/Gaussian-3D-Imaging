# Pre-clinical decision review — for joint ratification (assembled 2026-07-13)

Owner directive (07-12): final pre-clinical pipeline decisions are made TOGETHER after review of the
analyses. Every item below is therefore PROVISIONAL (or proposed) until marked ratified. Evidence
methodology throughout, per owner standards: equal-footing (identical box-crop + largest-component
cleanup on every arm before measuring), anatomy-FRONTAL real-capture-camera views, one variable per
arm, independent physical ground truth where available. Artifacts: `sessions/_sweep_eval/*`,
journal entries 07-10 → 07-13 in DAILY_NOTES.md.

## D1. OpenGL rasterizer backend (`milo_use_opengl: true`) — currently ON provisionally
- Why: the 2048-px cap that halved 12MP training resolution was OUR patch (missing libglvnd headers
  misdiagnosed 07-03 as "headless has no GL"); upstream MILo defaults to GL (no cap, 8-bit subpixel).
- Evidence: historic crash checkpoint runs 1,100+ stable iters at 4032 under GL. Weak branch (Andrew,
  full runs): head 12.90° @ +17% density vs 13.21° (CUDA -r2), crisper texture. Strong branch (face):
  11.69° vs 11.90° (run-noise tie) — backend safe where resolution doesn't change. Feet control:
  10.77° vs 12.33° (07-10 CUDA baseline; combined with D2).
- Cost: 12MP at -r1 trains ~4.5× longer (≈4.5-5 h weak-branch). OBJ export timeout raised to 7200 s.
- Risk: none observed across 5 full runs (train/refine/extract/bake). CUDA fallback = env knob.
- RECOMMEND: RATIFY ON. [ ] ratified  [ ] modified: ______

## D2. Per-image exposure embedding (`decoupled_appearance: true`) — currently ON provisionally
- Why: upstream-recommended for exposure variation; iPhone auto-exposure drifts through every orbit;
  we had never enabled it (audit miss #3).
- Evidence: weak branch C−B: head 12.48° vs 12.90°, junk verts −24% (whole) vs −8% (head) — absorbs
  exposure shifts geometry used to "explain." Strong branch D: 11.90° vs 11.62° baseline (run-noise
  tie), background exposure-ripples visibly flattened, stubble/detail PRESERVED (frontal check).
  Texture bake unaffected (samples raw source frames).
- Cost/risk: none measured; embedding is train-time only.
- RECOMMEND: RATIFY ON. [ ] ratified  [ ] modified: ______

## D3. Isolation defaults, branch-coupled (`subject_isolation: auto`, `subject_box_prune: auto`)
- Standing since 07-10 (predates the joint-review directive; formalized here).
- Evidence: no-isolation arm drowned in fused junk (post-processing cannot remove attached debris)
  and measured worse ON THE FACE (equal-footing 16.64° vs 14.57° head-only); mask+prune (the complete
  designed system) won every cut (13.21°, block removed at source, 100% capacity in-box); feet gate
  held (12.33° vs 12.21° tie @ 2.9× density). Strong branch stays OFF = proven v3 recipe (regression
  PASS 11.62° vs historic 12.30°).
- RECOMMEND: RATIFY (weak=mask+prune ON, strong=OFF). [ ] ratified  [ ] modified: ______

## D4. Consensus anchor policy (`--anchor-policy consensus`) — currently OFF (fixed order default)
- Why: 07-13 field batch — close-range, low-excitation counter captures showed VIO confidently ~6-8%
  SMALL (clean internal stats) while LiDAR+marker agreed within 1.4% and matched the independent
  96 mm design check (+0.19/+0.35% at marker scale vs −6.3/−7.4% at VIO scale). Fixed order shipped
  VIO (flagged review); consensus lets an agreeing pair outvote the deviant (marker>vio>lidar within
  the pair), keeps confidence=review on conflict. Owl regression: all-agree → behavior identical.
- Protocol implication to decide WITH this: for close-range small-anatomy captures (fingers, ears,
  lesions), promote the marker sheet from validation accessory to STANDARD capture companion; and/or
  add capture-app guidance for scale-observability motion (T3 family).
- RECOMMEND: RATIFY consensus as DEFAULT (it is strictly more evidence-driven; fixed remains a flag).
  [ ] ratified  [ ] modified: ______

## D5. T10 edge-aware flatness prior — tested, FAILS its A/B → recommend NOT adopting
- Single-variable (vs GL+DCA control on feet): table plane RMS 9.38 mm vs 8.89 (no benefit, slightly
  worse) and subject 11.30° vs 10.77° (slight detail cost). The wobble it targeted appears already
  mitigated by D2. Caveats: one capture; slab metric includes on-table objects.
- RECOMMEND: keep OFF permanently; close T10 as tested-not-adopted. [ ] ratified  [ ] modified: ______

## D6. Referral-video path (T16) — built + validated; decision = offer it externally (Tepole)
- Chain: ingest (orientation-hardened) → SfM (two-tier sharpness) → marker-PRIMARY anchor →
  reconstruction → metric OBJ with `scale_source: aruco_marker`, confidence capped "medium".
- Evidence: setter-vs-sensor 0.27% (owl); sim end-to-end −0.17%; REAL handheld 4K +0.38%; REAL
  factory-default (1080p/HDR/portrait) +0.79%; synthetic degradation 1080p +0.10%, 720p-heavy +0.41%.
  50 mm sheet certified at ≤0.6 m standoff; far-standoff probe (S1) still unrun (owner-deferred).
- RECOMMEND: approve external offer in the Tepole email as drafted. [ ] ratified  [ ] modified: ______

## D7. Metric-accuracy stance (for the Tepole conversation and beyond)
- App captures: three-way anchor (VIO/LiDAR/marker when present) + confidence gating; validated
  sub-1% in normal regimes; KNOWN WEAK REGIME: close-range + gentle motion (VIO 6-8% small, n=2) —
  mitigations = D4 + protocol note. NO calibration constant applied anywhere (marker stays
  independent); revisit only if the growing dataset shows a stable systematic (current hint: +0.2-0.8%
  lean across 4 marker-referenced checks, consistent with owner's ~49.9 mm print reading; treated
  as 50.0 by owner decision).
- Final gates unchanged (owner-reserved): clinical/anatomical validation — Natalie strong capture,
  Vectra surface-accuracy day (protocol ready: REFERENCE_SCAN_PROTOCOL.md).
- RECOMMEND: acknowledge stance. [ ] ratified  [ ] modified: ______

## Open items NOT requiring decisions now
- T9 (SAM2 tight masks): deferred; decision data = Natalie (in-box wall remnants are its target).
- T11 (measure-on-splat + uncertainty): deferred pending close-range pilot.
- T15 (wake/sleep economics): deferred by design until recipe-stable.
- Cosmetic fixed this cycle: anchor disagreement note now attributes the outlier correctly when a
  third anchor corroborates; capture_quality needs a poses-free fallback for external videos (minor).
- Remaining owner captures when convenient: S1-S3 + RESENT (validation breadth), T5 mode set,
  Natalie, real-handheld worse-lighting video.
