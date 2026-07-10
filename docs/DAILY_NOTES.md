# Daily notes

Running engineering journal. **Newest entry on top.**

---

## 2026-07-09 (later) — T1–T8 engineering sprint, FIRST ground-truth scale validation (−0.85% / 0.42 mm), transmit live

Agent-tier handoff (Opus 4.8 → Fable 5) mid-day; work continued from docs/TECH_ROADMAP.md with full
project context (the domain/tech doc split stands as documentation hygiene; the engineering agent works
with the application context in view — the "guardrail" concern behind the split was a misunderstanding).

**The sprint (T1–T8, all landed same-day; every piece through its own validation gate):**
- **T1 unified metric anchor** (`scripts/pose_ba/04_metric_anchor.py`): VIO-primary (Umeyama on camera
  centers + pairwise-baseline cross-validation) + LiDAR ray-median cross-check on one gauge-free model →
  `scale_sidecar.json` {primary, scale, agreement %, confidence}. Confidence logic distinguishes
  ATTRIBUTABLE disagreement (near-field LiDAR, auto-flagged) from UNEXPLAINED (forces review). Validated:
  exact 03b regression (S=0.078135/1.0% MAD); dual-anchor on the close-range sunglasses capture correctly
  attributes a 29.2% LiDAR bias at 0.20 m and selects VIO (1.7 mm residual). Consumers auto-discover the
  sidecar: stage-5 provenance (`metric_scale_anchor`) + OBJ `export_meta.json`.
- **T2 scale-validation harness** (`scripts/validate_scale.py`): ArUco corners → multi-view DLT in the
  metric model → scale error %, per-axis anisotropy, abs-mm. Synthetic self-test PASS (+2.30% injected →
  +2.38%). Printable exact-size sheet: `docs/aruco_scale_reference.png` (2×50 mm markers + 100 mm
  print-check bar; detector-validated).
- **T3 capture-quality score** (`scripts/capture_quality.py`): frames/sharpness/VIO-conditioning
  (eigen-diversity, turning, curvature variation)/angular coverage + guidance; reproduces the campaign's
  verdicts on the historical captures (ARKit feet diversity 0.102 = the bad-conditioning signature).
- **T4 capture-mode matrix** (subagent + review): {arkit1080, arkit4K, hqStills} × LiDAR toggle;
  arkit4K = real 12 MP `captureHighResolutionFrame` per keyframe with stream-res fallback. All files
  parse under Swift 6 (independently re-verified).
- **T6 auto-OBJ in driver**: every mesh reconstruction now auto-emits export/mesh.obj(+MTL+PNG, mm) with
  sidecar-sourced scale provenance — CONFIRMED end-to-end on the feet flagship (export_meta carries
  `lidar_ray_median / medium`). Sidecar discovery hardened for the swapped `metric/`↔`metric_sfm/` layout.
- **T7 photoreal harness** (subagent): first numbers ever — PSNR 32.97 / SSIM 0.928 / LPIPS 0.305
  (train-view label, MILo conventions) on the face flagship.
- **T8 retention knob**: `--simp_retention` routes all 6 distillation thresholds (default = stock).
- Validation gates caught 6 real bugs same-day: open3d transitive import, load_poses shape+convention,
  swapped-layout sidecar discovery, wrong T6 model path, xatlas API field, **mixed-res LiDAR mapping (below)**.

**FIELD DAY (owner device tests) — the milestone:**
- **Andrew capture (arkit4K pilot, 41 frames/15 s):** 12 MP still path WORKS on-device (35/41 true
  4032×3024, correct 2.1× K). First longer attempt CRASHED the app → root-caused: 12 MP PNG encoding
  outruns disk at the keyframe rate → unbounded FrameWriter backlog → jetsam. **Crash guards shipped**
  (still rate-limit ≥0.75 s + pendingAppends>4 backpressure, both degrade to stream fallback). Guards
  field-validated by the next capture (97 frames, no crash; 26% stills = throttle working).
- **Owl + 50 mm ArUco (the ground-truth test): markerless scale error −0.85%, median abs 0.42 mm
  (max 0.75), anisotropy 0.52%.** VIO↔LiDAR agreement 0.9% (confidence HIGH) after fixing a
  field-caught bug: the LiDAR anchor's color→depth pixel mapping used session-global color_resolution,
  wrong for mixed-res captures (bogus 135% "disagreement"); now per-image COLMAP camera dims (HQ-feet
  regression unchanged). **Three independent references (VIO, LiDAR, physical marker) converge within
  ~1% — beats the 2–5% literature floor; the excited-motion prescription (T3) is what conditions VIO
  this well.** Chain runtime: capture→number ≈ 4–5 min (no reconstruction needed for scale) →
  validation can piggyback on any capture containing the sheet. n=1: build a 10–15-capture validation
  set across standoff/motion/lighting for a defensible distribution.
- **Transmit live:** upload receiver deployed (systemd, 0.0.0.0:8902, fresh token, PIN 6250 gate,
  auth verified 401/401/200), Info.plist ATS exception added (iOS blocks plain-HTTP without it;
  IP endpoints can't use per-domain exceptions). Two throwaway transmits verified + deleted per owner.
- **`scripts/session_sfm.py`**: generic capture→SfM runner (dominant-res subset for mixed-res, shared-K,
  sequential); the missing substrate for T5 and for any new session. New-session convention: metric_sfm
  + a `metric` symlink for legacy-layout consumers (run.py, masks).

**In flight tonight:** Andrew full-chain rehearsal (stage 5 re-launched detached after a session restart
killed the first attempt at 21% — pre-steps durable). Natalie strong-capture (markerless; the validated
dual-anchor agreement is the quality gate). Then T5 benchmark harness.

**Andrew rehearsal OUTPUT verdict (owner review + diagnosis):** raw outputs looked bad — blurry
face, a floater "halo" over the head, heavy background. Diagnosis against the documented findings:
(1) capture class — 41 frames/15 s, non-fill-frame face (~500 px effective at training res, HALF the
old fill-frame captures despite 12 MP), cluttered specular lab → the #1 documented law (capture
dominates), output pre-declared throwaway; (2) REAL pipeline finding — the halo sits INSIDE the
convex-hull mask (58% frame coverage), where background is supervised and the opacity penalty is
inert → convex-hull isolation is INSUFFICIENT for a person in a cluttered scene (owner's mask
skepticism VALIDATED; T9/SAM2 promoted conditionally — judge on the next strong capture); (3) owner
viewed the raw ply files — the subject-centered review/ copies (the "prepped for viewing" artifacts)
exist and look far better (roughness 13.1°, subject isolated); the fair residue is that the subject
crop shares the coarse mask's root; (4) one config delta vs the feet flagship: dense_gaussians false
(recommended) vs true (flagship) — minor, testable. REMEDY = tonight's protocol-compliant fill-frame
capture; then re-judge the mask.

**Mask-robustness note (owner skepticism, on record):** the geometric mask on the Andrew face session
keeps head+torso+near context and cuts far background (coverage 0.58 median) — coarse-but-safe by
design (fails open, never cuts subject). Adequate for floater/background suppression; NOT a tight
anatomy silhouette. If tonight's run shows isolation underperforming on faces, the SAM2 refinement (T9,
backlogged) is the designed upgrade path.

## 2026-07-09 — Deliverables layer: sharp mesh appearance, OBJ export for FEA, capture-app upgrade, collaborator-spec verification

Focus shifted from "reconstruction quality" (settled 07-08) to **making the outputs usable downstream**
and **preparing the next capture round**. All decisions below have rationales; see also
PIPELINE_RECOMMENDATION.md, MESH_EXPORT_SPEC.md, CAPTURE_PROTOCOL_V2.md, CAPTURE_MODES_FINDINGS.md.

**Mesh appearance ("cartoonish" fix).** MILo bakes ONE diffuse color/vertex AVERAGED over all views →
blur. Built `scripts/bake_mesh_colors.py`: color each vertex/texel from the source frames by
ORTHOGONALITY to the surface normal (owner insight: a frontal camera is wrong for a curved side; the
best view looks down the local normal), hard-excluding grazing views, then MEDIAN-reject outliers
(shadow/specular/misregistered) and weighted-average the near-orthogonal inliers (adaptive cone: widen
only where near-orthogonal views are sparse). Rationale: keeps sharpness of few-best-views while being
poison-resistant. Validated on feet+face, no regression.

**v2 OBJ exporter** (`scripts/export_mesh_obj.py`): clean→largest-component→decimate(~200k tris,
FEA-mesher-ready)→xatlas UV→per-texel texture bake (same robust sampler; base layer = barycentric
vertex color so no black texels where a view can't see)→OBJ+MTL+PNG in **mm** + full-res metric PLY +
scale-provenance sidecar. Rationale: the downstream collaborator's analysis ingests OBJ triangulated
surfaces; PLY vertex-colors aren't standard OBJ appearance. Fixed two bugs live (black texels; xatlas
API field names). Loads valid (edge-manifold, UVs, texture) in a MeshLab-style reader.

**Flatness prior** (`--flatness_lambda`, edge-aware depth 2nd-difference weighted exp(-β|∇I|)): smooths
only textureless regions (the table wobble), leaving textured surfaces untouched. **Found + fixed a real
bug**: the mask gate was INVERTED (would have smoothed the subject and spared the background). Now
hard-protects the subject region (applies only outside the subject mask) + the texture-gate as a second
guard. STATUS: built, NOT yet A/B-verified → BACKLOGGED (see 07-09 decision below).

**Capture app upgrade.** Verified (multi-agent + adversarial) the ARKit capabilities: (a) 4K-stream and
LiDAR are NOT mutually exclusive — the stream tops at 1440p on Pro, and true high-res is 12 MP stills
via `captureHighResolutionFrame()` WHILE LiDAR/tracking keep running; (b) markerless absolute-scale
floor is ~2-5% (VIO-primary immune to LiDAR near-field bias; excited "figure-eight" motion ~halves VIO
error), so a physical ruler is demoted to VALIDATION-only. **Shipped:** configurable frame cap raised
60→360 frames / 20→120 s (KeyframeSelector+CaptureTuning+Settings; safety-stop+progress kept in sync) —
the one change that unblocks the strong-capture branch, low-risk config only. **Deferred (needs
on-device test):** 12 MP high-res-stills mode. **Not needed:** a LiDAR-off toggle — markerless is
testable server-side by ignoring the captured depth (no app change).

**Collaborator-spec verification.** Read the 4 collaborator papers (in docs/); multi-agent extraction +
ADVERSARIAL verification caught several unsupported numbers in the earlier MESH_EXPORT_SPEC (a
fabricated "1.05-1.79 mm" range, a "23k-58k hex" range, "pigs get scale from the grid" — actually the
ruler). Corrected. Key confirmed facts: their reference device is a single-shot 2-image stereo unit
(factory-metric, so THEY need no ruler; their MVS papers DO use a ruler at 0.6-2%); they lack dense
correspondence in the human case (grid is porcine-only) and are limited to overall distance+area — the
gap our dense texture could fill (UNPROVEN). Our many-view reconstruction genuinely exceeds a 2-image
stereo on coverage (hypothesis to demonstrate on the reference-scan day).

**Prior-art check (metric-from-splats).** Confirmed we ALREADY integrated the feed-forward metric-geometry
family: DA3 (chosen frontend, metric, pose-conditionable) + MapAnything (metric cross-check). The
campaign demoted DA3 to fallback (its metric is a learned prior, few-% off, no better than sensor
anchors). So the DUSt3R→MASt3R→MapAnything→DA3 lineage is known/settled; new-paper review (in flight)
is scoped to whether anything BEATS a LiDAR+VIO anchor on a LiDAR device (expected: no, for our case).

**DECISION — backlog SAM2 + flatness until capture/metric are resolved.** Rationale: (1) both are
REFINEMENTS, not core; the geometric mask already works adequately and the flatness prior is unproven;
(2) the high-value open work is capture methods + markerless metric, which must be validated first;
(3) sequencing them later keeps the near-term task list purely technical (helps the planned agent-tier
change). The metric plan (LiDAR ray-lock + VIO-primary + 4K stills + ruler-QA) does not need either.

## 2026-07-08 — Quality campaign: the capacity law, subject isolation, and the second nvdiffrast bug

Owner mandate: run the pipeline at full capacity ("as originally intended"), autonomously overnight.
Full record in PIPELINE_JOURNAL.md + SWEEP_RESULTS.md; rationale summary here.

**Root-cause discovery that reframed everything: the invisible "fast" schedule.** Every MILo run in the
repo's history had silently used the upstream-default accelerated schedule (`configs/fast`: densify
stops at iter 3,000 → tiny gaussian budgets). So EVERY prior verdict was measured under a
capacity-starved envelope. Fix: provenance now stamps `milo_schedule`; added `configs/quality` and
`configs/quality_mid` (the A6000-feasible full-capacity point). Also fixed the coupled bug that the
renderer switch (`regularization_from_iter`) must equal `densify_until_iter`, and the extraction
`--iteration` default (18000) that failed 30k runs.

**Second nvdiffrast bug (the R2' crash).** At full capacity the feet mesh exceeded nvdiffrast's
16.7M-triangle (2^24) limit → CUDA-700 in `triangleSetupKernel`; then even the scalable renderer hit a
"subtriangle count overflow" (density within one 2^24 chunk). Fix: `use_scalable_renderer:true` +
`max_triangles_in_batch 2^24→2^22`. Diagnosed via the checkpoint-resume + choke-point-instrumentation
method (each hypothesis test ~3 min, not 2.5 h). Validated 1,700+ iters past the crash.

**Process failure + fix.** A manual pause for the crash diagnosis + a session disconnect left the GPU
idle all night (no dead-man resume). New rule (JOURNAL §8.9): any pause of the autonomous runner must
schedule its own bounded auto-resume BEFORE diagnosis. Applied — the matrix then completed autonomously.

**THE CENTRAL FINDING — the capacity law.** The 2×2 (ARKit/HQ × λ0.2/λ0) at full capacity FALSIFIED the
prediction that "λ0-at-capacity wins." At capacity the depth-term effect shrinks to ~2°, and CAPACITY
itself dominates roughness: on WEAK captures (standoff feet, ≤57 frames) added capacity fits junk
regardless of λ (feet qualmid ≈33° both λ), while on STRONG captures (fill-frame face) capacity adds
real detail (+27-47% verts, true stubble relief). **Law: capacity must match input quality.** Fast+λ0
remained the feet optimum (14.5-15.8°, visible toes — owner-validated). So the pipeline has an
INPUT-QUALITY BRANCH and the capture protocol is the true centerpiece.

**Subject isolation — adopted (first-try win).** Photometric-loss mask (geometric: optical-axis center
+ metric cluster + projected hull) + out-of-mask opacity penalty. Result: floater extent −58%,
roughness unchanged, subject intact. Replaces LiDAR's incidental free-space-suppression role that λ0 lost.

## 2026-07-07 — Depth-free face baseline (owner-requested): clean result, reshapes the LiDAR question

Owner context: a pre-pipeline preliminary run (face photos → plain COLMAP SfM → MILo, no depth, no
dense, no DA3) was visibly CLEANER than our current outputs. Owner asked to reproduce that input
condition through the upgraded MILo as the depth-free arm — photos in `sessions/Previous face photos`
(3,258 full-fps Record3D JPGs, 1440×1920; the preliminary used a 169-frame sample; old recipe:
`scripts/run_pycolmap_from_record3d.py`).

**Run** (`scripts/face_depthfree_test.py`, results in SWEEP_RESULTS.md): 172 frames stride-19; pycolmap
PINHOLE + BA-refined focal (no metadata.json in folder → init fx 1367 from device-K scaling, refined to
1496), sequential matcher → **172/172 @ 0.79 px**. MILo: mesh reg ON, -r1 (1920 < 2048 cap),
depth_lambda **0**, dense **off** → 57.7k gaussians, 819k-vert mesh, **roughness 10.13° (best ever)**,
facial features legible in the mesh (eyes/brows/nose/cap buckle). Gauge-free = non-metric by design.

**Conclusions (with the owner's two standing conclusions):**
1. Poses are NOT the problem (again confirmed — 0.79 px from scratch).
2. The depth-free path through current MILo is clean → consistent with the LiDAR-surface-supervision
   diagnosis (the controlled term-isolation evidence stays the feet A/B: 21.1°→15.8°).
3. **DA3 is not required for reconstruction** — COLMAP sparse init suffices for MILo.
4. Honest confound: the face capture also FILLS THE FRAME (the pixel-coverage detail lever) and has
   3× the views — this test validates the path, it does not isolate the depth term.

**Shaping the improvement decision (owner will steer):** candidate architecture = **LiDAR as SCALE
ANCHOR only** (global, surface-neutral — 03b_relock proved a 1% MAD lock) + **photometric/mesh-reg for
the surface** (depth_lambda 0 or very low). Next controlled tests: depth_lambda {0.05, 0.1} on the feet;
HQ feet depth_lambda-0 (SfM poses already in place); subject isolation; fill-frame re-capture remains
the true detail lever. DA3's role shrinks to fallback (degenerate/low-texture captures where SfM fails).
Eval harness now durable at `scripts/eval_recon.py` (scratchpad copy was lost to session cleanup). Purpose: capture each day's decisions,
findings, and live experiment state so context survives resets. The durable references stay in
`RESULTS.md`, `EXPERIMENTS_BACKLOG.md`, `MILO_PLAN.md`, `POSE_BA_PLAN.md`, `COMPONENT_IO_REFERENCE.md`,
`00_BUILD_SPECIFICATION.md`; this file is the "what happened / where we are" log.

---

## 2026-07-05 — HQ-Depth done right (SfM poses), MILo crash root-caused, component stack mapped

### Goal (unchanged)
Metric **and** photorealistic 3D-Gaussian reconstruction **+ mesh** per capture, ~1 mm surface
deviation vs the Canfield Vectra reference. The metric **mesh is a first-class deliverable**. The
established 1 mm limiter is the ~12% LiDAR-vs-VIO near-field scale ambiguity (trust VIO where available).

### The experiment (corrected framing)
Two candidate **capture methods**, both processed at **tip-top**, compared on cloud **and** mesh:
- **ARKit** — metric VIO poses (convenient) + coarse 256×192 depth.
- **HQ-Depth ("HQ")** — the candidate *improved* method: hi-res color + raw LiDAR (320×240), **no
  online pose**; poses recovered **externally by SfM**. Premise (EXPERIMENTS_BACKLOG.md:83-90):
  sharper color → more SuperPoint detail → tighter SfM; better depth → better metric anchor.
- HQ is NOT "pose-sacrificing therefore inferior" — the missing pose is meant to be *supplied* by SfM.
- Prior state: ARKit ran tip-top (coherent cloud + MILo mesh). HQ had never been run as designed —
  it used DA3 monocular **ray-pose** (`use_ray_pose: true`), not SfM, and crashed before any mesh.

### Decisions today
1. **Reverted the mesh-reg bandage.** `mesh_regularization: false` (added last session to dodge the
   crash) drops the mesh deliverable AND degrades the cloud (mesh-reg shapes the cloud onto the
   surface), so even the cloud comparison was unfair. Not used. **MILo left completely untouched** —
   no finiteness guard, no instrumentation — fix the *cause* upstream first (owner's call, correct).
2. **Ran HQ the way it was designed** (SfM poses), reusing the existing 47 frames (no re-capture).

### HQ SfM pose-recovery (done today)
- `scripts/pose_ba/01_match.py` (SuperPoint + exhaustive LightGlue, 47 frames) +
  `02b_sfm_noseed.py` (unseeded incremental COLMAP SfM). Result: **47/47 frames registered from
  scratch, mean reprojection 1.50 px, 17,188 points.** (Log's "47/48 / 1 failed" is a cosmetic
  hardcoded "/48"; it is 100%.) → SfM works on smooth feet skin; HQ's hi-res-color premise holds.
- **New: `scripts/pose_ba/03b_relock_lidar.py`** — HQ counterpart to `03_relock.py`. HQ has no ARKit
  target, so it metric-locks the gauge-free SfM directly to the **raw LiDAR**: for each SfM point
  observed in a frame, ratio = LiDAR_depth / SfM_camera_depth at the observed pixel; **S = robust
  median**. Immune to the 7-DoF SfM gauge (per-observation ratio, no pose used to solve scale).
  Result: **S = 0.078135, MAD 1.0%** over 32,048 samples across all 47 frames; camera-center extent
  278×780×521 mm (sane handheld feet orbit). Median LiDAR depth ~0.85 m — inside the reliable range,
  so this sensor_depth lock is NOT near-field-biased for this capture.
- Injected as `metric/colmap` (DA3 artifacts preserved: `metric_da3/`, `output_da3_nomesh/`), then
  ran Stage 5 tip-top (`config/pipeline_a6000.yaml`: -r 1, dense, cuda, depth_lambda 0.2,
  **mesh reg ON**), MILo untouched.

### MILo crash — ROOT CAUSE (found + fixed 2026-07-05)
CUDA 700 in nvdiffrast at **iter 8001 = the first in-loop mesh build**. My first hypothesis —
incoherent DA3 poses → degenerate SDF/mesh — was **TESTED AND DISPROVEN**: the SfM-pose HQ run
(coherent poses, 47/47 @ 1.5px) crashed **identically** at iter 8001.

**Real cause** (found by resuming the iter-8000 checkpoint + instrumenting the exact mesh handed to
nvdiffrast): the mesh is **PERFECTLY VALID** — 0 NaN/Inf verts, 0 `norm_sdf` 0/0 edges, face indices
all in range, coords ~4 (unit-scaled). The crash is purely **nvdiffrast's CUDA rasterizer 2048-px/side
cap**. HQ renders the in-loop mesh at the full color resolution **4032×3024 (>2048)** → CUDA-700 illegal
address in `fineRasterKernel`. ARKit at **1920×1440 (<2048)** is under the cap → fine. (nvdiffrast
0.3.3's claimed >2048 auto-tiling fails on this compiled build.) This explains EVERYTHING the pose
theory forced: pose-independent, density-independent, HQ-always, ARKit-never.

**CONFIRMED:** resuming the same checkpoint at **-r 2 (HQ → 2016×1512, both <2048)** builds the mesh at
8001 and trains past it cleanly. Fast repro tool: MILo checkpoints at iter 8000 (`chkpnt8000.pth`);
`train.py --start_checkpoint chkpnt8000.pth` reproduces the 8001 mesh build in ~30 s (backed up to scratch).

**GENUINE FIX** (`milo_supervised.py`): cap MILo `-r` so `max(w,h) ≤ 2048` whenever mesh reg is on — a
hard rasterizer constraint, not a quality bandage. HQ → -r 2 (2016 px). Side benefit: equalizes render
res vs ARKit (2016 vs 1920) → a FAIRER capture-method comparison (isolates poses+depth from a res
confound). Diagnostic instrumentation reverted; MILo otherwise untouched.

**NOTE:** the SfM pose work was still necessary — HQ needs real poses regardless, and the metric-locked
SfM model is what MILo now trains on. The pose hypothesis being wrong doesn't undo that. **Lesson: a
plausible mechanism (from the doc review) that fits all prior evidence can still be wrong — cheap
direct reproduction (checkpoint resume + instrument) beat 1.5 hr of theorizing.**

### Component stack — how the pieces fit (in context of each other)

```
Stage 1 (iOS)  ──► RGB + LiDAR depth
   ├─ ARKit:    + metric VIO poses (coarse depth)
   └─ HQ-Depth: hi-res color + raw LiDAR, NO poses ─┐
Stage 2  DA3 (Depth-Anything-3 NESTED-GIANT-LARGE)  │
   Giant any-view branch (relative geom+poses) rescaled to a Large monocular-metric branch.
   pose-conditioned (locks to ARKit VIO scale) OR use_ray_pose (self-estimated, gauge-free ← HQ).
Pose-recovery arm (scripts/pose_ba) ◄─ substitutes for missing poses
   SuperPoint → LightGlue → hloc → COLMAP/pycolmap SfM+BA → gauge-free model
   → relock: 03_relock (ARKit target) | 03b_relock_lidar (LiDAR target ← HQ)  ─┐
Stage 3  metric ◄─ DA3 frontend OR SfM model                                    │
   3 anchors: sensor_depth (depth vs LiDAR) · camera_path (VIO) · ruler  → metric/colmap ◄┘
Stage 4  StableNormal_turbo → optional per-frame normal prior (OpenGL→OpenCV flip)
Stage 5  reconstruction
   ├─ gsplat (DEFAULT, robust; tolerates metric scale) → Open3D TSDF mesh
   └─ MILo (host=milo, ours) → in-loop learnable-SDF mesh:
        RaDe-GS rasterizes Gaussians→depth/normal;
        Gaussian pivots → Delaunay → marching-tetrahedra → SDF mesh;
        nvdiffrast rasterizes that mesh→depth/normal; the two forced to agree (mesh-reg loss).
   both take DN-Splatter/AGS-Mesh depth+normal supervision (LiDAR + normals), via ags_depth_normal_losses.py
Stage 6  point_cloud.ply + mesh.ply (metric, vs Vectra ~1mm)
```

Component one-liners:
- **DA3 (Stage 2)** — feed-forward depth+pose foundation model; its "metric" is a *learned prior*
  (few-% off), so Stage 3 re-locks. Runs at ~504 px internally (rescale K). Do NOT re-apply its /300
  metric formula on the NESTED output.
- **SuperPoint / LightGlue / hloc** — keypoints / learned matcher / harness for the SfM arm. `01_match`
  upsamples to 1600 px and lowers the match threshold to survive low-texture skin.
- **COLMAP / pycolmap** — incremental SfM + bundle adjustment; intrinsics FROZEN on short baselines
  (free focal is degenerate). From-scratch SfM is **7-DoF gauge-free** (arbitrary scale/rot/origin) →
  must be metric-relocked.
- **MILo (Stage 5)** — mesh-in-the-loop: differentiably extracts a mesh from the Gaussians every
  iteration and enforces Gaussian↔mesh consistency. Scene pre-scaled to ~unit (S=1/nerf_radius) for
  the rasterizer, inverted on output.
- **RaDe-GS ("radegs")** — the Gaussian rasterizer; closed-form ray-Gaussian depth/normal (not
  center-blended) → meaningful depth supervision.
- **nvdiffrast** — differentiable *mesh* rasterizer for the in-loop mesh; forced to CUDA context
  (no EGL) on this headless box; **fragile to NaN/Inf/overflow verts → CUDA 700** (our crash site).
- **base 3DGS (INRIA)** — common ancestor; tile-sort + alpha-composite + analytic backward;
  means2D-grad drives densification.
- **gsplat** — the alt/default Stage-5 host (permissive license, tolerates metric scale, TSDF mesh
  via Open3D). NOTE (owner directive 2026-07-05): **do NOT use gsplat as a fallback for HQ** — MILo
  must work on the HQ data; if it errors, diagnose and fix the real cause in MILo, not switch hosts.
- **DN-Splatter / AGS-Mesh** — the depth (EdgeAwareLogL1) + normal (L1/TV) supervision *recipe*
  (depth_lambda 0.2, normal_lambda 0.1, confidence-gated), ported into both hosts.
- **StableNormal_turbo (Stage 4)** — diffusion normal prior; camera-frame OpenCV; global sign is the
  classic silent-failure risk (validate on a planar frame). No anatomical prior (safe for abnormal).

Four connective truths:
1. DA3's metric is a prior → Stage 3 re-locks scale from physical anchors; the ~12% LiDAR-vs-VIO gap
   is the real 1 mm limiter, not pose drift.
2. **Pose coherence affects reconstruction QUALITY** (coherent = ARKit-conditioned or SfM → crisp
   fusion; incoherent = DA3 ray-pose → smeared cloud). NB: it did NOT cause the HQ mesh crash — that
   was the nvdiffrast 2048 resolution cap (see crash section). Poses matter for quality, not the crash.
3. Scene must be pre-scaled to ~unit for the INRIA-lineage rasterizer; gsplat tolerates metric scale
   (why it's the robust default). Our metric-lock only needs `metric/colmap` to be *truly metric*.
4. Supervision is a portable recipe (one file, both hosts): DN forms + AGS gating + StableNormal prior
   = how Stage-1 LiDAR + Stage-4 normals enter Stage 5.

### Current status / next
- **MILo crash ROOT-CAUSED + FIXED** (nvdiffrast 2048 cap → cap `-r`; see crash section).
- **HQ full run RE-LAUNCHED at -r 2 (auto-capped), mesh reg ON, from scratch** → cloud + mesh expected
  (~1–1.5 h). Prior crashed/degraded runs preserved: `output_da3_nomesh`, `output_dense_crashed`,
  `output_lowdens_crashed`, `output_sfm_crashed`.
- Then: evaluate HQ (cloud + mesh), fairly compare vs ARKit (render res 2016 vs 1920; **same subject,
  different capture instance** — so weight method quality + metric plausibility, NOT surface-truth ICP),
  then the knob-sweeps (density / mesh_config / depth_lambda / isolation), ARKit-vs-ARKit and HQ-vs-HQ
  separately, reviewable outputs for the owner.
- Key artifacts: SfM model `pose_ba/sfm_noseed`; LiDAR lock `scripts/pose_ba/03b_relock_lidar.py` +
  `metric/scale_report_sfm.json`; fix in `stages/stage5_reconstruction/milo_supervised.py`;
  component-review journal `wf_b854dd09-659/journal.jsonl`.

### Overnight knob-sweep batch (autonomous 2026-07-05 ~02:00)
- **Eval harness:** `scratch/eval_recon.py` (open3d headless EGL) → stats (n_gaussians, extent,
  object-box mm, mesh verts/faces, **roughness = mean dihedral deg**) + shaded-mesh & colored-cloud
  orbit renders. ARKit baseline measured: 406k gaussians, 4.1M-vert mesh, **roughness 21.1°** (bumpy,
  matches owner obs). Reusable for every A/B.
- **Baselines:** ARKit dense = `session_20260704_143210/output_arkit_dense` (-r1/1920, dense, mesh on);
  HQ dense = `session_20260704_143324/output` (-r2/2016, dense, mesh on, SfM poses).
- **Configs:** `pipeline_a6000.yaml` (baseline) · `_lowdensity` (dense off) · `_meshlowres`
  (mesh_config lowres → smoother mesh) · `_depth0` (depth_lambda 0). Driver now supports `mesh_config`.
- **Process:** Stage-5-only re-run (reuses metric/colmap); Stage 5 always writes `output/`, so preserve
  then relabel `output → output_<label>` after each. ARKit-vs-ARKit and HQ-vs-HQ separately (never cross
  the pose source for a knob A/B).
- **Sweep (each ~1.5-2h, 2 concurrent on the A6000):** [running] HQ dense baseline (-r2); [running]
  ARKit low-density; [queued] ARKit mesh-lowres · HQ low-density · depth0 A/B · (later) subject-isolation.
- Owner reviews `scratch/eval_*` renders + the `.ply` files to steer the next knobs.
