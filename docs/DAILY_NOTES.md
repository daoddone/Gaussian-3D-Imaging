# Daily notes

Running engineering journal. **Newest entry on top.** Purpose: capture each day's decisions,
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
