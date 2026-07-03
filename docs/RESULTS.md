# Real-data pipeline run — results

The full pipeline was run end-to-end on a real capture from an **iPhone 14 Pro
(iOS 26.5.2)** using the Stage 1 AnatomyCapture app: session
`session_20260703_145121`, a ~19 s hand-held LiDAR orbit of a face/region-sized
subject at ~25 cm. All heavy stages ran on the target **NVIDIA RTX A4000 (16 GB,
driver 535 / CUDA 12.2)**.

## Headline

| What | Result |
| --- | --- |
| Capture → metric surface (Stage 1 only, LiDAR + poses) | one coherent metric cloud; **99.7 % of points within 30 cm**, robust extent 23×10×26 cm |
| Stage 3 metric lock (reconstruction ↔ LiDAR, ICP) | **2.97 mm** RMSE, fitness 1.000 |
| Experiment A (single global scale) | **UNIFORM** residual (cv 0.30) — single-scale metric assumption holds |
| Stage 5 reconstruction (deliverable: full-res depth-only) | **1.83M gaussians, PSNR 25.5 dB**, metric mesh ~21k verts, **2.80 mm** median surface vs LiDAR |
| Accuracy target | ~1 mm vs gold standard — Stage 3 metric agreement is few-mm; the reconstruction is the appearance/mesh deliverable |

## Stage by stage

**Stage 1 — capture (AnatomyCapture, ARKit).** 60 frames: rgb 1920×1440, LiDAR
depth 256×192 (float32 m, NaN invalid), validity masks, ARKit `camera_to_world`
metric poses, intrinsics, timestamps. Validated by back-projecting the LiDAR
depth through the poses into one world cloud: 60 different camera poses put the
surface in the *same place* → the ARKit→OpenCV convention is correct
end-to-end (a sign error would scatter the frames).

**Stage 2 — front end (Depth Anything 3, `DA3NESTED-GIANT-LARGE`).**
Pose-conditioned on the ARKit metric poses (`use_ray_pose=False`,
`align_to_input_ext_scale=True`) so the front-end poses are exactly the metric
ARKit path. VRAM-calibrated on the A4000: **60 frames OOM** the giant model at
fp32; **48 frames fit** (peak 16.0/16.4 GB) — set as the cap. (bf16 weight-cast
would fit 60 but breaks DA3's internal mixed precision; fp16 autocast is a no-op.)

**Stage 3 — metric alignment.** With pose-conditioning the camera-path anchor is
ground-truth metric (scale 1.0). The depth anchor disagrees ~12 % (DA3 depth is
~13 % off at close range) → correctly **flagged `anchors_disagree`** but the
**anchor-priority policy** (`ruler > camera_path > sensor_depth`) applies the
reliable camera-path scale, giving the lowest reconstruction-to-LiDAR residual
(**2.97 mm** vs 4.69 mm if the depth scale were applied). Experiment A on this
scan: single-scale residual is **uniform** (cv 0.30, drift 0.36) → validates the
core metric premise.

**Stage 4 — normal prior (StableNormal).** Isolated venv (diffusers 0.28, hub
0.23) layered on the Stage 2 env, so DA3's `huggingface_hub` 1.22 is untouched.
**Sign gate passed**: converted normals vs depth-derived OpenCV normals agree at
**0.963 cosine** (median n_z −0.90). 60 normal maps produced.

**Stage 5 — reconstruction (gsplat, depth-supervised).** Ported the
DN-Splatter/AGS-Mesh recipe onto gsplat (in the Stage 2 env — disk-smart, no
nerfstudio/MILo toolchain; MILo shelved because it needs a second CUDA-11.8 env
the disk can't afford and ignores the LiDAR). Init from the Stage 3 metric cloud;
photometric (L1+SSIM) **+ EdgeAwareLogL1 metric depth supervision** against the
LiDAR (validity-masked) **+ optional normal supervision**; gsplat
`DefaultStrategy` densification; SH degree 3. First pass (7000 iters, 2×
downscale, depth-only): **SSIM 0.509 → 0.948**, densified 200k → **465k
gaussians**, Stage 6 exports = Gaussian splat `.ply`, metric TSDF mesh `.ply`
(~20k verts), preview renders. Peak VRAM only **2.8 GB**.

## Experiment B — does the learned normal prior help? (verdict: NO, here)

Controlled comparison at matched settings (downscale 2, 7000 iters; only the
StableNormal prior differs):

| Condition | PSNR | mesh verts | mesh→LiDAR median | mean |
| --- | --- | --- | --- | --- |
| depth-only | **25.58 dB** | 20,612 | **2.40 mm** | 8.12 mm |
| depth+normal | 24.76 dB | 15,665 (smoother) | 3.78 mm | 9.41 mm |

On this close-range (~25 cm) clinical-type capture the normal prior **slightly
hurts** both appearance (−0.8 dB) and surface accuracy (mesh sits ~1.4 mm
farther from the LiDAR): it over-smooths, exactly the wound-distortion risk the
spec warns about. Per the Section 8 decision rule ("if it does not clearly help,
remove Stage 4"), **Stage 4 stays disabled** (the config default). The best
reconstruction is depth-only; its **2.40 mm** median surface deviation matches
Stage 3's 2.97 mm ICP. (The high *mean* is inflated by a few far background/
floater verts; the median is the representative surface figure.)

The **definitive deliverable** in `output/` is therefore a full-resolution
**depth-only** run (Experiment-B-optimal config).

## Fidelity improvements made this pass

- **Pose-conditioning** DA3 on the ARKit metric poses → metric ground-truth poses
  + tighter multi-view geometry → ICP **3.44 → 2.97 mm**.
- **Anchor-policy fix**: prefer the ARKit VIO camera-path scale over the noisy
  close-range LiDAR-vs-learned-depth ratio (empirically the lower-residual scale;
  and it stops the depth anchor from corrupting the ground-truth metric poses).
- **Normal supervision** added to the trainer (depth-derived predicted normals vs
  the StableNormal prior, gated on Stage 4).

## Bugs found by adversarial review and fixed

- **SH `f_rest` order (HIGH):** the exported Gaussian `.ply` wrote view-dependent
  SH coefficients coefficient-major; INRIA/3DGS viewers expect channel-major →
  scrambled specular color. Fixed (`transpose(0,2,1)` before flatten).
- **Depth zero-bleed (MEDIUM):** bilinear downsampling of hole-filled depth bled
  the 0 fill into valid targets around holes → biased-small depth supervision.
  Fixed with mask-normalized interpolation.

## Environment built on the A4000 box (was bare — no package manager)

Miniforge → `pipeline_stage2_frontend` (PyTorch 2.5.1+cu124, Depth Anything 3,
Open3D, **gsplat 1.5.3** + cuda-nvcc) → `~/envs/stage4_normals` venv (StableNormal
deps, isolated). A cuDNN cu13/cu12 conflict from an xformers install was
diagnosed and fixed. `git` still not verified installed on this box.

## Config / code changes (committed to the working tree)

`config/pipeline.yaml`: `stage2.pose_conditioning: true`, `max_frames: 48`
(A4000-calibrated), `stage3.ransac_inlier_tol: 0.10` (real-data), anchor priority
in `report.py`, `stage5.host: gsplat` + gsplat params. New:
`stages/stage5_reconstruction/gsplat_recon.py` (the trainer). Fixed the Stage 4
StableNormal API. `sessions/` stays gitignored (data + outputs local).

## Outputs (under `sessions/session_20260703_145121/`)

- `output/` — **the deliverable**: full-resolution **depth-only** Gaussian splat
  (`point_cloud.ply`), metric TSDF mesh (`mesh.ply`), preview renders.
- `output_depth_only/` — depth-only @ half-res (Experiment B baseline).
- `output_d2_depthnormal/` — depth+normal @ half-res (Experiment B treatment).
- `output_fullres_depthnormal/` — depth+normal @ full-res (1.4M gaussians).
- `metric/` — scale-locked cloud, COLMAP model, `scale_report.json` (2.97 mm).
- `sensor_preview/sensor_cloud.ply` — the Stage-1-only LiDAR metric cloud.

## Limitations / next steps

- **~1 mm target** is for the full pipeline vs a Canfield Vectra gold standard;
  we have no reference here, so the reported figures are internal (LiDAR-vs-
  reconstruction ICP, PSNR/SSIM). A gold-standard benchmark is the validation
  step (Section 9).
- ARKit pose drift over the orbit caps achievable surface accuracy regardless of
  Stage 5 supervision.
- Depth resolution (256×192 LiDAR) limits fine mesh geometry; appearance is
  driven by the high-res color.
- Experiment B (does the normal prior help?) — the depth-only and depth+normal
  reconstructions are being compared this run; see the two `output*` folders.
