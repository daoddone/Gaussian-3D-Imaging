# Capture modes, markerless metric, segmentation — verified findings (2026-07-09)

From a multi-agent investigation (app map + ARKit APIs + metric scale + segmentation) with adversarial
verification. Raw result: docs/_capture_modes_workflow_raw.json. These correct/clarify earlier claims.

## 1. ARKit 4K + LiDAR concurrency — the "must we choose?" question: NO, we don't choose
- **Stream tops out at 1440p on iPhone Pro.** `ARWorldTrackingConfiguration`'s default video format AND
  `recommendedVideoFormatForHighResolutionFrameCapturing` are BOTH ~1920×1440@60. A `recommendedVideoFormatFor4KResolution`
  (3840×2160) exists but drops to 30fps and is NOT documented as depth-compatible → avoid.
- **True high-res = 12 MP STILLS via `captureHighResolutionFrame(completion:)`** (iOS 16+): pulls a
  4032×3024 still on demand WHILE the stream + LiDAR keep running uninterrupted (WWDC22 "Discover ARKit 6";
  depth read from `frame.sceneDepth.depthMap`, verified Apple Forums 805839; correctly registered on
  iPhone 16 Pro per thread 808028). So **LiDAR + high-res color coexist** — the "4K OR LiDAR" premise was wrong.
- **Implication for us:** resolution above ~2048 does NOT help the MESH (nvdiffrast caps at 2048 → we
  downsample anyway); it helps the SPLAT, SfM robustness, and texture bake. So the valuable high-res
  path is periodic 12 MP stills, not a 4K stream. STATUS: deferred (needs on-device implementation +
  test of `captureHighResolutionFrame` interleaving; see §5).

## 2. Markerless metric scale — feasible, but with an honest floor
- Monocular SfM is scale-ambiguous (7-DoF gauge); absolute scale MUST come from a metric reference.
- That reference can be a SENSOR (no physical marker): **ARKit VIO** (gravity-anchored metric camera path)
  or **LiDAR** (ray-median lock). **Markerless is achievable** — this is the improvement over Adrian's
  ruler-based MVS (their breast paper only skips a ruler because the Vectra is factory-metric).
- **Realistic accuracy floor (verified): ~2–5% absolute scale markerless, NOT a guaranteed 1 mm.**
  VIO with well-excited motion ~4.8% floor; LiDAR robust-locked at ≥250 mm ~1–2% / ±1 cm on >10 cm
  objects. Best case (LiDAR anchor + excited-motion VIO + temporal averaging) → ~2–4% ≈ a few mm on a
  10 cm feature — right at the edge of Adrian's ~2 mm bar, not comfortably under it.
- **Correction (verifier):** our internal "poses agree 1–2 mm" is RELATIVE PRECISION, not validated
  absolute SURFACE accuracy. Absolute accuracy is unproven until the Vectra/phantom test.
- **Strategy:** VIO-PRIMARY (immune to LiDAR near-field bias — our camera_path=1.000 choice was right),
  LiDAR-secondary cross-check, RULER = QA/validation only. **Prescribe excited capture motion**
  (figure-eight / varied-curvature orbit) — roughly halves VIO scale error (9.2%→4.8%).
- **So: markerless metric is a real, defensible improvement over their pipeline; the ruler stays as a
  validation cross-check, not a scale source.** 1 mm absolute is NOT guaranteed markerless — set
  expectations at few-percent and let the ruler/Vectra quantify it.

## 3. Segmentation in the loop — YES, as a refinement (answers "is there no segmentation today?")
- **Correct: there is NO segmentation today.** `make_subject_masks.py` is a GEOMETRIC mask — optical-axis
  center + 0.40 m metric cluster + projected-box convex hull. It protects a REGION, not a true anatomy
  outline (the flatness prior's "anatomy protection" is therefore approximate, not semantic).
- **Recommendation: ADD SAM2, server-side, as a refinement (not replacement).** Keep the geometric
  locator (robust at standoff where camera-hull culling fails); use its projected box as SAM2's BOX
  prompt on keyframes, then SAM2's VIDEO predictor propagates temporally-consistent silhouettes across
  the orbit — fully automatic, no human click, can only improve the mask. Runs on the A4000 at
  reconstruction time. Optional cheap on-device pass (Apple Vision) for live capture-framing UX.
- **What it buys:** tighter isolation (removes between-the-feet table + background the convex hull can't),
  exact anatomy protection for the flatness prior, better floater removal. STATUS: recommended next build.

## 3b. Capture-backend choice: ARKit vs HQ-Depth — re-derived (owner question, 2026-07-09)
**What each sensor stream is FOR (this is the crux):** Camera POSES come from SfM/COLMAP (pycolmap,
100% registration). The dense point cloud + mesh come from the MILo Gaussian-splat optimization driven by
the **RGB images + those SfM poses**. **LiDAR and VIO do NOT refine geometry** — surface supervision
against LiDAR was RETIRED (it stamped sensor noise; `depth_lambda:0` everywhere). Their ONLY job now is
**absolute metric SCALE** (a global gauge lock). So the backend choice is a **metric-scale + resolution**
question, not a reconstruction-quality question.

| | ARKit backend | HQ-Depth backend |
|---|---|---|
| RGB | ~1440p stream (→ 12 MP stills via T4) | ~1440p depth-capable stream (no 12 MP path; AVCapturePhoto+depth coexistence is RISKY) |
| LiDAR depth | ~256×192, temporally **smoothed** + confidence-gated | ~320×240, **raw** absolute unfiltered |
| Pose | **VIO metric camera path** (`camera.transform`) | **none** (SfM recovers it) |
| Scale anchors it enables | **LiDAR ray-lock AND VIO camera-path (TWO independent, cross-checking)** | LiDAR ray-lock ONLY |

**Why the plan leans ARKit now (a reasoned lean, to be confirmed in T5 — NOT a hard flip):**
1. The metric strategy is **VIO-primary + LiDAR cross-check** (two independent anchors → agreement =
   confidence, disagreement = flag a bad capture; VIO is immune to LiDAR near-field bias). **Only ARKit
   provides BOTH in one capture.** HQ-Depth gives LiDAR only — no VIO, so no within-capture cross-check.
2. HQ-Depth's one real edge — **raw, higher-res depth** — is **moot now that depth is scale-only**: scale
   is a robust median over thousands of ray comparisons, so 256×192-smoothed vs 320×240-raw barely moves
   it (that edge mattered when depth SUPERVISED the surface — which we retired).
3. **4K/12 MP + LiDAR concurrency is documented+reliable on ARKit** (`captureHighResolutionFrame` + sceneDepth);
   on HQ-Depth it's uncertain/risky (photo-output + depth-stream on one session).
4. With T4, **ARKit-4K is the most COMPLETE single capture: 12 MP RGB + LiDAR + VIO poses together** —
   HQ-Depth cannot add VIO, and can't reliably add 12 MP+depth.

**Reconciling with the earlier "we don't need ARKit's poses" reasoning:** that was correct *for
reconstruction* — SfM does poses, so we don't need VIO for pose accuracy. What CHANGED is that VIO's pose
became valuable for a different reason: as an **independent metric-scale anchor** (not for reconstruction).
So we're not contradicting the old logic; the old logic was about geometry, this is about scale.
**Caveat:** the best feet reconstruction to date WAS HQ-Depth (via SfM) — reconstruction quality is
backend-agnostic (RGB+SfM+MILo). So T5 must A/B ARKit vs HQ-Depth on SCALE accuracy specifically, with a
reference object, before finalizing. The lean is ARKit; the proof is the benchmark.

## 4. What actually SHIPPED this session (app)
- **Configurable frame cap, raised 60→360 frames / 20→120 s** (KeyframeSelector + CaptureTuning +
  Settings steppers; safety-stop + progress kept in sync via computed `budgetSeconds`). This is the ONE
  change that unblocks the strong-capture branch. Low-risk config only.

## 5. Deferred app modes (verified plans, need on-device testing — NOT shipped untested)
- **12 MP high-res stills** (`captureHighResolutionFrame` interleaved with the ARKit stream + sceneDepth):
  the real high-res path. Needs async still-capture + depth-pairing + FrameWriter integration, tested on
  device. Plan in docs/_capture_modes_workflow_raw.json (app_edit_plan).
- **LiDAR on/off toggle:** NOT NEEDED as an app toggle — markerless/VIO-only is testable SERVER-SIDE by
  ignoring the captured depth (use VIO poses only for scale). No app change required to run that experiment.
- **HQ 12 MP stills via AVCapturePhotoOutput:** coexistence with the depth stream on one AVCaptureSession
  is a flagged risk — needs device validation before shipping.

## Rationale for the ship/defer split
Frame-cap is safe config + unblocks our next experiments → shipped. The other "modes" are either
testable without app changes (markerless) or need device-tested feature work (12 MP stills). Shipping
untested async still-capture into the capture app would risk breaking captures — deferred with a
precise plan instead.
