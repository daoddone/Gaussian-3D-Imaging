# Measurement-from-splats — literature review (2026-07-09)

Scoped to: does anything beat our sensor-anchor (LiDAR ray-lock + VIO) for ABSOLUTE metric scale, and is
there a better way to take MEASUREMENTS from a splat/mesh? Reviewed the 3 PDFs the owner added + an
arxiv/named-method survey. Framing is domain-neutral; see DOMAIN_LAYER.md for context.

> **OWNER DECISION (2026-07-09):** The LiDAR+VIO sensor anchor is the SETTLED metric strategy. Everything
> in this doc is BACKLOG / future reference — nothing here is an active priority. Do not re-plumb scale.
>
> **SCALE-TRANSFER CAVEAT (owner point):** accuracy numbers from LARGE structures (aerial/street/room)
> do NOT transfer to close-range SMALL subjects. A "sub-percent" 1–2 cm RMSE on a 10 m building is a gross
> error on a 10 cm subject. Any method here must be RE-VALIDATED at our subject scale before adoption.

## Verdict on ABSOLUTE SCALE: keep the sensor anchor (do NOT re-plumb)
All three measurement papers get scale from mechanisms INFERIOR to ours for a markerless LiDAR-phone:
- Cho & Woo 2026 (forensic 3DGS): scale from ONE hand-tape-measured object, per-axis. Weaker/less
  repeatable than our LiDAR ray-median (~1% MAD).
- Deng & Qin 2026: no scale mechanism — inherits scale from upstream aerial triangulation.
- Wound3DAssist 2025: **ArUco markers** for scale — directly contradicts our markerless goal.
None addresses the ~12% LiDAR-vs-VIO disagreement or improves the ~2–5% markerless floor.

## The ONE genuinely-new capability worth adopting: measure-on-splat + uncertainty
**Deng & Qin 2026, "Accurate Point Measurement in 3DGS," arXiv:2603.24716** (OSU; open source
github.com/GDAOSU/3dgs_measurement_tool). Measures 3D points DIRECTLY from splat renderings by
multi-ray forward intersection (classical photogrammetry), BYPASSING the mesh, and returns a
per-measurement **uncertainty ellipsoid** (a-posteriori covariance).
- Why it matters to us: (a) our MILo MESH smooths away thin structures / sharp corners — they show
  measuring on the splat recovers exactly those (mesh gave ZERO points on sharp corners; their method
  0.013 m RMSE); (b) our pipeline has **no per-measurement uncertainty** today — a real gap for any
  "mm / few-%" measurement claim.
- Integration cost: MODERATE. Reuse only their ~30-line least-squares + covariance solver (their poses
  = our pycolmap SfM poses; the math is pose-source-agnostic); skip their Cesium/3D-Tiles web stack.
- CAVEAT: their evidence is UAV/meters-scale (1–2 cm ≈ sub-%); the "splat beats mesh" margin at
  close-range object scale is UNTESTED — pilot before committing.
- → **NEW BACKLOG TASK (T11): measure-on-splat point tool + per-measurement uncertainty.** P1/P2, after
  the metric-scale work; it's a MEASUREMENT capability, orthogonal to scale.

## Adopt-if-scope-grows: surface-aware metrics + a reconstruction-validation protocol
**Wound3DAssist 2025, arXiv:2508.17635** (CSIRO/QUT). A close precedent: monocular iPhone video →
textured mesh → clinical-grade measurement, ~1 mm avg surface error vs a Revopoint scanner reference.
Notably it evaluated GS and chose photogrammetry (Meshroom) for the final mesh — a caution, not a
blocker (our MILo path is different + newer). Useful pieces to MIRROR if/when we go beyond linear
distances: geodesic length/width over a fitted surface, area by triangle summation, **depth =
signed surface-deviation from a fitted reference surface**, and a **validation protocol** (Chamfer /
Hausdorff / Normal-Consistency vs a reference scan; segmentation DSC). This validation protocol is
exactly what our reference-scanner-comparison day should use.

## Free validation habits: Cho & Woo
- **Per-axis anisotropic-scale sanity check** (they saw ~0.27% X-vs-Z; a larger value flags a
  pose/scale problem) → add to the scale-validation harness (T2).
- **Report ABSOLUTE error (median) for thin features**, not % (small denominator explodes MAPE).

## Feed-forward geometry lineage survey (through Jul 9 2026) — CONFIRMS the sensor anchor
Reviewed: DUSt3R (2312.14132), MASt3R (2406.09756), MASt3R-SfM (2409.19152), Spann3R (2408.16061),
Fast3R (2501.13928), VGGT (2503.11651), MoGe/MoGe-2 (2410.19115/2507.02546), MapAnything (2509.13414),
π³/Pi3 (2507.13347), DA3 (2511.10647), AMB3R (2511.20343), + fusion (LiDAR-VGGT, MASt3R-Fusion 2509.20757).

**"Rudy et al. 2024" = a misremembered citation.** No splat-metrology paper by "Rudy" exists; it resolves
to **Deng & Qin 2026** (PI *Rongjun Qin* → "Rudy"), arXiv:2603.24716 — already the T11 measure-on-splat item.

**Decisive verdict: NO feed-forward method beats a well-locked LiDAR+VIO anchor for ABSOLUTE scale on a
LiDAR device.** They are the NO-depth-sensor solution. Why (the load-bearing point): their "1–4% metric"
numbers are IN-DOMAIN benchmarks; on an out-of-domain close-range subject the learned metric degrades —
this is the SAME mechanism as our ~12% VIO-vs-LiDAR gap. A LiDAR time-of-flight reading has NO train/test
domain gap. The SOTA phone wound paper even falls back to ArUco markers for scale — we already surpass
that with LiDAR+VIO. Replacing a physical anchor with a learned prior is a DOWNGRADE for absolute scale.

**Correct architecture (already ours): sensor-anchored, model-consumes-anchor.** MapAnything and AMB3R
ACCEPT metric depth / intrinsics / poses as optional inputs — feed them our ARKit K + poses + LiDAR depth
so the net refines dense structure while the SENSOR stays the scale authority. LiDAR-VGGT and MASt3R-Fusion
(+IMU/GNSS) are the field's templates for exactly this. We already use MapAnything only as a cross-check.

**The ONE worthwhile feed-forward change → T12: swap the metric cross-check MapAnything → AMB3R**
(2511.20343, CVPR'26 Highlight; ~1.7% avg metric-depth rel-err vs MapAnything 3.6%, better pose, and it
also does VO/SfM → a pose/SfM fallback in one model). Run SENSOR-CONDITIONED. NOT to beat LiDAR — for a
tighter outlier detector + a better fallback when LiDAR fails (range >5 m, dark, specular/transparent).
Incremental, low-priority; keep DA3 in its current pose-conditioned/fallback role.

**Skip (redundant overcomplication):** MoGe-2 / DA3-Metric as a SCALE source (monocular metric = weakest
estimator, worse than LiDAR); re-implementing feed-forward metric as the PRIMARY anchor (can't beat ~1%
physical lock, adds a domain-generalization failure mode we don't have); π³/Spann3R/Fast3R/MASt3R-SfM as
scale providers (all up-to-scale — they'd need our anchor anyway; pycolmap already covers pose, and pose
is not our ceiling).

**Real limiter unchanged:** the ~12% LiDAR-vs-VIO disagreement is a SENSOR-FUSION + CAPTURE-PROTOCOL
problem (T1/T3), not a reconstruction-network problem. No feed-forward model addresses it.

## Prior art already in our pipeline (settled — do not re-tread)
DA3 (feed-forward metric geometry frontend, optional/fallback) + MapAnything (metric cross-check, upgrade
candidate → AMB3R) integrated. Campaign finding stands: their learned metric is few-% off, no better than
sensor anchors → optional, not the scale source.

## Net recommendation (avoid overcomplication)
1. Absolute scale: KEEP LiDAR+VIO sensor anchor; add the anisotropy + absolute-thin-error checks to the
   validation harness. Do not adopt any paper's scale method.
2. Measurement: PILOT Deng & Qin's measure-on-splat + uncertainty (T11) — the one net-new capability.
3. Surface metrics (area/volume/deviation) + reference-scan validation protocol: adopt from
   Wound3DAssist only when scope requires it.
