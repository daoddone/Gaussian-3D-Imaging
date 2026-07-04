# Experiments — log & backlog

Running record of pose/capture experiments: what was tested, what it showed, and
what is deliberately deferred. Complements `docs/POSE_BA_PLAN.md` (the execution plan)
and `docs/RESULTS.md`.

Session under test: `sessions/session_20260703_145121` (48 keyframes, ARKit-unified
capture: single ARSession, ARKit pose + LiDAR depth, 1920×1440 color).

---

## DONE — Pose bundle-adjustment experiment (Priority 1)

**Question.** Does ARKit live-tracking pose drift cap reconstruction accuracy (the
suspected cause of soft splats / doubled mesh)?

**Method.** SuperPoint + exhaustive LightGlue matches over all 48 keyframes (`hloc`),
then three independent pose estimates compared in a common metric frame:
1. **ARKit** — phone poses as-is (what Stage 3 feeds the reconstruction).
2. **Seeded BA** — triangulate real tracks at *fixed* ARKit poses, then free all
   per-frame extrinsics in `pycolmap` bundle adjustment (intrinsics fixed).
3. **Unseeded SfM** — from-scratch incremental SfM (`reconstruction.main`) on the
   *same* features/matches, ARKit poses never used.

Re-locked to the Stage-3 metric frame via Umeyama on camera centers (`03_relock.py`),
so all three are directly comparable and inherit the LiDAR-locked scale.

**Result.**

| Estimate | reproj | vs ARKit rotation (median) | vs ARKit center (median) |
|---|---|---|---|
| ARKit (fixed-pose triangulation) | 1.71 px | — | — |
| Seeded BA | 1.56 px | 0.13° (max 0.27°) | 0.38 mm (max 1.1 mm) |
| Unseeded SfM (48/48 registered) | 1.45 px | 0.51° (max 0.91°) | 1.63 mm (max 4.3 mm) |

**Verdict — pose drift is NOT the accuracy ceiling.** You cannot triangulate 5025
points at 1.71 px reproj holding poses *fixed* if those poses were off by ~18 mm.
Freeing every extrinsic moved them <0.4 mm / <0.15°. A completely independent
from-scratch SfM that ignored ARKit still landed within ~1.6 mm / 0.5°. Three
methods agreeing to ~1–2 mm ⇒ the ARKit poses are already at the BA optimum for this
data. The earlier ~18 mm pairwise-ICP gap was ICP sliding on smooth skin + genuine
per-frame LiDAR *depth* disagreement, not pose error.

**Implication.** The blur/doubling has another cause. First identified structural
cause: the "known-blurry" deliverable (`output_depth_only`) was trained at
**downscale 2.0 (half res, 960×720) for only 7000 iters** → under-resolved and
under-densified. See the full-res A/B (`output_ab/`) and Priority 2.

### Sub-result — is the phone-pose SEED essential? (cheap add-on, user-requested)

**No, not for registration.** Unseeded from-scratch SfM registered **all 48/48**
frames on smooth skin (SuperPoint+LightGlue is strong enough here). The seed is a
robustness/accuracy convenience (seeded BA is tighter: 0.4 mm vs 1.6 mm and needs no
initialization search), not a requirement. Keep seeding: it costs nothing, guarantees
the metric gauge, and gives the tightest optimum — but the pipeline would not collapse
without it on captures of this quality.

Artifacts: `pose_ba/refined` (seeded), `pose_ba/sfm_noseed` (unseeded),
`metric_ba/` and `metric_ba_noseed/` (re-locked, reconstruction-ready).

---

## BACKLOG — Different capture method: LiDAR high-quality-depth framework

**Deferred (needs a capture-app change) — do NOT block current work on this.**

**Context / trade-off already decided.** The Apple capture stack forces a choice:
- **ARKit-unified (current):** one ARSession → ARKit pose + LiDAR depth, but color is
  ARKit's video resolution and depth is ARKit's real-time LiDAR.
- **LiDAR HQ-depth framework:** higher-quality depth + higher-res color, but **no
  online pose** (AVCapture and ARKit cannot share the rear camera — confirmed).

Current pipeline chose ARKit-unified because **pose is mandatory** and the phone poses
are the BA seed. The pose experiment above *validates* that choice: ARKit poses are
already at the BA optimum, so we lose nothing by taking them, and the seed rescues
pose recovery on smooth skin (though unseeded also works here).

**Experiment when ready.** Capture the *same* anatomy with the HQ-depth framework →
run the pipeline with **unseeded** BA (no poses available) → compare the deliverable
(sharpness, metric accuracy vs caliper ground truth, mesh doubling) against the
ARKit-unified result. Tests whether better depth + higher-res color outweighs losing
online pose (recovered from scratch, shown here to land within ~1.6 mm / 0.5°).

**Why it might win:** sharper/higher-res color → more SuperPoint detail → tighter SfM;
better depth → better metric anchor + depth supervision.
**Why it might not:** from-scratch SfM has no metric gauge (needs the depth anchor to
re-lock scale, as `03_relock` does), and unseeded is looser (1.6 mm vs 0.4 mm here).

**Prereq:** capture-app change to the HQ-depth framework. Genuinely separate piece of
work — logged here so it is not forgotten, explicitly not started now.

---

## QUEUED — Cross-validate on the hand session (owner-added, run after the current plan)

`sessions/session_20260703_203728` — a **hand** capture (56 RGB frames), raw (no `metric/`
yet, so it needs the FULL pipeline: Stage 2 DA3 → 3 metric → 4 normals → 5 recon, then the
pose-BA + scale diagnostics). Owner: run it AFTER finishing the current plan, for more data;
do not deviate from the plan to do it early.

**Specific diagnostic value (what to check, not just "re-run"):**
1. **Generality of the pose finding.** A hand has far more texture/relief than smooth face
   skin → SuperPoint+LightGlue should match *more easily*. Re-run the 3-method pose agreement
   (seeded BA / unseeded SfM / ARKit) and confirm poses again agree to ~mm (expect even tighter).
2. **Tests the LiDAR near-field-bias hypothesis directly.** The face's ~12% LiDAR-vs-VIO scale
   gap was attributed to LiDAR near-field bias (face at 0.13–0.25 m, below LiDAR's ~0.25 m
   reliable min). If the hand was held at a DIFFERENT working distance, the scale disagreement
   in its `metric/scale_report.json` should CHANGE predictably: larger distance → smaller
   disagreement (bias shrinks). This is a clean confirmation/refutation. **Check working
   distance + `anchors_disagree` %/flags in its scale report.**
3. **Blur/resolution + mesh doubling** reproduce on a second, differently-shaped subject?
   Run the full-res vs half-res comparison and the mesh de-doubling here too.

**PRE-REGISTERED PREDICTION (recorded before processing the hand — from raw LiDAR depth):**
Median working distance FACE = **203 mm** (178–247, below LiDAR's ~250 mm reliable min) →
near-field bias → observed **12%** scale disagreement. HAND = **323 mm** (281–343, at/above the
min) → less bias → **predicts a SMALLER LiDAR-vs-VIO scale disagreement than 12%**. If the hand's
`metric/scale_report.json` shows notably < 12% → near-field-bias explanation CONFIRMED. If it also
shows ~12% → explanation WRONG (systematic LiDAR/VIO issue), reopen the scale question. Config has
`stage3.flag_halts_pipeline: true`, so orchestrate.py will halt at Stage 3 on a flag — that halt is
where the scale_report lands; inspect it, then continue Stage 5 for the reconstruction/pose tests.
