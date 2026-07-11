# THE RECOMMENDED PIPELINE — final conclusions of the quality campaign (2026-07-07 → 07-09)

Definitive configuration, with the evidence chain. Full record: `PIPELINE_JOURNAL.md` (mechanisms,
root causes, lessons), `SWEEP_RESULTS.md` (every A/B), `EXPERIMENTS_BACKLOG.md` (what remains).
Campaign scope: ~20 reconstruction runs, 2 subjects × 2 capture methods, 7 root causes fixed,
4 falsified hypotheses, every verdict cross-checked on both pose paths.

## The pipeline

```
1. CAPTURE (the dominant lever — Protocol v2 in CAPTURE_GUIDANCE.md)
   Fill the frame • slow video-dense orbit • ≥25-30 cm standoff • known-size fiducial in frame.
   Either app path: ARKit (VIO poses; operationally simplest) or HQ-Depth (stills + raw LiDAR).

2. FRAME SELECTION      sharpest-per-temporal-window (Laplacian), ~150-400 frames
                        [proven: visible completeness gain at equal smoothness]

3. POSES                from-scratch SfM (pycolmap; PINHOLE; BA-refined focal) — or ARKit VIO.
                        DA3 = optional fallback only (not required for reconstruction).

4. METRIC SCALE         ANCHOR-ONLY: ruler fiducial > VIO camera-path > LiDAR ray-median lock
                        (scripts/pose_ba/03b_relock_lidar.py, 1.0% MAD proven).
                        NEVER per-pixel depth supervision (retired: worst arm in all 8 cells).

5. RECONSTRUCTION       MILo, mesh reg ON, scalable renderer (2^22 chunks), -r capped ≤2048,
                        depth_lambda = 0. SCHEDULE MATCHED TO INPUT QUALITY:
                          • strong capture (fill-frame, 150+ sharp views): quality_mid,
                            dense OFF, isolation OFF            (= the proven v3 recipe)
                          • weak/legacy capture (standoff, sparse frames): fast, dense ON,
                            COMPLETE ISOLATION SYSTEM ON (photometric mask + opacity penalty
                            + 3D box-prune mop-up; scripts/make_subject_masks.py + box.json)
                        [THE LAW: capacity must match input quality — excess capacity on weak
                         inputs manufactures junk detail regardless of every other setting.
                         dense/isolation/prune are BRANCH-COUPLED ("auto") — 07-10 trilogy.]

6. OUTPUT               metric splat (view-dependent, the appearance deliverable)
                        + metric mesh (geometry deliverable; texture-baking = backlogged polish)
```

## The evidence chain (each link measured, most on both pose paths)

| Verdict | Evidence |
|---|---|
| LiDAR surface supervision harms; worse with capacity | λ0.2→0: −5.3°/−3.7° (fast, both paths); at capacity λ0.2 ≈ 33-35° both paths |
| λ≈0.05 "whisper" unnecessary | R6a: 33.5° (no benefit); isolation solves floaters better |
| Subject isolation: adopt (weak branch) | floater axis 6.6→2.8 m (−58%), −14% gaussians, roughness unchanged (14.78→14.50), subject visually intact |
| Complete isolation = mask + 3D box-prune (07-10) | equal-footing: Andrew head 13.21° vs 14.57° (mask-only) vs 16.64° (none), hull-fill block removed AT SOURCE, 100% capacity in-box; feet gate holds (12.33° vs 12.21° at 2.9× density). Strong branch: NO isolation (v3 recipe; face regression 11.6° reproduces historic best) |
| Capacity law | feet: fast-λ0 = 14.5-15.8° w/ visible toes vs qualmid ≈32-33° (both λ, both paths); face: capacity → +27-47% verts, real stubble relief |
| Poses solved; SfM universal | 100% registration everywhere (0.72-1.5 px); 3 methods agree 1-2 mm |
| Metric-through-poses | subject dims agree ≤2% across λ arms; anchor lock 1.0% MAD |
| Image selection lever | v3 sharp-362 vs v1 blind-172: visibly more complete face, equal smoothness |
| Capture dominates | fill-frame face beats standoff feet in every configuration tried |

## Best artifacts produced (for reference/review)
- Feet (clinical case): `sessions/session_20260704_143324/output_hq_isolated_fast_l0/` — the
  recommended weak-capture config end-to-end (owner-validated class: "point cloud great, mesh decent").
- Face (strong capture): `sessions/face_depthfree_test/output_v3_quality_mid/` — sharpness-selected
  362 frames + capacity (the strong-capture branch demo).

## What remains (post-campaign backlog, priority order)
1. **Clinical re-capture per Protocol v2** (fill-frame feet + fiducial) → confirm the strong-input
   branch on the real clinical subject; unlock the ruler anchor.
2. **True accuracy claim**: fiducial-locked capture + Vectra/phantom reference → the ~1 mm test
   (all current numbers are internal-consistency, not ground-truth accuracy).
3. **Photoreal measurement**: radegs held-out PSNR/SSIM/LPIPS harness (the appearance goal is
   validated visually but still unmeasured numerically).
4. ~~Mesh texturing v1~~ SHIPPED 07-09: sharp top-3-view vertex-color bake (scripts/bake_mesh_colors.py, auto-runs per reconstruction -> mesh_textured.ply); v2 = UV texture map if still insufficient. Note: view-DEPENDENT appearance (specular) is fundamentally splat-only.
5. Strong-capture refinements when warranted: retention percentile (final gaussian count), dense
   mode with capped window, mesh_config highres.
