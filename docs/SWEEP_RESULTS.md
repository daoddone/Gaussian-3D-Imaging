# Knob-sweep results (2026-07-05, autonomous overnight session)

Reconstruction A/B results for owner review. Renders saved to `sessions/_sweep_eval/<name>/comparison.png`
(gitignored, on-disk — open in an image viewer; the `.ply` outputs are in each session's
`output_<label>/` for deep review in a splat/mesh viewer).

**Roughness metric = mean dihedral angle between adjacent mesh faces (deg); higher = bumpier.**
CAVEAT: dihedral conflates true surface roughness with mesh resolution (a finer mesh has smaller faces
that capture more micro-relief), so compare it alongside the renders, not in isolation. All current
reconstructions are **scene-scale** (feet + stand + floor), so the mm extents are the whole scene.

Fix applied this session: MILo mesh reg crashed on HQ because nvdiffrast's CUDA rasterizer caps at
2048 px/side and HQ renders at 4032; the driver now caps `-r` so max side ≤2048 (HQ → 2016). See
`DAILY_NOTES.md`.

---

## ARKit density A/B — session_20260704_143210 (VIO poses, -r1 / 1920 px)
| variant | gaussians | mesh verts | mesh faces | roughness° | dir |
|---|---|---|---|---|---|
| **dense** (baseline) | 406,436 | 4,102,775 | 8,179,770 | **21.09** | `output_arkit_dense/` |
| **low-density** | 46,915 | 408,532 | 814,695 | **18.43** | `output_arkit_lowdens/` |

**Finding:** lower density reduces roughness 21.1° → 18.4° (~13%) — partially confirms the owner's
hypothesis that dense drives bumpiness — but the effect is **modest**, and low-density costs ~10× mesh
detail. Bumpiness is **not solely** a density artifact; mesh_config / regularizers / capture distance
are the other levers to test. Render: `sessions/_sweep_eval/arkit_density/comparison.png`.

---

## ARKit vs HQ — capture-method comparison (dense, both mesh-reg ON)
| method | poses | render px | gaussians | mesh verts | roughness° | cloud extent (mm) | object-box (mm) |
|---|---|---|---|---|---|---|---|
| **ARKit** | VIO | 1920 | 406,436 | 4,102,775 | 21.09 | 2379×1998×2105 | 1094×719×1061 |
| **HQ** | SfM (LiDAR-locked) | 2016 | 732,347 | 2,579,412 | 18.34 | 3103×**5301**×2565 | 1307×**2177**×851 |

**Key observations:**
- **HQ is much more spread out** — 5.3 m cloud / 2.6 m mesh in one axis vs ARKit's ~2 m / 0.8 m. HQ's
  from-scratch SfM triangulated a long swath of background/floor along the capture path; ARKit's VIO
  fused a more compact scene. So **HQ especially needs SUBJECT ISOLATION** (crop the reconstruction to
  the feet) — now the top-priority knob.
- Roughness (HQ 18.3° < ARKit 21.1°) is confounded by resolution/extent/vert-count — trust the renders,
  not the number.
- **Different capture instances** (feet may have shifted) → this is a method-quality comparison, not a
  same-surface accuracy test. A fair "which is better" verdict needs the visual review + subject
  isolation. Render: `sessions/_sweep_eval/arkit_vs_hq/comparison.png`.
- Aside: HQ SfM-pose cloud (`output_hq_dense`) vs the earlier DA3-ray-pose cloud (`output_da3_nomesh`)
  is the "poses-matter-for-quality" A/B, available if wanted.

## ARKit mesh_config A/B — session_20260704_143210 (dense, same gaussians)
| variant | gaussians | mesh verts | roughness° | dir |
|---|---|---|---|---|
| default mesh | 406,436 | 4,102,775 | 21.09 | `output_arkit_dense` |
| **lowres** mesh_config | 411,260 | 553,825 | **41.35** | `output_arkit_meshlowres` |

**Finding (counterintuitive):** `mesh_config: lowres` makes the mesh **ROUGHER** (41° vs 21°), not
smoother — a coarser Delaunay tet grid extracts a cruder/blockier mesh of the SAME gaussian surface
(bigger faces → bigger angle jumps). So **mesh_config is the WRONG lever for smoothness**; it controls
mesh sampling fineness, not surface smoothness. The real smoothness lever is **DENSITY** (fewer
gaussians → genuinely smoother surface, 18.4°). Also confirms dihedral roughness is resolution-sensitive.
Render: `sessions/_sweep_eval/arkit_mesh/comparison.png`.

## HQ density A/B — session_20260704_143324 (SfM poses, -r2 / 2016)
| variant | gaussians | mesh verts | roughness° | dir |
|---|---|---|---|---|
| **dense** | 732,347 | 2,579,412 | 18.34 | `output_hq_dense` |
| **low-density** | 84,269 | 1,178,436 | 16.55 | `output_hq_lowdens` |

**Finding:** same as ARKit — density down → smoother mesh (18.3° → 16.6°). The density→smoothness lever
holds for BOTH capture methods. HQ stays spread (~2 m object-box in Y even at low density) → subject
isolation is orthogonal to density and still needed. Render: `sessions/_sweep_eval/hq_density/comparison.png`.

## ARKit LiDAR-supervision A/B — session_20260704_143210 (dense)
| variant | gaussians | mesh verts | roughness° | dir |
|---|---|---|---|---|
| **depth_lambda 0.2** (LiDAR on) | 406,436 | 4,102,775 | 21.09 | `output_arkit_dense` |
| **depth_lambda 0.0** (LiDAR off) | 591,973 | 5,075,161 | **15.75** | `output_arkit_depth0` |

**Finding (important):** turning OFF LiDAR supervision makes the mesh markedly SMOOTHER (21.1° → 15.8°,
~5° — a BIGGER effect than density's ~2-3°). ARKit's 256×192 LiDAR is noisy; supervising against it
injects sensor noise into the surface → bumpier mesh. **CAVEAT: smoother ≠ better** — without LiDAR the
surface may drift from true metric shape or over-smooth real relief ("faithful to the abnormal"). So the
lever isn't "LiDAR off" but "tune `depth_lambda` DOWN" (0.1 / 0.05) to cut sensor-noise bumpiness while
keeping metric grounding. Also note: no-LiDAR densified MORE (592k vs 406k) — the depth term was
constraining growth. Render: `sessions/_sweep_eval/arkit_lidar/comparison.png`.

## HQ LiDAR-supervision A/B — session_20260704_143324 (SfM poses, -r2, FAST schedule, 2026-07-07)
| variant | gaussians | roughness° | cloud extent (mm) | dir |
|---|---|---|---|---|
| depth_lambda 0.2 | 732,347 | 18.34 | 3103×5301×2565 | `output_hq_dense` |
| **depth_lambda 0.0** | 791,433 | **14.63** | 3134×5206×**6619** | `output_hq_depth0` |

**Completes the fast 2×2:** λ0 smooths on BOTH pose paths (ARKit −5.3°, HQ −3.7°) → noise-stamping is
pose-source-independent. **NEW INSIGHT:** λ0's cloud extent blew out in Z (2.6→6.6 m) — the LiDAR term
was doubling as a FREE-SPACE FLOATER SUPPRESSOR. Anchor-only architecture must replace that (subject
isolation / box-prune / small λ≈0.05). Object-box dims agree ~7-10% across arms (metric-through-poses
consistent, residual = floater statistics). Render: `sessions/_sweep_eval/hq_lidar/comparison.png`.
NOTE: all four cells measured at FAST — the quality-schedule matrix (R1-R4) re-tests them at full capacity.

**Subject-level follow-up (eval harness now auto-crops to the densest cluster):**
- Subject-mesh roughness: λ0.2 = 18.87° vs λ0 = **14.78°** → the smoothing holds ON THE SUBJECT, not just background.
- Subject-cluster dims agree across arms to ~0.3% (X) / ~2% (Z) → **metric-through-locked-poses confirmed at
  subject level** (the 11% Y delta is crop-radius sensitivity to floaters, not scale drift).
- **VISUAL verdict (renders): BOTH HQ feet arms are poor** — diffuse cloud, torn-sheet mesh, stand more
  legible than feet. λ0 is smoother junk. Root-cause candidate (NEW): **init density × schedule
  interaction** — ARKit's metric model bakes a DENSE DA3 init (100k pts); HQ-SfM initializes from only
  17k sparse SfM points, and the fast schedule's 3k-iter densify window can't grow a sparse init to
  coverage. Predicts R2/R4 (quality, 15k window) improve HQ disproportionately. If not: these feet
  captures (47 sparse stills, subject small in frame) are fundamentally weak inputs vs the face's 172
  frame-filling video frames.

## Quality-schedule internals (Face v2 live observation, 2026-07-07)
Densify trajectory on 172-view face: peak **16.4M** gaussians (47 GB VRAM solo — quality-dense runs
must be SERIALIZED through their peaks) → simp1 @15k: **1.8M** → simp2 @20k: **80,481** — only ~1.4×
the fast run's final 57.7k. **MILo's importance-distillation sets the final count nearly independent
of the peak**; the schedule's value is better *placement/refinement* of those final gaussians, not
count. → NEW R6 LEVER: the **simplification retention percentile** (importance-mass cutoff in
`init_cdf_mask`) — if granularity needs more FINAL gaussians, the knob is the prune criterion, not
schedule length. Also validated: `regularization_from_iter` must equal `densify_until_iter`
(renderer-switch KeyError otherwise; both now in `configs/quality`).

## Face schedule A/B — fast18k vs quality30k (same 172 frames, depth-free, 2026-07-07)
| variant | final gaussians | head-crop mesh verts | subject-mesh roughness° | verdict |
|---|---|---|---|---|
| fast 18k | 57,703 | 362,834 | **9.45** (smoother) | cleaner, waxier skin |
| quality 30k | 80,481 | 461,885 | 12.43 | **+27% verts; visible jaw/cheek micro-relief (matches real stubble)**, more floaters |

**Verdict:** the stock schedule buys a REAL but MODEST granularity gain (fine relief consistent with
actual skin texture), not a transformation → the owner's old-run gap is probably NOT schedule →
image-set selection (v3, sharpness-picked) is the leading candidate. Renders:
`sessions/_sweep_eval/face_schedule/` (incl. `face_tight_AB.png`). Owner judgment vs reference invited:
`sessions/face_depthfree_test/output{,_quality}/`.

**Feet VRAM reality:** stock-30k schedule OOM'd TWICE on the ARKit feet (dense @9k paired; non-dense
solo) → new `configs/quality_mid` (identical MS2 dynamics, densify window capped at 8k, simp 8k/12k,
mesh 12k→22k) is the A6000-feasible full-capacity point; feet matrix runs on it.

## Face image-set A/B — v1 blind-172 vs v3 sharpness-362 (both FAST, depth-free, 2026-07-07)
| variant | gaussians | head-crop verts | subject roughness° | visual |
|---|---|---|---|---|
| v1 blind stride-19 (172) | 57,703 | 362,834 | 9.45 | face has holes/cutoffs (mouth/chin) |
| **v3 sharpest-per-9 (362)** | 69,409 | 433,425 | 9.52 | **markedly more complete face** (full forehead→chin, intact mouth), tighter subject cluster |

**Verdict:** sharpness-aware dense frame selection = the INPUT lever for surface COMPLETENESS at equal
smoothness. Best explanation for the owner's old-run edge (their stride-5 set was temporally denser).
SfM gates: v3 registered 362/362 @ 0.717 px, focal solve consistent with v1 to <1 px. Renders:
`sessions/_sweep_eval/face_imageset/` (incl. `face_tight_v1v3.png`).
**Face program conclusion (all three arms):** poses ✓ → depth-free clean ✓ → schedule = modest texture
gain → image selection = visible completeness gain. Optional best-face combo (v3+quality) deferred to R6.

## R1'' — ARKit feet @ quality_mid, λ0.2 (vs fast baselines, 2026-07-07)
| variant | gaussians | mesh verts | subject roughness° |
|---|---|---|---|
| fast dense λ0.2 | 406,436 | 4.10M | 19.58 |
| fast non-dense λ0.2 (`output_arkit_lowdens`) | 46,915 | 0.41M | ~18.4 |
| **qualmid non-dense λ0.2 (R1'')** | 441,534 | 5.36M | **32.69** |

**Finding (diagnostic):** the capped-quality schedule grew the non-dense budget 9.4× (46.9k→441k) —
and with λ0.2 ON, the added capacity imprinted LiDAR sensor noise even MORE finely (roughness nearly
doubled). The depth-free face gained only +3° from the same schedule change → **capacity amplifies the
depth-term's noise-stamping**. Strongest confirmation yet of the λ mechanism. R3' (ARKit λ0 @ qualmid)
UN-GATED — the λ pair at capacity (R2' vs R4', R1'' vs R3') is now the decisive architecture test.
Subject cluster tighter than fast (883×624×761 mm) — scale consistent. Render: `_sweep_eval/arkit_schedule/`.

## R2' — HQ feet @ quality_mid, λ0.2 (2026-07-08, first run through the scalable-renderer fix)
| variant | gaussians | mesh verts (faces) | subject roughness° | subject cluster (mm) |
|---|---|---|---|---|
| fast λ0.2 | 732k | 2.6M (5.1M) | 18.87 | 1111×1119×973 |
| **qualmid λ0.2 (R2')** | 685k | **8.2M (16.5M)** | **35.06** | **691×563×550** |

**Two findings:** (1) capacity-amplified noise-stamping CONFIRMED on the HQ path (mirrors R1'' ARKit
19.6→32.7) — λ0.2 at capacity is unambiguously harmful; (2) the subject cluster TIGHTENED ~40% at
capacity → the sparse-SfM-init coherence problem was the densify window (init×window mechanism
confirmed) — HQ's path is exonerated. Run also validates root-cause-#7 fixes in production (16.5M-face
mesh trained + extracted cleanly). R4' (λ0 twin) = the decisive pair, auto-evaluating on completion.

## R4' — THE λ pair at capacity (HQ qualmid, 2026-07-08) — PREDICTION FALSIFIED, deeper law found
| variant | gaussians | mesh verts | subject roughness° | visual (subject renders) |
|---|---|---|---|---|
| fast λ0 (`output_hq_depth0`) | 791k | 2.7M | **14.78** | **recognizable feet, visible toes, smooth skin — best in campaign** |
| qualmid λ0.2 (R2') | 685k | 8.2M | 35.06 | craggy |
| **qualmid λ0 (R4')** | 777k | 7.4M | **32.65** | craggy, fragmented — rougher AND less legible than fast-λ0 |

**Falsified:** "λ0 at capacity wins dramatically" — the λ effect shrinks to ~2.4° at capacity.
**The deeper law: CAPACITY MUST MATCH INPUT QUALITY.** On weak captures (standoff feet, ≤57 frames)
added capacity fits junk regardless of λ — photometric starvation in the fast schedule was acting as
implicit regularization. On strong captures (fill-frame face) capacity adds real detail (+27% verts,
true stubble relief). λ0.2 remains worst everywhere → LiDAR-as-anchor-only unchanged.
**Practical:** weak/legacy captures → fast + λ0 (their optimum); strong captures → quality schedules.
Render: `_sweep_eval/hq_lambda_capacity/feet_tight_schedule_l0.png`. Remaining arms reinterpreted:
R3' = pose-independence check of capacity-roughness (~expect 30-34°); R6a = λ whisper at capacity
(~expect ≈R4'); R6b = best-face demo (strong inputs → expect good).

## Summary of levers (batch complete)
- **depth_lambda** (LiDAR weight) ↓ → SMOOTHER (~5°, the biggest single lever) but risks metric
  drift/over-smoothing. **Bumpiness is substantially LiDAR-sensor-noise, not just density.** Sweep
  0.2 / 0.1 / 0.05 next.
- **Density** ↓ → smoother (~2-3°); costs ~10× mesh/cloud detail.
- **mesh_config** lowres → rougher (dead end for smoothness).
- **Subject isolation** → top lever for HQ's background spread (untested; judgment-heavy — owner input).
- **Regularizers** (depth_ratio↓ / normal_weight↑) → proposed path to smoother-without-detail-loss.

## Depth-free face baseline — sessions/face_depthfree_test (2026-07-07, owner-requested)
Reproduces the owner's clean preliminary run (color photos → from-scratch pycolmap SfM → MILo; no
depth term, no dense_gaussians, no DA3) through the CURRENT MILo. Runner: `scripts/face_depthfree_test.py`.
172 frames (stride-19 of the 3,258-frame Record3D export ≈ the preliminary's 169), PINHOLE with
BA-refined focal (init 1367 → refined 1496), sequential matching.

| gate / result | value |
|---|---|
| SfM registration | **172/172**, 44,964 points |
| mean reprojection | **0.79 px** (tighter than the feet SfM's 1.50 px) |
| MILo (mesh reg ON, -r1/1920, depth_lambda 0, dense off) | 57,703 gaussians; 818,899-vert mesh |
| roughness | **10.13°** — best ever (vs 21.1° feet tip-top, 15.8° feet LiDAR-off) — and at ~5× FEWER mesh verts than the feet dense mesh, which per the mesh_config A/B biases the number ROUGHER, so the true gap is understated |
| visual | facial features legible in the MESH (eyes, brows, nose, cap buckle); smooth skin surfaces; background = the real wooden door + floaters. Renders: `sessions/_sweep_eval/face_depthfree/` |
| caveat | gauge-free (NON-metric — no depth anywhere); mm fields in stats are meaningless units |

**What it establishes:** (1) the depth-free config through current MILo produces clean, detailed
surfaces — consistent with the LiDAR-supervision diagnosis; (2) **COLMAP→MILo with no DA3 works
end-to-end** (COLMAP sparse init suffices); (3) the SfM front-end is excellent on a fill-frame capture.
**Honest confound:** vs the feet runs this also changed subject, view count (172 vs 47-57), and —
importantly — **capture framing (face fills the frame = the pixel-coverage detail lever)**. The
controlled evidence for the depth-term effect remains the same-capture feet A/B (21.1°→15.8°); this
test shows the depth-free *path* is viable and clean end-to-end, not the isolated term effect.

## Running / queued
- (done) ARKit depth_lambda-0 — logged above.
- (next, for owner to steer) **depth_lambda refine sweep** (0.05/0.1 on feet — find the metric-vs-noise
  sweet spot) · **SUBJECT ISOLATION** · **regularizer tuning** · LiDAR-as-anchor-only architecture
  (scale from 03b-style lock; surface from photometric+mesh-reg only).

## FINAL NIGHT (2026-07-09): R3', R6a, R6b, ISOLATION — campaign complete
| arm | result | verdict |
|---|---|---|
| R3' ARKit qualmid λ0 | 31.85° (vs λ0.2's 32.69°) | capacity-dominance is pose-independent; λ negligible at capacity |
| R6a HQ qualmid λ0.05 | 33.47° (vs λ0's 32.65°) | whisper adds nothing at capacity — retired |
| R6b face v3+qualmid | 9.52→12.3°, +47% gaussians, subject cluster grew | strong captures absorb capacity as coverage/detail (law confirmed) |
| **ISOLATION (fast+λ0)** | **14.50° (=), floater axis 6.6→2.8 m (−58%), −14% gaussians, subject visually intact** | **ADOPT — first-try success** |

**Campaign closed.** Definitive configuration + evidence chain: `docs/PIPELINE_RECOMMENDATION.md`.
