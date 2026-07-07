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

## Summary of levers (batch complete)
- **depth_lambda** (LiDAR weight) ↓ → SMOOTHER (~5°, the biggest single lever) but risks metric
  drift/over-smoothing. **Bumpiness is substantially LiDAR-sensor-noise, not just density.** Sweep
  0.2 / 0.1 / 0.05 next.
- **Density** ↓ → smoother (~2-3°); costs ~10× mesh/cloud detail.
- **mesh_config** lowres → rougher (dead end for smoothness).
- **Subject isolation** → top lever for HQ's background spread (untested; judgment-heavy — owner input).
- **Regularizers** (depth_ratio↓ / normal_weight↑) → proposed path to smoother-without-detail-loss.

## Running / queued
- (running) **ARKit depth_lambda-0** (LiDAR off) — ETA ~09:00; will give the LiDAR-supervision A/B.
- (next, for owner to steer) **SUBJECT ISOLATION** · **regularizer tuning** · feature-preserving smoothing.
