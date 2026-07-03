# Metric photorealistic radiance-field reconstruction of patient anatomy

A staged pipeline that turns a short smartphone LiDAR video of a patient into a
**metrically accurate, photorealistic 3D reconstruction** (a view-dependent
Gaussian-splat radiance field) **and a metric surface mesh**, for clinical
documentation, health-record referencing, and longitudinal tracking. Accuracy
target: ~1 mm surface deviation against a gold-standard reference.

This is the **research and publication** pipeline. Full brief:
[`00_BUILD_SPECIFICATION.md`](00_BUILD_SPECIFICATION.md); start-here rules:
[`KICKOFF.md`](KICKOFF.md).

Two design commitments run through everything:
- **Metric by construction and by validation** — geometry is produced at metric
  scale and independently cross-checked against physical measurements (Stage 3).
- **Faithful to the abnormal** — no healthy-anatomy prior anywhere (Sapiens2 is
  deliberately excluded), so wounds and deformities are never regularized toward
  "normal."

## Pipeline (each arrow is files on disk)

The whole pipeline has been **run end-to-end on a real iPhone 14 Pro capture** —
see [`docs/RESULTS.md`](docs/RESULTS.md).

| Stage | Role | Component | Status |
| --- | --- | --- | --- |
| 1 · capture | iPhone LiDAR → `capture/` | AnatomyCapture (ARKit, Swift) | **built** (ran on device; capture validated) |
| 2 · frontend | poses + dense geometry → `frontend/` | Depth Anything 3 (pose-conditioned) | **built & run** |
| 3 · metric | scale-lock + validate → `metric/` | custom (Umeyama + robust depth + ICP) | **built, verified & run** (2.97 mm) |
| 4 · normals | optional prior → `normals/` | StableNormal | **built & run** (sign gate 0.96) |
| 5 · reconstruction | splats + mesh | **gsplat** depth+normal supervised (MILo alt flagged) | **built & run** |
| 6 · outputs | splat + mesh + renders → `output/` | gsplat + Open3D TSDF | **produced** |

Stages run in **isolated environments** (never merged — their deps conflict) and
communicate **only through files** following [`io_contracts/`](io_contracts/).
The metric-validation slice (Stage 3) is unit-verified; the full reconstruction
runs on the A4000 (see `docs/RESULTS.md` for the real-data run and metrics).

## Repository layout

```
├── orchestrate.py            # coordinator: runs each stage as a subprocess (Section 6)
├── config/pipeline.yaml      # enabled stages, env names, Stage 3 thresholds
├── io_contracts/             # frozen file-format contracts (the real interface)
├── common/                   # dependency-light shared helpers (stdlib + numpy ONLY)
├── stages/
│   ├── stage1_capture/       # CAPTURE_SPEC.md + README (Swift app, deferred)
│   ├── stage2_frontend/      # DA3 wrapper + env + README
│   ├── stage3_metric/        # metric alignment — built & verified
│   ├── stage4_normals/       # StableNormal wrapper + env + README (optional)
│   └── stage5_reconstruction/# MILo dataset prep + env + README (H1/H2 flagged)
├── experiments/              # Section 8 week-one experiments A and B
├── tests/                    # orientation self-test, contract tests, synthetic E2E
├── docs/COMPONENT_IO_REFERENCE.md   # per-component I/O from the doc review
└── sessions/<id>/            # working data (gitignored): capture/frontend/metric/normals/output
```

## Coordinate convention (load-bearing)

OpenCV **everywhere**: camera looks down **+z**, x right, y down; depth
increases along +z; meters, float32. Every pose file declares its `pose_type`.
Convert **once** per boundary via `common/conventions.py`, then re-run the
orientation self-test. The only real rotation conversions in the whole pipeline
are at Stage 1 (ARKit OpenGL→OpenCV) and Stage 4 (StableNormal normal frame);
every COLMAP / DA3 / MILo camera boundary is identity or a pure c2w↔w2c inverse.

## Run

```bash
# one session, all enabled stages, each in its own env
python orchestrate.py --session sessions/<id> --config config/pipeline.yaml

# a single stage
python orchestrate.py --session sessions/<id> --only stage3_metric
```

Coordinator exit codes: `0` ok · `3` Stage 3 flagged a scale disagreement (halts
if `flag_halts_pipeline`) · `4` Stage 5 dataset prepared but host not built/
ported yet (tasks H1/H2).

## Verify (no GPU, no conda needed)

```bash
python tests/run_all.py           # 25 checks: unit + orientation + Stage 3 E2E + Experiment A
python -m common.orientation_selftest
```

The end-to-end check generates a synthetic session with a **known** scale and
confirms Stage 3 recovers it (both anchors → 1.05, residual ~0.02 mm, metric
cloud matches truth to ~0.002 mm).

## Where to start (Section 7 build order) — current state

1. **Stage 3 core + Experiment A on existing scans** — ✅ built & verified; needs
   real `capture/` + `frontend/` data to run for real.
2. **Experiment B (normal prior)** — harness built; the definitive verdict needs
   a reconstructor (Stage 5) + surface reference — flagged in the harness.
3. **Stage 5 upgrades** — dataset prep built; **H1 compile** and **H2
   supervision port** are flagged as human-in-the-loop (see Stage 5 README).
4. **Stage 2 frontend** — wrapper scaffolded against the DA3 API.
5. **Stage 1 capture app** — deferred (produces the real test data).
6. **Validation** against the Canfield Vectra reference.

## Week-one experiments (Section 8)

```bash
python experiments/experiment_a_single_scale.py --sessions sessions/scanA ... --config config/pipeline.yaml
python experiments/experiment_b_normal_prior.py --sessions sessions/scanA ... --config config/pipeline.yaml
```

See [`experiments/README.md`](experiments/README.md).

## Component I/O and known convention conflicts

[`docs/COMPONENT_IO_REFERENCE.md`](docs/COMPONENT_IO_REFERENCE.md) is the
consolidated per-component interface reference from reviewing each project's
docs. It lists the exact conversions each wrapper must perform and the
convention conflicts to watch (DA3 conf range undocumented; StableNormal OpenGL
normals; DN-Splatter inverted confidence; MILo `mesh_normals` frame; etc.).
These are the items to confirm against real code/data as each stage comes online.

## Data needed to run for real (open, from the team)

- Existing capture data (Record3D / Spectacular AI / already-contract) with, per
  scan: `rgb/`, sensor `depth/`, `confidence/`, `intrinsics.json`, and
  (ideally) `poses.json` so the camera-path anchor is live.
- Any physical reference (ruler in frame, or Canfield Vectra / ground-truth
  geometry) so residuals are meaningful in real units.

## License note

Deferred; several components carry non-commercial research licenses, acceptable
for publication. Revisit before any commercial use.
