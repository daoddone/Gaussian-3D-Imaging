# Stage 3 — Metric alignment and validation

Custom, model-agnostic module that guarantees the reconstruction is at true
physical scale and emits the pipeline's headline accuracy number
(`final_residual_meters`). It reads only the frozen file contract, so it works
unchanged if Stage 2 is swapped. Full interface: [`METRIC_CONTRACT.md`](METRIC_CONTRACT.md).

## Run

```bash
conda run -n pipeline_stage3_metric \
  python stages/stage3_metric/run.py \
  --session sessions/<session_id> --config config/pipeline.yaml
```

Exit code `0` = pass; `3` = the session was flagged and `stage3.flag_halts_pipeline`
is true (the coordinator halts and reports).

## What it does

1. Estimates a single scale from up to three independent physical anchors:
   - **sensor depth** — front-end depth resampled onto the sensor grid via
     intrinsics (same physical camera, so no reprojection), then a robust
     RANSAC + least-squares fit of `sensor ≈ scale·front (+ offset)`;
   - **camera path** — closed-form Umeyama similarity between the front end's
     camera centers and the Stage 1 metric camera centers (depth-independent);
   - **physical ruler** — optional, needs a known + measured size.
2. Compares the anchors. Within `agreement_threshold_percent` → apply the
   **median** (pass). Beyond it → prefer the **physical** anchors and **flag**
   (physical measurement outranks the model's learned scale — never silently
   averaged).
3. Applies the similarity to the point cloud and camera poses (optionally
   ICP-refined against the back-projected sensor cloud) and writes the outputs.

## Inputs / Outputs

**Reads:** `frontend/points.ply`, `frontend/poses.json`, `frontend/intrinsics.json`,
`frontend/depth/*.npy`; `capture/depth/*.npy`, `capture/confidence/*.png`,
`capture/intrinsics.json`, `capture/poses.json`, and the capture `README`
(to learn whether the camera-path anchor is present).

**Writes** (`io_contracts/metric_output.md`): `metric/points_metric.ply`,
`metric/colmap/sparse/0/{cameras,images,points3D}.bin` (world-to-camera; camera
model uses the color-resolution intrinsics so it matches the RGB frames Stage 5
optimizes; `points3D` left empty because Stage 5 gets `points_metric.ply` as its
init cloud separately), `metric/scale_report.json`.

## Modules

| File | Role |
| --- | --- |
| `align.py` | Umeyama similarity; robust 1-D depth scale/offset fit; applying a similarity to points and to world-to-camera poses. Pure numpy, unit-tested. |
| `anchors.py` | The three anchors; intrinsics handling; front→sensor depth resampling. |
| `report.py` | Scale-decision rules and the frozen `scale_report.json` schema. |
| `icp.py` | Optional Open3D ICP refinement against the back-projected sensor cloud. |
| `run.py` | Entry point / orchestration. |

## Week-one experiments

`experiments/` (repo root) drives **Experiment A** (does a single global scale
make the reconstruction metric?) using this stage on existing scans. See
[`experiments/README.md`](../../experiments/README.md).

## Notes / assumptions to revisit with real data

- **Frame correspondence** between `capture/` and `frontend/` is by shared
  6-digit frame id. If Stage 2 renumbers its selected keyframes, add an
  id-mapping step; the depth and camera-path anchors currently intersect on
  common ids and report how many matched.
- **MapAnything cross-check** (independent metric second opinion) runs in its
  own Python 3.12 environment and is invoked out-of-band, not from this env.
- The COLMAP camera-model resolution/intrinsics choice must match the images
  the Stage 5 host loads — reconcile with MILo's expected dataset layout
  (`docs/COMPONENT_IO_REFERENCE.md`).
