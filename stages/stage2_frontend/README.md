# Stage 2 — Front end (Depth Anything 3)

Recovers a camera pose per frame and a dense metric point cloud from the Stage 1
color frames, using **Depth Anything 3**, nested giant+metric model
`DA3NESTED-GIANT-LARGE` (outputs geometry already in meters). It is the
geometric scaffold for everything downstream.

## Setup (isolated env)

```bash
conda env create -f stages/stage2_frontend/environment.yml
git clone https://github.com/ByteDance-Seed/Depth-Anything-3
conda run -n pipeline_stage2_frontend pip install -e ./Depth-Anything-3
conda run -n pipeline_stage2_frontend pip install -e ./common
# Only if the Gaussian head is used (infer_gs): the PINNED gsplat commit
conda run -n pipeline_stage2_frontend pip install --no-build-isolation \
  git+https://github.com/nerfstudio-project/gsplat.git@0b4dddf04cb687367602c01196913cde6a743d70
```

## Run

```bash
conda run -n pipeline_stage2_frontend \
  python stages/stage2_frontend/run.py --session sessions/<id> --config config/pipeline.yaml
```

## I/O (see `io_contracts/frontend_output.md`)

**Reads:** `capture/rgb/*.png` (and, if `stage2.pose_conditioning`,
`capture/poses.json` + `capture/intrinsics.json`).

**Writes:** `frontend/poses.json` (world_to_camera, OpenCV), `frontend/intrinsics.json`
(per-frame K at the model's output resolution), `frontend/depth/*.npy` (meters),
`frontend/conf/*.npy` ([0,1]), `frontend/points.ply`, `frontend/colmap/sparse/0/`.

## Key component facts (from `docs/COMPONENT_IO_REFERENCE.md`)

- `prediction.extrinsics` are **world-to-camera, OpenCV** — an *identity*
  boundary with our contract (no axis flip; only c2w↔w2c inversion if ever
  needed). Shape is `(N,3,4)` or `(N,4,4)` by build — the wrapper branches on
  `.shape`.
- **DA3NESTED depth is already meters.** Do **not** apply the `DA3METRIC`
  `focal*net/300` formula — that would double-scale.
- `prediction.conf` range is **undocumented**; the wrapper normalizes per-frame
  to `[0,1]` with a robust percentile clip. **Validate** against real conf maps.
- DA3 is **0-indexed** by input order; the wrapper maps output `i` → the i-th
  input frame id, keeping numbering aligned with Stage 1.
- Returned `K`/`depth` are at the model's **processed resolution** (≈504), not
  the capture resolution — `intrinsics.json` records that resolution.

## Options (`config/pipeline.yaml` → `stage2`)

`use_ray_pose` (more accurate poses), `pose_conditioning` (condition on Stage 1
poses/intrinsics; sets `align_to_input_ext_scale`), `free_geometry_refinement`
(optional test-time refinement — needs the Free-Geometry repo + `peft`; **not
wired yet**, the wrapper warns and proceeds).

## Swap candidates

MapAnything (`facebook/map-anything`, metric by design) or AMB3R. These double as
the Stage 3 metric cross-check.
