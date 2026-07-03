# Stage 4 — Surface-normal prior (StableNormal) · OPTIONAL

A per-frame surface-normal map that gently regularizes the reconstruction in
smooth, textureless regions where the image is ambiguous and the depth sensor is
jittery. **This stage is a flag** — it can be kept or removed without damaging
the pipeline, and whether it earns its place is settled by Section 8
Experiment B. It is added at a modest fixed weight and confidence-gated, so its
downside is bounded: it can help or do little, not actively distort.

StableNormal carries **no anatomical assumption** (unlike the excluded
Sapiens2), which is why it is acceptable here for documenting abnormal anatomy.

## Setup / Run

```bash
conda env create -f stages/stage4_normals/environment.yml
conda run -n pipeline_stage4_normals pip install -e ./common
conda run -n pipeline_stage4_normals \
  python stages/stage4_normals/run.py --session sessions/<id> --config config/pipeline.yaml
```

## I/O (see `io_contracts/normals_output.md`)

**Reads:** `capture/rgb/*.png` (prediction); `capture/confidence/*.png` (only if
confidence-tied weighting is enabled).

**Writes:** `normals/000001.npy` `[H,W,3]` unit vectors in `[-1,1]`, **camera
frame, OpenCV**; optional `normals_weight/000001.npy` `[H,W]` in `[0,1]`.

## The convention conversion (the classic silent failure)

StableNormal emits normals in the **OpenGL** camera frame (+Y up, +Z toward the
camera). Our contract is **OpenCV** (+Y down, +Z into the scene). The wrapper
**negates the Y and Z channels** and renormalizes, so a camera-facing surface
has `n_z < 0`, consistent with `normals_from_depth` and the orientation
self-test.

> ⚠️ The source convention is *inferred* (from diffusers/Marigold defaults), not
> documented. **Validate the global sign on one known planar frame** before a
> batch run. Also confirm the sign MILo's normal-supervision term expects
> (`docs/COMPONENT_IO_REFERENCE.md` gap: MILo `mesh_normals` frame undocumented).

## Three normal sources compared in Experiment B

1. none, 2. `normals_from_depth.py` (bias-free, straight from metric depth,
DSINE-style Z-only concerns avoided), 3. StableNormal (this wrapper). See
[`experiments/experiment_b_normal_prior.py`](../../experiments/experiment_b_normal_prior.py).

## Swap candidate

DSINE (lighter feed-forward). Its output needs only the **Z** channel negated
(X/Y already match OpenCV) — medium-confidence, validate empirically.
