# Stage 5 — Reconstruction host (MILo)

Builds the actual reconstruction: optimizes surface-aligned Gaussian splats to
reproduce the captured images while respecting the metric depth and the optional
normal prior, and extracts a surface mesh **in the loop** (not as a lossy
afterthought). Produces the Stage 6 deliverables.

Chosen host: **MILo (Mesh-In-the-Loop Gaussian Splatting)** — state-of-the-art
in-loop mesh with an order of magnitude fewer vertices; already renders
differentiable depth+normal maps, which is the seam the ported supervision
attaches to.

## Run

```bash
conda run -n pipeline_stage5_reconstruction \
  python stages/stage5_reconstruction/run.py --session sessions/<id> --config config/pipeline.yaml
```

`run.py` does the **routine** part now — assembling the MILo dataset from the
Stage 3 metric outputs (an identity boundary):

```
sessions/<id>/reconstruction_input/
├── images/000001.png ...        # symlinks to capture/rgb, names match images.bin
└── sparse/0/{cameras,images,points3D}.bin   # the metric-locked model (Stage 3)
```

The metric COLMAP model already carries the baked init cloud in `points3D.bin`
(Stage 3), so MILo initializes from **metric** geometry even if it ignores an
external `.ply`. If the built+ported host is not present, `run.py` **halts with
a flag** (exit code 4) rather than pretending to reconstruct.

## Inputs / Outputs

**Reads:** `metric/colmap/sparse/0/` (metric cameras + init points), `capture/rgb/*.png`,
`metric/points_metric.ply`, `capture/depth/` + `capture/confidence/` (depth
supervision + masking), `normals/` (+ optional `normals_weight/`) if Stage 4 on.

**Writes** (`io_contracts/reconstruction_output.md`): `output/point_cloud.ply`
(Gaussian splat), `output/mesh.ply` (+ optional `.obj`), `output/renders/`,
`output/provenance.json`.

---

## ⚠️ The two hard tasks (NOT routine — need a human engineer)

Per `KICKOFF.md`, these are the hardest parts of the whole project and are
flagged deliberately. Do not present them as scaffolding.

### H1 — Compile MILo's pinned toolchain from source against the A4000

Python 3.9 / CUDA 11.8 / PyTorch 2.3.1, plus building from source: the
differentiable rasterizer submodules (2DGS/3DGS variants), a nearest-neighbor
helper, and `nvdiffrast`. From-source CUDA compilation is a common failure
point (nvcc/host-compiler/arch mismatches, submodule ABI).

**What I need from you:** confirm the exact MILo commit to pin; access to build
on the A4000 with the CUDA 11.8 toolkit; and a decision on whether to build in
this conda env or a container. I will drive the build and surface each error, but
expect to iterate on it together.

### H2 — Port the DN-Splatter / AGS-Mesh depth+normal supervision into MILo

This is a translation of **loss-term concepts** across two frameworks, not a
code copy. The recipe to port (from `docs/COMPONENT_IO_REFERENCE.md`):

1. **Metric depth supervision** — an edge-aware / gradient-aware depth loss
   (AGS-Mesh `EdgeAwareLogL1`, `depth_lambda ≈ 0.2`) driven by the Stage 1
   metric depth, trusting the sensor on smooth regions, less at edges. Attach to
   MILo's differentiable `mesh_depth`.
2. **Normal supervision** (if Stage 4 enabled) — a cosine (L1 + TV) loss between
   MILo's rendered normals and the Stage 4 normal maps, `normal_lambda ≈ 0.1`,
   at the fixed modest weight (× optional trust weight).
3. **Confidence masking** — discard unreliable depth using both the Stage 1
   mask and AGS-Mesh consistency masking. ⚠️ DN-Splatter's confidence is
   **inverted** vs our contract (it expects `255 - ours`; 0 = keep).
4. **Metric initialization** — init splats from the metric point cloud
   (`points3D.bin`), the root fix for floaters.

**Open questions to resolve first (from the research gaps):**
- Confirm MILo's `mesh_normals` **coordinate frame** (camera vs world). Our
  Stage 4 normals are camera-frame OpenCV; rotate by `R_c2w`/`R_w2c` exactly
  once if MILo's are world-frame. Get this wrong and the normal loss silently
  fights the geometry.
- If using the DN-Splatter code path directly, set `normal_format='dsine'`
  (not `omnidata`, which re-applies a Y/Z flip) and feed `255 - confidence`.
- Decide init strategy: bake our points into `points3D.bin` (done in Stage 3) vs
  accept SfM points.

The ported module is expected at `stages/stage5_reconstruction/supervision/`;
`run.py` checks for it and stays flagged until it exists.

### Optional upgrades (after H1/H2)

- **MCMC densification** (same lineage as MILo, `3dgs-mcmc`): note the source
  `noise_lr = 5e5` (the README's `5e-5` is a typo).
- Gaussian PLY (Stage 6) must serialize INRIA-standard fields (log-scale,
  logit-opacity, WXYZ quats); apply `exp`/`sigmoid` when reading into a gsplat
  head — double-activation is a silent bug.

## Fallback host

If porting into MILo proves too heavy: the **DN-Splatter / AGS-Mesh** host
already has depth+normal supervision and confidence masking as first-class
features (runs on `gsplat`), at the cost of a post-hoc rather than in-loop mesh.
