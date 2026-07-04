# MILo integration plan (Stage-5 mesh-quality upgrade)

Reconnaissance/planning only (2026-07-03). Nothing cloned/built/run. Grounded in a full
read of stages/stage5_reconstruction/{gsplat_recon.py,run.py,README.md,environment.yml},
common/, config/pipeline.yaml + web research. See [docs/EXPERIMENTS_BACKLOG.md] and the
memory finding-pose-not-ceiling-scale-ambiguity.

## What MILo is
*MILo: Mesh-In-the-Loop Gaussian Splatting*, Guédon et al., SIGGRAPH Asia 2025 (TOG).
arXiv 2506.24096 · https://anttwo.github.io/milo/ · official repo https://github.com/Anttwo/MILo
(Anttwo = Antoine Guédon, author of SuGaR / Gaussian Frosting; the `bransantiago/MILo` in
search is a mirror — ignore). Extracts a mesh at EVERY training iteration, differentiably,
from Gaussian params: Gaussians as pivots for a Delaunay tetrahedralization (9 pivots each),
9 learnable SDF values per Gaussian (decoupled from opacity/scale/rot), Marching Tetrahedra.
Gradients flow mesh->Gaussians. SOTA mesh quality at ~10x fewer vertices (~4-7M vs 15-16M).
Chosen over 2DGS/Surfels (post-hoc TSDF/Poisson, coarser), SuGaR (its predecessor), GOF
(heavier, not in-loop). MILo will NOT fix image quality — it is a mesh upgrade on good poses.

## THREE GATES (one is a business decision)
1. **LICENSE (go/no-go, owner decision).** MILo inherits the INRIA Gaussian-Splatting
   research/non-commercial license via 3DGS/RaDe-GS/GOF (repo issue #42). For a clinical
   pipeline that may be commercialized this is gating. gsplat (current host) is unrestricted.
   If non-commercial is unacceptable -> do NOT build MILo; use a permissive mesher (2DGS,
   Apache-2.0) on the existing gsplat splats instead.
2. **VRAM.** Dense mode ~17GB -> OOMs on the 16GB A4000 (paper used 24GB 4090). Base
   (non-dense) mode only, ~10GB, capped Gaussians, downscaled renders (issues #38, #43).
3. **Depth port is NET-NEW.** MILo uses no external depth (its "depth loss" is Gaussian-vs-mesh
   self-consistency). Our LiDAR edge_aware_logl1 is a new signal to inject, not a swap.

## Build recipe (separate `milo` conda env; NO system CUDA here so nvcc must come from conda)
- `conda create -n milo python=3.9`; install `pytorch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1
  pytorch-cuda=11.8 mkl=2023.1.0` (-c pytorch -c nvidia).
- `conda install -c nvidia/label/cuda-11.8.0 cuda-toolkit` (provides nvcc + dev headers).
- `conda install -c conda-forge cmake ninja gmp cgal=5.6 eigen`.
- Env gotcha: `export CUDA_HOME=$CONDA_PREFIX; PATH=$CUDA_HOME/bin:$PATH; TORCH_CUDA_ARCH_LIST=8.6`
  (A4000=sm_86). Driver 535 runs cu11.8 fine. gcc 11.4 is within cu11.8's <=gcc11 limit.
- `git clone --recursive https://github.com/Anttwo/MILo`; build submodules: diff-gaussian-
  rasterization{,_ms,_gof}, simple-knn, fused-ssim, nvdiffrast, tetra_triangulation (CGAL).
- **Known build failures + fixes:** (a) nvcc/CUDA_HOME unset -> use CONDA_PREFIX (above);
  (b) CGAL `Parallel_if_available_tag` error in tetra_triangulation (GOF issue #16) -> edit
  triangulation.cpp `typedef CGAL::Sequential_tag Concurrency_tag;`, pin cgal=5.6; (c)
  nvdiffrast segfault headless (#34) -> use RasterizeCudaContext or EGL; (d) set arch to 8.6
  to avoid multi-arch bloat.

## Depth-supervision port (the hard task)
- Poses/intrinsics/init already handled: run.py::prepare_milo_dataset writes COLMAP
  reconstruction_input/ (our OpenCV world_to_camera == COLMAP native; Stage-3 metric points
  baked into points3D.bin).
- Add `stages/stage5_reconstruction/supervision/ags_depth_normal_losses.py` (framework-agnostic;
  port edge_aware_logl1 + depth_to_normal verbatim from gsplat_recon.py — pure torch).
  run.py::_host_ready() already checks for this file.
- Inject into MILo train loop where it has rendered Gaussian depth D:
  `loss += depth_lambda * edge_aware_logl1(D, gt_lidar, rgb, mask)` (depth_lambda~0.2). Reuse
  gsplat_recon.load_dataset mask-normalized depth resize (prevents 0-hole bleed).
- **Top risk (silent):** RaDe-GS/GOF dataloaders often recenter+rescale scene to a unit cube ->
  LiDAR meters won't match rendered depth; audit + disable normalization or scale LiDAR too and
  invert on mesh export. Also confirm depth semantics (metric z, not disparity/NDC/ray-dist) and
  normal frame (RaDe-GS renders world-frame normals; our Stage-4 normals are camera-frame ->
  rotate by R_c2w once).

## Port findings (verified by reading the cloned repo, 2026-07-03)
- **Normalization trap AVOIDED (verified).** MILo uses stock 3DGS `getNerfppNorm` (scene/
  dataset_readers.py): `nerf_normalization = {translate:-center, radius:diag*1.1}` feeds only
  the position LR scale (`create_from_pcd(pcd, cameras_extent)`) and densification thresholds —
  it does NOT rescale points or cameras. **MILo trains in METRIC coordinates**, so LiDAR depth
  (m) matches rendered depth directly; no unit-cube conversion needed. This de-risks the port's
  #1 danger.
- **Injection points (train.py):** loop renders with `require_depth=True` -> `render_pkg` has
  rendered depth; `Ll1 = l1_loss(image, gt_image)`. Add
  `loss += depth_lambda * edge_aware_logl1(render_pkg[depth], cam.lidar_depth, image, cam.mask)`.
  MILo already has depth-order + depth-normal + mesh-in-loop regularizers to sit alongside.
- **Data plumbing:** extend scene/cameras.py `Camera` to hold `lidar_depth`+`mask`; load in
  scene/dataset_readers.py from our `capture/depth/{fid}.npy` + `capture/confidence/{fid}.npy`
  (reuse gsplat_recon.load_dataset mask-normalized resize). Match by COLMAP image name.
- **Run:** `train.py -s <colmap_dataset> -m <out> --imp_metric indoor --rasterizer radegs`
  (indoor for close-up face/hand); then `mesh_extract_sdf.py`. TODO at port time: confirm radegs
  rendered depth is metric z-depth (not disparity/NDC) before differencing against LiDAR.
- **Env gotcha resolved:** MILo's `mkl=2023.1.0` pin is UNSATISFIABLE with torch 2.3.1
  (mkl2023.1 needs llvm-openmp>=16; torch2.3.1 needs <16). Drop the mkl pin. Build the env with
  scripts/pose_ba/milo_build_env2.sh + milo_build_submodules.sh. tetra_triangulation's
  `find_library(cnpy)` is unused (target links only CUDA/Torch/CGAL) — cnpy NOT required; and its
  Delaunay is sequential (no CGAL Concurrency_tag fix needed).

## Architecture
Keep MILo as an ALTERNATIVE Stage-5 host (config stage5.host: milo), NOT a replacement. Add
`milo_supervised.py` exposing reconstruct(dataset_dir, capture_dir, normals_dir, output_dir,
options) [signature run.py already calls]; run MILo train + mesh_extract, then write the SAME
output contract: point_cloud.ply (re-serialize into INRIA fields via gsplat_recon.export_ply
layout), mesh.ply (in metric world space — undo any normalization), renders/, provenance
(stage5_host: milo, rasterizer, commit, lambdas, counts). gsplat stays the default host.

## Effort / recommendation
- **Interim (do first, ~1/2 day, ~0 risk):** improve current Open3D TSDF in
  gsplat_recon.export_mesh_and_renders (voxel 0.004 -> ~0.0015-0.002, tighten sdf_trunc &
  depth_trunc, add per-view depth max-clip + edge/discontinuity mask before integrate()). The
  "doubled" mesh is most likely TSDF fusing a 2nd shell from depth-edge/back-face leak; this
  likely fixes most of it far cheaper than MILo, and re-extracts from existing splats (no retrain).
- **MILo:** high effort (1-3d build + 2-4d port + 16GB tuning), medium-high risk (immature repo,
  license, VRAM). Two-track conditional go: (1) ship finer-TSDF interim now; (2) resolve the
  LICENSE question first — if non-commercial is unacceptable, stop and use 2DGS/permissive mesher;
  (3) if acceptable, green-light MILo as an experimental host on a pinned commit, kept out of
  production default until it beats the improved TSDF within 16GB.

Sources: arXiv 2506.24096; anttwo.github.io/milo; github.com/Anttwo/MILo (issues #34,#38,#42,#43);
GOF tetra_triangulation CGAL issue #16; ACM TOG 10.1145/3763339.

---

## BUILD LOG — EXECUTED 2026-07-03/04 (H1 + H2 DONE, host wired)

MILo is BUILT and depth-supervision is PORTED + validated. Env `milo`; repo `third_party/MILo`.
Reproduce with scripts/pose_ba/milo_build_env2.sh -> milo_build_submodules.sh -> milo_fix_tetra.sh.

**Blockers hit + fixes (all real, all resolved):**
1. `mkl=2023.1.0` pin UNSATISFIABLE (needs llvm-openmp>=16; torch2.3.1 needs <16). -> drop mkl pin.
2. conda `pytorch-cuda=11.8` resolved to a CPU-only torch. -> `pip install --force-reinstall
   torch==2.3.1+cu118 ... --index-url https://download.pytorch.org/whl/cu118` (GPU, avail True).
3. tetra_triangulation cmake FAILED: cmake 4.x rejects pybind11 v2.9.2's old cmake_minimum_required.
   -> `conda install cmake<4` (got 3.31.8). (cnpy find_library is unused; Delaunay is sequential
   so no CGAL Concurrency_tag fix needed.)
4. Runtime: nvdiffrast JIT-compiles its OpenGL plugin -> `fatal error: EGL/egl.h` (headless #34).
   -> scene/mesh.py MeshRasterizer default `use_opengl=True` -> `False` (nvdiffrast CUDA context).
5. Runtime: `cudaErrorInvalidConfiguration` in the _ms rasterizer backward at iter 0. ROOT CAUSE:
   our METRIC scenes are ~0.1 units; the INRIA-lineage rasterizer overflows at that scale (gsplat
   tolerated it). -> train at scene scaled to ~unit (S = 1/nerf_radius), re-metric outputs /S.
   This is baked into milo_supervised.py (auto-scale) and threaded into the depth loss.

**Depth port (H2) — 3 flag-guarded edits to milo/train.py + supervision/ags_depth_normal_losses.py:**
- args `--lidar_depth_dir/--lidar_depth_lambda/--lidar_depth_scale`; loss block after the base
  photometric loss uses `render_pkg["expected_depth"]` (metric z) + on-the-fly LiDAR load by
  `image_name` (no camera/dataset-class edits). Piggybacks on the depth render active from
  regularization_from_iter=3000 (forcing it earlier breaks the _ms densification, which needs
  'area_max'). LiDAR metres *= lidar_depth_scale (S) to match the scaled render depth.
- NaN-safety: iPhone LiDAR marks invalid px as NaN; `nan_to_num` in load_lidar_depth (BEFORE the
  mask-normalized resize, else d*m = NaN*0 poisons the map) AND in edge_aware_logl1 (masked
  selection over a tensor with NaN makes backward compute 0*NaN=NaN).
- Validated: loss finite past iter 3000, LiDAR term contributes (~+0.07 to loss), 22 it/s.

**Host wiring:** milo_supervised.reconstruct(dataset_dir, capture_dir, normals_dir, output_dir,
options) runs MILo in its env via subprocess (auto-scale up, train+mesh_extract, scale outputs
back to metric, write point_cloud.ply + mesh.ply + provenance). run.py _host_ready() now checks
the milo env + compiled submodules + the two H2 files (NOT `import milo`, wrong env). Set
config stage5.host: milo to select it. Mesh -> <out>/mesh_learnable_sdf.ply (learnable-SDF
Marching-Tetrahedra); gaussians -> <out>/point_cloud/iteration_N/point_cloud.ply.
