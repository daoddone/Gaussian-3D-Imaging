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
