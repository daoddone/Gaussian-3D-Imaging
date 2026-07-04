#!/usr/bin/env bash
# The real deliverable: full resolution, 30k iters, on the seeded-BA (optimum) poses.
# Changes from the blurry baseline (downscale 2.0 / 7k): full res + 4.3x iters +
# use all init points (no 200k random drop). Densification (DefaultStrategy, refine
# to 15k) now runs its full schedule -> far more, smaller splats -> sharp.
set -e
cd "/home/paperspace/Documents/VS Code Projects/3D-Gaussian"
ENV="$HOME/miniforge3/envs/pipeline_stage2_frontend"
ENVPY="$ENV/bin/python"
export CUDA_HOME="$ENV"; export PATH="$ENV/bin:$PATH"; export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"
S="sessions/session_20260703_145121"

"$ENVPY" stages/stage5_reconstruction/gsplat_recon.py --session "$S" \
  --iters 30000 --downscale 1.0 --depth-lambda 0.2 --sh-degree 3 \
  --max-init-points 300000 \
  --colmap-dir "$S/metric_ba/colmap/sparse/0" \
  --init-ply "$S/metric/points_metric.ply" \
  --output-dir "$S/output_full30k"
echo "############ FULL 30k DONE ############"
