#!/usr/bin/env bash
# Render + score all three splats at matched FULL resolution (48 views each).
set -e
cd "/home/paperspace/Documents/VS Code Projects/3D-Gaussian"
ENV="$HOME/miniforge3/envs/pipeline_stage2_frontend"
ENVPY="$ENV/bin/python"
export CUDA_HOME="$ENV"; export PATH="$ENV/bin:$PATH"; export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"
S="sessions/session_20260703_145121"

echo "############ BASELINE  (half-res trained: downscale 2.0, 7k iters) ############"
"$ENVPY" scripts/pose_ba/eval_splat.py "$S/output_depth_only/point_cloud.ply" "$S/metric/colmap/sparse/0"
echo "############ ARM A     (ARKit poses,  full-res, 7k iters) ############"
"$ENVPY" scripts/pose_ba/eval_splat.py "$S/output_ab/arkit_fr7k/point_cloud.ply" "$S/metric/colmap/sparse/0"
echo "############ ARM B     (seeded-BA poses, full-res, 7k iters) ############"
"$ENVPY" scripts/pose_ba/eval_splat.py "$S/output_ab/ba_fr7k/point_cloud.ply" "$S/metric_ba/colmap/sparse/0"
echo "############ EVAL DONE ############"
