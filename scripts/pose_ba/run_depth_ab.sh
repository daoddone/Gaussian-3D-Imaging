#!/usr/bin/env bash
set -e
cd "/home/paperspace/Documents/VS Code Projects/3D-Gaussian"
ENV="$HOME/miniforge3/envs/pipeline_stage2_frontend"; ENVPY="$ENV/bin/python"
export CUDA_HOME="$ENV" PATH="$ENV/bin:$PATH" LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"
S="sessions/session_20260703_145121"
echo "###### depth_lambda 0.0 (no LiDAR depth supervision), full-res 7k, seeded-BA poses ######"
"$ENVPY" stages/stage5_reconstruction/gsplat_recon.py --session "$S" \
  --iters 7000 --downscale 1.0 --depth-lambda 0.0 --max-init-points 300000 \
  --colmap-dir "$S/metric_ba/colmap/sparse/0" --init-ply "$S/metric/points_metric.ply" \
  --output-dir "$S/output_ab/depth0_fr7k" 2>&1 | grep -aE "step  6999|DONE|Error|Traceback"
echo "###### eval depth0 vs source ######"
"$ENVPY" scripts/pose_ba/eval_splat.py "$S/output_ab/depth0_fr7k/point_cloud.ply" "$S/metric_ba/colmap/sparse/0" 2>&1 | grep -aE "PSNR|SSIM|render/source"
echo "###### DEPTH-AB DONE ######"
