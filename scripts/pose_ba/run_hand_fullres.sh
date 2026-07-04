#!/usr/bin/env bash
set -e
cd "/home/paperspace/Documents/VS Code Projects/3D-Gaussian"
ENV="$HOME/miniforge3/envs/pipeline_stage2_frontend"; ENVPY="$ENV/bin/python"
export CUDA_HOME="$ENV" PATH="$ENV/bin:$PATH" LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
S="sessions/session_20260703_203728"
echo "###### HAND full-res 7k (seeded-BA poses) ######"
"$ENVPY" stages/stage5_reconstruction/gsplat_recon.py --session "$S" \
  --iters 7000 --downscale 1.0 --depth-lambda 0.2 --max-init-points 300000 \
  --colmap-dir "$S/metric_ba/colmap/sparse/0" --init-ply "$S/metric/points_metric.ply" \
  --output-dir "$S/output_fullres" 2>&1 | grep -aE "step  6999|DONE|Error|Traceback"
echo "###### eval HAND full-res ######"
"$ENVPY" scripts/pose_ba/eval_splat.py "$S/output_fullres/point_cloud.ply" "$S/metric_ba/colmap/sparse/0" 2>&1 | grep -aE "PSNR|SSIM"
echo "###### de-doubled mesh (voxel 0.002) ######"
"$ENVPY" scripts/pose_ba/remesh.py "$S/output_fullres/point_cloud.ply" "$S/metric_ba/colmap/sparse/0" "$S/output_fullres/mesh_dedoubled.ply" 0.002 0.5 0.02 0.05 2>&1 | grep -aE "remesh"
echo "###### HAND FULLRES DONE ######"
