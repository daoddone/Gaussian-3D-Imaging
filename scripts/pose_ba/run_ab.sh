#!/usr/bin/env bash
# Strict pose A/B at FULL resolution. Both arms: identical init ply + settings;
# ONLY --colmap-dir (poses) differs. Also serves as the resolution test vs the
# half-res baseline (output_depth_only, downscale 2.0).
set -e
cd "/home/paperspace/Documents/VS Code Projects/3D-Gaussian"
ENV="$HOME/miniforge3/envs/pipeline_stage2_frontend"
ENVPY="$ENV/bin/python"
# gsplat JIT-compiles CUDA kernels at runtime -> needs nvcc on PATH + CUDA_HOME.
# conda-activate normally sets these; a direct-python launch does not.
export CUDA_HOME="$ENV"
export PATH="$ENV/bin:$PATH"
export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"
SESS="sessions/session_20260703_145121"
INIT="$SESS/metric/points_metric.ply"        # SAME init for both arms
ITERS=7000
DS=1.0                                          # full resolution 1920x1440

echo "=================  ARM A: ARKit poses  (full res, ${ITERS} it)  ================="
"$ENVPY" stages/stage5_reconstruction/gsplat_recon.py --session "$SESS" \
  --iters $ITERS --downscale $DS --depth-lambda 0.2 \
  --colmap-dir "$SESS/metric/colmap/sparse/0" \
  --init-ply "$INIT" \
  --output-dir "$SESS/output_ab/arkit_fr7k" 2>&1 | grep -aE "stage5|Error|OOM|CUDA|Traceback" | tail -40

echo "=================  ARM B: seeded-BA poses  (full res, ${ITERS} it)  ============="
"$ENVPY" stages/stage5_reconstruction/gsplat_recon.py --session "$SESS" \
  --iters $ITERS --downscale $DS --depth-lambda 0.2 \
  --colmap-dir "$SESS/metric_ba/colmap/sparse/0" \
  --init-ply "$INIT" \
  --output-dir "$SESS/output_ab/ba_fr7k" 2>&1 | grep -aE "stage5|Error|OOM|CUDA|Traceback" | tail -40

echo "=================  A/B DONE  ================="
