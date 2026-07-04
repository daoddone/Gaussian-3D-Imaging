#!/usr/bin/env bash
set -e
ENV="$HOME/miniforge3/envs/milo"
export CUDA_HOME="$ENV" PATH="$ENV/bin:$PATH" LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
REPO="/home/paperspace/Documents/VS Code Projects/3D-Gaussian"; S="$REPO/sessions/session_20260703_145121"
DS="$S/reconstruction_input_scaled"; OUT="$S/output_milo_dense"
cd "$REPO/third_party/MILo/milo"; rm -rf "$OUT"
echo "###### MILo DENSE (sunglasses: --dense_gaussians + veryhighres, -r2, depth-sup) ######"
"$ENV/bin/python" train.py -s "$DS" -m "$OUT" --imp_metric indoor --rasterizer radegs --quiet \
  --dense_gaussians --mesh_config veryhighres -r 2 --data_device cpu \
  --lidar_depth_dir "$S/capture" --lidar_depth_lambda 0.2 --lidar_depth_scale 10
echo "###### MESH EXTRACT (veryhighres) ######"
"$ENV/bin/python" mesh_extract_sdf.py -s "$DS" -m "$OUT" --rasterizer radegs --config veryhighres
echo "###### SG DENSE DONE ######"
grep -c "" /dev/null; find "$OUT" -name "mesh_learnable_sdf.ply" | head
