#!/usr/bin/env bash
set -e
ENV="$HOME/miniforge3/envs/milo"
export CUDA_HOME="$ENV" PATH="$ENV/bin:$PATH" LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "/home/paperspace/Documents/VS Code Projects/3D-Gaussian/third_party/MILo/milo"
echo "### stock MILo train (face, radegs, indoor) ###"
"$ENV/bin/python" train.py -s "/home/paperspace/Documents/VS Code Projects/3D-Gaussian/sessions/session_20260703_145121/reconstruction_input" -m "/home/paperspace/Documents/VS Code Projects/3D-Gaussian/sessions/session_20260703_145121/output_milo_stock" --imp_metric indoor --rasterizer radegs --quiet
echo "### STOCK MILO EXIT $? ###"
