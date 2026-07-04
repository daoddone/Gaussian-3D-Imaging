#!/usr/bin/env bash
set -e
cd "/home/paperspace/Documents/VS Code Projects/3D-Gaussian"
ENV="$HOME/miniforge3/envs/pipeline_stage2_frontend"
export CUDA_HOME="$ENV" PATH="$ENV/bin:$PATH" LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"
# DA3-nested fp32 on 48 frames peaks ~14.8GB (right at the 16GB edge); reclaim the
# ~1.1GB reserved-but-unallocated + reduce fragmentation so it fits (bf16 stays OFF:
# it would corrupt DA3 depth precision and contaminate the scale test).
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
S="sessions/session_20260703_203728"
echo "###### HAND PIPELINE: orchestrate Stage 2->3->5 (--no-conda, all in stage2 env) ######"
"$ENV/bin/python" orchestrate.py --session "$S" --config config/pipeline.yaml --no-conda 2>&1
echo "###### HAND PIPELINE EXIT $? ######"
