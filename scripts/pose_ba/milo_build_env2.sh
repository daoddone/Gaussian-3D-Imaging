#!/usr/bin/env bash
set -e
CONDA="$HOME/miniforge3/bin/conda"
echo "### torch 2.3.1 + pytorch-cuda 11.8 (no mkl pin: mkl2023.1 needs llvm-openmp>=16, torch2.3.1 needs <16) ###"
$CONDA install -n milo -c pytorch -c nvidia pytorch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 pytorch-cuda=11.8 -y
echo "### cuda-toolkit 11.8 (nvcc + dev headers) ###"
$CONDA install -n milo -c "nvidia/label/cuda-11.8.0" cuda-toolkit -y
echo "### build deps (cmake ninja gmp cgal eigen) ###"
$CONDA install -n milo -c conda-forge cmake ninja gmp "cgal=5.6" eigen -y
echo "### verify ###"
"$HOME/miniforge3/envs/milo/bin/python" -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'avail', torch.cuda.is_available())"
"$HOME/miniforge3/envs/milo/bin/nvcc" --version | tail -1
echo "### MILO ENV DONE ###"
