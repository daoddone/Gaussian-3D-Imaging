#!/usr/bin/env bash
set -e
CONDA="$HOME/miniforge3/bin/conda"
echo "### create milo env (py3.9) ###"
$CONDA create -n milo python=3.9 -y
echo "### torch 2.3.1 + pytorch-cuda 11.8 ###"
$CONDA install -n milo -c pytorch -c nvidia pytorch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 pytorch-cuda=11.8 mkl=2023.1.0 -y
echo "### cuda-toolkit 11.8 (nvcc + dev headers) ###"
$CONDA install -n milo -c "nvidia/label/cuda-11.8.0" cuda-toolkit -y
echo "### build deps (cmake ninja gmp cgal eigen) ###"
$CONDA install -n milo -c conda-forge cmake ninja gmp "cgal=5.6" eigen -y
echo "### verify ###"
"$HOME/miniforge3/envs/milo/bin/python" -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'avail', torch.cuda.is_available())"
"$HOME/miniforge3/envs/milo/bin/nvcc" --version | tail -1
echo "### MILO ENV DONE ###"
