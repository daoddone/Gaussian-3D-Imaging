#!/usr/bin/env bash
set -e
REPO="/home/paperspace/Documents/VS Code Projects/3D-Gaussian"
MILO="$REPO/third_party/MILo"
ENV="$HOME/miniforge3/envs/milo"
export CUDA_HOME="$ENV" PATH="$ENV/bin:$PATH" LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"
export CPATH="$ENV/include:${CPATH:-}" TORCH_CUDA_ARCH_LIST="8.6"
PY="$ENV/bin/python"; PIP="$ENV/bin/pip"
cd "$MILO"
echo "### nvcc / torch ###"; "$ENV/bin/nvcc" --version | tail -1; "$PY" -c "import torch;print('torch',torch.__version__,'cuda',torch.version.cuda)"
echo "### requirements ###"; "$PIP" install -q -r requirements.txt
for m in diff-gaussian-rasterization_ms diff-gaussian-rasterization diff-gaussian-rasterization_gof simple-knn fused-ssim; do
  echo "### build $m ###"; "$PIP" install "./submodules/$m"
done
echo "### tetra_triangulation (cmake+make) ###"
cd submodules/tetra_triangulation
TORCH_CMAKE=$("$PY" -c "import torch,os;print(os.path.join(os.path.dirname(torch.__file__),'share','cmake'))")
cmake . -DCMAKE_PREFIX_PATH="$TORCH_CMAKE;$ENV" -DCMAKE_BUILD_TYPE=Release
make -j4
"$PIP" install -e .
cd "$MILO"
echo "### nvdiffrast ###"; "$PIP" install -e ./submodules/nvdiffrast
echo "### VERIFY ###"
"$PY" - <<'PY'
for m in ["diff_gaussian_rasterization","diff_gaussian_rasterization_ms","diff_gaussian_rasterization_gof","simple_knn","fused_ssim","nvdiffrast.torch"]:
    try: __import__(m); print("OK",m)
    except Exception as e: print("FAIL",m,type(e).__name__,str(e)[:90])
try:
    import tetranerf.utils.extension; print("OK tetranerf")
except Exception as e: print("FAIL tetranerf",type(e).__name__,str(e)[:90])
PY
echo "### MILO SUBMODULES DONE ###"
