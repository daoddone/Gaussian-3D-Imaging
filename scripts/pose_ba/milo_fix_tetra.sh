#!/usr/bin/env bash
set -e
REPO="/home/paperspace/Documents/VS Code Projects/3D-Gaussian"
MILO="$REPO/third_party/MILo"; ENV="$HOME/miniforge3/envs/milo"
export CUDA_HOME="$ENV" PATH="$ENV/bin:$PATH" LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"
export CPATH="$ENV/include:${CPATH:-}" TORCH_CUDA_ARCH_LIST="8.6"
echo "### downgrade cmake to <4 (pybind11 2.9.2 rejects cmake 4.x cmake_minimum_required) ###"
"$HOME/miniforge3/bin/conda" install -n milo -c conda-forge "cmake<4" -y
"$ENV/bin/cmake" --version | head -1
echo "### tetra_triangulation (clean rebuild) ###"
cd "$MILO/submodules/tetra_triangulation"
rm -rf CMakeCache.txt CMakeFiles _deps build bin
TORCH_CMAKE=$("$ENV/bin/python" -c "import torch,os;print(os.path.join(os.path.dirname(torch.__file__),'share','cmake'))")
cmake . -DCMAKE_PREFIX_PATH="$TORCH_CMAKE;$ENV" -DCMAKE_BUILD_TYPE=Release
make -j4
"$ENV/bin/pip" install -e .
cd "$MILO"
echo "### nvdiffrast ###"; "$ENV/bin/pip" install -e ./submodules/nvdiffrast
echo "### VERIFY ###"
"$ENV/bin/python" - <<'PY'
for m in ["diff_gaussian_rasterization","diff_gaussian_rasterization_ms","diff_gaussian_rasterization_gof","simple_knn","fused_ssim","nvdiffrast.torch"]:
    try: __import__(m); print("OK",m)
    except Exception as e: print("FAIL",m,type(e).__name__,str(e)[:100])
try:
    import tetranerf.utils.extension; print("OK tetranerf")
except Exception as e: print("FAIL tetranerf",type(e).__name__,str(e)[:100])
PY
echo "### MILO TETRA FIX DONE ###"
