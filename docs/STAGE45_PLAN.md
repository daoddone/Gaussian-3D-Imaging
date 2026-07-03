## Consolidated execution plan — fidelity first, then Stage 4 → 5 → 6

**Absolute paths**
- Repo: `/home/paperspace/Documents/VS Code Projects/3D-Gaussian`
- Session: `/home/paperspace/Documents/VS Code Projects/3D-Gaussian/sessions/session_20260703_145121`
- Metric COLMAP: `.../metric/colmap/sparse/0/` (cameras/images/points3D.bin present)

### Phase 0 — Fidelity fixes (existing `pipeline_stage2_frontend` env, ZERO new disk)
Do these before building any heavy env so the DA3 work lands while disk is still free.

**0a. Fix the Stage 4 crash (edit only, no run yet)** — `stages/stage4_normals/run.py:92-93`
```python
# was: out = predictor(img, output_type="np")
#      normals_gl = np.asarray(getattr(out, "prediction", out)).squeeze()
pil = predictor(img, data_type="indoor")            # current-main API: no output_type, returns PIL
normals_gl = np.asarray(pil).astype(np.float32) / 127.5 - 1.0   # [H,W,3] in ~[-1,1]
```
Keep the existing `_to_opencv_normals` diag(1,-1,-1) flip.

**0b. Re-run DA3 frontend pose-conditioned + bf16** (the ~15% scale fix + 60-frame fit)
```bash
conda activate pipeline_stage2_frontend
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```
In `stages/stage2_frontend/run.py`: after load add `model = model.to(device).to(torch.bfloat16); model.device = device`, and set `use_ray_pose=False`. Call `inference()` with `extrinsics` = (N,4,4) **world_to_camera OpenCV** (invert ARKit c2w), `intrinsics` = (N,3,3) at **native 1920x1440** (fx=fy=1392.8737, cx=959.78, cy=721.59, pass unchanged), `align_to_input_ext_scale=True`, `process_res=504`. bf16 reclaims ~3 GB → 60 frames fit; if still ~98% VRAM drop to `process_res=448`.

**0c. Gate:** re-run existing LiDAR ICP on the new depth. Residual must stay ~3.44 mm (camera-path scale ~0.355), not 4.69 mm. This locks the metric scale that Stage 5 depth loss requires.

**0d. Rebuild Stage 3 metric COLMAP** from the conditioned output so `points3D.bin` + poses carry the camera-path scale verbatim. All downstream depth PNGs must match these units.

### Phase 1 — Stage 4 normals (venv layered on stage2, ~0.5 GB, NOT a new conda env)
```bash
conda activate pipeline_stage2_frontend
python -m venv --system-site-packages ~/envs/stage4_normals   # reuses stage2 torch, saves ~6GB
source ~/envs/stage4_normals/bin/activate
pip install "diffusers==0.28.0" "transformers==4.36.1" "accelerate==0.30.1" \
            "huggingface_hub==0.23.0" "safetensors>=0.4"
```
Smoke-test one frame: `torch.hub.load("Stable-X/StableNormal","StableNormal_turbo",trust_repo=True)`, `pred(img, data_type="indoor")`. Then run the **dot-product sign gate** against `stages/stage4_normals/normals_from_depth.py` on a near-planar frame — median dot must be strongly positive; if negative, drop the diag(1,-1,-1) flip. Weights (~2.5-4 GB) go to HF cache; watch disk. Isolation check: `conda run -n pipeline_stage2_frontend python -c "import huggingface_hub as h;print(h.__version__)"` must still print 1.22 (DA3 untouched).

### Phase 2 — Reclaim disk, then build Stage 5 host: DN-Splatter / AGS-Mesh
```bash
conda clean -a -y && pip cache purge          # reclaim several GB BEFORE the heavy env
df -h /                                         # confirm headroom (need ~10GB for the env)
```
**Disk-driven env choice:** with only ~12 GB free, install dn-splatter INTO `pipeline_stage2_frontend` (reuses torch 2.5.1+cu124, saves the ~6 GB torch reinstall) rather than a fresh conda env. Trade-off: nerfstudio may downgrade torch — pin and verify.
```bash
conda activate pipeline_stage2_frontend
conda install -y -c "nvidia/label/cuda-12.4.0" cuda-nvcc cuda-cudart-dev   # nvcc absent everywhere
export CUDA_HOME=$CONDA_PREFIX
export TORCH_CUDA_ARCH_LIST=8.6               # RTX A4000 sm_86
pip install setuptools==69.5.1
pip install dn-splatter                        # pulls nerfstudio + gsplat; gsplat ext JITs first run
python -c "import torch;print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
# MUST print 2.5.1 12.4 True. If nerfstudio downgraded to 2.1.2+cu118:
#   pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
ns-install-cli
```
If disk after cleanup allows ≥15 GB, prefer a fresh `dnsplat` conda env instead (cleaner isolation from DA3), same steps + `pip install torch==2.5.1 ... cu124` first. Skip base-gsplat smoke test unless the ns-train build fails — DN-Splatter itself validates COLMAP + scale, and gsplat `--depth_loss` is only sparse SfM depth (does not exercise LiDAR).

**Data prep (coolermap / normal-nerfstudio layout under `.../metric`):**
- LiDAR depth: per-frame `"depth_file_path"` in `transforms.json` → sensor depth; float32 meters, NaN→0, NEAREST-upsample 256x192 → 1920x1440; `"depth_unit_scale_factor": 1.0`.
- Normals: write StableNormal npy as `[0,1]` PNGs in `metric/normals_from_pretrain/*.png` (dataparser globs .png only).
- Confidence: `metric/depth_normals_mask/*.jpg`, **255 = KEEP valid LiDAR, 0 = DISCARD** (model keeps where conf>0). Do NOT feed DN's `depth_normal_consistency.py` output raw — it writes the opposite polarity (255=bad); invert it.

**Train (AGS-Mesh so confidence masks are consumed):**
```bash
ns-train ags-mesh \
  --pipeline.model.use-depth-loss True \
  --pipeline.model.depth-loss-type EdgeAwareLogL1 --pipeline.model.depth-lambda 0.2 \
  --pipeline.model.use-normal-loss True --pipeline.model.use-normal-tv-loss True \
  --pipeline.model.normal-supervision mono \
  --data "sessions/session_20260703_145121/metric" \
  normal-nerfstudio --normal-format dsine --load-normals True --load-depths True \
  --load-depth-confidence-masks True --depth-unit-scale-factor 1.0
```
Run a ~2000-step smoke pass first; render pred-vs-gt depth/normals to confirm dsine polarity and confidence polarity on-box (gating only fully activates after 7000/15000 steps).

### Phase 3 — Stage 6 deliverables
```bash
gs-mesh o3dtsdf --load-config outputs/metric/ags-mesh/*/config.yml --output-dir metric/mesh/
ns-export gaussian-splat --load-config outputs/metric/ags-mesh/*/config.yml --output-dir metric/splat/
```
Start TSDF voxel coarse, refine toward sub-mm only if VRAM/time allow. MILo stays shelved — build it only if the AGS-Mesh TSDF surface is inadequate.