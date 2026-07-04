# Pose-Prior Bundle Adjustment -> Metric Re-lock -> Depth-only A/B -> MILo

Ordered, commit-ready execution plan for `session_20260703_145121`. (Full copy written to `docs/POSE_BA_PLAN.md`.)

**Goal.** ARKit poses drift; drift shows up in Stage 5 as soft splats + a doubled mesh. (1) Refine poses with pose-prior BA (hloc SuperPoint+LightGlue, ARKit poses as init, sequential+loop matching), (2) re-lock metric scale through Stage 3 (BA is up-to-gauge), (3) re-run the SAME depth-only gsplat recon on refined poses and A/B vs `output_depth_only`, (4) build MILo cleanly in its own env and port our EdgeAwareLogL1 depth supervision.

**Platform.** RTX A4000, driver 535 / CUDA 12.2, Miniforge, disk 250 GB — every stage gets its own conda env.

## 0. Ground truth (verified on disk)
```
REPO=/home/paperspace/Documents/VS Code Projects/3D-Gaussian
SESS=$REPO/sessions/session_20260703_145121
```
- capture/rgb/ = 60 PNGs 000001.png..000060.png @1920x1440; capture/depth/ = 60 *.npy 256x192 float32 metric meters (NaN invalid); capture/confidence/ masks (>=128 valid).
- capture/poses.json = ARKit camera_to_world OpenCV metric-but-drifting, 60 frames. intrinsics K fx=fy=1392.8736572 cx=959.7799683 cy=721.5903320.
- metric/colmap/sparse/0 = Stage-3 model: 48 images (subset of 60; Stage2 max_frames=48 dropped 000003,8,...,58), 1 shared PINHOLE cam, world_to_camera, scale 1.0. **Tracks EMPTY (0 observations) -> cannot BA directly.**
- Baseline already exists: output_depth_only/ (iters=15000, downscale=1.0, depth_lambda=0.2, 48 views).
- Only env present: pipeline_stage2_frontend (gsplat 1.5.3, torch 2.5.1+cu124, open3d 0.19, pycolmap 4.1.0). colmap binary NOT on PATH.

**Invariants:** 48 names in images.bin are authoritative — read them from the model and use the SAME list for extraction/pairs/matching/triangulation (never process all 60). hloc keys features by path relative to image_dir, so with image_dir=capture/rgb keys are `000001.png` == pair names == model names; never pass absolute paths. BA is 7-DoF gauge-free; at ~25cm working distance 1% scale = 2.5mm — never read metric off the BA model, never feed it to Stage 5/MILo without the Stage-3 re-lock.

## 1. Env `gs-ba` (BA stage) — see ba_env_commands
pycolmap BA runs CPU/Ceres (seconds on 48 frames); only GPU consumer is torch (SuperPoint/LightGlue). Do NOT `conda install colmap` here (bundles conflicting pycolmap+CUDA). Install torch BEFORE hloc. Pin pycolmap==4.1.0 (matches the version reading our model; 4.x BA API differs from older releases).

## 2. Features + pairs + matching (env gs-ba) -> scripts/pose_ba/01_match.py
Primary: **exhaustive** matching (48 frames = ~1128 pairs, cheap, captures all loop-closure that cancels drift). Robust alt = UNION of hand-written sequential window (hloc has NO pairs_from_sequential, issue #339) + pairs_from_poses.main(model=REF, num_matched=10, rotation_threshold=45) + NetVLAD pairs_from_retrieval.
```python
from pathlib import Path; import pycolmap
from hloc import extract_features, match_features, pairs_from_exhaustive
SESS=Path("/home/paperspace/Documents/VS Code Projects/3D-Gaussian/sessions/session_20260703_145121")
IMAGES=SESS/"capture/rgb"; REF=SESS/"metric/colmap/sparse/0"; WORK=SESS/"pose_ba"; WORK.mkdir(exist_ok=True)
names=sorted(im.name for im in pycolmap.Reconstruction(str(REF)).images.values())   # 48
fc=extract_features.confs["superpoint_max"]                        # 1600px, 4096 kpts
mc=match_features.confs["superpoint+lightglue"]
mc={**mc,"model":{**mc["model"],"filter_threshold":0.05}}          # low-texture face insurance
pairs=WORK/"pairs.txt"; feats=WORK/"feats-superpoint-max.h5"; matches=WORK/"matches-splg.h5"
pairs_from_exhaustive.main(pairs, image_list=names)
extract_features.main(fc, IMAGES, image_list=names, feature_path=feats)
match_features.main(mc, pairs, features=feats, matches=matches)
```
Do NOT downscale images — keypoints and PINHOLE K are both full-res.

## 3. Triangulate at fixed ARKit poses, then joint BA (env gs-ba) -> scripts/pose_ba/02_triangulate_ba.py
triangulation.main triangulates at FIXED poses (does NOT remove drift). Drift removal = bundle_adjustment with per-frame extrinsics free. pycolmap 4.1.0: extrinsics knob is **refine_rig_from_world** (NOT refine_extrinsics, absent in 4.x). Keep intrinsics FIXED (short baseline makes focal degenerate with depth/scale; free focal buries pose error -> soft-splat/doubled-mesh signature). Legacy images.bin auto-makes 1 rig + 48 frames on 1 sensor -> refine_sensor_from_rig=False (gauge-redundant).
```python
from hloc import triangulation; import pycolmap
tri=triangulation.main(sfm_dir=WORK/"triangulated", reference_model=REF, image_dir=IMAGES,
    pairs=pairs, features=feats, matches=matches, skip_geometric_verification=False,
    estimate_two_view_geometries=False, verbose=True)      # real tracks, fixed poses
rec=pycolmap.Reconstruction(str(WORK/"triangulated"))
o=pycolmap.BundleAdjustmentOptions()
o.refine_focal_length=False; o.refine_principal_point=False; o.refine_extra_params=False
o.refine_rig_from_world=True     # per-frame extrinsics: removes ARKit drift
o.refine_sensor_from_rig=False; o.refine_points3D=True; o.print_summary=True
pycolmap.bundle_adjustment(rec,o); rec.write(str(WORK/"refined"))
print("mean reproj:", rec.compute_mean_reprojection_error())
```
Gauge fallback if it errors: BundleAdjustmentConfig -> add_image all, set_constant_cam_intrinsics, fix_gauge(TWO_CAMS_FROM_WORLD), create_default_bundle_adjuster(o,cfg,rec).solve().
**Checkpoint:** mean reproj < ~1.5px and len(rec.reg_image_ids())==48; if frames dropped, lower filter_threshold or add union pairs.

## 4. Re-lock metric scale via Stage 3 (env pipeline_stage2_frontend)
BA is gauge-free (COLMAP issues #595/#3102). A global 7-DoF similarity cannot re-introduce the local per-frame drift BA removed, so re-lock restores metric gauge without undoing correction. Use the VALIDATED Stage 3 (do not hand-roll Umeyama): it fits align.umeyama(BA centers -> ARKit centers) for R,t and adds an INDEPENDENT LiDAR depth_anchor for scale (authoritative post-BA). EXPECT the camera-path residual to increase — that residual IS the removed drift, not a failure.
```bash
mamba activate pipeline_stage2_frontend; cd "$REPO"
cp $SESS/frontend/poses.json $SESS/frontend/poses.json.prearkit
cp $SESS/frontend/points.ply $SESS/frontend/points.ply.prearkit
cp -r $SESS/metric/colmap/sparse/0 $SESS/metric/colmap/sparse/0.arkit_backup
python scripts/pose_ba/03_to_frontend.py            # BA refined -> frontend contract
python stages/stage3_metric/run.py --session sessions/session_20260703_145121 --config config/pipeline.yaml
```
03_to_frontend.py: read pose_ba/refined via pycolmap (world_to_camera), C.save_poses(FE/poses.json, {stem:{R,t}}, pose_type=WORLD_TO_CAMERA, convention=OPENCV); plyio.write_ply(FE/points.ply, xyz, colors). Then inspect metric/scale_report.json (depth anchor inlier fraction >= 0.30; anchors_disagree is a soft flag; only no_anchor_available hard-halts).

## 5. Depth-only Stage 5 A/B (env pipeline_stage2_frontend)
Separate session root so arm B never clobbers baseline. Hold everything constant except poses; match baseline flags EXACTLY (iters=15000 downscale=1.0 depth_lambda=0.2 sh_degree=3 normal=0). Call gsplat_recon.py directly (NOT run.py). Trainer init = metric/points_metric.ply (subsampled to 200k cap); points3D.bin ignored.
```bash
SRC=sessions/session_20260703_145121; DST=${SRC}_refined; mkdir -p "$DST"
ln -sfn "$PWD/$SRC/capture" "$DST/capture"
mkdir -p "$DST/metric/colmap/sparse"; cp -r "$SRC/metric/colmap/sparse/0" "$DST/metric/colmap/sparse/0"
cp "$SRC/metric/points_metric.ply" "$DST/metric/"
python stages/stage5_reconstruction/gsplat_recon.py --session "$DST" --iters 15000 --downscale 1.0 --depth-lambda 0.2 --sh-degree 3
```
Compare $DST/output/mesh.ply vs $SRC/output_depth_only/ (doubled-mesh separation + soft-splat halo should shrink); cross-check reproj err and depth-anchor inlier/scale before vs after BA. Optional clean control: rebuild BOTH arms' init by sensor-depth backprojection under their own poses so arms differ ONLY in poses.

## 6. MILo — clean build (env milo)
Python 3.9, torch 2.3.1+cu118, supply your own CUDA 11.8 nvcc (pytorch-cuda=11.8 is runtime-only -> rasterizers fail). Default branch master. .gitmodules uses git@ SSH -> rewrite to https before --recursive.
- Dataset: cp metric/colmap/sparse/0/*.bin -> milo_dataset/sparse/0/; ln rgb -> milo_dataset/images/ (names match images.bin).
- Port loss into milo/train.py: reuse render_full(..., compute_expected_depth=True); render_pkg['expected_depth'] is (1,H,W) metric world depth with gradient. Since Stage 3 re-locked scale, units match LiDAR — NO per-frame scale/shift solve. Add --depth_lambda(0.2)/--sensor_depth_dir/--sensor_conf_dir; get_sensor_depth() loads capture/depth/{stem}.npy, masks NaN+conf>=128, mask-normalized resize to cam (H,W) [num=interp(d*m); den=interp(m); d=num/den.clamp_min(1e-6)] because MILo downscales (1600px cap); gate iteration>=500; add depth_lambda*edge_aware_logl1(pred,gt_d,rgb_hw,mask) before backward. Copy edge_aware_logl1 verbatim from stages/stage5_reconstruction/gsplat_recon.py. Read expected_depth ONLY from render_full.
- Train: train.py -s milo_dataset -m milo_out --imp_metric indoor --rasterizer radegs --depth_lambda 0.2 --sensor_depth_dir capture/depth --sensor_conf_dir capture/confidence --mesh_config highres --data_device cuda; then mesh_extract_sdf.py -> mesh_learnable_sdf.ply. Headless nvdiffrast: PYOPENGL_PLATFORM=egl or CUDA raster context. simple-knn FLT_MAX -> add #include <float.h>.

## Execution order
1. gs-ba install. 2. 01_match.py. 3. 02_triangulate_ba.py (checkpoint reproj+48). 4. 03_to_frontend.py + stage3/run.py (inspect scale_report). 5. gsplat_recon.py on _refined. 6. milo build -> dataset -> port loss -> train+extract.