# Component I/O Reference

Engineering reference for every third-party component in the 3D-Gaussian pipeline, organized by stage. For each component: install, inputs, outputs, key API, coordinate convention, the **exact conversions** the stage wrapper must perform to satisfy our contracts, and gotchas.

## Our pipeline contracts (the target every mapping below must hit)

- **Coordinate convention:** OpenCV EVERYWHERE (camera looks down +Z, X right, Y down). Convert **exactly once** per boundary.
- **Units/dtype/format:** meters, float32. Per-frame numeric arrays as `.npy`; color as lossless `.png`. Frame numbering zero-padded 6 digits, **1-indexed** (`000001`).
- **Pose files** declare `pose_type` (`'camera_to_world'` or `'world_to_camera'`).
- **Stage 1 `capture/`:** `rgb/*.png`, `depth/*.npy [H,W]` meters (NaN=invalid), `confidence/*.png` (255 valid / 0 invalid), `intrinsics.json` (K 3x3 + resolutions), `poses.json` (camera_to_world, OpenCV), `timestamps.json`.
- **Stage 2 `frontend/`:** `poses.json` (world_to_camera), `intrinsics.json` (per-frame K), `depth/*.npy`, `conf/*.npy [0..1]`, `points.ply` (plain), `colmap/sparse/0/{cameras,images,points3D}.bin`.
- **Stage 3 `metric/`:** `points_metric.ply`, `colmap/sparse/0/*.bin` (world_to_camera), `scale_report.json`.
- **Stage 4 `normals/`:** `000001.npy [H,W,3]` unit vectors in [-1,1], CAMERA frame, OpenCV; optional `normals_weight/*.npy [H,W]` in [0,1].
- **Stage 5 (MILo):** reads COLMAP sparse-model dataset + rgb + init point cloud + sensor depth/confidence (+ normals if enabled).
- **Stage 6 `output/`:** `point_cloud.ply` (Gaussian splat), `mesh.ply/.obj`, `renders/`, `provenance.json`.

### Convention cheat-sheet (per boundary)

| Component | Native camera frame | Native pose direction | Native quat order | Conversion to our contract |
|---|---|---|---|---|
| Apple ARKit (S1) | +X right, +Y up, **−Z fwd** (OpenGL) | camera→world | (matrix) | `T_c2w_opencv = T_c2w_arkit @ diag(1,−1,−1,1)`; transpose out of simd column-major |
| Depth Anything 3 (S2) | OpenCV | world→camera | COLMAP WXYZ (via pycolmap) | **passthrough** (identity) |
| Free Geometry (S2) | OpenCV | world→camera | COLMAP WXYZ | **passthrough** |
| MapAnything (S3) | OpenCV/RDF | **camera→world** | **XYZW (scalar-last)** | invert to w2c; reorder quat to WXYZ for COLMAP |
| Open3D (S3/S6) | OpenCV | matrices only | none | identity camera frame; watch depth_scale/trunc |
| StableNormal (S4) | +X right, **+Y up, +Z toward cam** (OpenGL) | n/a (per-pixel) | none | negate **Y and Z** channels |
| DSINE (S4) | +X right, +Y down, **+Z toward cam** | n/a | none | negate **Z** channel only |
| MILo (S5) | OpenCV (COLMAP) | world→camera | WXYZ | **passthrough** |
| DN-Splatter (S5 loss) | OpenCV internal | world→camera (COLMAP) | WXYZ | passthrough; `normal_format="dsine"`; invert confidence |
| 3dgs-mcmc (S5) | OpenCV (COLMAP) | world→camera | WXYZ | **passthrough** |
| gsplat (ref only) | OpenCV | world→camera (viewmats) | WXYZ | passthrough; invert if source is c2w |
| COLMAP bin (I/O) | OpenCV | world→camera | WXYZ (scalar-first) | identity; invert only if source is c2w |

---

# Stage 1 — Capture (`capture/`)

## Apple AVFoundation LiDAR + ARKit (on-device front-end)

**Confidence: medium** (Apple docs are JS-rendered; facts pulled from backing JSON API. Several invalid-sentinel and confidence facts are inferred — verify on hardware).

### Install
Apple SDK, no package install. Requires a LiDAR device (iPhone 12 Pro+, iPad Pro 11" 3rd gen+, iPad Pro 12.9" 5th gen+). `import AVFoundation` (depth), `import ARKit` (pose), `import CoreVideo` (CVPixelBuffer), `import simd`. Needs `NSCameraUsageDescription`; physical device only (no Simulator).

### Inputs
- Device: `AVCaptureDevice.default(.builtInLiDARDepthCamera, for: .video, position: .back)`.
- `AVCaptureDepthDataOutput` (set `isFilteringEnabled = false` for raw sensor depth) + `AVCaptureVideoDataOutput`, time-synced via `AVCaptureDataOutputSynchronizer`.
- Pose comes **only** from the ARKit path (`ARWorldTrackingConfiguration`, default `worldAlignment = .gravity`) via `ARFrame.camera`. AVFoundation depth capture alone yields **no** world pose. Typical fused capture uses `ARFrame` (`capturedImage` + `sceneDepth` + `camera`) sharing one timestamp.

### Outputs
- `AVDepthData.depthDataMap` (CVPixelBuffer). Force meters with `depthData.converting(toDepthDataType: kCVPixelFormatType_DepthFloat32)`. Disparity types relate by `depth = 1/disparity`.
- `cameraCalibrationData.intrinsicMatrix` (`matrix_float3x3`, **column-major**), `intrinsicMatrixReferenceDimensions` (the px size K is relative to).
- `ARCamera.transform` (`simd_float4x4`, camera→world, column-major), `.intrinsics`, `.imageResolution`, `.trackingState`.
- Per-pixel confidence: **not** in AVDepthData (only frame-level `depthDataQuality/Accuracy`). Use `ARFrame.sceneDepth.confidenceMap` (`ARConfidenceLevel` .low=0/.medium=1/.high=2).

### Key API
```swift
let device = AVCaptureDevice.default(.builtInLiDARDepthCamera, for: .video, position: .back)
depthOut.isFilteringEnabled = false
let depth = d.depthData.converting(toDepthDataType: kCVPixelFormatType_DepthFloat32)
let map = depth.depthDataMap            // Float32 meters
let K   = depth.cameraCalibrationData?.intrinsicMatrix
frame.camera.transform                  // simd_float4x4 camera->world
frame.camera.trackingState              // gate on == .normal
```

### Coordinate convention
ARKit **world** frame (gravity): right-handed, **+Y up** ((0,−1,0) is down), −Z is device-facing at session start, origin at first-run device pose. ARKit **camera-local**: +X right, +Y up, **−Z forward** (OpenGL-style). `ARCamera.transform` is **camera→world**. simd matrices are **column-major** (transpose for row-major numpy).

### Mapping to our contract (EXACT conversions)
- **Depth → `capture/depth/000001.npy`:** convert to DepthFloat32, read plane 0 as `[H,W]` Float32 (watch `bytesPerRow` padding). Already meters. Map LiDAR no-return `0 → NaN` (**inferred**, verify).
- **RGB → `capture/rgb/000001.png`:** convert YCbCr CVPixelBuffer → RGB, save lossless PNG. LiDAR depth (~256×192) is far lower res than color — record the saved resolution.
- **Intrinsics → `capture/intrinsics.json`:** transpose out of simd column-major → `K=[[fx,0,ox],[0,fy,oy],[0,0,1]]`. Scale to the **saved** image size per stream: `sx=W_save/W_ref`, `sy=H_save/H_ref`; `fx*=sx; ox*=sx; fy*=sy; oy*=sy`. Depth-K and color-K differ — scale each to the array it indexes. Principal point is already top-left-origin (no y-flip).
- **Pose → `capture/poses.json` (`camera_to_world`, OpenCV):** transpose `ARCamera.transform` to row-major, then convert camera basis OpenGL→OpenCV by **negating the 2nd and 3rd rotation columns** (rotate 180° about camera X): `T_c2w_opencv = T_c2w_arkit @ diag(1,−1,−1,1)`. Translation unchanged. World frame stays ARKit gravity-aligned. For Stage 2/3, invert **after** the flip: `T_w2c = inv(T_c2w_opencv)`.
- **Confidence → `capture/confidence/000001.png`:** from `sceneDepth.confidenceMap`; map high(2)→255, low(0)→0 per policy.
- **Timestamps → `capture/timestamps.json`:** synchronizer/ARFrame timestamp (seconds), 6-digit frame index.

### Gotchas
- simd is **column-major** — forgetting to transpose K/transform is the classic bug (`fx` in `columns.0`, `ox` in `columns.2`).
- Intrinsics are **resolution-relative** to `intrinsicMatrixReferenceDimensions` / `imageResolution`, which differ from both saved PNG size and depth-map size — scale per stream or reprojection is wrong.
- `isFilteringEnabled=false` semantics inferred; expect holes (0/NaN).
- `0→NaN`, per-pixel confidence (ARDepthData), and camera −Z-forward are **inferred / from docs not directly read** — verify on hardware.
- `builtInLiDARDepthCamera` is nil on non-LiDAR devices; gate `trackingState == .normal`.

---

# Stage 2 — Geometry frontend (`frontend/`)

## Depth Anything 3 — DA3NESTED-GIANT-LARGE

**Confidence: high.** Primary Stage-2 engine. Native output is **already OpenCV w2c + metric** for the nested model → most maps are near-identity (only layout, no rotation).

### Install
```
pip install xformers 'torch>=2' torchvision
git clone https://github.com/ByteDance-Seed/Depth-Anything-3 && cd Depth-Anything-3
pip install -e .          # src-layout: import from depth_anything_3.api
pip install --no-build-isolation git+https://github.com/nerfstudio-project/gsplat.git@0b4dddf04cb687367602c01196913cde6a743d70   # PINNED, for gs/infer_gs paths
# colmap export needs pycolmap; weights auto-download on from_pretrained
```

### Inputs
`DepthAnything3.from_pretrained('depth-anything/DA3NESTED-GIANT-LARGE').to('cuda')`, then `model.inference(image=[...], ...)`. Key args: `image` (list of HxWx3 uint8 RGB / PIL / paths), optional `extrinsics (N,4,4) w2c OpenCV`, optional `intrinsics (N,3,3)` in **original** pixel units (auto-scaled internally), `align_to_input_ext_scale=True`, `process_res=504` (upper-bound resize), `export_dir`, `export_format='mini_npz'`, `conf_thresh_percentile=40.0`.

### Outputs — `Prediction` dataclass (numpy, float32)
- `depth (N,H,W)` — **metric meters** for nested (`is_metric` truthy).
- `conf (N,H,W)|None` — **range undocumented, NOT guaranteed [0,1]**.
- `extrinsics` — world→camera, OpenCV. **Shape ambiguous:** docs say `(N,3,4)`, `specs.py` comment says `(N,4,4)` — check `.shape` at runtime.
- `intrinsics (N,3,3)` — pixel-unit K at the **processed** HxW (=`processed_images` size, not original).
- `processed_images (N,H,W,3) uint8`, `sky`, `scale_factor`, `is_metric`, `gaussians` (WXYZ world quats if `infer_gs`), `aux`.
- **Export formats** (combine with `-`): `glb, npz, mini_npz, gs_ply, gs_video, colmap, feat_vis, depth_vis`. `colmap` writes real `.bin` via pycolmap: PINHOLE, `cam_from_world` (w2c), qvec `(qw,qx,qy,qz)`. Plain `'ply'` token is **not confirmed** in dispatch.

### Coordinate convention
Extrinsics = **world→camera, OpenCV** — exactly our contract, no axis flip. `c2w = affine_inverse(w2c)`. Depth along +Z, metric meters (nested). Gaussian quats WXYZ. Metric formula `metric_depth = focal_px * net_output / 300.0` applies to **DA3METRIC-LARGE only**, NOT nested (already meters).

### Mapping to our contract (EXACT conversions)
- **`frontend/poses.json` (world_to_camera):** `prediction.extrinsics[i]`, slice `[:3,:4]` (handle 3x4-vs-4x4), write R|t with `pose_type='world_to_camera'`. **No inversion, no flip.**
- **`frontend/intrinsics.json`:** `prediction.intrinsics[i]` direct (pixel-unit K at depth resolution). Record that resolution; rescale K if you keep original RGB res.
- **`frontend/depth/000001.npy` [meters]:** `prediction.depth[i].astype(float32)`. **Do NOT apply the /300 formula** (DA3METRIC only). DA3 is 0-indexed by input order → wrapper adds 1 for our 1-indexed 6-digit name.
- **`frontend/conf/000001.npy` [0..1]:** `prediction.conf[i]` — **REQUIRED non-trivial normalization** (min-max per-frame or percentile map using `conf_thresh_percentile=40` as invalid cutoff); DA3 conf is not [0,1].
- **`frontend/colmap/sparse/0/*.bin`:** `export_format='colmap'` — **direct match** (PINHOLE, w2c, WXYZ qvec). No conversion.
- **`frontend/points.ply` (plain):** via `export_format='glb'` vertices or unproject depth+K+w2c yourself (no guaranteed plain-ply token).
- **NaN-invalid:** DA3 never emits NaN → wrapper sets `depth[conf<thresh]=NaN`.
- **Feeding our Stage-1 c2w poses as conditioning:** invert c2w→w2c **once** before `inference(extrinsics=...)`; pass intrinsics in original pixel units.

### Gotchas
gsplat must be the pinned commit. `extrinsics` shape inconsistent across docs. `conf` range undocumented. Depth/K/conf are at the **internal 504 processed resolution**, not original — rescale K for original-res alignment. Only NESTED/METRIC are metric.

---

## Free Geometry (test-time LoRA refinement of DA3-GIANT)

**Confidence: high.** A fork of the DA3 package (pip `depth-anything-3`, CLI `da3`) adding per-scene PEFT-LoRA refinement. Inference API and all I/O are identical to DA3.

### Install
Same as DA3, plus `pip install peft` (**imported but MISSING from requirements/pyproject** — install manually or LoRA import fails). Base weights `depth-anything/DA3-GIANT-1.1`; adapters `PeterDAI/Free-Geometry`.

### Inputs
Two interfaces: **(A) refinement** (`scripts/train_da3.py`, images-only, no poses/GT) is **bound to a hardcoded `DATASET_REGISTRY` {eth3d,7scenes,scannetpp,hiroom,dtu}** — no arbitrary-folder path; you must add a small Dataset class returning `{'image_files':[...]}`. **(B) inference** via `LoRADepthAnything3(base_model, lora_path, lora_rank, lora_alpha).inference(...)` (merges LoRA, then identical to DA3 `inference`).

### Outputs
Identical `Prediction` to DA3, **except depth is up-to-scale** for DA3-GIANT-1.1 (not metric). COLMAP exporter rescales intrinsics to **original** image size and stores `cam_from_world`.

### Coordinate convention
Identical to DA3: world→camera, OpenCV, COLMAP WXYZ. Passthrough, no axis flip.

### Mapping to our contract (EXACT conversions)
- Poses/COLMAP/intrinsics: same passthrough as DA3. Intrinsics at ~504 processed res → rescale `K[0,:]*=W_target/W_proc; K[1,:]*=H_target/H_proc`.
- **`depth/*.npy` meters:** NOT metric on its own. Either run pose-conditioned inference with input extrinsics + `align_to_input_ext_scale=True`, OR apply a Stage-3 metric scale afterward. `depth[conf<thresh]=NaN`, float32.
- **`conf/*.npy` [0..1]:** normalize (raw, undocumented range).
- **points.ply / points_metric.ply:** `utils/export/glb._depths_to_world_points_with_colors(depth, intrinsics, extrinsics_w2c, images, conf, thresh)` or `export_format='glb'`.
- **Frame numbering:** DA3 uses `os.path.basename` as COLMAP `image.name` → name inputs `000001.png` etc.
- **Normals (Stage 4):** not produced — out of scope.
- **Integration note:** refinement can't ingest `capture/` directly — add a Dataset class (RGB paths only), train, then `LoRADepthAnything3` for Stage-2 artifacts.

### Gotchas
`peft` missing from deps. Refinement registry-bound. Depth up-to-scale. conf undocumented. extrinsics shape both (N,3,4)/(N,4,4). Heavy VRAM (1.15B). LoRA rank/alpha auto-read from `adapter_config.json` at inference (overrides CLI). Normals not produced.

---

# Stage 3 — Metric (`metric/`)

## MapAnything (facebookresearch/map-anything) — independent metric cross-check

**Confidence: high.** Feed-forward **metric** multi-view reconstruction. Outputs are already meters; used to cross-check DA3 scale.

### Install
```
conda create -n mapanything python=3.12 -y && conda activate mapanything
pip install -e ".[all]"    # pycolmap/rerun; plain -e . is core-only
# weights: facebook/map-anything (CC-BY-NC) or facebook/map-anything-apache (Apache-2.0)
```

### Inputs
`MapAnything.from_pretrained("facebook/map-anything").to(device)`; `views = load_images(folder_or_list)` (**resizes**, default long-side ~518, dims multiple of 14). `model.infer(views, memory_efficient_inference=True, use_amp=True, amp_dtype="bf16", apply_mask=True, mask_edges=True, ...)`. Optional per-view geometry (`intrinsics` XOR `ray_directions`; `depth_z` needs one of them; `camera_poses` cam2world; `is_metric_scale`). If any view has poses, view 0 must too.

### Outputs — list of per-view dicts (metric meters, float32 after `.cpu().numpy()`)
`pts3d` (world), `pts3d_cam`, `depth_z (B,H,W,1)`, `depth_along_ray`, `ray_directions`, `intrinsics (B,3,3)` at **processed** res, `camera_poses (B,4,4)` **cam2world OpenCV/RDF**, `cam_trans`, `cam_quats (B,4)` **XYZW (scalar-last)**, `metric_scaling_factor (B,)` (internal, not cross-model comparable), `mask (B,H,W,1) bool`, `conf (B,H,W)` (~[0,1]), `img_no_norm`.

### Coordinate convention
Camera **OpenCV/RDF** (+X right, +Y down, +Z forward) — matches ours, **no axis flip**. Poses are **camera→world**. Quat order **XYZW** (identity `[0,0,0,1]`) — **NOT** COLMAP WXYZ. World frame is model-internal (anchored to view 0), metric but not gravity-aligned. K/depth at processed resolution.

### Mapping to our contract (EXACT conversions)
- **Stage 1 `capture/poses.json` (camera_to_world):** `camera_poses` direct, `pose_type='camera_to_world'`. If building from quat, keep **XYZW** in `quaternion_to_rotation_matrix`.
- **Stage 2/3 `poses.json` (world_to_camera):** invert via `closed_form_pose_inverse(camera_poses[None])[0]`; `pose_type='world_to_camera'`.
- **COLMAP `.bin`:** call `export_predictions_to_colmap(preds, ...)` — does cam2world→world2cam internally. If you write COLMAP quats yourself, **reorder XYZW→WXYZ** (pycolmap Rigid3d handles it).
- **Depth → `.npy` meters:** `depth_z[0].squeeze(-1)` float32; `depth[~mask]=np.nan`.
- **Confidence:** `mask.astype(uint8)*255` for PNG; `conf` (clip [0,1]) for Stage-2 conf.npy.
- **Intrinsics:** valid only at processed res — either `load_images(resize_mode="fixed_size", size=(W,H))` or rescale K (`fx,fy,cx,cy *= W_out/W_proc, H_out/H_proc`) and resample depth. Record the resolution used.
- **`points_metric.ply`:** stack `pts3d[0]` (or `depthmap_to_world_frame(depth_z,K,c2w)`) over `mask`; colors from Stage-1 rgb.
- **`scale_report.json` cross-check:** store per-frame `median(depth_z[mask])` meters and ratio to DA3 median metric depth (target ~1.0). Use median depth (or `||Δcam_trans||` baseline), **not** `metric_scaling_factor`.

### Gotchas
Python 3.12. Images internally **resized** → K/depth/points at processed res. Quat **XYZW** ≠ COLMAP WXYZ. `metric_scaling_factor` is internal, not comparable. Single-image metric less reliable. `facebook/map-anything` is **CC-BY-NC** (non-commercial) — use `-apache` for commercial.

---

## Open3D — point-cloud I/O, ICP, TSDF fusion (Stage 3 metric, optional Stage 6 mesh)

**Confidence: high.** Camera frame is **OpenCV** (no flip). Matrices only, no quaternions.

### Install
`pip install open3d` (or `open3d-cpu`, x86_64-Linux only). Check `o3d.__version__` (~0.19.x).

### Inputs / Outputs
- **PLY:** `read_point_cloud(...)`, `write_point_cloud(..., write_ascii=False)` (default **binary** PLY; only `.points` set → "plain" x,y,z). No units/CRS metadata.
- **numpy interop:** `Vector3dVector` is **float64 only** (float32 up-cast + copy); colors must be **[0,1]**.
- **ICP:** `registration_icp(source, target, max_correspondence_distance, init=eye(4), estimation_method=TransformationEstimationPointToPoint(with_scaling=False), criteria=ICPConvergenceCriteria(max_iteration=30))`. Returns `.transformation` (4x4 float64, source→target), `.fitness`, `.inlier_rmse`.
- **TSDF:** `ScalableTSDFVolume(voxel_length, sdf_trunc, color_type)`, `RGBDImage.create_from_color_and_depth(color, depth, depth_scale=1000.0, depth_trunc=3.0, convert_rgb_to_intensity=True)`, `volume.integrate(rgbd, PinholeCameraIntrinsic(w,h,fx,fy,cx,cy), extrinsic)`. **`extrinsic` is world→camera.** `extract_triangle_mesh()` / `extract_point_cloud()`.

### Coordinate convention
Camera +X right, **+Y down**, +Z into scene = **OpenCV**, no flip. `integrate` extrinsic = **world→camera**. Matrices 4x4 float64 row-major. ICP `init`/`.transformation` map source→target. `metric_depth = raw/depth_scale`, then `> depth_trunc` dropped; 0 = invalid.

### Mapping to our contract (EXACT conversions)
- **PLY:** default binary; set only `.points` (+optional `.colors`) for plain. Coords already meters. Open3D holds float64 → `np.asarray(...).astype(np.float32)` on export for our float32 contract.
- **Color:** ours is PNG uint8 [0,255]; Open3D wants [0,1] → `rgb/255.0` in, `*255`+round out.
- **ICP scale:** default PointToPoint is **rigid only** — will NOT recover metric scale. Use `TransformationEstimationPointToPoint(with_scaling=True)` (Umeyama), or bake known scale into `init` as `diag(s,s,s,1)`.
- **TSDF extrinsic:** Stage-2/3 poses are world→camera → pass **directly** (no inversion, no flip). Stage-1 poses are camera→world → pass `np.linalg.inv(pose)`.
- **Depth:** our `.npy` is float32 meters, NaN=invalid. Set **`depth_scale=1.0`** (default 1000 assumes uint16 mm), **`convert_rgb_to_intensity=False`** (keep RGB), and **raise `depth_trunc`** to scene max (default 3.0 m silently drops far depth). **Replace NaN with 0** before `o3d.geometry.Image`; also zero where confidence==0 / conf below threshold.
- **Mesh out (Stage 6):** `extract_triangle_mesh(); compute_vertex_normals(); write_triangle_mesh('mesh.ply'|'.obj', ...)`.
- **Normals (Stage 4):** poor fit — Open3D gives **sparse per-point world-frame** normals with ambiguous sign, not dense `[H,W,3]` camera-frame. Would need reprojection + `R_wc @ n` + `orient_normals_towards_camera_location`.

### Gotchas
`depth_scale=1000` and `depth_trunc=3.0` are the two silent-failure knobs. `convert_rgb_to_intensity=True` drops color. ICP default only 30 iters. `Vector3dVector` float64-only. `o3d.registration` moved to `o3d.pipelines.registration` at v0.10.

---

## COLMAP binary sparse-model format (cameras/images/points3D.bin)

**Confidence: high.** Reference impl `scripts/python/read_write_model.py`. Reimplement in `common/colmap_io.py` (numpy + stdlib `struct` only).

### Byte layout (little-endian `<` throughout)
- **cameras.bin:** `<Q` num_cameras; per cam `<iiQQ` [camera_id(int32), model_id(int32), width(**uint64**), height(**uint64**)] then `<d`×num_params. PINHOLE id=1 params `[fx,fy,cx,cy]`; SIMPLE_PINHOLE id=0 `[f,cx,cy]`; OPENCV id=4 `[fx,fy,cx,cy,k1,k2,p1,p2]`.
- **images.bin:** `<Q` num_reg_images; per image 64B header `<idddddddi` [image_id(int32), qvec `[qw,qx,qy,qz]`, tvec, camera_id(int32)]; then `name` as raw UTF-8 chars terminated by single `\x00` (**no length prefix**); `<Q` num_points2D; per point2D `<ddq` [x, y, **point3D_id int64 signed, −1=none**].
- **points3D.bin:** `<Q` num_points; per point `<QdddBBBd` [point3D_id(**uint64**), xyz(float64), rgb(uint8×3), error(float64)], `<Q` track_length, per track `<ii` [image_id(int32), point2D_idx(int32)].

### Coordinate convention
**World→camera.** `X_cam = R(qvec) @ X_world + tvec`. Quaternion **Hamilton, scalar-first `[qw,qx,qy,qz]`**. Camera axes X right, Y down, Z forward = **OpenCV** — no flip. Camera center in world `= −R^T @ tvec`.

### Mapping to our contract (EXACT conversions)
- **Stage 2 poses (world_to_camera):** `qvec = rotmat2qvec(R_wc)` (`[w,x,y,z]`), `tvec = t_wc`. Direct.
- **Stage 3 (world_to_camera + metric scale s):** apply s to **translations and points only** (`tvec_metric = s*t_wc; points3D.xyz *= s`); rotation/qvec unchanged.
- **Stage 1 poses (camera_to_world) → invert once:** `R_wc = R_cw.T; t_wc = -R_cw.T @ C`, then `qvec=rotmat2qvec(R_wc)`.
- **Reading back to c2w:** `R_cw = R(qvec).T; C = -R(qvec).T @ tvec`.
- **Intrinsics:** PINHOLE (id=1) `[fx,fy,cx,cy]` + width/height from resolutions; per-frame K (Stage 2) → one Camera per image with distinct camera_id.
- **Points/color:** id uint64 1-based; xyz float64 (upcast from float32); rgb sampled from lossless PNG; error 0.0/1.0 if synthetic; track_length may be 0.
- **Image names:** zero-padded 6-digit `000001.png`, null-terminated UTF-8, must match rgb filenames MILo loads.

### Gotchas
Two int types for point3D_id: **int64 signed (−1 sentinel) in images.bin**, **uint64 in points3D.bin**. width/height are **uint64** not int32. Name has no length prefix, single `\x00`. `rotmat2qvec` normalizes sign to `qw>=0`. Empty files still need the 8-byte count=0. Only place a conversion is needed is Stage-1 c2w→w2c inversion — **no COLMAP↔OpenGL flip** (COLMAP already = OpenCV).

---

# Stage 4 — Normals (`normals/`)

Target: `[H,W,3]` float32 unit vectors in [-1,1], **CAMERA frame, OpenCV** (+X right, +Y down, +Z into scene). Both normal estimators below emit a **different Z (and StableNormal also Y) sign** than our contract — this is the critical convention conflict for Stage 4. **Verify the global sign empirically on one known frame** for whichever estimator is chosen.

## StableNormal (Stable-X/StableNormal)

**Confidence: high** (convention inferred from diffusers Marigold).

### Install
`pip install torch==2.2.0 diffusers==0.28.0 transformers==4.36.1 xformers==0.0.24 accelerate==0.30.1 huggingface-hub==0.23.0 numpy==1.26.4 Pillow==10.3.0` (CUDA/fp16 required). Loaded via `torch.hub`, no repo pip install.

### Inputs / Outputs
`torch.hub.load("Stable-X/StableNormal", "StableNormal_turbo", trust_repo=True)` (fast) or `"StableNormal"` (quality). `predictor(img: PIL, resolution=1024, match_input_resolution=True, data_type=DataType.INDOOR)` → PIL RGB uint8 (visualization, quantized). **Raw floats (recommended):** `predictor.model(img, match_input_resolution=True, output_type="np").prediction` → `[N,H,W,3]` float32 unit vectors in **[-1,1]**.

### Coordinate convention
**OpenGL camera frame** (Marigold): +X right, **+Y up, +Z toward observer/camera**. Camera-facing surface → native `(0,0,+1)` (blue). NOT documented by StableNormal — from diffusers v0.28 Marigold defaults.

### Mapping to our contract (EXACT conversions)
1. Load once; per frame open Stage-3 rgb as PIL RGB.
2. `n = pred.model(img, match_input_resolution=True, output_type="np").prediction[0]` (**avoid uint8 PIL path** — quantizes to 8-bit).
3. **Axis fix — negate Y and Z:** `n[...,1] *= -1; n[...,2] *= -1` (OpenGL→OpenCV camera differs by negating Y and Z). Camera-facing surface becomes `(0,0,-1)`.
4. **Renormalize** (LANCZOS + fp16 make non-unit): `n /= max(||n||, 1e-6)`, cast float32.
5. Save `normals/000001.npy` `[H,W,3]`; ensure `match_input_resolution=True` so H,W equal the intrinsics-K resolution.
6. `data_type`: use `"indoor"`/`"outdoor"`; **avoid `"object"`** unless single-object (masks background to non-unit `(1,1,1)` → treat as invalid).
- `normals_weight/*.npy`: **no native source** (no per-pixel confidence surfaced) — omit or derive a validity mask.

### Gotchas
Convention inferred — validate one frame (only ambiguity is a global sign flip). GPU + fp16 only. `trust_repo=True`. Pin `yoso_version='yoso-normal-v0-3'` (keep ≤ v1.5 for scenes; > 1.5 switches to `nirne` package, object-only). resize forces multiples of 64; always renormalize. uint8 PIL loses precision.

## DSINE (baegwangbin/DSINE) — lighter feed-forward alternative

**Confidence: medium** (Z-sign is the riskiest fact).

### Install
`git clone https://github.com/baegwangbin/DSINE`; `pip install torch torchvision opencv-python Pillow numpy geffnet`; download `dsine.pt` (Google Drive) → `projects/dsine/checkpoints/exp001_cvpr2024/`. Lighter: `torch.hub.load('hugoycj/DSINE-hub','DSINE', trust_repo=True)` (third-party, "deprecated but provided").

### Inputs / Outputs
RGB uint8 → float32/255 → `(B,3,H,W)`; **pad to multiple of 32** (`get_padding`), ImageNet-normalize. `intrins` = full **3x3 K** `(B,3,3)` in pixels (shift cx,cy by padding); or synthesize from fov=60. `model(img, intrins)[-1]` → `(B,3,H,W)`, crop off padding, permute → `[H,W,3]`, **unit vectors [-1,1]**. No confidence channel; dense (no NaN).

### Coordinate convention
Camera frame, unit vectors. README example: camera-facing wall → **`(0,0,1)`**, meaning **+Z points toward the camera**; X=right, Y=down match OpenCV. (README internally inconsistent — labels "right-handed (right,down,front)" but gives `(0,0,1)`; the concrete example is operative.)

### Mapping to our contract (EXACT conversions)
- **Input K:** our OpenCV 3x3 drops straight into `intrins` (`fx=K[0,0]` etc), **no conversion**; add padding offsets to cx,cy. Prefer per-frame K from Stage-2/Stage-1 intrinsics; fall back fov=60. (K is pose-type-independent.)
- **Output:** `n = model(img,intrins)[-1]`, crop padding, permute→`[H,W,3]`, float32. Keep **raw [-1,1]** (do NOT apply `(n+1)/2` or ×255 — visualization only).
- **Axis fix — negate Z only:** `n_opencv = (nx, ny, -nz)`. DSINE +Z=toward-camera vs OpenCV +Z=into-scene; X/Y already match. Camera-facing wall → `(0,0,-1)`.
- Save `normals/000001.npy`; already unit-norm, no NaN handling.
- `normals_weight`: **no DSINE source** — omit.

### Gotchas
**Z-sign is medium confidence** — validate one frame (if pipeline wants visible normals pointing away, skip the negation). Pad to /32, crop back, shift cx,cy. `intrins` is a **3x3 K**, not `(fx,fy,cx,cy)`. No confidence channel. hub path is third-party/deprecated; maintained path is `test_minimal.py` + Drive `dsine.pt`.

---

# Stage 5 — MILo (mesh + Gaussian reconstruction)

## MILo (Anttwo/MILo)

**Confidence: high.** Default branch **`master`** (`main` 404s). Consumes a COLMAP sparse-model dataset; native convention = OpenCV/COLMAP w2c — passthrough.

### Install
Python 3.9, **CUDA 11.8 only** (export CPATH/LD_LIBRARY_PATH/PATH **before** cmake). `pytorch==2.3.1 torchvision==0.18.1 pytorch-cuda=11.8`. `python install.py` builds CUDA submodules (3× diff-gaussian-rasterization `_ms`/RaDe-GS/`_gof`, simple-knn, fused-ssim, nvdiffrast). **`tetra_triangulation`** needs CGAL+GMP (conda-forge) + cmake — main build friction. `--depth_order` needs DepthAnythingV2 `vitl` checkpoint.

### Inputs
`-s <DATASET_DIR>` COLMAP layout: `sparse/0/{cameras,images,points3D}.bin` (SIMPLE_PINHOLE/PINHOLE), `images/` (filenames must match `images.bin`). Pose: `R = qvec2rotmat(qvec).T` (stored transposed, glm/CUDA), true transform **world→camera**, qvec `(qw,qx,qy,qz)`. Init cloud from `points3D` (regenerated as `points3D.ply`) — **does NOT read an external user .ply**. **NO native sensor-depth/confidence/external-normal ingestion** in `train.py`; only monocular `--depth_order` (DepthAnythingV2 ordinal, not our sensor depth). Units = whatever COLMAP uses (metric input → meters).

### Outputs
- Gaussian: `point_cloud/iteration_18000/point_cloud.ply` (hard-coded final iter). Standard INRIA 3DGS PLY: x,y,z, nx/ny/nz (**zeros**), f_dc_0..2, f_rest_0..44 (SH deg 3), opacity (**logit**), scale_0..2 (**log**), rot_0..3 (**WXYZ**, un-normalized). World frame, meters (metric input), float32.
- Meshes (triangle PLY + vertex RGB): `mesh_learnable_sdf.ply` / `mesh_integration_sdf.ply` / `mesh_regular_tsdf_res*.ply` (extraction scripts must reuse the training `--mesh_config`).
- In-loop differentiable `mesh_depth`, `mesh_normals` tensors (via `milo.functional MeshRenderer`, `return_depth/return_normals=True`).

### Key API
Train: `python train.py -s <DATASET> -m <MODEL_DIR> --imp_metric <indoor|outdoor> --rasterizer <radegs|gof>` (from `./milo`). Flags: `--dense_gaussians`, `--mesh_config`, `--depth_order`, `--decoupled_appearance`, `--eval`. Differentiable Gaussians→mesh: `sample_gaussians_on_surface / extract_gaussian_pivots / compute_initial_sdf_values / compute_delaunay_triangulation / extract_mesh / MeshRenderer`.

### Coordinate convention
Standard 3DGS/COLMAP = **OpenCV**, world→camera, qvec WXYZ. Output Gaussian/mesh in SfM world frame. Matches our contract, no flip.

### Mapping to our contract (EXACT conversions)
- **Input (Stage 3 → `-s` dir):** our `metric/colmap/sparse/0/*.bin` are already w2c OpenCV = exactly MILo's expectation. Symlink/copy `sparse/0` and rgb → `images/` (6-digit names matching `images.bin`). **Identity boundary.** Metric Stage-3 → MILo trains in meters.
- **Init cloud:** MILo uses `points3D`, not our `points_metric.ply` — to control init, bake our points into `points3D.bin`.
- **Sensor depth/confidence:** **no native hook** — wire manually into `train.py` loss or compare rendered `mesh_depth [H,W]` meters vs our depth masked by confidence. `--depth_order` does NOT use it.
- **Normals:** no native external-normal supervision — attach against `MeshRenderer(return_normals=True)`. **`mesh_normals` frame (camera vs world) is undocumented** — confirm from source; if world-frame, rotate our camera-frame normals by `R_c2w` (or `mesh_normals` by `R_w2c`) exactly once, and match sign. `normals_weight` → per-pixel loss weight.
- **Output → Stage 6:** copy `point_cloud/iteration_18000/point_cloud.ply` → `output/point_cloud.ply` (no transform); chosen mesh → `output/mesh.ply` (+ `.obj` via trimesh); renders from `eval/mesh_nvs`. Record `--rasterizer/--mesh_config/iteration` in `provenance.json`.

### Gotchas
Branch `master`. CUDA 11.8 only. tetra_triangulation build friction. Final iter hard-coded 18000. Mesh extraction must reuse training `--mesh_config`. VGGT pose "coming soon" (not landed). Init from points3D, not user .ply. No sensor-depth/normal supervision without code changes. `mesh_normals` frame undocumented.

## DN-Splatter / AGS-Mesh (depth+normal supervision recipe, loss-concept port)

**Confidence: high.** Two strategies (`"dn-splatter"`, `"ags-mesh"`) in `regularization_strategy.py`. Port as a **loss concept** into the Stage-5 trainer.

### Install
`pip install git+https://github.com/maturk/dn-splatter` (needs nerfstudio + gsplat; `ns-train dn-splatter` / `ns-train ags-mesh`).

### Inputs (on-disk, CoolerMap/COLMAP path)
`images/`, `colmap/sparse/0/*.bin` (w2c, OpenCV), `mono_depth/*_aligned.npy` (or `sensor_depth`), `normals_from_pretrain/*.png` (uint8 RGB [0,1] encoding), optional confidence. **Frame association by natsorted filename order** (hard assert lengths match). Depth `.npy` meters (after scale) or `.png` uint16 mm (`SCALE_FACTOR=0.001`). Normal `.png` well-supported; `.npy` path TODO/unverified.

### Outputs (conceptual)
Scalar reg loss = edge-aware depth (`EdgeAwareLogL1`) + normal (L1 + TV, or AGS angular-masked) + scale/flatten (`mean(min(exp(scales)))`). Config: `depth_lambda` (default 0.0, **must set >0**; 0.2, or 0.5 Replica), `normal_lambda=0.1`, `depth_tolerance=0.1 m`.

### Coordinate convention
Internal normals = **OpenCV camera frame**, stored as `(n+1)/2` in [0,1]. **`normal_format` conflict:** `"omnidata"` applies `diag(1,-1,-1)` (Y/Z flip, treats file as OpenGL); **`"dsine"` applies NO flip** (assumes already OpenCV). COLMAP w2c read, then dataparser converts to nerfstudio OpenGL internally. **AGS confidence inverted:** `confidence = 1 - file/255` (0=confident/keep, 255=drop).

### Mapping to our contract (EXACT conversions)
- **Our Stage-4 normals → DN:** our convention already = DN internal. Encode `(n+1)/2 * 255` uint8 RGB PNG → `normals_from_pretrain/`, natsort-aligned 1:1 with `images/`. **CRITICAL: set `normal_format="dsine"`** (NOT `"omnidata"`, which would wrongly flip our Y/Z). `normals_weight` has no direct hook (multiply per-pixel normal L1 by it — custom).
- **Our depth → DN:** already meters → `depth_unit_scale_factor=1.0`, `is_euclidean_depth=False` (we store z-depth). Place as `mono_depth/*_aligned.npy`. **Replace NaN with 0** (valid mask is `gt>0.1`; NaN poisons LogL1). Note the 0.1 m floor drops true depth < 10 cm.
- **Our confidence → AGS:** DN expects **inverted** (0=keep, 255=drop) → save `255 - our_confidence`. Filtering active at step ≥ 7000.
- **Poses/COLMAP:** our Stage-3 `metric/colmap/sparse/0/*.bin` (w2c OpenCV) pass through unchanged; dataparser does the OpenCV↔OpenGL boundary once.
- **Weights to port:** `depth_lambda=0.2` (0.5 Replica), `normal_lambda=0.1`, `depth_tolerance=0.1`, HuberL1 `tresh=0.2`, EdgeAwareLogL1, L1+TV normals, flatten loss. AGS gating: depth conf ≥ step 7000, normal conf (angular > 0.1 rad) + Laplacian-edge mask ≥ step 15000.

### Gotchas
`depth_lambda` must be >0 (assert). DN scales depth term `×(1+depth_lambda)`, AGS `×depth_lambda`. Confidence **inverted** vs us. `normal_format` must be `"dsine"`. Normal `.npy` loader unverified. `depth_tolerance=0.1` drops <10 cm. NaN→0. Association by natsorted filename.

## 3dgs-mcmc (ubc-vision) — MCMC densification alternative for Stage 5

**Confidence: high.** Fork of graphdeco-inria 3DGS; same COLMAP input + 62-field PLY output as MILo.

### Install
Python 3.8, CUDA 11.7, torch 1.13.1+cu117 (old — likely build-fail on sm_90+/CUDA 12). `git clone --recursive`; build submodules incl. MCMC relocation kernel `compute_relocation_cuda`.

### Inputs / Outputs
Same as vanilla 3DGS: `-s <dir>` with `images/` + `sparse/0/*.bin` (= our Stage-3 metric). `--cap_max N` **effectively required** (default -1 aborts). `--init_type sfm` seeds from points3D. Output `point_cloud/iteration_<N>/point_cloud.ply` — 62 float32 fields: x,y,z, nx/ny/nz (**zeros**), f_dc_0..2, f_rest_0..44, opacity (**logit**), scale_0..2 (**log**), rot_0..3 (**WXYZ**, un-normalized). Renders named 5-digit sequential index.

### Coordinate convention
Inherited COLMAP = **OpenCV**, world→camera, WXYZ. Output means in COLMAP world frame. No axis flip Stage-3→Stage-5.

### Mapping to our contract (EXACT conversions)
- **Input:** point `-s` at `images/` (our 6-digit PNGs) + `sparse/0/*.bin` (our metric w2c OpenCV) — passthrough. `--init_type sfm`, set `--cap_max`. Depth/conf/normals have **no hook** here.
- **Output → `output/point_cloud.ply`:** structurally satisfies it. Consumer conversions: `opacity=sigmoid(opacity); scale=exp(scale); quat normalize (WXYZ); color=f_dc*0.28209+0.5`. Ignore nx/ny/nz (zeros). Renders are **5-digit sequential**, not our 6-digit frame IDs — remap via `cameras.json image_name`.

### Gotchas
`noise_lr` default = **5e5 = 500000.0** (README table's "5e-5" is WRONG — trust source). `cap_max` mandatory. `opacity_reg` scene-dependent (0.001 Deep Blending, else 0.01). Old CUDA/torch pin. PLY normals are placeholder zeros. Inria research-only license.

## gsplat (nerfstudio-project) — REFERENCE ONLY

**Confidence: high.** The rasterizer/densifier DA3's Gaussian head pins (commit 0b4dddf), **NOT** our Stage-5 renderer (Stage 5 = MILo). Documented here to reason about DA3's Gaussian head.

### Inputs / Outputs
`rasterization(means, quats(**WXYZ**, auto-normalized), scales(**LINEAR meters**), opacities(**[0,1] activated**), colors, viewmats(**world→camera**), Ks(pixels), width, height, sh_degree=..., render_mode='RGB'|'RGB+ED'|'D', ...)` → `(render_colors, render_alphas, meta)`. Depth channel = metric **camera-frame +z** meters, no NaN sentinel. `MCMCStrategy(cap_max, noise_lr=5e5, ...)` mutates params in place; needs current means `lr`. **No PLY writer** (caller's job).

### Coordinate convention
`viewmats` = **world→camera, OpenCV/COLMAP**. Quat **WXYZ**. Depth metric camera +z. Matches our contract.

### Mapping to our contract (EXACT conversions)
- **Our data → gsplat:** Stage-2/3 poses (w2c OpenCV) → viewmats **direct**, no flip/inversion. Stage-1 c2w → `viewmats = torch.linalg.inv(camtoworlds)` (invert once). Ks = our K in pixels. means/scales in meters float32. Quats must be **WXYZ** (reorder if any tooling is XYZW).
- **gsplat → our data:** render_colors clamp [0,1]→PNG; depth channel → `depth/*.npy` float32 meters (map zero-alpha → NaN via `render_alphas`); alpha → confidence.
- **Splat → INRIA PLY (Stage 6):** gsplat live values are **linear scale / activated opacity**; INRIA PLY stores **log(scale) / logit(opacity)**, SH split f_dc(3)+f_rest(45), quats WXYZ. Convert `scale→log, opacity→logit` on write; `exp/sigmoid` on read.

### Gotchas
JIT-compiles CUDA on first import. **Activation trap:** `rasterization()` wants linear scales / [0,1] opacities, but the reference trainer stores log/logit and applies exp/sigmoid at the call site — don't double-activate. Quats WXYZ. MCMC needs means `lr` (`noise = noise_lr*lr`). No PLY/mesh writer. Depth has no NaN sentinel. Pinned commit 0b4dddf touches only viewer/2DGS, not the rasterization/MCMC public API (medium-high confidence).

---

# Stage 6 — Output (`output/`)

Assembled from MILo (primary) or 3dgs-mcmc outputs:
- `point_cloud.ply` ← MILo `iteration_18000/point_cloud.ply` (or 3dgs-mcmc), world-frame meters, standard INRIA 3DGS layout (log-scale, logit-opacity, WXYZ quats, zeroed normals).
- `mesh.ply/.obj` ← MILo mesh (world-frame meters, vertex RGB); `.obj` via trimesh/Open3D.
- `renders/` ← MILo `eval/mesh_nvs` or gsplat/3dgs-mcmc renders — **remap 5-digit sequential indices to our 6-digit frame IDs via `cameras.json`**.
- `provenance.json` ← our responsibility (rasterizer, mesh_config, iteration, scale_report ref). Not emitted by any component.
