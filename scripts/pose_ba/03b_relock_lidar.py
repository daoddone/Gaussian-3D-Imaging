#!/usr/bin/env python3
"""Metric-lock the gauge-free unseeded SfM to the LiDAR (sensor_depth), the HQ counterpart
to 03_relock.py.

03_relock aligns the gauge-free BA model to an ARKit metric COLMAP target (Umeyama on camera
centers). The HQ-Depth capture has NO ARKit poses, so its ONLY absolute metric reference is the
raw LiDAR depth. This script recovers the single SfM->metric scale S directly from that depth:

  for each SfM 3D point observed in a frame, compare its SfM camera-frame depth to the LiDAR
  depth at the SAME observed pixel; S = robust median( lidar_depth / sfm_depth ).

Scale is derived per-observation (a ratio), so it is immune to the SfM gauge freedom
(rotation/translation/global-scale) — no pose is used to solve it. Then poses (tvec*=S) and
points (xyz*=S) are scaled and written as a metric COLMAP model that Stage 5 (MILo) ingests.

Reads/writes with common.colmap_io (the exact reader/writer the pipeline uses), so the emitted
metric/colmap is byte-identical in format to what Stage 3 produces and Stage 5/MILo consumes.
Writes into <session>/metric_sfm/ (does NOT touch the DA3 metric/); the caller swaps it in.
Run in the gs-ba env (has numpy + PIL).
"""
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path("/home/paperspace/Documents/VS Code Projects/3D-Gaussian")
sys.path.insert(0, str(REPO))
from common import colmap_io, plyio
from common.conventions import quat_to_rotmat

SESS = REPO / "sessions/session_20260704_143324"
SFM = SESS / "pose_ba" / "sfm_noseed"
OUT = SESS / "metric_sfm"
CAP = json.load(open(SESS / "capture/intrinsics.json"))
DEPTH_DIR = SESS / "capture/depth"
CONF_DIR = SESS / "capture/confidence"

color_w, color_h = CAP["color_resolution"]
depth_w, depth_h = CAP["depth_resolution"]
sx, sy = depth_w / color_w, depth_h / color_h          # color-pixel -> depth-pixel
DMIN, DMAX = 0.15, 3.0

imgs = colmap_io.read_images_binary(SFM / "images.bin")
pts = colmap_io.read_points3D_binary(SFM / "points3D.bin")
cams = colmap_io.read_cameras_binary(SFM / "cameras.bin")
print(f"[lock] SfM model: {len(imgs)} images, {len(pts)} points, {len(cams)} camera(s)")


def load_conf(fid):
    p = CONF_DIR / f"{fid}.png"
    if not p.exists():
        return None
    from PIL import Image
    a = np.asarray(Image.open(p))
    if a.ndim == 3:
        a = a[..., 0]
    return a >= 128


# ---- 1. solve the SfM->metric scale from the LiDAR ------------------------- #
ratios, per_frame = [], {}
for img in imgs.values():
    fid = Path(img["name"]).stem
    dpath = DEPTH_DIR / f"{fid}.npy"
    if not dpath.exists():
        continue
    depth = np.load(dpath).astype(float)               # (240,320) meters, NaN = invalid
    conf = load_conf(fid)
    R = quat_to_rotmat(img["qvec"])                     # world-to-camera (w,x,y,z)
    t = np.asarray(img["tvec"], float)
    xys, p3d = img["xys"], img["point3D_ids"]
    n = 0
    for k in range(len(xys)):
        pid = int(p3d[k])
        if pid not in pts:
            continue
        X = np.asarray(pts[pid]["xyz"], float)
        zc = float((R @ X + t)[2])
        if zc <= 1e-6:
            continue
        ud, vd = int(round(float(xys[k][0]) * sx)), int(round(float(xys[k][1]) * sy))
        if not (0 <= ud < depth_w and 0 <= vd < depth_h):
            continue
        d = depth[vd, ud]
        if not np.isfinite(d) or d < DMIN or d > DMAX:
            continue
        if conf is not None and not conf[vd, ud]:
            continue
        ratios.append(d / zc)
        n += 1
    per_frame[fid] = n

ratios = np.asarray(ratios)
if ratios.size < 100:
    raise SystemExit(f"[lock] too few depth samples ({ratios.size}); cannot metric-lock")
S = float(np.median(ratios))
mad = float(np.median(np.abs(ratios - S)))
frames_hit = sum(1 for v in per_frame.values() if v > 0)
print(f"[lock] {ratios.size} (point,frame) samples over {frames_hit}/{len(imgs)} frames")
print(f"[lock] SfM->metric scale S = {S:.6f}   MAD = {mad:.6f} ({100*mad/S:.1f}% of S)")
pf = list(per_frame.values())
print(f"[lock] per-frame samples: min={min(pf)} median={int(np.median(pf))} max={max(pf)}")

# ---- 2. scale + write the metric COLMAP model + init cloud ----------------- #
# uniform scale about the origin: qvec unchanged, tvec*=S, points xyz*=S. Keep the SfM camera
# (its K == device K, and the poses were solved against it, so poses<->K stay consistent).
new_imgs = {}
for iid, img in imgs.items():
    new_imgs[iid] = {"qvec": img["qvec"],
                     "tvec": [float(v) * S for v in img["tvec"]],
                     "camera_id": img["camera_id"], "name": img["name"]}  # no xys -> npt 0

new_pts, xyz_all, rgb_all = {}, [], []
for i, (pid, p) in enumerate(sorted(pts.items()), start=1):
    xyz = np.asarray(p["xyz"], float) * S
    rgb = tuple(int(c) for c in np.asarray(p["rgb"])[:3])
    new_pts[i] = {"xyz": xyz, "rgb": rgb, "error": 0.0, "track": []}
    xyz_all.append(xyz)
    rgb_all.append(rgb)
xyz_all = np.asarray(xyz_all)
rgb_all = np.asarray(rgb_all, dtype=np.uint8)

out_sparse = OUT / "colmap" / "sparse" / "0"
colmap_io.write_model(out_sparse, cams, new_imgs, new_pts)
plyio.write_ply(OUT / "points_metric.ply", xyz_all, colors=rgb_all, binary=True)

# sanity: camera-center + object extents in mm (feet capture, camera moving tens of cm)
centers = np.array([-quat_to_rotmat(im["qvec"]).T @ np.asarray(im["tvec"], float) for im in new_imgs.values()])
cam_extent_mm = (centers.max(0) - centers.min(0)) * 1000
obj_extent_mm = (xyz_all.max(0) - xyz_all.min(0)) * 1000
print(f"[lock] metric camera-center extent: {cam_extent_mm.round(0)} mm")
print(f"[lock] metric point-cloud extent:   {obj_extent_mm.round(0)} mm ({len(xyz_all)} pts)")

report = {
    "method": "sfm_lidar_metric_lock",
    "note": "unseeded SfM (02b) metric-locked to raw LiDAR depth (sensor_depth); HQ has no ARKit/VIO anchor",
    "sfm_source": str(SFM.relative_to(REPO)),
    "scale_S": S, "scale_MAD": mad, "scale_MAD_pct": 100 * mad / S,
    "depth_samples": int(ratios.size), "frames_used": frames_hit,
    "n_images": len(new_imgs), "n_points_init": int(len(xyz_all)),
    "camera_center_extent_mm": cam_extent_mm.tolist(),
    "point_extent_mm": obj_extent_mm.tolist(),
    "intrinsics": "sfm_shared_K(==device_K)", "convention": "world_to_camera OpenCV (w,x,y,z)",
}
(OUT / "scale_report_sfm.json").write_text(json.dumps(report, indent=2))
print(f"[lock] wrote {out_sparse}")
print(f"[lock] wrote {OUT / 'points_metric.ply'} + scale_report_sfm.json")
