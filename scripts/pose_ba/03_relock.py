#!/usr/bin/env python3
"""Pose-BA step 3 (env pipeline_stage2_frontend): re-lock the gauge-free BA output
to the metric frame that the reconstruction actually consumes.

BA is 7-DoF gauge-free (its global scale/rotation/translation drifts from the seed),
so it must be re-metriced. A GLOBAL similarity cannot re-introduce the LOCAL per-frame
drift that BA removed, so this restores the metric gauge without undoing the fix.

CRITICAL: we align to metric/colmap/sparse/0 (the exact model the A-arm trains on),
NOT to the raw capture_poses JSON. Stage 3 applies its own metric-anchor similarity to
the ARKit poses (~8 mm shift), so capture_poses and metric/colmap live in DIFFERENT
frames. Aligning to metric/colmap makes the A/B differ ONLY by BA's per-frame pose
refinement and inherits Stage 3's LiDAR-locked metric scale.

Usage: 03_relock.py [--model pose_ba/refined] [--out metric_ba]
"""
import sys
import json
import argparse
from pathlib import Path

sys.path.insert(0, "/home/paperspace/Documents/VS Code Projects/3D-Gaussian")
import numpy as np

from common import colmap_io, plyio
from common import conventions as C
from common.file_layout import SessionLayout
from stages.stage3_metric import align

import os
SESS = Path("/home/paperspace/Documents/VS Code Projects/3D-Gaussian") / os.environ.get(
    "PBA_SESS", "sessions/session_20260703_145121")
lay = SessionLayout(SESS)

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="pose_ba/refined", help="session-relative BA model dir")
ap.add_argument("--out", default="metric_ba", help="session-relative output dir")
args = ap.parse_args()

REFINED = SESS / args.model
TARGET = SESS / "metric" / "colmap" / "sparse" / "0"   # the exact ARKit model the A-arm trains on


def _stem(n):
    return Path(n).stem


def _center(img):
    R = C.quat_to_rotmat(img["qvec"])
    t = np.asarray(img["tvec"], float)
    return -R.T @ t


ba_imgs = colmap_io.read_images_binary(REFINED / "images.bin")
ba_pts = colmap_io.read_points3D_binary(REFINED / "points3D.bin")
tgt_imgs = colmap_io.read_images_binary(TARGET / "images.bin")

ba = {im["name"]: (C.quat_to_rotmat(im["qvec"]), np.asarray(im["tvec"], float)) for im in ba_imgs.values()}
tgt_center = {_stem(im["name"]): _center(im) for im in tgt_imgs.values()}

common = sorted((n for n in ba if _stem(n) in tgt_center), key=_stem)
if len(common) < 3:
    raise SystemExit(f"[03] only {len(common)} matched cameras; BA likely dropped frames")

ba_centers = np.array([-ba[n][0].T @ ba[n][1] for n in common])
ark_centers = np.array([tgt_center[_stem(n)] for n in common])

s, R, t = align.umeyama(ba_centers, ark_centers, with_scale=True)
resid = align.similarity_residual(ba_centers, ark_centers, s, R, t)
print(f"[03] re-lock {args.model} -> metric/colmap frame: scale={s:.5f}  "
      f"center-fit residual={1000 * resid:.2f} mm  ({len(common)} cams)")

Rt = {}
for n in common:
    R2, t2 = align.apply_similarity_to_w2c_pose(ba[n][0], ba[n][1], s, R, t)
    Rt[_stem(n)] = (R2, t2)

ci = json.load(open(lay.capture_intrinsics))
K = np.asarray(ci["K"], float)
res = tuple(ci["color_resolution"])
fids = sorted(Rt)
cams, imgs = colmap_io.build_pinhole_model(fids, {f: K for f in fids}, Rt, {f: f"{f}.png" for f in fids},
                                           res, shared_intrinsics=True)
out = SESS / args.out / "colmap" / "sparse" / "0"
colmap_io.write_model(out, cams, imgs, points3D={})

if ba_pts:
    pts = align.apply_similarity_points(np.array([ba_pts[p]["xyz"] for p in ba_pts]), s, R, t)
    cols = np.array([ba_pts[p]["rgb"] for p in ba_pts])
    plyio.write_ply(SESS / args.out / "points_ba.ply", pts.astype(np.float32), colors=cols)

print(f"[03] wrote {out} ({len(fids)} poses) + {args.out}/points_ba.ply ({len(ba_pts)} BA pts)")
