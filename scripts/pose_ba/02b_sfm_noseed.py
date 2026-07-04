#!/usr/bin/env python3
"""Pose-BA step 2b (env gs-ba): UNSEEDED from-scratch incremental SfM.

Same SuperPoint features + LightGlue matches as the seeded arm, but the ARKit poses
are NOT used at all -- COLMAP builds the reconstruction from scratch. This tests two
things the seeded arm cannot:
  1. Is the phone-pose SEED essential to recover geometry on smooth low-texture skin?
     (if from-scratch registers all 48 frames cleanly -> seed not essential)
  2. Independent cross-check of pose quality: if unseeded SfM lands near the ARKit
     poses while ignoring them entirely, that CORROBORATES the ARKit poses are good.

Intrinsics are fixed to the known ARKit K (fair vs the seeded arm, which also fixes K),
so the only variable is the pose seed.
"""
import os
from pathlib import Path
import pycolmap
from hloc import reconstruction

REPO = Path("/home/paperspace/Documents/VS Code Projects/3D-Gaussian")
SESS = REPO / os.environ.get("PBA_SESS", "sessions/session_20260703_145121")
IMAGES = SESS / "capture/rgb"
WORK = SESS / "pose_ba"
pairs = WORK / "pairs.txt"; feats = WORK / "feats.h5"; matches = WORK / "matches.h5"
OUT = WORK / "sfm_noseed"
REF = SESS / "metric/colmap/sparse/0"

# restrict to the keyframes that actually have features/matches (image_dir has all frames)
_ref = pycolmap.Reconstruction(str(REF))
names = sorted(im.name for im in _ref.images.values())

# read THIS session's intrinsics from the reference model (do NOT hardcode a session's K)
_cam = list(_ref.cameras.values())[0]
_p = list(_cam.params)                                    # PINHOLE: fx, fy, cx, cy
fx, fy, cx, cy = (_p + _p[:1] * 4)[:4] if len(_p) >= 4 else (_p[0], _p[0], _p[1], _p[2])
img_opts = pycolmap.ImageReaderOptions()
try:
    img_opts.camera_model = "PINHOLE"
    img_opts.camera_params = f"{fx},{fy},{cx},{cy}"
    print(f"[02b] fixed intrinsics from ref model: fx={fx:.2f} fy={fy:.2f} cx={cx:.2f} cy={cy:.2f}")
except Exception as e:
    print("[02b] note: could not preset intrinsics via ImageReaderOptions:", e)

mapper_opts = {"ba_refine_focal_length": False, "ba_refine_principal_point": False,
               "ba_refine_extra_params": False}

kw = dict(sfm_dir=OUT, image_dir=IMAGES, pairs=pairs, features=feats, matches=matches,
          image_list=names, camera_mode=pycolmap.CameraMode.SINGLE, verbose=False)
try:
    model = reconstruction.main(image_options=img_opts, mapper_options=mapper_opts, **kw)
except TypeError as e:
    print("[02b] signature fallback (no image/mapper options):", e)
    model = reconstruction.main(**kw)

n_reg = len(model.reg_image_ids())
n_pts = model.num_points3D()
print(f"[02b] UNSEEDED SfM: {n_reg}/48 images registered, {n_pts} pts")
if n_reg < 48:
    print(f"[02b] *** {48 - n_reg} frames FAILED to register -> seed helps robustness on smooth skin ***")
else:
    print("[02b] all 48 registered from scratch -> seed NOT essential for registration")
print(f"[02b] wrote {OUT}")
