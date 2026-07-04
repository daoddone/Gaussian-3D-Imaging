#!/usr/bin/env python3
"""Pose-BA step 1 (env gs-ba): SuperPoint features + exhaustive LightGlue matching.

Runs on the 48 keyframes that are actually in the reference COLMAP model (NOT all
60 rgb pngs). Exhaustive matching over 48 frames (~1128 pairs) is cheap and
captures every loop-closure pair that cancels the ARKit drift. filter_threshold
is lowered to 0.05 as insurance against smooth low-texture skin starving matches.
"""
import os
from pathlib import Path
import pycolmap
from hloc import extract_features, match_features, pairs_from_exhaustive

REPO = Path("/home/paperspace/Documents/VS Code Projects/3D-Gaussian")
SESS = REPO / os.environ.get("PBA_SESS", "sessions/session_20260703_145121")
IMAGES = SESS / "capture/rgb"
REF = SESS / "metric/colmap/sparse/0"
WORK = SESS / "pose_ba"; WORK.mkdir(exist_ok=True)

names = sorted(im.name for im in pycolmap.Reconstruction(str(REF)).images.values())
print(f"[01] {len(names)} keyframes: {names[0]} .. {names[-1]}")

fc = extract_features.confs["superpoint_max"]                    # 1600px, 4096 kpts
mc = match_features.confs["superpoint+lightglue"]
mc = {**mc, "model": {**mc["model"], "filter_threshold": 0.05}}  # low-texture insurance

pairs = WORK / "pairs.txt"; feats = WORK / "feats.h5"; matches = WORK / "matches.h5"
pairs_from_exhaustive.main(pairs, image_list=names)
extract_features.main(fc, IMAGES, image_list=names, feature_path=feats)
match_features.main(mc, pairs, features=feats, matches=matches)
print(f"[01] done: {feats.name}, {matches.name}, {sum(1 for _ in open(pairs))} pairs")
