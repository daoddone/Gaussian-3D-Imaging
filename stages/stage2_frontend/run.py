#!/usr/bin/env python3
"""Stage 2 entry point: front end (camera poses + dense geometry).

    python stages/stage2_frontend/run.py --session <session_dir> --config <pipeline.yaml>

Runs Depth Anything 3 (nested giant+metric model DA3NESTED-GIANT-LARGE) on the
Stage 1 color frames and writes the Stage 2 output contract
(io_contracts/frontend_output.md):

    frontend/poses.json          (world_to_camera, OpenCV)
    frontend/intrinsics.json     (per-frame K at the model's output resolution)
    frontend/depth/*.npy         (metric meters)
    frontend/conf/*.npy          (normalized to [0,1])
    frontend/points.ply          (fused dense cloud, unprojected)
    frontend/colmap/sparse/0/    (cameras/images/points3D.bin, world_to_camera)

Runs in its OWN environment (pipeline_stage2_frontend); this file is written
against the DA3 API but requires that environment (torch + GPU) to execute.

Component facts (see docs/COMPONENT_IO_REFERENCE.md):
  * prediction.extrinsics are WORLD-TO-CAMERA in the OpenCV convention — an
    identity boundary with our contract (no axis flip). Shape is (N,3,4) or
    (N,4,4) depending on build; we branch on .shape.
  * DA3NESTED depth is already meters — do NOT apply the DA3METRIC /300 formula.
  * prediction.conf range is undocumented; we normalize per-frame to [0,1].
  * DA3 is 0-indexed by input order; we map output i -> the i-th input frame id
    so frame numbering stays aligned with Stage 1.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[1]
for p in (str(_ROOT), str(_HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
import yaml

from common import conventions as C
from common import colmap_io, plyio
from common.file_layout import SessionLayout


def _normalize_conf(conf, lo_pct=2.0, hi_pct=98.0):
    """Map an arbitrary-range confidence map to [0,1] via robust percentiles.

    DA3's conf range is undocumented, so we clip to the [2,98] percentiles and
    rescale. This is a heuristic to satisfy the [0,1] contract; validate against
    real conf maps and revisit if the model documents its range.
    """
    conf = np.asarray(conf, dtype=np.float32)
    finite = np.isfinite(conf)
    if not finite.any():
        return np.zeros_like(conf)
    lo = np.percentile(conf[finite], lo_pct)
    hi = np.percentile(conf[finite], hi_pct)
    if hi <= lo:
        return np.clip(conf, 0, 1).astype(np.float32)
    out = (conf - lo) / (hi - lo)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def _extrinsic_to_Rt(ext_i):
    """Return (R, t) world_to_camera from a (3,4) or (4,4) extrinsic."""
    ext_i = np.asarray(ext_i, dtype=float)
    R = ext_i[:3, :3]
    t = ext_i[:3, 3]
    return R, t


def _unproject_cloud(depth, K, R_w2c, t_w2c, conf=None, conf_thresh=0.1, stride=4):
    """Back-project a depth map to world points (subsampled), OpenCV convention."""
    H, W = depth.shape
    vs, us = np.mgrid[0:H:stride, 0:W:stride]
    z = depth[vs, us]
    m = np.isfinite(z) & (z > 0)
    if conf is not None:
        m &= conf[vs, us] >= conf_thresh
    us = us[m]; vs = vs[m]; z = z[m]
    x = (us - K[0, 2]) / K[0, 0] * z
    y = (vs - K[1, 2]) / K[1, 1] * z
    pts_cam = np.stack([x, y, z], axis=1)
    R_c2w, t_c2w = C.invert_pose(R_w2c, t_w2c)
    return (R_c2w @ pts_cam.T).T + t_c2w


def run(session_dir, config_path):
    cfg = yaml.safe_load(open(config_path))
    s2 = cfg.get("stage2", {})
    layout = SessionLayout(session_dir)

    rgb_ids = SessionLayout.list_frames(layout.capture_rgb, ".png")
    if not rgb_ids:
        raise SystemExit(f"[stage2] no rgb frames under {layout.capture_rgb}")
    image_paths = [str(layout.capture_rgb / f"{fid}.png") for fid in rgb_ids]

    # optional pose conditioning from Stage 1
    ext_in = ixt_in = None
    if s2.get("pose_conditioning") and layout.capture_poses.exists() and layout.capture_intrinsics.exists():
        cap = C.load_poses(layout.capture_poses)
        K = np.asarray(json.load(open(layout.capture_intrinsics))["K"], dtype=float)
        exts, ixts = [], []
        for fid in rgb_ids:
            if fid not in cap["poses"]:
                exts = ixts = None
                break
            R_wc, t_wc = C.to_world_to_camera(cap["poses"][fid]["R"], cap["poses"][fid]["t"], cap["pose_type"])
            E = np.eye(4); E[:3, :3] = R_wc; E[:3, 3] = t_wc
            exts.append(E); ixts.append(K)
        if exts is not None:
            ext_in = np.stack(exts); ixt_in = np.stack(ixts)

    # ---- run Depth Anything 3 (requires the stage2 env: torch + GPU) --------
    from depth_anything_3.api import DepthAnything3  # noqa: E402

    model_id = "depth-anything/" + s2.get("model", "DA3NESTED-GIANT-LARGE")
    model = DepthAnything3.from_pretrained(model_id).to(device="cuda")
    pred = model.inference(
        image=image_paths,
        extrinsics=ext_in,
        intrinsics=ixt_in,
        use_ray_pose=bool(s2.get("use_ray_pose", True)),
        align_to_input_ext_scale=ext_in is not None,
    )

    if s2.get("free_geometry_refinement"):
        print("[stage2] WARNING: Free Geometry refinement requested but not wired here. "
              "It needs the Free-Geometry repo + peft and a LoRADepthAnything3 wrap; "
              "see docs/COMPONENT_IO_REFERENCE.md. Proceeding without it.")

    depth = np.asarray(pred.depth)                 # (N,H,W) meters (nested = metric)
    conf = getattr(pred, "conf", None)
    extr = np.asarray(pred.extrinsics)             # (N,3,4) or (N,4,4), w2c OpenCV
    intr = np.asarray(pred.intrinsics)             # (N,3,3) at output resolution
    N, H, W = depth.shape

    # ---- write per-frame outputs -------------------------------------------
    layout.frontend_depth.mkdir(parents=True, exist_ok=True)
    layout.frontend_conf.mkdir(parents=True, exist_ok=True)

    poses = {}
    K_by_frame = {}
    Rt_w2c = {}
    all_pts = []
    for i, fid in enumerate(rgb_ids[:N]):
        d = depth[i].astype(np.float32)
        np.save(layout.frontend_depth / f"{fid}.npy", d)
        c = _normalize_conf(conf[i]) if conf is not None else np.ones((H, W), np.float32)
        np.save(layout.frontend_conf / f"{fid}.npy", c)

        R_wc, t_wc = _extrinsic_to_Rt(extr[i])
        poses[fid] = {"R": R_wc, "t": t_wc}
        K_by_frame[fid] = np.asarray(intr[i], dtype=float)
        Rt_w2c[fid] = (R_wc, t_wc)
        all_pts.append(_unproject_cloud(d, K_by_frame[fid], R_wc, t_wc, conf=c))

    # poses.json (world_to_camera)
    C.save_poses(layout.frontend_poses, poses, pose_type=C.WORLD_TO_CAMERA)
    # intrinsics.json (per-frame K, at model output resolution W x H)
    with open(layout.frontend_intrinsics, "w") as fh:
        json.dump({"convention": "OpenCV", "resolution": [int(W), int(H)],
                   "K": {fid: K_by_frame[fid].tolist() for fid in K_by_frame}}, fh, indent=2)
    # fused dense cloud
    cloud = np.concatenate(all_pts, axis=0) if all_pts else np.zeros((0, 3))
    plyio.write_ply(layout.frontend_points, cloud, binary=True)
    # COLMAP sparse model (world_to_camera, PINHOLE), built with our validated writer
    names = {fid: f"{fid}.png" for fid in rgb_ids[:N]}
    cameras, images = colmap_io.build_pinhole_model(
        list(rgb_ids[:N]), K_by_frame, Rt_w2c, names, (W, H), shared_intrinsics=False)
    colmap_io.write_model(layout.frontend_colmap, cameras, images, points3D={})

    print(f"[stage2] wrote frontend/ for {N} frames at {W}x{H}; "
          f"cloud={cloud.shape[0]} pts; model={model_id}")
    return 0


def main():
    ap = argparse.ArgumentParser(description="Stage 2: Depth Anything 3 front end")
    ap.add_argument("--session", required=True)
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    return run(args.session, args.config)


if __name__ == "__main__":
    sys.exit(main())
