#!/usr/bin/env python3
"""Generate a synthetic capture+frontend session with a KNOWN ground-truth scale.

This is a test fixture, not real data. It lets us verify Stage 3 end-to-end
before any iPhone capture or Depth Anything 3 inference exists: the metric world
is defined in meters, the front-end reconstruction is a scaled/rotated/translated
copy of it, and Stage 3 must recover the applied scale (= TRUE_SCALE) and land
the residual near zero.

Geometry: a small sphere (face-sized) viewed by cameras orbiting it.
  metric world (Stage 1 truth):   P_m,  cameras camera_to_world in meters
  front-end world (Stage 2):       P_f = (1/TRUE_SCALE) * R0 P_m + t0
So sensor_depth = TRUE_SCALE * front_depth, and the camera-path Umeyama fit
recovers scale = TRUE_SCALE. Both anchors should agree -> status "pass".

Usage:
    python tests/make_synthetic_session.py --out sessions/synthetic_demo --scale 1.05
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np

from common import conventions as C
from common import plyio
from common.file_layout import SessionLayout, frame_id


def sphere_points(center, radius, n_u=220, n_v=440):
    u = np.linspace(0.02, np.pi - 0.02, n_u)
    v = np.linspace(0, 2 * np.pi, n_v, endpoint=False)
    uu, vv = np.meshgrid(u, v, indexing="ij")
    uu = uu.ravel(); vv = vv.ravel()
    dirs = np.stack([np.sin(uu) * np.cos(vv), np.sin(uu) * np.sin(vv), np.cos(uu)], axis=1)
    pts = np.asarray(center) + radius * dirs
    # simple colour by direction so points.ply carries colour
    cols = ((dirs * 0.5 + 0.5) * 255).astype(np.uint8)
    return pts, cols


def look_at_c2w(center_of_scene, cam_pos):
    f = center_of_scene - cam_pos
    f = f / np.linalg.norm(f)
    tmp = np.array([0.0, 1.0, 0.0])
    if abs(f @ tmp) > 0.99:
        tmp = np.array([1.0, 0.0, 0.0])
    r = np.cross(tmp, f); r /= np.linalg.norm(r)
    d = np.cross(f, r)
    R_c2w = np.stack([r, d, f], axis=1)  # columns: x(right), y(down), z(forward)
    return R_c2w, cam_pos.copy()


def K_matrix(fx, fy, cx, cy):
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])


def render_sphere_depth(center_w, radius, R_w2c, t_w2c, K, W, H):
    """Analytic ray-sphere depth render (exact, smooth). Returns planar-z depth
    [H,W] float32 in meters, NaN where the ray misses the sphere.

    A point-splat render would quantise depth and bias the sensor-vs-front ratio
    near the silhouette; the analytic render keeps sensor = TRUE_SCALE * front
    exact per pixel, which is what the depth anchor must recover.
    """
    us, vs = np.meshgrid(np.arange(W), np.arange(H))
    x = (us - K[0, 2]) / K[0, 0]
    y = (vs - K[1, 2]) / K[1, 1]
    d = np.stack([x, y, np.ones_like(x, dtype=float)], axis=2)
    dn = d / np.linalg.norm(d, axis=2, keepdims=True)     # unit ray dirs
    c = R_w2c @ np.asarray(center_w, dtype=float) + t_w2c  # sphere center, cam frame
    b = np.einsum("ijk,k->ij", dn, c)
    cc = float(c @ c) - radius ** 2
    disc = b * b - cc
    hit = disc > 0
    t_near = np.where(hit, b - np.sqrt(np.maximum(disc, 0.0)), np.nan)
    hit = hit & (t_near > 0)
    depth = np.where(hit, t_near * dn[:, :, 2], np.nan).astype(np.float32)
    return depth


def save_confidence(path, depth):
    from PIL import Image
    mask = (np.isfinite(depth)).astype(np.uint8) * 255
    Image.fromarray(mask, mode="L").save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="sessions/synthetic_demo")
    ap.add_argument("--scale", type=float, default=1.05, help="TRUE_SCALE (front*scale=metric)")
    ap.add_argument("--frames", type=int, default=8)
    args = ap.parse_args()

    TRUE_SCALE = float(args.scale)
    a = 1.0 / TRUE_SCALE  # metric -> front scale

    lay = SessionLayout(args.out)
    for d in (lay.capture_rgb, lay.capture_depth, lay.capture_confidence,
              lay.frontend_depth, lay.frontend):
        d.mkdir(parents=True, exist_ok=True)
    lay.frontend_colmap.parent.mkdir(parents=True, exist_ok=True)

    # metric world geometry (meters): a ~9 cm-radius sphere 0.45 m away
    center = np.array([0.0, 0.0, 0.0])
    P_m, cols = sphere_points(center, radius=0.09)

    # a fixed metric->front similarity (rotation R0, translation t0, scale a)
    ang = 0.6
    R0 = np.array([[np.cos(ang), 0, np.sin(ang)], [0, 1, 0], [-np.sin(ang), 0, np.cos(ang)]])
    t0 = np.array([0.2, -0.1, 0.3])
    P_f = (a * (R0 @ P_m.T).T) + t0

    # intrinsics: sensor/depth res (320x240); color res (640x480) = 2x
    depth_res = (320, 240)
    color_res = (640, 480)
    K_sensor = K_matrix(300.0, 300.0, 160.0, 120.0)      # at depth res
    K_color = K_matrix(600.0, 600.0, 320.0, 240.0)       # at color res (2x)
    front_res = (512, 384)
    K_front = K_matrix(480.0, 480.0, 256.0, 192.0)       # at front res

    cap_poses = {}
    cap_K = {}
    fe_poses = {}
    fe_K = {}
    dist = 0.45
    for i in range(args.frames):
        theta = 2 * np.pi * i / args.frames
        cam_pos = center + dist * np.array([np.sin(theta), 0.15, -np.cos(theta)])
        R_c2w, Cc = look_at_c2w(center, cam_pos)
        # metric camera (Stage 1)
        R_w2c, t_w2c = C.invert_pose(R_c2w, Cc)
        fid = frame_id(i + 1)
        # sensor depth (metric), analytic ray-sphere with sensor K
        dep_s = render_sphere_depth(center, 0.09, R_w2c, t_w2c, K_sensor, *depth_res)
        np.save(lay.capture_depth / f"{fid}.npy", dep_s)
        save_confidence(lay.capture_confidence / f"{fid}.png", dep_s)
        # a placeholder rgb (constant); Stage 3 does not read pixels
        from PIL import Image
        Image.fromarray(np.zeros((color_res[1], color_res[0], 3), np.uint8)).save(
            lay.capture_rgb / f"{fid}.png")
        cap_poses[fid] = {"R": R_c2w, "t": Cc}
        # per-frame device K: simulate ~1% autofocus breathing on the focal length across the orbit
        Kc = K_color.copy()
        breathe = 1.0 + 0.01 * np.sin(theta)
        Kc[0, 0] *= breathe
        Kc[1, 1] *= breathe
        cap_K[fid] = Kc

        # front-end camera: transform metric camera by the similarity
        C_f = a * (R0 @ Cc) + t0
        R_c2w_f = R0 @ R_c2w
        R_w2c_f, t_w2c_f = C.invert_pose(R_c2w_f, C_f)
        # the front-end "sphere" is the similarity image of the metric sphere:
        # center S(center)=a*R0@center+t0, radius a*0.09
        center_f = a * (R0 @ center) + t0
        dep_f = render_sphere_depth(center_f, a * 0.09, R_w2c_f, t_w2c_f, K_front, *front_res)
        np.save(lay.frontend_depth / f"{fid}.npy", dep_f)
        fe_poses[fid] = {"R": R_w2c_f, "t": t_w2c_f}
        fe_K[fid] = K_front

    # write capture/ metadata
    C.save_poses(lay.capture_poses, cap_poses, pose_type=C.CAMERA_TO_WORLD)
    with open(lay.capture_intrinsics, "w") as fh:
        json.dump({"convention": "OpenCV", "color_resolution": list(color_res),
                   "depth_resolution": list(depth_res),
                   "intrinsic_matrix_applies_to": "color",
                   "K": K_color.tolist(),
                   "K_per_frame": {fid: cap_K[fid].tolist() for fid in cap_K}}, fh, indent=2)
    with open(lay.capture_timestamps, "w") as fh:
        json.dump({"unit": "seconds",
                   "timestamps": {frame_id(i + 1): round(i * 0.1, 3) for i in range(args.frames)}}, fh, indent=2)
    lay.capture_readme.write_text(
        f"Synthetic session. Convention: OpenCV. color={color_res}, depth={depth_res}. "
        f"Pose stream present. TRUE_SCALE(front->metric)={TRUE_SCALE}.\n")

    # write frontend/ metadata
    C.save_poses(lay.frontend_poses, fe_poses, pose_type=C.WORLD_TO_CAMERA)
    with open(lay.frontend_intrinsics, "w") as fh:
        json.dump({"convention": "OpenCV", "resolution": list(front_res),
                   "K": {fid: fe_K[fid].tolist() for fid in fe_K}}, fh, indent=2)
    plyio.write_ply(lay.frontend_points, P_f, colors=cols, binary=True)

    print(f"Wrote synthetic session to {args.out}")
    print(f"  TRUE_SCALE (front->metric) = {TRUE_SCALE}  => Stage 3 should recover this")
    print(f"  frames={args.frames}, sphere r=0.09 m at {dist} m")
    return 0


if __name__ == "__main__":
    sys.exit(main())
