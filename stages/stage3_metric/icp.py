"""Optional Iterative Closest Point (ICP) refinement (METRIC_CONTRACT step 7).

After the similarity transform places the reconstruction in the metric world,
ICP can sharpen the alignment between the scaled reconstruction and the point
cloud formed by back-projecting the Stage 1 sensor depth. ICP is rigid (no
scale), so it only nudges rotation/translation; the recovered metric scale is
untouched.

Requires Open3D and the Stage 1 camera path (to back-project sensor depth into
a consistent world). If either is missing, refinement is skipped and the caller
proceeds with the similarity-only alignment.
"""
from __future__ import annotations

import numpy as np

from common import conventions as C
from common.file_layout import SessionLayout

try:
    from . import anchors
except ImportError:
    import anchors


def open3d_available():
    try:
        import open3d  # noqa: F401
        return True
    except Exception:
        return False


def backproject_sensor_cloud(layout: SessionLayout, cfg, max_points=200000,
                             max_frames=20):
    """Back-project valid Stage 1 sensor depth into the metric world frame.

    Uses the capture (metric) poses, which must be camera_to_world in the
    OpenCV convention. Returns (N,3) world points, or None if unavailable.
    """
    if not (layout.capture_poses.exists() and layout.capture_intrinsics.exists()):
        return None
    cap = C.load_poses(layout.capture_poses)
    cap_intr = anchors.load_capture_intrinsics(layout.capture_intrinsics)
    K = cap_intr["K_sensor"]
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

    d_cfg = cfg["stage3"]["depth"]
    dmin = float(d_cfg["min_valid_depth_m"])
    dmax = float(d_cfg["max_valid_depth_m"])

    frames = SessionLayout.list_frames(layout.capture_depth, ".npy")
    if not frames:
        return None
    if len(frames) > max_frames:
        idx = np.linspace(0, len(frames) - 1, max_frames).astype(int)
        frames = [frames[i] for i in idx]

    chunks = []
    for fid in frames:
        if fid not in cap["poses"]:
            continue
        depth = np.load(layout.capture_depth / f"{fid}.npy").astype(float)
        valid = np.isfinite(depth) & (depth > dmin) & (depth < dmax)
        conf = layout.capture_confidence / f"{fid}.png"
        if conf.exists():
            valid &= anchors._load_confidence(conf)
        if not np.any(valid):
            continue
        vs, us = np.where(valid)
        z = depth[vs, us]
        x = (us - cx) / fx * z
        y = (vs - cy) / fy * z
        pts_cam = np.stack([x, y, z], axis=1)
        R = cap["poses"][fid]["R"]
        t = cap["poses"][fid]["t"]
        Rc2w, tc2w = C.to_camera_to_world(R, t, cap["pose_type"])
        pts_world = (Rc2w @ pts_cam.T).T + tc2w
        chunks.append(pts_world)

    if not chunks:
        return None
    cloud = np.concatenate(chunks, axis=0)
    if cloud.shape[0] > max_points:
        rng = np.random.default_rng(7)
        sel = rng.choice(cloud.shape[0], size=max_points, replace=False)
        cloud = cloud[sel]
    return cloud


def refine(source_points, target_points, max_corr_dist, max_iter):
    """Point-to-point ICP aligning source to target. Returns (R, t, info)."""
    import open3d as o3d

    src = o3d.geometry.PointCloud()
    src.points = o3d.utility.Vector3dVector(np.asarray(source_points, dtype=float))
    tgt = o3d.geometry.PointCloud()
    tgt.points = o3d.utility.Vector3dVector(np.asarray(target_points, dtype=float))

    result = o3d.pipelines.registration.registration_icp(
        src, tgt, float(max_corr_dist), np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=int(max_iter)),
    )
    T = np.asarray(result.transformation)
    R = T[:3, :3]
    t = T[:3, 3]
    info = {"fitness": float(result.fitness), "inlier_rmse": float(result.inlier_rmse)}
    return R, t, info
