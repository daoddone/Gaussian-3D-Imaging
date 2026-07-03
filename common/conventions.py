"""Coordinate-convention helpers and conversions.

The pipeline uses the computer-vision (OpenCV) convention **everywhere**: the
camera looks down its +z axis, x points right, y points down. Depth increases
away from the camera along +z. See Section 5 of the build specification.

Camera poses are always stored with an explicit ``pose_type`` that is one of
``"world_to_camera"`` or ``"camera_to_world"`` (see ``io_contracts/``), so a
reader is never left guessing. Normalise at a boundary with
:func:`to_world_to_camera` / :func:`to_camera_to_world`, and get the metric
camera position regardless of storage with :func:`camera_center`.

Definitions used throughout:
    world_to_camera:  X_cam   = R @ X_world + t
    camera_to_world:  X_world = R @ X_cam   + t
The two are exact inverses of one another.

This module is numpy-only by the dependency-light rule.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

WORLD_TO_CAMERA = "world_to_camera"
CAMERA_TO_WORLD = "camera_to_world"
OPENCV = "OpenCV"

# Flip that turns an OpenGL-style camera frame (x right, y up, looks down -z,
# used by Apple's ARKit) into the OpenCV camera frame (x right, y down, +z).
# Equivalent to a 180-degree rotation about the camera's own x axis.
_GL_TO_CV = np.diag([1.0, -1.0, -1.0])


# --------------------------------------------------------------------------- #
# Rotation <-> quaternion (COLMAP quaternion order is w, x, y, z)
# --------------------------------------------------------------------------- #
def quat_to_rotmat(q) -> np.ndarray:
    """Quaternion (w, x, y, z) -> 3x3 rotation matrix."""
    w, x, y, z = (float(v) for v in q)
    n = w * w + x * x + y * y + z * z
    if n < 1e-12:
        return np.eye(3)
    s = 2.0 / n
    wx, wy, wz = s * w * x, s * w * y, s * w * z
    xx, xy, xz = s * x * x, s * x * y, s * x * z
    yy, yz, zz = s * y * y, s * y * z, s * z * z
    return np.array(
        [
            [1.0 - (yy + zz), xy - wz, xz + wy],
            [xy + wz, 1.0 - (xx + zz), yz - wx],
            [xz - wy, yz + wx, 1.0 - (xx + yy)],
        ]
    )


def rotmat_to_quat(R) -> np.ndarray:
    """3x3 rotation matrix -> quaternion (w, x, y, z), with w >= 0.

    Uses the branch-stable Shepperd method. Matches COLMAP's (w, x, y, z) order.
    """
    R = np.asarray(R, dtype=float)
    m00, m01, m02 = R[0]
    m10, m11, m12 = R[1]
    m20, m21, m22 = R[2]
    tr = m00 + m11 + m22
    if tr > 0.0:
        S = np.sqrt(tr + 1.0) * 2.0
        w = 0.25 * S
        x = (m21 - m12) / S
        y = (m02 - m20) / S
        z = (m10 - m01) / S
    elif m00 > m11 and m00 > m22:
        S = np.sqrt(1.0 + m00 - m11 - m22) * 2.0
        w = (m21 - m12) / S
        x = 0.25 * S
        y = (m01 + m10) / S
        z = (m02 + m20) / S
    elif m11 > m22:
        S = np.sqrt(1.0 + m11 - m00 - m22) * 2.0
        w = (m02 - m20) / S
        x = (m01 + m10) / S
        y = 0.25 * S
        z = (m12 + m21) / S
    else:
        S = np.sqrt(1.0 + m22 - m00 - m11) * 2.0
        w = (m10 - m01) / S
        x = (m02 + m20) / S
        y = (m12 + m21) / S
        z = 0.25 * S
    q = np.array([w, x, y, z], dtype=float)
    if q[0] < 0:
        q = -q
    return q / np.linalg.norm(q)


# --------------------------------------------------------------------------- #
# Pose-type normalisation
# --------------------------------------------------------------------------- #
def invert_pose(R, t):
    """Invert a rigid transform (R, t). world<->camera are inverses."""
    R = np.asarray(R, dtype=float)
    t = np.asarray(t, dtype=float).reshape(3)
    Rin = R.T
    return Rin, -Rin @ t


def to_world_to_camera(R, t, pose_type):
    """Return (R, t) as world_to_camera regardless of the input pose_type."""
    R = np.asarray(R, dtype=float)
    t = np.asarray(t, dtype=float).reshape(3)
    if pose_type == WORLD_TO_CAMERA:
        return R, t
    if pose_type == CAMERA_TO_WORLD:
        return invert_pose(R, t)
    raise ValueError(f"unknown pose_type: {pose_type!r}")


def to_camera_to_world(R, t, pose_type):
    """Return (R, t) as camera_to_world regardless of the input pose_type."""
    R = np.asarray(R, dtype=float)
    t = np.asarray(t, dtype=float).reshape(3)
    if pose_type == CAMERA_TO_WORLD:
        return R, t
    if pose_type == WORLD_TO_CAMERA:
        return invert_pose(R, t)
    raise ValueError(f"unknown pose_type: {pose_type!r}")


def camera_center(R, t, pose_type) -> np.ndarray:
    """Metric camera position in world coordinates, for either pose_type.

    world_to_camera: C = -R^T t.   camera_to_world: C = t.
    """
    R = np.asarray(R, dtype=float)
    t = np.asarray(t, dtype=float).reshape(3)
    if pose_type == WORLD_TO_CAMERA:
        return -R.T @ t
    if pose_type == CAMERA_TO_WORLD:
        return t.copy()
    raise ValueError(f"unknown pose_type: {pose_type!r}")


# --------------------------------------------------------------------------- #
# Apple -> OpenCV pose conversion (used by the Stage 1 offline alignment)
# --------------------------------------------------------------------------- #
def opencv_c2w_from_arkit(transform4x4) -> tuple:
    """Convert an ARKit ``ARCamera.transform`` to an OpenCV camera_to_world pose.

    ARKit reports a 4x4 camera-to-world matrix where the camera looks down its
    -z axis with +y up (the OpenGL/graphics convention). Flipping the camera's
    y and z axes (a 180-degree rotation about camera x) yields the OpenCV
    convention. Camera position is unchanged.

    Returns (R, t) as an OpenCV camera_to_world pose.
    """
    T = np.asarray(transform4x4, dtype=float).reshape(4, 4)
    R = T[:3, :3] @ _GL_TO_CV
    t = T[:3, 3]
    return R, t


# --------------------------------------------------------------------------- #
# poses.json I/O (schema defined in io_contracts/*_output.md)
# --------------------------------------------------------------------------- #
def load_poses(path):
    """Read a poses.json file.

    Returns a dict with keys:
        ``convention`` (str), ``pose_type`` (str),
        ``poses`` -> {frame_id (str): {"R": 3x3 ndarray, "t": (3,) ndarray}}.
    """
    path = Path(path)
    with open(path, "r") as fh:
        raw = json.load(fh)
    pose_type = raw.get("pose_type")
    if pose_type not in (WORLD_TO_CAMERA, CAMERA_TO_WORLD):
        raise ValueError(f"{path}: missing or invalid pose_type {pose_type!r}")
    poses = {}
    for fid, p in raw["poses"].items():
        R = np.asarray(p["R"], dtype=float).reshape(3, 3)
        t = np.asarray(p["t"], dtype=float).reshape(3)
        poses[str(fid)] = {"R": R, "t": t}
    return {
        "convention": raw.get("convention", OPENCV),
        "pose_type": pose_type,
        "poses": poses,
    }


def save_poses(path, poses, pose_type, convention=OPENCV):
    """Write a poses.json file. ``poses`` maps frame_id -> {"R","t"}."""
    out = {"convention": convention, "pose_type": pose_type, "poses": {}}
    for fid, p in sorted(poses.items()):
        R = np.asarray(p["R"], dtype=float).reshape(3, 3)
        t = np.asarray(p["t"], dtype=float).reshape(3)
        out["poses"][str(fid)] = {"R": R.tolist(), "t": t.tolist()}
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2)


def camera_centers(loaded_poses):
    """Given the dict from :func:`load_poses`, return (frame_ids, Nx3 centers).

    frame_ids are sorted; the i-th center corresponds to the i-th id.
    """
    pose_type = loaded_poses["pose_type"]
    fids = sorted(loaded_poses["poses"].keys())
    centers = np.zeros((len(fids), 3), dtype=float)
    for i, fid in enumerate(fids):
        p = loaded_poses["poses"][fid]
        centers[i] = camera_center(p["R"], p["t"], pose_type)
    return fids, centers
