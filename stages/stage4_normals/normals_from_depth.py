"""Compute surface normals directly from a metric depth map (no learned model).

This is the most bias-free normal source (Stage 4 swap candidate) and condition
(2) of Section 8 Experiment B. Normals are expressed in the camera frame in the
OpenCV convention and oriented to face the camera (n_z < 0), matching the Stage 4
output contract (`io_contracts/normals_output.md`).

Pure numpy.
"""
from __future__ import annotations

import numpy as np


def normals_from_depth(depth, K, valid=None):
    """depth [H,W] meters (NaN/invalid allowed), K 3x3 -> normals [H,W,3].

    Back-projects each pixel to a 3-D point, estimates the surface normal as the
    normalized cross product of the local surface tangents, and orients it to
    face the camera. Invalid/edge pixels get a zero vector.
    """
    depth = np.asarray(depth, dtype=float)
    H, W = depth.shape
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

    if valid is None:
        valid = np.isfinite(depth) & (depth > 0)
    z = np.where(valid, depth, np.nan)

    us, vs = np.meshgrid(np.arange(W), np.arange(H))
    X = (us - cx) / fx * z
    Y = (vs - cy) / fy * z
    P = np.stack([X, Y, z], axis=2)  # [H,W,3]

    # tangents along image axes
    dPdx = np.full_like(P, np.nan)
    dPdy = np.full_like(P, np.nan)
    dPdx[:, 1:-1, :] = (P[:, 2:, :] - P[:, :-2, :]) * 0.5
    dPdy[1:-1, :, :] = (P[2:, :, :] - P[:-2, :, :]) * 0.5

    n = np.cross(dPdx, dPdy)
    norm = np.linalg.norm(n, axis=2, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        n_unit = n / norm

    # orient to face the camera: in the OpenCV camera frame the camera looks
    # down +z, so a surface facing it should have n_z < 0.
    flip = n_unit[..., 2] > 0
    n_unit[flip] *= -1.0

    out = np.zeros((H, W, 3), dtype=np.float32)
    good = np.isfinite(n_unit).all(axis=2) & (norm[..., 0] > 0)
    out[good] = n_unit[good].astype(np.float32)
    return out


def angular_error_degrees(normals_a, normals_b, valid=None):
    """Mean angular error (degrees) between two normal maps over valid pixels."""
    a = np.asarray(normals_a, dtype=float)
    b = np.asarray(normals_b, dtype=float)
    na = np.linalg.norm(a, axis=2)
    nb = np.linalg.norm(b, axis=2)
    m = (na > 1e-6) & (nb > 1e-6)
    if valid is not None:
        m &= valid
    if not np.any(m):
        return float("nan"), 0
    dots = np.einsum("ijk,ijk->ij", a, b)[m] / (na[m] * nb[m])
    dots = np.clip(dots, -1.0, 1.0)
    ang = np.degrees(np.arccos(dots))
    return float(ang.mean()), int(m.sum())
