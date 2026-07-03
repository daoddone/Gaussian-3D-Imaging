"""Alignment primitives for Stage 3 (numpy-only, unit-testable).

Two standard tools, per METRIC_CONTRACT.md:
  * the Umeyama closed-form similarity fit (scale + rotation + translation)
    between two matched point sets, used for the camera-path anchor;
  * a robust 1-D scale/offset fit (RANSAC + least-squares refit) used for the
    depth anchor, which ignores outliers rather than being dragged by them.

Also: applying a world similarity transform to points and to world-to-camera
camera poses (scaling the world scales camera positions but must keep camera
orientation a pure rotation).
"""
from __future__ import annotations

import numpy as np


# --------------------------------------------------------------------------- #
# Umeyama similarity (Umeyama, 1991): find s, R, t minimising
#   sum_i || dst_i - (s R src_i + t) ||^2
# --------------------------------------------------------------------------- #
def umeyama(src, dst, with_scale=True):
    """Return (s, R, t) mapping src -> dst. src, dst are (N,3) matched points."""
    src = np.asarray(src, dtype=float)
    dst = np.asarray(dst, dtype=float)
    if src.shape != dst.shape or src.ndim != 2:
        raise ValueError("src and dst must be matched (N,D) arrays")
    n, dim = src.shape
    if n < dim:
        raise ValueError(f"need at least {dim} correspondences, got {n}")

    mu_src = src.mean(axis=0)
    mu_dst = dst.mean(axis=0)
    src_c = src - mu_src
    dst_c = dst - mu_dst

    cov = (dst_c.T @ src_c) / n
    U, Dvals, Vt = np.linalg.svd(cov)
    Smat = np.eye(dim)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        Smat[-1, -1] = -1.0
    R = U @ Smat @ Vt

    if with_scale:
        var_src = (src_c ** 2).sum() / n
        s = float(np.trace(np.diag(Dvals) @ Smat) / var_src) if var_src > 0 else 1.0
    else:
        s = 1.0
    t = mu_dst - s * R @ mu_src
    return s, R, t


def similarity_residual(src, dst, s, R, t):
    """Mean Euclidean distance after applying (s,R,t) to src, vs dst."""
    pred = (s * (R @ src.T)).T + t
    return float(np.linalg.norm(pred - dst, axis=1).mean())


# --------------------------------------------------------------------------- #
# Robust 1-D fit for the depth anchor:  sensor ~= scale * front (+ offset)
# --------------------------------------------------------------------------- #
def robust_depth_fit(front, sensor, fit_offset=False, ransac_iters=200,
                     inlier_tol=0.02, rng=None):
    """Robustly fit sensor ~= scale * front (+ offset) over paired depths.

    front, sensor : 1-D arrays of paired depth values (meters), same length.
    fit_offset    : if True also fit an additive offset b. NOTE: scale and
                    offset are degenerate over a narrow depth range, so leave
                    this off unless the scene spans a wide depth range; the
                    physical model of sensor-vs-front agreement is a pure scale.
    inlier_tol    : relative residual |r|/sensor below which a pair is an inlier.

    Method: initialise the scale from the robust MEDIAN of per-pixel ratios (a
    well-conditioned, outlier-resistant estimate), then run a few reweighted
    refinements on the inlier set. This avoids the tilt an unconstrained RANSAC
    line fit suffers on narrow-range, edge-noisy depth. ``ransac_iters``/``rng``
    are accepted for API stability but the median-initialised path is
    deterministic.

    Returns dict: scale, offset, inlier_fraction, residual_meters, n_used,
    n_inliers.
    """
    front = np.asarray(front, dtype=float).ravel()
    sensor = np.asarray(sensor, dtype=float).ravel()
    good = np.isfinite(front) & np.isfinite(sensor) & (front > 0) & (sensor > 0)
    front = front[good]
    sensor = sensor[good]
    n = front.size
    if n < 3:
        return {"scale": None, "offset": 0.0, "inlier_fraction": 0.0,
                "residual_meters": None, "n_used": int(n), "n_inliers": 0}

    def residual_rel(s, b):
        return np.abs(sensor - (s * front + b)) / np.maximum(sensor, 1e-6)

    s = float(np.median(sensor / front))   # robust scale initialisation
    b = 0.0
    inl = residual_rel(s, b) < inlier_tol
    for _ in range(3):                      # reweighted refinement on inliers
        if int(inl.sum()) < 3:
            break
        f = front[inl]
        se = sensor[inl]
        if fit_offset:
            # centre the regressor so scale and offset decorrelate numerically
            fmean = f.mean()
            A = np.stack([f - fmean, np.ones_like(f)], axis=1)
            sol, *_ = np.linalg.lstsq(A, se, rcond=None)
            s = float(sol[0])
            b = float(sol[1] - sol[0] * fmean)
        else:
            s = float((f * se).sum() / (f * f).sum())
            b = 0.0
        inl = residual_rel(s, b) < inlier_tol

    resid_m = float(np.abs(sensor[inl] - (s * front[inl] + b)).mean()) if inl.any() else None
    return {
        "scale": float(s),
        "offset": float(b),
        "inlier_fraction": float(inl.mean()),
        "residual_meters": resid_m,
        "n_used": int(n),
        "n_inliers": int(inl.sum()),
    }


# --------------------------------------------------------------------------- #
# Applying a world similarity transform S(x) = s R x + t
# --------------------------------------------------------------------------- #
def apply_similarity_points(points, s, R, t):
    """Apply S(x) = s R x + t to (N,3) points."""
    points = np.asarray(points, dtype=float)
    return (s * (R @ points.T)).T + np.asarray(t, dtype=float)


def apply_similarity_to_w2c_pose(R_wc, t_wc, s, R, t):
    """Update a world-to-camera pose when the WORLD is transformed by S=sRx+t.

    Camera center moves as C' = s R C + t; camera orientation rotates as
    R_wc' = R_wc R^T; both scene and camera frame end up metric, so the new
    extrinsic rotation stays a pure rotation (no scale leaks into it).
    """
    R_wc = np.asarray(R_wc, dtype=float)
    t_wc = np.asarray(t_wc, dtype=float).reshape(3)
    R = np.asarray(R, dtype=float)
    t = np.asarray(t, dtype=float).reshape(3)
    C = -R_wc.T @ t_wc                 # old camera center (front-end world)
    C_new = s * (R @ C) + t            # new camera center (metric world)
    R_wc_new = R_wc @ R.T
    t_wc_new = -R_wc_new @ C_new
    return R_wc_new, t_wc_new


def transform_centers(centers, s, R, t):
    """Convenience: apply S to a set of camera centers (N,3)."""
    return apply_similarity_points(centers, s, R, t)
