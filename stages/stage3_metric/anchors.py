"""The three physical anchors of Stage 3 (METRIC_CONTRACT.md, section "The three
physical anchors").

  1. Sensor depth  — the Stage 1 LiDAR depth on valid pixels; per-frame it is
     compared to the front end's own depth and a single robust scale is fit.
  2. Camera path   — the Stage 1 metric camera positions matched against the
     front end's estimated camera positions via a closed-form similarity fit;
     independent of the noisy depth.
  3. Physical ruler (optional) — an object of known size in frame; the only true
     physical ground truth. Usually absent.

Each anchor returns a small dict recording availability, its scale estimate, how
much data it used, and its own residual. Reading only the frozen file contract
keeps this module independent of which front end produced the reconstruction.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from common import conventions as C
from common.file_layout import SessionLayout

try:  # works both as a package import and as a direct-script sibling import
    from . import align
except ImportError:
    import align


# --------------------------------------------------------------------------- #
# small I/O helpers (PIL used only for the confidence PNG; kept out of common/)
# --------------------------------------------------------------------------- #
def _load_confidence(path):
    """Read a confidence/validity PNG -> boolean 'valid' mask (True where 255)."""
    from PIL import Image  # stage-local dependency, not in common/

    arr = np.asarray(Image.open(path))
    if arr.ndim == 3:
        arr = arr[..., 0]
    return arr >= 128


def _scale_K(K, from_res, to_res):
    """Scale a 3x3 intrinsic matrix from one resolution to another."""
    K = np.asarray(K, dtype=float).copy()
    sx = to_res[0] / from_res[0]
    sy = to_res[1] / from_res[1]
    K[0, 0] *= sx
    K[0, 2] *= sx
    K[1, 1] *= sy
    K[1, 2] *= sy
    return K


def load_capture_intrinsics(path):
    """Return dict with K_color, color_res (w,h), depth_res (w,h), K_sensor."""
    with open(path) as fh:
        d = json.load(fh)
    K_color = np.asarray(d["K"], dtype=float)
    color_res = tuple(d["color_resolution"])          # (w, h)
    depth_res = tuple(d.get("depth_resolution", color_res))
    applies_to = d.get("intrinsic_matrix_applies_to", "color")
    if applies_to == "color":
        K_sensor = _scale_K(K_color, color_res, depth_res)
    else:  # matrix already at depth resolution
        K_sensor = K_color.copy()
    return {"K_color": K_color, "color_res": color_res,
            "depth_res": depth_res, "K_sensor": K_sensor}


def load_frontend_intrinsics(path):
    """Return ({frame_id: K 3x3}, resolution (w,h))."""
    with open(path) as fh:
        d = json.load(fh)
    res = tuple(d["resolution"])
    Ks = {str(fid): np.asarray(K, dtype=float) for fid, K in d["K"].items()}
    return Ks, res


def load_capture_per_frame_intrinsics(path):
    """Return ({frame_id: K_color 3x3}, color_res (w,h)) from capture/intrinsics.json's
    optional 'K_per_frame' map: the TRUE device intrinsics per frame (they track focus
    breathing / OIS, unlike the single first-frame K). Returns (None, None) when the field
    is absent — i.e. legacy single-K captures, so callers fall back to the old behaviour."""
    with open(path) as fh:
        d = json.load(fh)
    per = d.get("K_per_frame")
    if not per:
        return None, None
    color_res = tuple(d["color_resolution"])
    Ks = {str(fid): np.asarray(K, dtype=float) for fid, K in per.items()}
    return Ks, color_res


# --------------------------------------------------------------------------- #
# per-pixel resampling of front-end depth onto the sensor grid
# --------------------------------------------------------------------------- #
def _bilinear_sample(img, u, v):
    """Bilinear sample img[H,W] at float coords (u,v); NaN outside the image."""
    H, W = img.shape
    out = np.full(u.shape, np.nan, dtype=float)
    u0 = np.floor(u).astype(int)
    v0 = np.floor(v).astype(int)
    u1 = u0 + 1
    v1 = v0 + 1
    valid = (u0 >= 0) & (v0 >= 0) & (u1 < W) & (v1 < H)
    if not np.any(valid):
        return out
    uu = u[valid]; vv = v[valid]
    a = uu - u0[valid]; b = vv - v0[valid]
    i00 = img[v0[valid], u0[valid]]
    i01 = img[v0[valid], u1[valid]]
    i10 = img[v1[valid], u0[valid]]
    i11 = img[v1[valid], u1[valid]]
    out[valid] = (i00 * (1 - a) * (1 - b) + i01 * a * (1 - b)
                  + i10 * (1 - a) * b + i11 * a * b)
    return out


def sample_front_at_sensor(front_depth, K_front, sensor_shape, K_sensor):
    """Sample the front-end depth map at each sensor-depth pixel location.

    Both depth maps describe the SAME physical camera per frame (LiDAR fused
    with the color camera), so a sensor pixel maps to a front-end pixel through
    normalized image coordinates: no reprojection or pose is needed. Returns an
    array of front-end depth on the sensor grid (NaN where out of bounds).
    """
    Hs, Ws = sensor_shape
    vs, us = np.mgrid[0:Hs, 0:Ws].astype(float)
    x = (us - K_sensor[0, 2]) / K_sensor[0, 0]
    y = (vs - K_sensor[1, 2]) / K_sensor[1, 1]
    uf = K_front[0, 0] * x + K_front[0, 2]
    vf = K_front[1, 1] * y + K_front[1, 2]
    return _bilinear_sample(front_depth, uf, vf)


# --------------------------------------------------------------------------- #
# anchor 1: sensor depth
# --------------------------------------------------------------------------- #
def depth_anchor(layout: SessionLayout, cfg, max_pairs=300000):
    """Fit a single robust scale from front-end depth to sensor depth."""
    out = {"available": False, "scale_estimate": None, "points_used": 0,
           "inlier_fraction": 0.0, "residual_meters": None, "frames_used": 0,
           "per_frame_residual_meters": {}}

    cap_intr_path = layout.capture_intrinsics
    fe_intr_path = layout.frontend_intrinsics
    if not (cap_intr_path.exists() and fe_intr_path.exists()):
        out["note"] = "missing intrinsics"
        return out

    cap_intr = load_capture_intrinsics(cap_intr_path)
    fe_Ks, fe_res = load_frontend_intrinsics(fe_intr_path)

    sensor_frames = set(SessionLayout.list_frames(layout.capture_depth, ".npy"))
    front_frames = set(SessionLayout.list_frames(layout.frontend_depth, ".npy"))
    common = sorted(sensor_frames & front_frames)
    if not common:
        out["note"] = "no overlapping depth frames between capture/ and frontend/"
        return out

    d_cfg = cfg["stage3"]["depth"]
    dmin = float(d_cfg["min_valid_depth_m"])
    dmax = float(d_cfg["max_valid_depth_m"])

    all_front = []
    all_sensor = []
    frame_pairs = {}  # fid -> (front_arr, sensor_arr) for per-frame residual
    frames_used = 0
    for fid in common:
        sensor = np.load(layout.capture_depth / f"{fid}.npy").astype(float)
        front = np.load(layout.frontend_depth / f"{fid}.npy").astype(float)
        conf_path = layout.capture_confidence / f"{fid}.png"
        valid = np.isfinite(sensor) & (sensor > dmin) & (sensor < dmax)
        if conf_path.exists():
            valid &= _load_confidence(conf_path)
        if not np.any(valid):
            continue
        K_front = fe_Ks.get(fid, next(iter(fe_Ks.values())))
        front_on_sensor = sample_front_at_sensor(front, K_front, sensor.shape, cap_intr["K_sensor"])
        pair_valid = valid & np.isfinite(front_on_sensor) & (front_on_sensor > dmin) & (front_on_sensor < dmax)
        if not np.any(pair_valid):
            continue
        f = front_on_sensor[pair_valid]
        s = sensor[pair_valid]
        all_front.append(f)
        all_sensor.append(s)
        frame_pairs[fid] = (f, s)
        frames_used += 1

    if frames_used == 0:
        out["note"] = "no valid depth pairs"
        return out

    front_all = np.concatenate(all_front)
    sensor_all = np.concatenate(all_sensor)

    # cap total pairs for the global fit (reproducible subsample)
    rng = np.random.default_rng(2026)
    if front_all.size > max_pairs:
        sel = rng.choice(front_all.size, size=max_pairs, replace=False)
        front_fit = front_all[sel]
        sensor_fit = sensor_all[sel]
    else:
        front_fit = front_all
        sensor_fit = sensor_all

    fit = align.robust_depth_fit(
        front_fit, sensor_fit,
        fit_offset=bool(d_cfg["fit_offset"]),
        ransac_iters=int(d_cfg["ransac_iterations"]),
        inlier_tol=float(d_cfg["ransac_inlier_tol"]),
        rng=rng,
    )
    if fit["scale"] is None:
        out["note"] = "robust fit failed"
        return out

    s = fit["scale"]
    b = fit["offset"]
    # per-frame residual after the global fit (Experiment A drift check)
    for fid, (f, se) in frame_pairs.items():
        out["per_frame_residual_meters"][fid] = float(np.abs(se - (s * f + b)).mean())

    out.update({
        "available": True,
        "scale_estimate": s,
        "offset_meters": b,
        "points_used": int(front_all.size),
        "inlier_fraction": fit["inlier_fraction"],
        "residual_meters": fit["residual_meters"],
        "frames_used": frames_used,
    })
    return out


# --------------------------------------------------------------------------- #
# anchor 2: camera path
# --------------------------------------------------------------------------- #
def camera_path_anchor(layout: SessionLayout, cfg):
    """Recover scale (+ rotation/translation) by matching camera positions.

    Fits the closed-form similarity between the front end's camera centers
    (frontend/poses.json) and Stage 1's metric camera centers (capture/poses.json)
    over the common frames. Returns the similarity so Stage 3 can place the
    reconstruction in the metric world frame.
    """
    out = {"available": False, "scale_estimate": None, "frames_used": 0,
           "residual_meters": None, "R": None, "t": None}

    if not (layout.frontend_poses.exists() and layout.capture_poses.exists()):
        out["note"] = "camera-path anchor unavailable (missing frontend or capture poses)"
        return out

    fe = C.load_poses(layout.frontend_poses)
    cap = C.load_poses(layout.capture_poses)
    fe_ids, fe_centers = C.camera_centers(fe)
    cap_ids, cap_centers = C.camera_centers(cap)

    fe_map = dict(zip(fe_ids, fe_centers))
    cap_map = dict(zip(cap_ids, cap_centers))
    common = sorted(set(fe_map) & set(cap_map))
    if len(common) < 3:
        out["note"] = f"only {len(common)} common frames (need >= 3)"
        return out

    src = np.array([fe_map[f] for f in common])   # front-end centers
    dst = np.array([cap_map[f] for f in common])  # metric centers
    s, R, t = align.umeyama(src, dst, with_scale=True)
    resid = align.similarity_residual(src, dst, s, R, t)

    out.update({
        "available": True,
        "scale_estimate": float(s),
        "frames_used": len(common),
        "residual_meters": float(resid),
        "R": R.tolist(),
        "t": t.tolist(),
        # source (front-end) centroid, so a caller can recompute t for a
        # different applied scale while keeping the centroids aligned.
        "src_centroid": src.mean(axis=0).tolist(),
    })
    return out


# --------------------------------------------------------------------------- #
# anchor 3: physical ruler (optional, usually absent)
# --------------------------------------------------------------------------- #
def ruler_anchor(layout: SessionLayout, cfg):
    """Physical object of known size. Requires out-of-band measured size, so it
    is unavailable unless both known and measured sizes are provided."""
    out = {"available": False, "scale_estimate": None,
           "known_size_meters": None, "measured_size_meters": None}
    ruler_cfg = cfg["stage3"].get("ruler", {})
    known = ruler_cfg.get("known_size_meters")
    measured = ruler_cfg.get("measured_size_meters")
    if known and measured:
        out.update({
            "available": True,
            "known_size_meters": float(known),
            "measured_size_meters": float(measured),
            "scale_estimate": float(known) / float(measured),
        })
    return out
