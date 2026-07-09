#!/usr/bin/env python3
"""Subject-isolation stage 1: generate per-frame subject masks by projecting a robust 3D subject box.

Design (docs/EXPERIMENTS_BACKLOG.md implementation notes): MILo densification is GRADIENT-driven, so
isolation must remove the background from the PHOTOMETRIC loss (stage 2 wires the mask into MILo);
these masks are that signal. Subject detection is geometry-only (no segmentation model):

  1. SUBJECT CENTER c* = least-squares intersection of the camera optical axes
     (argmin_c sum_i ||(I - d_i d_i^T)(c - o_i)||^2) — the orbit converges on the subject, so c* is
     background-independent (works for standoff captures where camera-hull culling fails).
  2. SUBJECT BOX = percentile box (with generous pad; owner prefers under-cropping) of the metric
     init points within a radius of c* (cluster crop rejects far background).
  3. MASK_i = filled convex hull of the box's 8 corners projected into frame i (+ pixel margin).

Outputs <session>/subject_masks/{fid}.png (255=subject region, 0=background) at color resolution,
plus subject_masks/box.json (c*, box, params) for the box-prune in stage 2.

Usage (any env with numpy+cv2, e.g. gs-ba):
  python scripts/make_subject_masks.py --session sessions/session_20260704_143324 [--pad 0.15]
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path("/home/paperspace/Documents/VS Code Projects/3D-Gaussian")
sys.path.insert(0, str(REPO))
from common import colmap_io  # noqa: E402
from common.conventions import quat_to_rotmat  # noqa: E402


def optical_axis_center(Rs, ts):
    """Least-squares point nearest all camera optical axes (rays o_i + s*d_i)."""
    A = np.zeros((3, 3))
    b = np.zeros(3)
    for R, t in zip(Rs, ts):
        o = -R.T @ t                      # camera center (world)
        d = R.T @ np.array([0.0, 0.0, 1.0])  # +z optical axis (world, OpenCV w2c convention)
        d = d / np.linalg.norm(d)
        P = np.eye(3) - np.outer(d, d)
        A += P
        b += P @ o
    return np.linalg.solve(A, b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", required=True)
    ap.add_argument("--pad", type=float, default=0.15, help="box pad fraction (generous = under-crop)")
    ap.add_argument("--radius-m", type=float, default=0.40,
                    help="ABSOLUTE metric radius (m) around c* for the subject cluster — valid because "
                         "the model is metric-locked; clinical subjects fit ~0.4 m (feet ~0.3 m)")
    ap.add_argument("--percentile", type=float, default=1.0, help="box percentile trim per side")
    args = ap.parse_args()

    sess = REPO / args.session if not Path(args.session).is_absolute() else Path(args.session)
    sparse = sess / "metric" / "colmap" / "sparse" / "0"
    imgs = colmap_io.read_images_binary(sparse / "images.bin")
    cams = colmap_io.read_cameras_binary(sparse / "cameras.bin")
    pts = colmap_io.read_points3D_binary(sparse / "points3D.bin")
    xyz = np.array([p["xyz"] for p in pts.values()], dtype=float)
    print(f"[masks] {len(imgs)} frames, {len(xyz)} metric init points")

    Rs, ts = [], []
    for im in imgs.values():
        Rs.append(quat_to_rotmat(im["qvec"]))
        ts.append(np.asarray(im["tvec"], float))
    c = optical_axis_center(Rs, ts)
    print(f"[masks] optical-axis center c* = {np.round(c, 4).tolist()}")

    # ABSOLUTE metric cluster crop around c* (init points are majority background — floor/stand —
    # so relative crops keep a scene-sized box; the metric lock makes a physical radius valid)
    d = np.linalg.norm(xyz - c, axis=1)
    keep = d < args.radius_m
    if keep.sum() < 200:
        print(f"[masks] WARNING: only {keep.sum()} pts within {args.radius_m} m of c* — widening to 0.6 m")
        keep = d < 0.6
    cl = xyz[keep]
    lo = np.percentile(cl, args.percentile, axis=0)
    hi = np.percentile(cl, 100 - args.percentile, axis=0)
    pad = args.pad * (hi - lo)
    lo, hi = lo - pad, hi + pad
    print(f"[masks] cluster {keep.sum()}/{len(xyz)} pts; subject box (m): "
          f"{np.round(hi - lo, 3).tolist()} at [{np.round(lo, 3).tolist()} .. {np.round(hi, 3).tolist()}]")

    # NOTE: box-CORNER projection fails close-in (cameras sit nearly inside the padded box -> the
    # near-face corners project outside the frame -> full-frame hulls). Project the CLUSTER POINTS
    # instead: a compact in-frame silhouette; dilate for the pad. The 3D box (box.json) still drives
    # the stage-2 gaussian box-prune.
    out_dir = sess / "subject_masks"
    out_dir.mkdir(exist_ok=True)

    coverage = []
    for im in imgs.values():
        fid = Path(im["name"]).stem
        cam = cams[im["camera_id"]]
        fx, fy, cx, cy = cam["params"][:4]
        W, H = int(cam["width"]), int(cam["height"])
        R = quat_to_rotmat(im["qvec"])
        t = np.asarray(im["tvec"], float)
        pc = (R @ cl.T).T + t                      # cluster points, camera frame
        front = pc[:, 2] > 0.05
        mask = np.zeros((H, W), np.uint8)
        if front.sum() >= 10:
            z = pc[front, 2]
            u = fx * pc[front, 0] / z + cx
            v = fy * pc[front, 1] / z + cy
            inb = (u > -0.2 * W) & (u < 1.2 * W) & (v > -0.2 * H) & (v < 1.2 * H)
            pix = np.stack([np.clip(u[inb], 0, W - 1), np.clip(v[inb], 0, H - 1)], 1).astype(np.int32)
            if len(pix) >= 10:
                hull = cv2.convexHull(pix.reshape(-1, 1, 2))
                cv2.fillConvexPoly(mask, hull, 255)
                k = max(3, int(0.04 * max(W, H)))   # ~4% dilation = the generous pad in 2D
                mask = cv2.dilate(mask, np.ones((k, k), np.uint8))
        if mask.max() == 0:
            mask[:] = 255                            # degenerate view -> fail OPEN (supervise everything)
        cv2.imwrite(str(out_dir / f"{fid}.png"), mask)
        coverage.append(float((mask > 0).mean()))

    cov = np.array(coverage)
    print(f"[masks] wrote {len(coverage)} masks -> {out_dir}")
    print(f"[masks] subject-region coverage: min {cov.min():.2f} median {np.median(cov):.2f} max {cov.max():.2f} "
          f"(1.0 = whole frame; very low values risk over-cropping)")
    (out_dir / "box.json").write_text(json.dumps({
        "center_optical_axes": c.tolist(), "box_lo": lo.tolist(), "box_hi": hi.tolist(),
        "pad_frac": args.pad, "radius_m": args.radius_m,
        "coverage_median": float(np.median(cov)),
        "note": "masks: 255=subject (photometric supervised), 0=background (masked out in stage-2 loss)",
    }, indent=2))
    print(f"[masks] wrote box.json")


if __name__ == "__main__":
    main()
