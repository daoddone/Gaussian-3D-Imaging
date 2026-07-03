#!/usr/bin/env python3
"""Stage 3 entry point: metric alignment and validation.

Standard signature (Section 3.5):
    python stages/stage3_metric/run.py --session <session_dir> --config <pipeline.yaml>

Reads the Stage 2 front-end outputs plus the Stage 1 sensor depth/confidence/
camera path, recovers the true metric scale from up to three independent
physical anchors, applies the chosen similarity transform to the point cloud and
camera poses, and writes:

    metric/points_metric.ply
    metric/colmap/sparse/0/{cameras,images,points3D}.bin
    metric/scale_report.json

Exit codes: 0 on pass; 3 when the session is flagged AND
config stage3.flag_halts_pipeline is true (the coordinator halts and reports).
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

# --- bootstrap imports so this works as a script or as a package -----------
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

import align
import anchors
import icp as icp_mod
import report as report_mod


def _load_config(path):
    with open(path) as fh:
        return yaml.safe_load(fh)


def _build_similarity(applied_scale, camera_anchor):
    """World similarity S(x)=s R x + t placing the reconstruction in the metric
    world. Rotation/translation come from the camera-path anchor when available
    (to align to the Stage 1 metric frame); scale is the decided applied_scale.

    The camera-path Umeyama fit solved (s_cam, R, t_cam) with
    t_cam = mu_dst - s_cam R mu_src. When the applied consensus scale differs
    from s_cam, recompute t so the centroids stay aligned at the applied scale:
    t = t_cam + (s_cam - s_applied) R mu_src. Depth-only fallback: scale about
    the origin, identity rotation."""
    s = float(applied_scale)
    if camera_anchor.get("available") and camera_anchor.get("R") is not None:
        R = np.asarray(camera_anchor["R"], dtype=float)
        t = np.asarray(camera_anchor["t"], dtype=float)
        s_cam = float(camera_anchor["scale_estimate"])
        mu_src = np.asarray(camera_anchor.get("src_centroid", np.zeros(3)), dtype=float)
        t = t + (s_cam - s) * (R @ mu_src)
    else:
        R = np.eye(3)
        t = np.zeros(3)
    return s, R, t


def _transform_pointcloud(layout, s, R, t):
    """Apply the similarity to frontend/points.ply -> metric/points_metric.ply.

    Returns (info, metric_points, metric_colors) so the metric cloud can also be
    baked into the COLMAP points3D.bin.
    """
    src = layout.frontend_points
    if not src.exists():
        return {"written": False, "note": "frontend/points.ply missing"}, None, None
    cloud = plyio.read_ply(src)
    pts = align.apply_similarity_points(cloud["points"], s, R, t)
    colors = cloud.get("colors")
    normals = cloud.get("normals")
    if normals is not None:  # rotate normals only, then renormalise
        n = (R @ np.asarray(normals, dtype=float).T).T
        norm = np.linalg.norm(n, axis=1, keepdims=True)
        normals = n / np.maximum(norm, 1e-9)
    layout.metric.mkdir(parents=True, exist_ok=True)
    plyio.write_ply(layout.metric_points, pts, colors=colors, normals=normals, binary=True)
    return {"written": True, "num_points": int(pts.shape[0])}, pts, colors


def _bake_points3D(metric_points, metric_colors, max_points):
    """Subsample the metric cloud into a COLMAP points3D dict (empty tracks)."""
    if metric_points is None or len(metric_points) == 0:
        return {}
    pts = np.asarray(metric_points, dtype=float)
    n = pts.shape[0]
    if n > max_points:
        rng = np.random.default_rng(99)
        sel = rng.choice(n, size=max_points, replace=False)
        pts = pts[sel]
        cols = None if metric_colors is None else np.asarray(metric_colors)[sel]
    else:
        cols = None if metric_colors is None else np.asarray(metric_colors)
    points3D = {}
    for i in range(pts.shape[0]):
        rgb = (128, 128, 128) if cols is None else tuple(int(c) for c in cols[i][:3])
        points3D[i + 1] = {"xyz": pts[i], "rgb": rgb, "error": 0.0, "track": []}
    return points3D


def _transform_poses_to_colmap(layout, cfg, s, R, t, metric_points=None, metric_colors=None):
    """Transform front-end world-to-camera poses and write the metric COLMAP
    model. Prefers the color-resolution intrinsics (shared) so the camera model
    matches the RGB frames the reconstruction host optimizes; falls back to the
    front-end per-frame intrinsics. Optionally bakes the metric cloud into
    points3D.bin as reconstruction-host init geometry."""
    if not layout.frontend_poses.exists():
        return {"written": False, "note": "frontend/poses.json missing"}

    fe = C.load_poses(layout.frontend_poses)
    frame_ids = sorted(fe["poses"].keys())

    Rt_w2c = {}
    for fid in frame_ids:
        R_wc, t_wc = C.to_world_to_camera(fe["poses"][fid]["R"], fe["poses"][fid]["t"], fe["pose_type"])
        R_wc2, t_wc2 = align.apply_similarity_to_w2c_pose(R_wc, t_wc, s, R, t)
        Rt_w2c[fid] = (R_wc2, t_wc2)

    # choose intrinsics
    if layout.capture_intrinsics.exists():
        cap = anchors.load_capture_intrinsics(layout.capture_intrinsics)
        K = cap["K_color"]
        resolution = cap["color_res"]
        K_by_frame = {fid: K for fid in frame_ids}
        shared = True
    else:
        K_by_frame, resolution = anchors.load_frontend_intrinsics(layout.frontend_intrinsics)
        shared = False

    names = {fid: f"{fid}.png" for fid in frame_ids}
    cameras, images = colmap_io.build_pinhole_model(
        frame_ids, K_by_frame, Rt_w2c, names, resolution, shared_intrinsics=shared)

    colmap_cfg = cfg["stage3"].get("colmap", {})
    if colmap_cfg.get("bake_points", True):
        points3D = _bake_points3D(metric_points, metric_colors,
                                  int(colmap_cfg.get("max_points", 100000)))
    else:
        points3D = {}

    out_dir = layout.metric_colmap
    colmap_io.write_model(out_dir, cameras, images, points3D=points3D)
    return {"written": True, "frames": len(frame_ids), "shared_intrinsics": shared,
            "points3D_baked": len(points3D)}


def run(session_dir, config_path):
    cfg = _load_config(config_path)
    s3 = cfg["stage3"]
    layout = SessionLayout(session_dir)
    session_id = Path(session_dir).name
    layout.metric.mkdir(parents=True, exist_ok=True)

    # ---- 1. compute the three anchors ------------------------------------
    depth = anchors.depth_anchor(layout, cfg)
    camera = anchors.camera_path_anchor(layout, cfg)
    ruler = anchors.ruler_anchor(layout, cfg)

    # gate the depth anchor on inlier fraction
    min_inlier = float(s3["min_depth_inlier_fraction"])
    if depth.get("available") and depth.get("inlier_fraction", 0.0) < min_inlier:
        depth["available"] = False
        depth["note"] = (f"inlier_fraction {depth.get('inlier_fraction'):.3f} "
                         f"< min {min_inlier}; depth anchor treated as unavailable")

    # ---- 2. decide the applied scale -------------------------------------
    anchor_scales = {}
    if depth.get("available"):
        anchor_scales["sensor_depth"] = depth["scale_estimate"]
    if camera.get("available"):
        anchor_scales["camera_path"] = camera["scale_estimate"]
    if ruler.get("available"):
        anchor_scales["physical_ruler"] = ruler["scale_estimate"]

    decision = report_mod.decide_scale(anchor_scales, float(s3["agreement_threshold_percent"]))
    applied_scale = decision["applied_scale"]

    # ---- 3. build + optionally ICP-refine the similarity transform -------
    s, Rmat, tvec = _build_similarity(applied_scale, camera)
    icp_info = {"ran": False}
    final_residual = depth.get("residual_meters") if depth.get("available") else camera.get("residual_meters")

    if s3.get("icp", {}).get("enabled") and icp_mod.open3d_available():
        target = icp_mod.backproject_sensor_cloud(layout, cfg)
        src_cloud = None
        if target is not None and layout.frontend_points.exists():
            fe_pts = plyio.read_ply(layout.frontend_points)["points"]
            src_cloud = align.apply_similarity_points(fe_pts, s, Rmat, tvec)
        if src_cloud is not None and target is not None:
            R_icp, t_icp, info = icp_mod.refine(
                src_cloud, target,
                s3["icp"]["max_correspondence_distance_m"], s3["icp"]["max_iterations"])
            # compose ICP (rigid) after the similarity: x -> R_icp(sRx+t)+t_icp
            Rmat = R_icp @ Rmat
            tvec = R_icp @ tvec + t_icp
            icp_info = {"ran": True, **info}
            final_residual = info["inlier_rmse"]
        else:
            icp_info = {"ran": False, "note": "insufficient data for ICP"}
    elif s3.get("icp", {}).get("enabled"):
        icp_info = {"ran": False, "note": "open3d unavailable; ICP skipped"}

    # ---- 4. apply transform and write outputs ----------------------------
    pc_info, metric_pts, metric_cols = _transform_pointcloud(layout, s, Rmat, tvec)
    colmap_info = _transform_poses_to_colmap(layout, cfg, s, Rmat, tvec, metric_pts, metric_cols)

    generated = datetime.now().strftime("%m-%d-%Y %H:%M") + " local"
    report = report_mod.build_report(
        session_id=session_id,
        front_end_model=cfg.get("stage2", {}).get("model", "unknown"),
        generated=generated,
        depth=depth, camera=camera, ruler=ruler,
        decision=decision,
        final_residual_meters=final_residual,
    )
    report["icp"] = icp_info
    report["outputs"] = {"points_metric": pc_info, "colmap": colmap_info}
    # keep the Experiment-A per-frame residuals accessible for analysis
    if depth.get("per_frame_residual_meters"):
        report["anchors"]["sensor_depth"]["per_frame_residual_meters"] = depth["per_frame_residual_meters"]

    with open(layout.metric_scale_report, "w") as fh:
        json.dump(report, fh, indent=2)

    # ---- 5. report + exit code -------------------------------------------
    print(f"[stage3] session={session_id} status={report['status']} "
          f"applied_scale={applied_scale:.5f} ({decision['applied_scale_source']}) "
          f"final_residual_m={final_residual}")
    if report["flags"]:
        print(f"[stage3] flags: {report['flags']}")
    print(f"[stage3] wrote {layout.metric_scale_report}")

    # A single available anchor still gives a valid metric lock (the documented
    # depth-only / no-ARKit fallback) — it is flagged as un-cross-checked but is
    # a SOFT flag that should not halt. Genuine disagreement or no anchor is a
    # HARD flag that halts when flag_halts_pipeline is set.
    HARD_FLAGS = {"anchors_disagree", "no_anchor_available", "no_physical_anchor"}
    hard = any(f in HARD_FLAGS for f in report["flags"])
    halts = report["status"] == "flag" and hard and bool(s3.get("flag_halts_pipeline", True))
    return 3 if halts else 0


def main():
    ap = argparse.ArgumentParser(description="Stage 3: metric alignment and validation")
    ap.add_argument("--session", required=True, help="path to sessions/<session_id>")
    ap.add_argument("--config", required=True, help="path to config/pipeline.yaml")
    args = ap.parse_args()
    return run(args.session, args.config)


if __name__ == "__main__":
    sys.exit(main())
