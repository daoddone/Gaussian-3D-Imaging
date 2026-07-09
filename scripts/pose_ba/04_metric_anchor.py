#!/usr/bin/env python3
"""Pose-BA step 4 — UNIFIED metric-scale anchor for gauge-free SfM/BA models (task T1).

Computes ABSOLUTE metric scale for a gauge-free COLMAP model from BOTH device-sensor anchors,
cross-checks them, selects a primary, applies it, and emits a machine-readable scale sidecar:

  anchor A — VIO camera path (PRIMARY when available; ARKit captures only):
      Umeyama similarity between the SfM camera centers and the metric VIO centers from
      capture/poses.json (same physical trajectory -> scale = baseline ratio). Immune to the
      depth sensor's near-field bias. Cross-validated internally by the median pairwise
      camera-baseline ratio (catches degenerate similarity fits).

  anchor B — LiDAR ray-median (cross-check; primary for captures without VIO):
      for each SfM point observed in a frame, ratio of sensor depth at the observed pixel to
      the SfM camera-frame depth; scale = robust median over all (point,frame) samples.
      Gauge-immune (no pose used). Same estimator as 03b_relock_lidar.py, generalized.

Why both: agreement between two INDEPENDENT physical anchors is the scale-confidence signal
(disagreement flags a bad capture or near-field LiDAR bias — the known ~12% failure mode).
The sidecar (scale_sidecar.json) records primary/scale/agreement/confidence for downstream
consumers (stage-5 provenance, OBJ export_meta). See docs/TECH_ROADMAP.md §3.

Supersedes the hardcoded 03b for new work (03b kept for history). Run in the gs-ba env.

Usage:
  04_metric_anchor.py --session sessions/<S> [--sfm pose_ba/sfm_noseed] [--out metric_sfm]
                      [--no-apply] [--agree-warn 3.0]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path("/home/paperspace/Documents/VS Code Projects/3D-Gaussian")
sys.path.insert(0, str(REPO))
from common import colmap_io, plyio                      # noqa: E402
from common import conventions as C                      # noqa: E402
from stages.stage3_metric import align                   # noqa: E402

DMIN, DMAX = 0.15, 3.0        # usable LiDAR range (near-field bias below ~0.25 m — tracked below)


def sfm_camera_centers(imgs):
    """{stem: center} + {stem: (R,t)} for a COLMAP images dict."""
    centers, poses = {}, {}
    for im in imgs.values():
        R = C.quat_to_rotmat(im["qvec"])
        t = np.asarray(im["tvec"], float)
        stem = Path(im["name"]).stem
        centers[stem] = -R.T @ t
        poses[stem] = (R, t)
    return centers, poses


def vio_anchor(sess: Path, sfm_centers: dict):
    """Umeyama SfM->VIO similarity + pairwise-baseline cross-validation."""
    out = {"available": False, "method": "vio_camera_path_umeyama"}
    pose_file = sess / "capture" / "poses.json"
    if not pose_file.exists():
        out["note"] = "no capture/poses.json (capture without VIO)"
        return out
    ids, centers = C.camera_centers(C.load_poses(pose_file))
    vio = {Path(i).stem: c for i, c in zip(ids, centers)}
    common = sorted(set(vio) & set(sfm_centers))
    if len(common) < 3:
        out["note"] = f"only {len(common)} frames common to SfM and VIO (need >=3)"
        return out
    src = np.array([sfm_centers[f] for f in common])       # gauge-free
    dst = np.array([vio[f] for f in common])               # metric
    s, R, t = align.umeyama(src, dst, with_scale=True)
    resid = align.similarity_residual(src, dst, s, R, t)   # meters, metric frame

    # internal cross-validation: median of pairwise baseline ratios (scale without any fit)
    d_src = np.linalg.norm(src[:, None] - src[None, :], axis=-1)
    d_dst = np.linalg.norm(dst[:, None] - dst[None, :], axis=-1)
    iu = np.triu_indices(len(common), k=1)
    num, den = d_dst[iu], d_src[iu]
    ok = den > 1e-6
    pairwise = float(np.median(num[ok] / den[ok])) if ok.any() else None

    out.update({
        "available": True, "scale": float(s), "frames_used": len(common),
        "umeyama_residual_mm": float(1000 * resid),
        "pairwise_median_scale": pairwise,
        "umeyama_vs_pairwise_pct": (100 * abs(pairwise - s) / s) if pairwise else None,
        "R": np.asarray(R).tolist(), "t": np.asarray(t).tolist(),
    })
    return out


def lidar_anchor(sess: Path, imgs, pts):
    """Robust median of (sensor depth / SfM camera-depth) over all observations (03b estimator)."""
    out = {"available": False, "method": "lidar_ray_median"}
    cap_intr = sess / "capture" / "intrinsics.json"
    depth_dir = sess / "capture" / "depth"
    conf_dir = sess / "capture" / "confidence"
    if not (cap_intr.exists() and depth_dir.exists()):
        out["note"] = "no capture depth (intrinsics.json / depth/ missing)"
        return out
    cap = json.load(open(cap_intr))
    color_w, color_h = cap["color_resolution"]
    depth_w, depth_h = cap["depth_resolution"]
    sx, sy = depth_w / color_w, depth_h / color_h

    from PIL import Image

    def load_conf(fid):
        p = conf_dir / f"{fid}.png"
        if not p.exists():
            return None
        a = np.asarray(Image.open(p))
        return (a[..., 0] if a.ndim == 3 else a) >= 128

    ratios, depths, frames_hit = [], [], 0
    for img in imgs.values():
        fid = Path(img["name"]).stem
        dpath = depth_dir / f"{fid}.npy"
        if not dpath.exists():
            continue
        depth = np.load(dpath).astype(float)
        conf = load_conf(fid)
        R = C.quat_to_rotmat(img["qvec"])
        t = np.asarray(img["tvec"], float)
        xys, p3d = img.get("xys"), img.get("point3D_ids")
        if xys is None or len(xys) == 0:
            continue
        n = 0
        for k in range(len(xys)):
            pid = int(p3d[k])
            if pid not in pts:
                continue
            zc = float((R @ np.asarray(pts[pid]["xyz"], float) + t)[2])
            if zc <= 1e-6:
                continue
            ud, vd = int(round(float(xys[k][0]) * sx)), int(round(float(xys[k][1]) * sy))
            if not (0 <= ud < depth_w and 0 <= vd < depth_h):
                continue
            d = depth[vd, ud]
            if not np.isfinite(d) or d < DMIN or d > DMAX:
                continue
            if conf is not None and not conf[vd, ud]:
                continue
            ratios.append(d / zc)
            depths.append(d)
            n += 1
        frames_hit += n > 0
    ratios = np.asarray(ratios)
    if ratios.size < 100:
        out["note"] = f"too few depth samples ({ratios.size})"
        return out
    S = float(np.median(ratios))
    mad = float(np.median(np.abs(ratios - S)))
    med_depth = float(np.median(depths))
    out.update({
        "available": True, "scale": S, "mad_pct": 100 * mad / S,
        "samples": int(ratios.size), "frames_used": int(frames_hit),
        "median_sample_depth_m": med_depth,
        "nearfield_warning": bool(med_depth < 0.25),
    })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", required=True)
    ap.add_argument("--sfm", default="pose_ba/sfm_noseed", help="session-relative gauge-free model dir")
    ap.add_argument("--out", default="metric_sfm")
    ap.add_argument("--no-apply", action="store_true", help="sidecar only; do not write the scaled model")
    ap.add_argument("--agree-warn", type=float, default=3.0, help="agreement %% above which confidence drops")
    args = ap.parse_args()

    sess = (REPO / args.session) if not Path(args.session).is_absolute() else Path(args.session)
    sfm_dir = sess / args.sfm
    if not (sfm_dir / "images.bin").exists():
        raise SystemExit(f"[anchor] no COLMAP model at {sfm_dir}")
    imgs = colmap_io.read_images_binary(sfm_dir / "images.bin")
    pts = colmap_io.read_points3D_binary(sfm_dir / "points3D.bin")
    cams = colmap_io.read_cameras_binary(sfm_dir / "cameras.bin")
    print(f"[anchor] {sess.name}: SfM model {len(imgs)} images / {len(pts)} points ({args.sfm})")

    centers, _ = sfm_camera_centers(imgs)
    vio = vio_anchor(sess, centers)
    lid = lidar_anchor(sess, imgs, pts)
    for name, a in (("VIO", vio), ("LiDAR", lid)):
        if a["available"]:
            extra = (f"resid={a['umeyama_residual_mm']:.1f}mm frames={a['frames_used']}"
                     if name == "VIO" else
                     f"MAD={a['mad_pct']:.1f}% samples={a['samples']} med_depth={a['median_sample_depth_m']:.2f}m")
            print(f"[anchor] {name}: scale={a['scale']:.6f}  {extra}")
        else:
            print(f"[anchor] {name}: unavailable — {a.get('note')}")

    # ---- selection + confidence -------------------------------------------------------- #
    notes = []
    if vio["available"]:
        primary, primary_name = vio, "vio_camera_path"
    elif lid["available"]:
        primary, primary_name = lid, "lidar_ray_median"
    else:
        raise SystemExit("[anchor] NO metric anchor available — cannot scale this model")
    S = primary["scale"]

    agreement_pct = None
    if vio["available"] and lid["available"]:
        agreement_pct = 100 * abs(vio["scale"] - lid["scale"]) / S
        if agreement_pct > args.agree_warn:
            notes.append(f"anchors disagree by {agreement_pct:.1f}% "
                         f"(known near-field LiDAR bias signature if capture is close-range)")
    if lid.get("nearfield_warning"):
        notes.append("LiDAR samples are near-field (<0.25 m median) — LiDAR scale may read biased")
    if vio.get("umeyama_vs_pairwise_pct") and vio["umeyama_vs_pairwise_pct"] > 1.0:
        notes.append(f"VIO umeyama vs pairwise-baseline differ {vio['umeyama_vs_pairwise_pct']:.1f}% "
                     f"(possible degenerate trajectory — low motion excitation?)")

    # Disagreement is only tolerable when it is ATTRIBUTABLE (near-field LiDAR bias, flagged above);
    # an unexplained cross-anchor disagreement must force human review regardless of per-anchor fit.
    unexplained_disagreement = (agreement_pct is not None and agreement_pct > args.agree_warn
                                and not lid.get("nearfield_warning"))
    if vio["available"] and lid["available"] and agreement_pct <= args.agree_warn:
        confidence = "high"
    elif unexplained_disagreement:
        confidence = "review"
    elif (vio["available"] and vio["umeyama_residual_mm"] < 10 and
          (vio.get("umeyama_vs_pairwise_pct") or 0) <= 1.0) or \
         (not vio["available"] and lid["available"] and lid["mad_pct"] <= 3.0
          and not lid.get("nearfield_warning")):
        confidence = "medium"
    else:
        confidence = "review"

    print(f"[anchor] PRIMARY={primary_name} scale={S:.6f} "
          f"agreement={f'{agreement_pct:.1f}%' if agreement_pct is not None else 'n/a'} "
          f"confidence={confidence}")
    for n in notes:
        print(f"[anchor] NOTE: {n}")

    # ---- apply (scale-only about the origin, 03b convention) --------------------------- #
    out_dir = sess / args.out
    if not args.no_apply:
        new_imgs = {iid: {"qvec": im["qvec"], "tvec": [float(v) * S for v in im["tvec"]],
                          "camera_id": im["camera_id"], "name": im["name"]}
                    for iid, im in imgs.items()}
        new_pts, xyz_all, rgb_all = {}, [], []
        for i, (pid, p) in enumerate(sorted(pts.items()), start=1):
            xyz = np.asarray(p["xyz"], float) * S
            rgb = tuple(int(c) for c in np.asarray(p["rgb"])[:3])
            new_pts[i] = {"xyz": xyz, "rgb": rgb, "error": 0.0, "track": []}
            xyz_all.append(xyz)
            rgb_all.append(rgb)
        xyz_all = np.asarray(xyz_all)
        colmap_io.write_model(out_dir / "colmap" / "sparse" / "0", cams, new_imgs, new_pts)
        plyio.write_ply(out_dir / "points_metric.ply", xyz_all.astype(np.float32),
                        colors=np.asarray(rgb_all, dtype=np.uint8), binary=True)
        ext = (xyz_all.max(0) - xyz_all.min(0)) * 1000
        print(f"[anchor] wrote metric model -> {out_dir}/colmap/sparse/0  "
              f"(point extent {ext.round(0).tolist()} mm)")

    sidecar = {
        "schema": 1,
        "task": "T1 unified metric-scale anchor (docs/TECH_ROADMAP.md §3)",
        "session": str(sess.relative_to(REPO)) if sess.is_relative_to(REPO) else str(sess),
        "sfm_source": args.sfm,
        "primary_anchor": primary_name,
        "scale": S,
        "confidence": confidence,
        "anchor_agreement_pct": agreement_pct,
        "anchors": {"vio_camera_path": vio, "lidar_ray_median": lid},
        "applied": not args.no_apply,
        "notes": notes,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "scale_sidecar.json").write_text(json.dumps(sidecar, indent=2))
    print(f"[anchor] wrote {out_dir / 'scale_sidecar.json'}")


if __name__ == "__main__":
    main()
