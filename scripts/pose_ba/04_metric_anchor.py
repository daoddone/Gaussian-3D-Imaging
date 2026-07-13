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


def marker_anchor(sess: Path, imgs, cams, marker_mm: float):
    """anchor C — ArUco marker of KNOWN size (T16): PRIMARY for external videos (no VIO/LiDAR).

    Detects DICT_4X4_50 corners on the model's source frames, triangulates each corner across
    views in the GAUGE-FREE model (validate_scale's robust DLT, 0.27 mm median demonstrated when
    used as a checker), and sets S = known_size / measured_gauge_size (meters convention, matching
    vio/lidar anchors). The marker VALIDATES app captures; it SETS scale only when sensors are absent
    or --anchor marker is forced.
    """
    out = {"available": False, "method": "aruco_marker_dlt"}
    import cv2
    sys.path.insert(0, str(REPO / "scripts"))
    from validate_scale import projection_matrix, triangulate_robust  # noqa: E402

    imdir = None
    for cand in (sess / "sfm_images", sess / "capture" / "rgb"):
        if cand.exists():
            imdir = cand
            break
    if imdir is None:
        out["note"] = "no image dir (sfm_images/ or capture/rgb)"
        return out

    dic = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    det = cv2.aruco.ArucoDetector(dic, cv2.aruco.DetectorParameters())
    # per (marker_id, corner_idx): list of (K, R, t, uv) observations across views
    obs = {}
    n_det_views = 0
    for im in imgs.values():
        p = imdir / im["name"]
        if not p.exists():
            continue
        g = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if g is None:
            continue
        corners, ids, _ = det.detectMarkers(g)
        if ids is None or not len(ids):
            continue
        n_det_views += 1
        cam = cams[im["camera_id"]]
        fx, fy, cx, cy = [float(v) for v in cam["params"][:4]]
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])
        R = C.quat_to_rotmat(im["qvec"])
        t = np.asarray(im["tvec"], float)
        P = projection_matrix(K, R, t)
        for mid, quad in zip(ids.flatten().tolist(), corners):
            for ci in range(4):
                obs.setdefault((mid, ci), []).append((P, quad[0][ci]))
    if not obs:
        out["note"] = f"no ArUco detections in {n_det_views} views"
        return out

    sides_gauge = []
    per_marker = {}
    mids = sorted({m for m, _ in obs})
    for mid in mids:
        pts3d = []
        for ci in range(4):
            o = obs.get((mid, ci), [])
            if len(o) < 3:
                pts3d = []
                break
            p3, _ = triangulate_robust(o)
            if p3 is None:
                pts3d = []
                break
            pts3d.append(p3)
        if len(pts3d) != 4:
            continue
        q = np.asarray(pts3d)
        s = [np.linalg.norm(q[i] - q[(i + 1) % 4]) for i in range(4)]
        sides_gauge.extend(s)
        per_marker[int(mid)] = {"views": len(obs[(mid, 0)]), "side_gauge_mean": float(np.mean(s))}
    if not sides_gauge:
        out["note"] = "markers detected but corner triangulation failed (need >=3 views/corner)"
        return out

    side_med = float(np.median(sides_gauge))
    scatter_pct = float(100 * np.std(sides_gauge) / side_med)
    out.update({
        "available": True,
        "scale": (marker_mm / 1000.0) / side_med,      # gauge -> meters, same convention as VIO/LiDAR
        "marker_mm": marker_mm,
        "markers": per_marker,
        "views_with_detections": n_det_views,
        "side_scatter_pct": scatter_pct,
    })
    return out


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


def lidar_anchor(sess: Path, imgs, pts, cams):
    """Robust median of (sensor depth / SfM camera-depth) over all observations (03b estimator).

    Pixel mapping color->depth uses EACH IMAGE'S OWN camera dims from the COLMAP model, NOT the
    session-global color_resolution: mixed-resolution captures (12 MP stills + stream-res fallbacks)
    have per-frame sizes, and a global mapping samples the wrong depth pixels (field-caught
    2026-07-09: produced a bogus 135% 'disagreement' on the first mixed-res capture)."""
    out = {"available": False, "method": "lidar_ray_median"}
    cap_intr = sess / "capture" / "intrinsics.json"
    depth_dir = sess / "capture" / "depth"
    conf_dir = sess / "capture" / "confidence"
    if not (cap_intr.exists() and depth_dir.exists()):
        out["note"] = "no capture depth (intrinsics.json / depth/ missing)"
        return out
    cap = json.load(open(cap_intr))
    depth_w, depth_h = cap["depth_resolution"]

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
        cam = cams[img["camera_id"]]
        sx, sy = depth_w / float(cam["width"]), depth_h / float(cam["height"])
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
    ap.add_argument("--marker-mm", type=float, help="ArUco side length (mm): enables the marker anchor "
                    "(PRIMARY for external videos without VIO/LiDAR; cross-check otherwise)")
    ap.add_argument("--anchor-policy", choices=["fixed", "consensus"], default="fixed",
                    help="consensus: with 3 anchors, an agreeing pair outvotes a deviant "
                         "(PROVISIONAL; default 'fixed' = VIO > marker > LiDAR)")
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
    lid = lidar_anchor(sess, imgs, pts, cams)
    mrk = marker_anchor(sess, imgs, cams, args.marker_mm) if args.marker_mm else \
        {"available": False, "note": "no --marker-mm given"}
    for name, a in (("VIO", vio), ("LiDAR", lid), ("MARKER", mrk)):
        if a["available"]:
            extra = (f"resid={a['umeyama_residual_mm']:.1f}mm frames={a['frames_used']}" if name == "VIO"
                     else f"MAD={a['mad_pct']:.1f}% samples={a['samples']} med_depth={a['median_sample_depth_m']:.2f}m"
                     if name == "LiDAR"
                     else f"scatter={a['side_scatter_pct']:.2f}% markers={len(a['markers'])} views={a['views_with_detections']}")
            print(f"[anchor] {name}: scale={a['scale']:.6f}  {extra}")
        else:
            print(f"[anchor] {name}: unavailable — {a.get('note')}")

    # ---- selection + confidence -------------------------------------------------------- #
    # Fixed order (default): VIO > MARKER > LiDAR. Marker is the designed PRIMARY for external
    # videos, where it is the only anchor.
    # CONSENSUS policy (--anchor-policy consensus; PROVISIONAL pending owner ratification,
    # motivated by the 07-13 validation batch: close-range/low-excitation captures showed VIO
    # confidently ~6-8% small while LiDAR+marker agreed within 1.4% and matched the independent
    # 96 mm design check): when >=2 anchors agree within --agree-warn and another deviates beyond
    # it, primary = the agreeing pair's preferred member (marker > vio > lidar); the deviant is
    # flagged in notes instead of silently winning by fixed order.
    notes = []
    avail = [(n, a) for n, a in (("vio_camera_path", vio), ("aruco_marker", mrk),
                                 ("lidar_ray_median", lid)) if a["available"]]
    if not avail:
        raise SystemExit("[anchor] NO metric anchor available — cannot scale this model "
                         "(external video needs --marker-mm with the printed sheet in frame)")
    PREF = {"aruco_marker": 0, "vio_camera_path": 1, "lidar_ray_median": 2}
    primary = primary_name = None
    if args.anchor_policy == "consensus" and len(avail) >= 3:
        pairs = []
        for i in range(len(avail)):
            for j in range(i + 1, len(avail)):
                (n1, a1), (n2, a2) = avail[i], avail[j]
                d = 100 * abs(a1["scale"] - a2["scale"]) / max(a1["scale"], a2["scale"])
                pairs.append((d, n1, n2))
        pairs.sort()
        d, n1, n2 = pairs[0]
        others = [n for n, _ in avail if n not in (n1, n2)]
        if d <= args.agree_warn and others:
            dev = others[0]
            dev_scale = dict(avail)[dev]["scale"]
            pair_mean = (dict(avail)[n1]["scale"] + dict(avail)[n2]["scale"]) / 2
            dev_pct = 100 * abs(dev_scale - pair_mean) / pair_mean
            if dev_pct > args.agree_warn:
                primary_name = min((n1, n2), key=lambda n: PREF[n])
                primary = dict(avail)[primary_name]
                notes.append(f"CONSENSUS: {n1}+{n2} agree ({d:.1f}%); {dev} deviates {dev_pct:.1f}% "
                             f"and was OUTVOTED (fixed order would have picked "
                             f"{min((n for n, _ in avail), key=lambda n: PREF[n] if n != 'vio_camera_path' else -1)})")
    if primary is None:
        for want in ("vio_camera_path", "aruco_marker", "lidar_ray_median"):
            hit = dict(avail).get(want)
            if hit is not None:
                primary, primary_name = hit, want
                break
    S = primary["scale"]

    agreement_pct = None
    if vio["available"] and lid["available"]:
        agreement_pct = 100 * abs(vio["scale"] - lid["scale"]) / S
        if agreement_pct > args.agree_warn:
            # attribute correctly when a third anchor exists (07-13 field batch: LiDAR+marker agreed
            # and VIO was the deviant — the old text blamed near-field LiDAR bias unconditionally)
            if mrk["available"] and 100 * abs(mrk["scale"] - lid["scale"]) / lid["scale"] <= args.agree_warn:
                notes.append(f"VIO vs LiDAR disagree by {agreement_pct:.1f}% and MARKER corroborates "
                             f"LiDAR — VIO scale is the likely outlier (close-range/low-excitation regime)")
            else:
                notes.append(f"anchors disagree by {agreement_pct:.1f}% "
                             f"(known near-field LiDAR bias signature if capture is close-range)")
    if lid.get("nearfield_warning"):
        notes.append("LiDAR samples are near-field (<0.25 m median) — LiDAR scale may read biased")
    if vio.get("umeyama_vs_pairwise_pct") and vio["umeyama_vs_pairwise_pct"] > 1.0:
        notes.append(f"VIO umeyama vs pairwise-baseline differ {vio['umeyama_vs_pairwise_pct']:.1f}% "
                     f"(possible degenerate trajectory — low motion excitation?)")

    # Marker as an additional independent cross-check (or the primary, for external videos)
    marker_agreement_pct = None
    if mrk["available"] and primary is not mrk:
        marker_agreement_pct = 100 * abs(mrk["scale"] - S) / S
        if marker_agreement_pct > args.agree_warn:
            notes.append(f"marker disagrees with {primary_name} by {marker_agreement_pct:.1f}% "
                         f"(check the sheet's 100 mm print bar — print scaling is the usual culprit)")
    if mrk["available"] and mrk.get("side_scatter_pct", 99) > 1.5:
        notes.append(f"marker corner scatter {mrk['side_scatter_pct']:.2f}% (loose triangulation)")

    # Disagreement is only tolerable when it is ATTRIBUTABLE (near-field LiDAR bias, flagged above);
    # an unexplained cross-anchor disagreement must force human review regardless of per-anchor fit.
    unexplained_disagreement = (
        (agreement_pct is not None and agreement_pct > args.agree_warn
         and not lid.get("nearfield_warning"))
        or (marker_agreement_pct is not None and marker_agreement_pct > args.agree_warn)
    )
    second_anchor_agrees = (
        (vio["available"] and lid["available"] and agreement_pct <= args.agree_warn)
        or (marker_agreement_pct is not None and marker_agreement_pct <= args.agree_warn)
    )
    if second_anchor_agrees and not unexplained_disagreement:
        confidence = "high"
    elif unexplained_disagreement:
        confidence = "review"
    elif (vio["available"] and vio["umeyama_residual_mm"] < 10 and
          (vio.get("umeyama_vs_pairwise_pct") or 0) <= 1.0) or \
         (primary is mrk and mrk["side_scatter_pct"] <= 1.5 and mrk["views_with_detections"] >= 5) or \
         (not vio["available"] and primary is lid and lid["mad_pct"] <= 3.0
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
        "marker_agreement_pct": marker_agreement_pct,
        "anchors": {"vio_camera_path": vio, "lidar_ray_median": lid, "aruco_marker": mrk},
        "applied": not args.no_apply,
        "notes": notes,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "scale_sidecar.json").write_text(json.dumps(sidecar, indent=2))
    print(f"[anchor] wrote {out_dir / 'scale_sidecar.json'}")


if __name__ == "__main__":
    main()
