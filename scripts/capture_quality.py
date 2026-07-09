#!/usr/bin/env python3
"""Task T3 — capture-quality score + excited-motion guidance (docs/TECH_ROADMAP.md §5).

Scores a capture BEFORE reconstruction so weak inputs are caught at the source (the campaign's
central law: capacity must match input quality — no reconstruction knob rescues a weak capture),
and so the VIO scale anchor (T1) gets the trajectory conditioning it needs (varied-curvature
"excited" motion materially improves VIO scale observability; a straight slide or static hover
is degenerate).

Components (all reuse existing pipeline machinery — no new estimators):
  frames      count vs the strong-capture branch target (150-400 sharp frames)
  sharpness   gradient-energy per frame (bake_mesh_colors.frame_sharpness); flags blurry frames
              relative to the session median (absolute thresholds are camera-dependent)
  motion      VIO trajectory analysis (ARKit captures): path length, direction diversity
              (eigen-spread of camera centers: line -> orbit -> 3D), total turning + curvature
              variation (the "excited motion" signal), speed uniformity
  coverage    angular sweep around the subject (optical-axis intersection center from
              make_subject_masks) — how much of the subject the orbit actually saw

Output: <capture>/capture_quality.json + human-readable verdicts + concrete guidance strings.
Captures without VIO poses (HQ backend) get frames+sharpness+(coverage if an SfM model is given).

Run in gs-ba env:
  capture_quality.py --session sessions/<S>            # uses <S>/capture
  capture_quality.py --session sessions/<S> --sfm pose_ba/sfm_noseed   # coverage w/o VIO
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path("/home/paperspace/Documents/VS Code Projects/3D-Gaussian")
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
from common import conventions as C                      # noqa: E402
from common import colmap_io                             # noqa: E402
from make_subject_masks import optical_axis_center       # noqa: E402


def frame_sharpness(img_gray_small):
    """Gradient energy (same metric as bake_mesh_colors.frame_sharpness; inlined because that
    module imports open3d, which the gs-ba env intentionally lacks)."""
    gy, gx = np.gradient(np.asarray(img_gray_small, dtype=np.float32))
    return float((gx * gx + gy * gy).mean())

STRONG_MIN, STRONG_MAX = 150, 400


def verdict(value, good, fair, higher_is_better=True):
    if higher_is_better:
        return "good" if value >= good else ("fair" if value >= fair else "low")
    return "good" if value <= good else ("fair" if value <= fair else "low")


def motion_metrics(centers):
    """Trajectory conditioning for VIO scale (T1). centers: (N,3) metric."""
    if len(centers) < 5:
        return None
    seg = np.diff(centers, axis=0)
    seglen = np.linalg.norm(seg, axis=1)
    ok = seglen > 1e-6
    path_len = float(seglen.sum())
    ext = centers.max(0) - centers.min(0)

    # direction diversity: eigen-spread of centers (line ~ (1,0,0); orbit ~ (1,1,eps); 3D ~ (1,1,1))
    cov = np.cov((centers - centers.mean(0)).T)
    ev = np.sort(np.linalg.eigvalsh(cov))[::-1]
    ev = ev / max(ev[0], 1e-12)
    diversity_2d, diversity_3d = float(ev[1]), float(ev[2])

    # turning: angle between consecutive segment directions (curvature variation = "excitation")
    d = seg[ok] / seglen[ok, None]
    cosang = np.clip((d[:-1] * d[1:]).sum(1), -1, 1)
    turn = np.arccos(cosang)
    total_turn_deg = float(np.degrees(turn.sum()))
    curvature_var = float(np.degrees(turn.std()))

    speed_cv = float(seglen[ok].std() / max(seglen[ok].mean(), 1e-9))
    return {
        "path_length_m": round(path_len, 3),
        "extent_m": [round(float(v), 3) for v in ext],
        "direction_diversity_2d": round(diversity_2d, 3),   # >=0.15 fair, >=0.35 good
        "direction_diversity_3d": round(diversity_3d, 3),
        "total_turning_deg": round(total_turn_deg, 1),      # ~360 = one orbit; more = excited
        "curvature_variation_deg": round(curvature_var, 2),
        "speed_cv": round(speed_cv, 3),
        "verdicts": {
            "diversity": verdict(diversity_2d, 0.35, 0.15),
            "turning": verdict(total_turn_deg, 360, 180),
            "excitation": verdict(curvature_var, 8.0, 4.0),
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", required=True)
    ap.add_argument("--sfm", help="optional session-relative COLMAP dir for coverage w/o VIO")
    args = ap.parse_args()
    sess = (REPO / args.session) if not Path(args.session).is_absolute() else Path(args.session)
    cap = sess / "capture"
    rgb = cap / "rgb"
    report = {"session": sess.name, "guidance": []}
    G = report["guidance"]

    # ---- frames ---------------------------------------------------------------- #
    frames = sorted(rgb.glob("*.png")) + sorted(rgb.glob("*.jpg"))
    n = len(frames)
    fv = verdict(n, STRONG_MIN, 60)
    report["frames"] = {"count": n, "strong_target": [STRONG_MIN, STRONG_MAX], "verdict": fv}
    if fv != "good":
        G.append(f"only {n} frames — strong-capture branch wants {STRONG_MIN}-{STRONG_MAX}; "
                 f"capture a longer, slower orbit (frame cap is now 360/120 s in Settings)")

    # ---- sharpness ------------------------------------------------------------- #
    if frames:
        from PIL import Image
        sh = []
        for f in frames:
            g = Image.open(f).convert("L")
            g = g.resize((max(g.width // 8, 1), max(g.height // 8, 1)))
            sh.append(frame_sharpness(np.asarray(g)))
        sh = np.asarray(sh)
        med = float(np.median(sh))
        blurry = int((sh < 0.5 * med).sum())
        bv = verdict(100 * blurry / max(n, 1), 5, 15, higher_is_better=False)
        report["sharpness"] = {"median_gradient_energy": round(med, 2),
                               "blurry_frames": blurry, "blurry_pct": round(100 * blurry / max(n, 1), 1),
                               "verdict": bv}
        if bv != "good":
            G.append(f"{blurry}/{n} frames are blurry (<50% of median sharpness) — slow the orbit; "
                     f"avoid sweeping fast between viewpoints")

    # ---- motion (VIO) ---------------------------------------------------------- #
    poses_file = cap / "poses.json"
    centers = None
    if poses_file.exists():
        ids, cc = C.camera_centers(C.load_poses(poses_file))
        centers = np.asarray(cc)
        m = motion_metrics(centers)
        report["motion_vio"] = m
        if m:
            if m["verdicts"]["diversity"] != "good":
                G.append("camera path is nearly a line/arc — VIO scale is poorly conditioned; "
                         "use a FIGURE-EIGHT / varied-curvature orbit (this alone ~halves VIO scale error)")
            if m["verdicts"]["turning"] != "good":
                G.append(f"total turning {m['total_turning_deg']:.0f}° (<360°) — complete at least a "
                         f"full orbit around the subject")
    else:
        report["motion_vio"] = None
        report.setdefault("notes", []).append("no capture/poses.json (no VIO) — motion conditioning "
                                              "unavailable; scale will rely on the LiDAR anchor only")

    # ---- coverage (needs poses: VIO or SfM) ------------------------------------- #
    Rs, ts = None, None
    if args.sfm:
        model = sess / args.sfm
        if (model / "images.bin").exists():
            imgs = colmap_io.read_images_binary(model / "images.bin")
            Rs = [C.quat_to_rotmat(im["qvec"]) for im in imgs.values()]
            ts = [np.asarray(im["tvec"], float) for im in imgs.values()]
    elif poses_file.exists():
        loaded = C.load_poses(poses_file)
        w2c = loaded["pose_type"] == "world_to_camera"
        Rs, ts = [], []
        for fid, p in loaded["poses"].items():
            R, t = np.asarray(p["R"], float), np.asarray(p["t"], float)
            if not w2c:                      # camera_to_world -> invert to w2c for the axis solver
                R, t = R.T, -R.T @ t
            Rs.append(R)
            ts.append(t)
    if Rs:
        c_subj = optical_axis_center(Rs, ts)
        cams = np.asarray([-np.asarray(R).T @ np.asarray(t) for R, t in zip(Rs, ts)])
        rel = cams - c_subj
        az = np.degrees(np.arctan2(rel[:, 1], rel[:, 0]))
        az_sorted = np.sort(az)
        gaps = np.diff(np.append(az_sorted, az_sorted[0] + 360))
        az_span = float(360 - gaps.max())
        cv2v = verdict(az_span, 270, 150)
        report["coverage"] = {"azimuth_span_deg": round(az_span, 1),
                              "largest_gap_deg": round(float(gaps.max()), 1), "verdict": cv2v}
        if cv2v != "good":
            G.append(f"angular coverage {az_span:.0f}° with a {gaps.max():.0f}° gap — orbit further "
                     f"around the subject; large gaps become holes/occlusions")

    # ---- overall ---------------------------------------------------------------- #
    verdicts = [report["frames"]["verdict"]]
    if report.get("sharpness"):
        verdicts.append(report["sharpness"]["verdict"])
    if report.get("motion_vio"):
        verdicts += list(report["motion_vio"]["verdicts"].values())
    if report.get("coverage"):
        verdicts.append(report["coverage"]["verdict"])
    overall = "STRONG" if all(v == "good" for v in verdicts) else \
              ("WEAK" if verdicts.count("low") >= 2 else "FAIR")
    report["overall"] = overall
    report["expected_capacity_branch"] = "quality" if overall == "STRONG" else "fast"

    out = cap / "capture_quality.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"[T3] {sess.name}: OVERALL {overall} (expected branch: {report['expected_capacity_branch']})")
    for k in ("frames", "sharpness", "motion_vio", "coverage"):
        if report.get(k):
            print(f"[T3]   {k}: {json.dumps({kk: vv for kk, vv in report[k].items() if kk != 'per_marker'})}")
    for g in G:
        print(f"[T3]   GUIDANCE: {g}")
    print(f"[T3] wrote {out}")


if __name__ == "__main__":
    main()
