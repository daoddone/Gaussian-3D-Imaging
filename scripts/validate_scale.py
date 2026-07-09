#!/usr/bin/env python3
"""Task T2 — absolute-scale VALIDATION harness (docs/TECH_ROADMAP.md §5).

Quantifies the END-TO-END absolute-scale error of a metric reconstruction against a physical
reference object of known size placed in the capture. The pipeline's scale comes from device
sensors (T1: VIO-primary + LiDAR cross-check, scale_sidecar.json); this harness MEASURES how good
that markerless scale actually is — the reference object VALIDATES, it never SETS, the scale.

Auto path (recommended: a printed ArUco marker or ChArUco board of known size, rigid + matte):
  1. detect marker corners in every source image (cv2.aruco);
  2. triangulate each corner via multi-view DLT using the METRIC COLMAP poses/intrinsics
     (outlier views dropped at >3 px reprojection, then re-solved);
  3. compare measured corner-to-corner distances to the known geometry.
Reports (per Cho & Woo validation habits, MEASUREMENT_LITERATURE.md):
  * scale error % (median measured/known − 1) — the headline number;
  * PER-AXIS scale ratios + anisotropy % (marker u vs v sides);
  * ABSOLUTE errors in mm alongside % (percent explodes on thin/small features);
  * triangulation residuals + views-per-corner (measurement quality gates).
If a T1 scale_sidecar.json is found near the model, the result is appended to it as
"reference_validation" — closing the loop: sensor anchor (T1) -> physical validation (T2).
The measured size can also feed stage3's ruler_anchor (known/measured) if re-anchoring is wanted.

Manual path: --known-mm + --measured-mm (a distance measured in any viewer) -> plain % report.
Self-test:  --selftest synthesizes cameras+marker, injects a known scale error, and must recover
it to <0.2% — proves the math without a marker capture (run before trusting real results).

Run in the gs-ba env (cv2 + numpy). Examples:
  validate_scale.py --colmap sessions/S/metric_sfm/colmap/sparse/0 --images sessions/S/capture/rgb \
                    --marker-mm 50
  validate_scale.py --selftest
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path("/home/paperspace/Documents/VS Code Projects/3D-Gaussian")
sys.path.insert(0, str(REPO))
from common import colmap_io                      # noqa: E402
from common.conventions import quat_to_rotmat     # noqa: E402


# --------------------------------------------------------------------------- #
# geometry core (shared by real + selftest paths)
# --------------------------------------------------------------------------- #
def projection_matrix(K, R, t):
    return K @ np.hstack([R, t.reshape(3, 1)])


def triangulate_dlt(obs):
    """obs: list of (P 3x4, uv). Returns (X 3, per-view reproj px). Needs >=2 views."""
    A = []
    for P, uv in obs:
        u, v = float(uv[0]), float(uv[1])
        A.append(u * P[2] - P[0])
        A.append(v * P[2] - P[1])
    _, _, Vt = np.linalg.svd(np.asarray(A))
    Xh = Vt[-1]
    X = Xh[:3] / Xh[3]
    res = []
    for P, uv in obs:
        x = P @ np.append(X, 1.0)
        res.append(float(np.linalg.norm(x[:2] / x[2] - np.asarray(uv, float))))
    return X, np.asarray(res)


def triangulate_robust(obs, px_thresh=3.0):
    X, res = triangulate_dlt(obs)
    keep = res < px_thresh
    if keep.sum() >= 2 and keep.sum() < len(obs):
        X, res = triangulate_dlt([o for o, k in zip(obs, keep) if k])
    return X, res


def measure_markers(corners3d, known_mm):
    """corners3d: {marker_id: (4,3) metric METERS, order TL,TR,BR,BL (cv2.aruco)}.
    Returns per-marker + aggregate measurements vs the known side length."""
    per_marker, u_scales, v_scales, side_errs_mm, sides_mm = {}, [], [], [], []
    for mid, C in corners3d.items():
        if C is None or len(C) != 4:
            continue
        d = lambda a, b: float(np.linalg.norm(C[a] - C[b])) * 1000.0  # noqa: E731  mm
        u_sides = [d(0, 1), d(3, 2)]          # top, bottom  (marker u axis)
        v_sides = [d(1, 2), d(0, 3)]          # right, left  (marker v axis)
        allsides = u_sides + v_sides
        sides_mm += allsides
        side_errs_mm += [s - known_mm for s in allsides]
        u_scales.append(np.mean(u_sides) / known_mm)
        v_scales.append(np.mean(v_sides) / known_mm)
        per_marker[int(mid)] = {"sides_mm": [round(s, 3) for s in allsides],
                                "diagonals_mm": [round(d(0, 2), 3), round(d(1, 3), 3)]}
    if not sides_mm:
        return None
    sides = np.asarray(sides_mm)
    su, sv = float(np.mean(u_scales)), float(np.mean(v_scales))
    return {
        "known_side_mm": known_mm,
        "measured_side_mm_median": float(np.median(sides)),
        "scale_error_pct": float((np.median(sides) / known_mm - 1.0) * 100.0),
        "abs_error_mm_median": float(np.median(np.abs(side_errs_mm))),
        "abs_error_mm_max": float(np.max(np.abs(side_errs_mm))),
        "per_axis_scale": {"u": su, "v": sv,
                           "anisotropy_pct": float(200.0 * abs(su - sv) / (su + sv))},
        "n_markers": len(per_marker), "n_sides": int(sides.size),
        "per_marker": per_marker,
    }


# --------------------------------------------------------------------------- #
# real-capture path
# --------------------------------------------------------------------------- #
def run_real(args):
    import cv2

    colmap = (REPO / args.colmap) if not Path(args.colmap).is_absolute() else Path(args.colmap)
    imdir = (REPO / args.images) if not Path(args.images).is_absolute() else Path(args.images)
    imgs = colmap_io.read_images_binary(colmap / "images.bin")
    cams = colmap_io.read_cameras_binary(colmap / "cameras.bin")
    print(f"[T2] model: {len(imgs)} images  |  marker side = {args.marker_mm} mm  dict={args.dict}")

    dic = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, f"DICT_{args.dict}"))
    detector = cv2.aruco.ArucoDetector(dic, cv2.aruco.DetectorParameters())

    # per-(marker,corner) multi-view observations
    obs = {}
    n_imgs_hit = 0
    for im in imgs.values():
        cam = cams[im["camera_id"]]
        fx, fy, cx, cy = cam["params"][:4]
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], float)
        R = quat_to_rotmat(im["qvec"])
        t = np.asarray(im["tvec"], float)
        P = projection_matrix(K, R, t)
        p = imdir / im["name"]
        if not p.exists():
            cand = list(imdir.glob(Path(im["name"]).stem + ".*"))
            if not cand:
                continue
            p = cand[0]
        g = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if g is None:
            continue
        corners, ids, _ = detector.detectMarkers(g)
        if ids is None:
            continue
        n_imgs_hit += 1
        for mid, quad in zip(ids.ravel(), corners):
            for ci in range(4):
                obs.setdefault((int(mid), ci), []).append((P, quad[0][ci]))

    if not obs:
        raise SystemExit("[T2] no ArUco detections — check --dict / marker visibility / image dir")
    print(f"[T2] detections in {n_imgs_hit}/{len(imgs)} images; "
          f"{len({m for m, _ in obs})} marker(s)")

    corners3d, residuals, views_per_corner = {}, [], []
    for (mid, ci), o in sorted(obs.items()):
        if len(o) < args.min_views:
            continue
        X, res = triangulate_robust(o)
        corners3d.setdefault(mid, [None] * 4)[ci] = X
        residuals += res.tolist()
        views_per_corner.append(len(o))
    corners3d = {m: np.asarray(c) for m, c in corners3d.items() if all(x is not None for x in c)}
    if not corners3d:
        raise SystemExit(f"[T2] no marker with all 4 corners seen in >={args.min_views} views")

    rep = measure_markers(corners3d, args.marker_mm)
    rep.update({
        "mode": "aruco_multiview_dlt",
        "colmap": str(colmap), "images": str(imdir),
        "images_with_detections": n_imgs_hit,
        "views_per_corner_median": int(np.median(views_per_corner)),
        "triangulation_residual_px_median": float(np.median(residuals)),
        "note": ("scale error = end-to-end absolute-scale error of the SENSOR-anchored "
                 "reconstruction vs the physical reference (reference validates, never sets, scale)"),
    })
    finish(rep, colmap)


def finish(rep, colmap_or_none):
    print(f"[T2] measured {rep['measured_side_mm_median']:.2f} mm vs known {rep['known_side_mm']:.1f} mm"
          f"  ->  SCALE ERROR {rep['scale_error_pct']:+.2f}%")
    print(f"[T2] abs err median {rep['abs_error_mm_median']:.2f} mm (max {rep['abs_error_mm_max']:.2f});"
          f" per-axis u={rep['per_axis_scale']['u']:.4f} v={rep['per_axis_scale']['v']:.4f}"
          f" anisotropy={rep['per_axis_scale']['anisotropy_pct']:.2f}%")
    out = None
    if colmap_or_none is not None:
        # write next to the metric model; append into the T1 sidecar if present
        for anc in Path(colmap_or_none).resolve().parents:
            sc = anc / "scale_sidecar.json"
            if sc.exists():
                s = json.loads(sc.read_text())
                s["reference_validation"] = {k: rep[k] for k in
                                             ("scale_error_pct", "abs_error_mm_median",
                                              "per_axis_scale", "known_side_mm",
                                              "measured_side_mm_median")}
                sc.write_text(json.dumps(s, indent=2))
                print(f"[T2] appended reference_validation -> {sc}")
                out = anc / "scale_validation.json"
                break
        out = out or Path(colmap_or_none).resolve().parents[2] / "scale_validation.json"
    else:
        out = REPO / "scale_validation.json"
    out.write_text(json.dumps(rep, indent=2))
    print(f"[T2] wrote {out}")


# --------------------------------------------------------------------------- #
# synthetic self-test: inject a known scale error, require recovery to <0.2%
# --------------------------------------------------------------------------- #
def run_selftest():
    rng = np.random.default_rng(7)
    known_mm = 50.0
    inj = 1.023                                   # simulate a +2.3% sensor-scale error
    noise_px = 0.3

    # TRUE world: one 50mm marker on z=0, corners TL,TR,BR,BL
    s = known_mm / 1000.0
    C_true = np.array([[-s / 2, s / 2, 0], [s / 2, s / 2, 0],
                       [s / 2, -s / 2, 0], [-s / 2, -s / 2, 0]])
    K = np.array([[1400, 0, 720], [0, 1400, 960], [0, 0, 1]], float)

    # camera ring (TRUE metric); observations rendered from TRUE geometry
    Ps_scaled, obs2d = [], []
    for ang in np.linspace(0, 2 * np.pi, 14, endpoint=False):
        cpos = np.array([0.45 * np.cos(ang), 0.45 * np.sin(ang), 0.55])
        z = -cpos / np.linalg.norm(cpos)                       # look at origin
        x = np.cross(np.array([0.0, 0.0, 1.0]), z)
        x /= np.linalg.norm(x)
        y = np.cross(z, x)
        R = np.stack([x, y, z])                                # world->cam rows
        t = -R @ cpos
        P_true = projection_matrix(K, R, t)
        uv = []
        for X in C_true:
            xh = P_true @ np.append(X, 1.0)
            uv.append(xh[:2] / xh[2] + rng.normal(0, noise_px, 2))
        obs2d.append(uv)
        # the "reconstruction" lives in a frame scaled by inj: centers*inj => t*inj
        Ps_scaled.append(projection_matrix(K, R, t * inj))

    obs = {(0, ci): [(Ps_scaled[v], obs2d[v][ci]) for v in range(len(Ps_scaled))]
           for ci in range(4)}
    corners3d = {}
    for (mid, ci), o in obs.items():
        X, res = triangulate_robust(o)
        corners3d.setdefault(mid, [None] * 4)[ci] = X
    rep = measure_markers({0: np.asarray(corners3d[0])}, known_mm)

    got = rep["scale_error_pct"]
    want = (inj - 1) * 100
    ok_scale = abs(got - want) < 0.2
    ok_aniso = rep["per_axis_scale"]["anisotropy_pct"] < 0.5
    print(f"[T2:selftest] injected {want:+.2f}% -> recovered {got:+.2f}%  "
          f"(tolerance 0.2%)  {'PASS' if ok_scale else 'FAIL'}")
    print(f"[T2:selftest] anisotropy {rep['per_axis_scale']['anisotropy_pct']:.3f}% "
          f"(<0.5%)  {'PASS' if ok_aniso else 'FAIL'}")
    if not (ok_scale and ok_aniso):
        raise SystemExit(1)
    print("[T2:selftest] harness math validated")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--colmap", help="METRIC colmap sparse dir (e.g. sessions/S/metric_sfm/colmap/sparse/0)")
    ap.add_argument("--images", help="source images dir (e.g. sessions/S/capture/rgb)")
    ap.add_argument("--marker-mm", type=float, help="known ArUco side length in mm")
    ap.add_argument("--dict", default="4X4_50", help="cv2.aruco dictionary suffix (default 4X4_50)")
    ap.add_argument("--min-views", type=int, default=3)
    ap.add_argument("--known-mm", type=float, help="manual mode: known distance")
    ap.add_argument("--measured-mm", type=float, help="manual mode: distance measured in a viewer")
    args = ap.parse_args()

    if args.selftest:
        run_selftest()
    elif args.known_mm and args.measured_mm:
        err = (args.measured_mm / args.known_mm - 1) * 100
        rep = {"mode": "manual", "known_side_mm": args.known_mm,
               "measured_side_mm_median": args.measured_mm,
               "scale_error_pct": err,
               "abs_error_mm_median": abs(args.measured_mm - args.known_mm),
               "abs_error_mm_max": abs(args.measured_mm - args.known_mm),
               "per_axis_scale": {"u": args.measured_mm / args.known_mm,
                                  "v": args.measured_mm / args.known_mm, "anisotropy_pct": 0.0},
               "n_markers": 0, "n_sides": 1}
        finish(rep, Path(args.colmap) if args.colmap else None)
    elif args.colmap and args.images and args.marker_mm:
        run_real(args)
    else:
        ap.error("need --selftest, or --colmap+--images+--marker-mm, or --known-mm+--measured-mm")


if __name__ == "__main__":
    main()
