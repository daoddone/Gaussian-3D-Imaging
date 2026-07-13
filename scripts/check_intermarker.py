#!/usr/bin/env python3
"""Independent metric check: inter-marker center distance vs the sheet design (96.00 mm at 254 dpi).

Triangulates both ArUco markers' centers in the session's gauge-free SfM model, applies the
sidecar scale, and compares to the design ground truth. For VIO-scaled sessions this is fully
independent of the marker; for marker-primary sessions it still checks print geometry +
reconstruction (shares the print's linear scale).

Usage: check_intermarker.py --session sessions/<S> [--gt-mm 96.0]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path("/home/paperspace/Documents/VS Code Projects/3D-Gaussian")
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
from common import colmap_io                      # noqa: E402
from common.conventions import quat_to_rotmat     # noqa: E402
from validate_scale import projection_matrix, triangulate_robust  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", required=True)
    ap.add_argument("--gt-mm", type=float, default=96.0)
    args = ap.parse_args()
    import cv2

    sess = (REPO / args.session) if not Path(args.session).is_absolute() else Path(args.session)
    sparse = sess / "pose_ba" / "sfm_noseed"
    imgs = colmap_io.read_images_binary(sparse / "images.bin")
    cams = colmap_io.read_cameras_binary(sparse / "cameras.bin")
    sc = json.loads((sess / "metric_sfm" / "scale_sidecar.json").read_text())
    imdir = sess / "sfm_images" if (sess / "sfm_images").exists() else sess / "capture" / "rgb"

    det = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50),
                                  cv2.aruco.DetectorParameters())
    obs = {}
    for im in imgs.values():
        g = cv2.imread(str(imdir / im["name"]), cv2.IMREAD_GRAYSCALE)
        if g is None:
            continue
        cs, ii, _ = det.detectMarkers(g)
        if ii is None:
            continue
        cam = cams[im["camera_id"]]
        fx, fy, cx, cy = [float(v) for v in cam["params"][:4]]
        P = projection_matrix(np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]]),
                              quat_to_rotmat(im["qvec"]), np.asarray(im["tvec"], float))
        for mid, quad in zip(ii.flatten().tolist(), cs):
            for ci in range(4):
                obs.setdefault((mid, ci), []).append((P, quad[0][ci]))
    need = [(m, c) for m in (0, 1) for c in range(4)]
    if any(len(obs.get(k, [])) < 3 for k in need):
        raise SystemExit(f"[imk] insufficient marker observations in {sess.name}")
    c3d = {m: np.mean([triangulate_robust(obs[(m, c)])[0] for c in range(4)], axis=0) for m in (0, 1)}
    d_mm = float(np.linalg.norm(c3d[0] - c3d[1]) * sc["scale"] * 1000)
    err = 100 * (d_mm - args.gt_mm) / args.gt_mm
    print(f"[imk] {sess.name}: inter-marker {d_mm:.2f} mm vs {args.gt_mm:.2f} -> {err:+.2f}% "
          f"(primary={sc['primary_anchor']}, confidence={sc['confidence']})")


if __name__ == "__main__":
    main()
