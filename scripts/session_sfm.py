#!/usr/bin/env python3
"""Generic per-session from-scratch SfM (gauge-free), for any capture session.

Generalizes the proven face-runner recipe (scripts/face_depthfree_test.py stage_sfm — pycolmap,
PINHOLE single shared camera, BA-refined focal, sequential matching for video-order captures) to
arbitrary sessions, so new captures flow: capture -> session_sfm -> 04_metric_anchor (VIO+LiDAR
scale sidecar) -> stage 5 / validate_scale.

Mixed-resolution captures (arkit4K writes 12 MP stills with stream-res fallbacks): SfM uses a
single shared camera, so we select the DOMINANT resolution subset by default (or --res WxH).
Output: <session>/pose_ba/sfm_noseed (the layout 04_metric_anchor expects) + sfm_images/ (the
image subset the model refers to — pass as --images downstream).

Run in the gs-ba env:
  session_sfm.py --session sessions/<S> [--res 1920x1440] [--min-inliers 50]
"""
import argparse
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

REPO = Path("/home/paperspace/Documents/VS Code Projects/3D-Gaussian")
sys.path.insert(0, str(REPO))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", required=True)
    ap.add_argument("--res", help="use frames of this WxH (default: dominant size)")
    ap.add_argument("--min-inliers", type=int, default=50)
    args = ap.parse_args()

    import pycolmap
    from PIL import Image

    sess = (REPO / args.session) if not Path(args.session).is_absolute() else Path(args.session)
    rgb = sess / "capture" / "rgb"
    frames = sorted(rgb.glob("*.png")) + sorted(rgb.glob("*.jpg"))
    if not frames:
        raise SystemExit(f"[sfm] no frames in {rgb}")

    sizes = {f: Image.open(f).size for f in frames}
    if args.res:
        w, h = map(int, args.res.lower().split("x"))
        target = (w, h)
    else:
        target = Counter(sizes.values()).most_common(1)[0][0]
    subset = [f for f in frames if sizes[f] == target]
    print(f"[sfm] {sess.name}: {len(subset)}/{len(frames)} frames at {target[0]}x{target[1]} "
          f"(sizes present: {dict(Counter(sizes.values()))})")
    if len(subset) < 10:
        raise SystemExit("[sfm] too few frames at the chosen resolution")

    # shared-K init from the per-frame device intrinsics (median fx of the subset)
    intr = json.load(open(sess / "capture" / "intrinsics.json"))
    per_k = intr.get("K_per_frame", {})
    import numpy as np
    fxs = []
    for f in subset:
        k = per_k.get(f.stem)
        if k:
            fxs.append(float(k[0][0]))
    if fxs:
        fx = float(np.median(fxs))
    else:
        fx = float(np.asarray(intr["K"], float)[0][0])
    W, H = target
    cx, cy = W / 2.0, H / 2.0
    print(f"[sfm] shared-K init PINHOLE fx=fy={fx:.1f} cx={cx} cy={cy} (BA refines focal)")

    # image subset dir the COLMAP model will reference
    imdir = sess / "sfm_images"
    imdir.mkdir(exist_ok=True)
    for old in imdir.iterdir():
        old.unlink()
    for f in subset:
        (imdir / f.name).symlink_to(f.resolve())

    out = sess / "pose_ba"
    out.mkdir(exist_ok=True)
    db = out / "sfm_noseed.db"
    if db.exists():
        db.unlink()
    sfm_dir = out / "sfm_noseed"
    if sfm_dir.exists():
        shutil.rmtree(sfm_dir)

    reader = pycolmap.ImageReaderOptions()
    reader.camera_model = "PINHOLE"
    reader.camera_params = f"{fx},{fx},{cx},{cy}"
    pycolmap.extract_features(database_path=str(db), image_path=str(imdir),
                              camera_mode=pycolmap.CameraMode.SINGLE, reader_options=reader)
    pycolmap.match_sequential(database_path=str(db))

    opts = pycolmap.IncrementalPipelineOptions()
    opts.min_model_size = 3
    opts.min_num_matches = 15
    opts.ba_refine_focal_length = True
    opts.ba_refine_extra_params = False
    opts.ba_refine_principal_point = False
    mapper = opts.get_mapper()
    mapper.init_min_num_inliers = args.min_inliers
    mapper.init_min_tri_angle = 4.0

    maps = pycolmap.incremental_mapping(database_path=str(db), image_path=str(imdir),
                                        output_path=str(out / "_maps"), options=opts)
    if not maps:
        raise SystemExit("[sfm] NO model reconstructed")
    best = max(maps, key=lambda i: maps[i].num_images())
    rec = maps[best]
    cam = list(rec.cameras.values())[0]
    print(f"[sfm] best model: {rec.num_images()}/{len(subset)} images, {rec.num_points3D()} points")
    try:
        print(f"[sfm] mean reprojection error: {rec.compute_mean_reprojection_error():.3f} px")
    except Exception:
        pass
    print(f"[sfm] refined K: {[round(float(x), 2) for x in cam.params]}")

    sfm_dir.mkdir(parents=True)
    rec.write(str(sfm_dir))
    shutil.rmtree(out / "_maps", ignore_errors=True)
    print(f"[sfm] wrote {sfm_dir}  (next: 04_metric_anchor.py --session {args.session})")


if __name__ == "__main__":
    main()
