#!/usr/bin/env python3
"""Depth-free fidelity baseline: face photos -> from-scratch pycolmap SfM -> MILo (no depth term).

Reproduces the owner's clean preliminary run (scripts/run_pycolmap_from_record3d.py: color frames ->
COLMAP SfM with jointly-refined focal -> MILo, no depth supervision, no dense_gaussians, near-zero
background) through the CURRENT MILo host. Purpose (docs/SWEEP_RESULTS.md): the sweep showed LiDAR
depth supervision is the biggest mesh-bumpiness source; this is the depth-free arm on a subject the
owner has a reference for. Also doubles as the "no DA3" configuration (init = COLMAP sparse points).

NOTE: no depth anywhere -> the result is gauge-free (NON-metric). Pure fidelity test by design.

Conditions matched to the preliminary run:
  * ~169 frames (stride sampling of the full-fps Record3D export; the folder has 3258, owner used 169)
  * PINHOLE, single camera, principal point fixed, focal INIT estimated (no metadata.json in the
    folder; fx ~= 2871 * 1920/4032 = 1367 from the known device K) then ba_refine_focal_length=True
    -- same "solved from scratch and jointly refined" as the old script.
  * sequential matcher (video), init_min_num_inliers=50, init_min_tri_angle=4.0, min_num_matches=15
  * MILo: mesh reg ON (default), radegs, imp_metric indoor, -r 1 (1920 <= 2048 nvdiffrast cap),
    dense_gaussians=False, depth_lambda=0.0 (MILo's LiDAR block is gated on lambda>0 -> fully off).

Usage (two envs):
  ~/miniforge3/envs/gs-ba/bin/python scripts/face_depthfree_test.py --stage sfm
  python3 scripts/face_depthfree_test.py --stage milo
"""
import argparse
import shutil
import sqlite3
import sys
from pathlib import Path

REPO = Path("/home/paperspace/Documents/VS Code Projects/3D-Gaussian")
PHOTOS = REPO / "sessions/Previous face photos"
SESS = REPO / "sessions/face_depthfree_test"
FX_INIT = 2871.0 * 1920.0 / 4032.0   # ~1367; device K scaled to this resolution (refined in BA)
CX, CY = 720.0, 960.0                 # image center (1440x1920 portrait)

# Variants isolate ONE input variable each (v1 = the preliminary-matched baseline):
#   v1: 172 frames, blind stride 19 (matches the owner's original 169-frame sample scale)
#   v3: ~362 frames, SHARPEST frame per 9-frame window (more views + blur-aware selection)
VARIANTS = {
    "v1": {"dataset": SESS / "dataset", "colmap_out": SESS / "colmap_out",
           "window": 19, "sharpness": False},
    "v3": {"dataset": SESS / "dataset_v3", "colmap_out": SESS / "colmap_out_v3",
           "window": 9, "sharpness": True},
}
DATASET = VARIANTS["v1"]["dataset"]   # set per-run in __main__
COLMAP_OUT = VARIANTS["v1"]["colmap_out"]


def sorted_jpgs(d: Path):
    return sorted([p for p in d.iterdir() if p.suffix.lower() == ".jpg"],
                  key=lambda p: int(p.stem) if p.stem.isdigit() else 10**12)


def select_frames(files, window: int, sharpness: bool):
    """One frame per consecutive `window`-frame group; blind = first of window, sharp = max
    Laplacian variance (motion-blur rejection — video frames vary hugely in sharpness)."""
    if not sharpness:
        return files[::window]
    import cv2
    import numpy as np
    picked = []
    for i in range(0, len(files), window):
        group = files[i:i + window]
        best, best_v = group[0], -1.0
        for f in group:
            img = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (360, 480), interpolation=cv2.INTER_AREA)
            v = float(cv2.Laplacian(img, cv2.CV_64F).var())
            if v > best_v:
                best, best_v = f, v
        picked.append(best)
    return picked


def enforce_pinhole(database_path: Path):
    """Guard from the old script: pycolmap sometimes writes SIMPLE_RADIAL; force PINHOLE (id 1)."""
    conn = sqlite3.connect(str(database_path))
    rows = conn.execute("SELECT camera_id, model FROM cameras").fetchall()
    changed = False
    for camera_id, model in rows:
        if int(model) != 1:
            conn.execute("UPDATE cameras SET model = 1 WHERE camera_id = ?", (camera_id,))
            changed = True
    if changed:
        conn.execute("DELETE FROM two_view_geometries")
        print("[sfm] adjusted camera model(s) to PINHOLE")
    conn.commit()
    conn.close()


def stage_sfm(window: int, sharpness: bool):
    import pycolmap

    images_dir = DATASET / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    for old in images_dir.iterdir():
        if old.is_file():
            old.unlink()
    files = sorted_jpgs(PHOTOS)
    selected = select_frames(files, window, sharpness)
    for f in selected:
        shutil.copy2(f, images_dir / f.name)
    print(f"[sfm] sampled {len(selected)}/{len(files)} frames "
          f"(window {window}, {'sharpest-per-window' if sharpness else 'blind stride'})")

    COLMAP_OUT.mkdir(parents=True, exist_ok=True)
    db = COLMAP_OUT / "database.db"
    if db.exists():
        db.unlink()
    for p in COLMAP_OUT.iterdir():
        if p.is_dir() and p.name.isdigit():
            shutil.rmtree(p)

    reader_options = pycolmap.ImageReaderOptions()
    reader_options.camera_model = "PINHOLE"
    reader_options.camera_params = f"{FX_INIT},{FX_INIT},{CX},{CY}"
    print(f"[sfm] intrinsics init PINHOLE fx=fy={FX_INIT:.1f} cx={CX} cy={CY} (BA refines focal)")

    pycolmap.extract_features(
        database_path=str(db), image_path=str(images_dir),
        camera_mode=pycolmap.CameraMode.SINGLE, reader_options=reader_options)
    enforce_pinhole(db)
    pycolmap.match_sequential(database_path=str(db))

    options = pycolmap.IncrementalPipelineOptions()
    options.min_model_size = 3
    options.min_num_matches = 15
    options.ba_refine_focal_length = True
    options.ba_refine_extra_params = False
    options.ba_refine_principal_point = False
    mapper_options = options.get_mapper()
    mapper_options.init_min_num_inliers = 50
    mapper_options.init_min_tri_angle = 4.0

    maps = pycolmap.incremental_mapping(
        database_path=str(db), image_path=str(images_dir),
        output_path=str(COLMAP_OUT), options=options)
    if not maps:
        raise SystemExit("[sfm] NO model reconstructed")
    best_i = max(maps, key=lambda i: maps[i].num_images())
    rec = maps[best_i]
    cam = list(rec.cameras.values())[0]
    print(f"[sfm] {len(maps)} model(s); best #{best_i}: {rec.num_images()}/{len(selected)} images, "
          f"{rec.num_points3D()} points")
    try:
        print(f"[sfm] mean reprojection error: {rec.compute_mean_reprojection_error():.3f} px")
    except Exception:
        pass
    print(f"[sfm] refined camera params: {[round(float(x), 2) for x in cam.params]} "
          f"(init fx {FX_INIT:.1f})")

    sparse0 = DATASET / "sparse" / "0"
    if sparse0.exists():
        shutil.rmtree(sparse0)
    sparse0.mkdir(parents=True)
    rec.write(str(sparse0))
    print(f"[sfm] wrote {sparse0}")


def stage_milo(schedule: str, variant: str):
    sys.path.insert(0, str(REPO))
    sys.path.insert(0, str(REPO / "stages" / "stage5_reconstruction"))
    import milo_supervised as m

    suffix = "" if variant == "v1" else f"_{variant}"
    out = SESS / (f"output{suffix}" if schedule == "fast" else f"output{suffix}_{schedule}")
    options = {
        "depth_lambda": 0.0,               # DEPTH-FREE (the point of this test)
        "dense_gaussians": False,          # matches the preliminary run
        "data_device": "cuda",
        "imp_metric": "indoor",
        "milo_resolution": 1,              # 1920 max side, under the 2048 cap
        "milo_crop_pad": 0.10,
        "milo_schedule": schedule,         # "fast" (MS2 accelerated) or "quality" (stock 30k/densify-15k)
    }
    if schedule != "fast":
        options["mesh_config"] = schedule  # matching mesh schedule (configs/mesh/<schedule>.yaml)
    prov = m.reconstruct(
        dataset_dir=str(DATASET),
        capture_dir=str(DATASET / "images"),   # unused: depth_lambda=0 gates the LiDAR block off
        normals_dir=None,
        output_dir=str(out),
        options=options,
    )
    print(f"[milo] provenance: {prov}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["sfm", "milo"], required=True)
    ap.add_argument("--schedule", choices=["fast", "quality", "quality_mid"], default="fast",
                    help="MILo training schedule (quality = stock 30k, densify until 15k)")
    ap.add_argument("--variant", choices=list(VARIANTS), default="v1",
                    help="input-image variant (v1 = 172 blind-stride; v3 = ~362 sharpest-per-window)")
    args = ap.parse_args()
    v = VARIANTS[args.variant]
    DATASET, COLMAP_OUT = v["dataset"], v["colmap_out"]
    if args.stage == "sfm":
        stage_sfm(v["window"], v["sharpness"])
    else:
        stage_milo(args.schedule, args.variant)
