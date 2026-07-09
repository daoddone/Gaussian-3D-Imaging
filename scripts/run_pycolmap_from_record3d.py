#!/usr/bin/env python3
"""
Record3D -> pycolmap sparse reconstruction.

Default behavior uses a moderate stride (frame_step=5), which is typically a
better quality/runtime balance on shared Tesla T4 sessions.
"""

import argparse
import json
import sqlite3
import shutil
from pathlib import Path

import pycolmap


def sorted_jpgs(rgb_dir: Path):
    imgs = [p for p in rgb_dir.iterdir() if p.suffix.lower() == ".jpg"]
    return sorted(imgs, key=lambda p: int(p.stem) if p.stem.isdigit() else p.stem)


def copy_frames(src_rgb: Path, dst_images: Path, frame_step: int) -> int:
    dst_images.mkdir(parents=True, exist_ok=True)
    # Ensure reruns with different frame_step values do not mix stale frames.
    for old in dst_images.iterdir():
        if old.is_file() and old.suffix.lower() == ".jpg":
            old.unlink()
    files = sorted_jpgs(src_rgb)
    selected = files[::frame_step]
    for f in selected:
        shutil.copy2(f, dst_images / f.name)
    return len(selected)


def remove_existing_mapping(output_root: Path) -> None:
    db = output_root / "database.db"
    if db.exists():
        db.unlink()
    if output_root.exists():
        for p in output_root.iterdir():
            if p.is_dir() and p.name.isdigit():
                shutil.rmtree(p)


def run_reconstruction(
    images_dir: Path,
    output_root: Path,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    matcher: str,
) -> None:
    database_path = output_root / "database.db"

    reader_options = pycolmap.ImageReaderOptions()
    reader_options.camera_model = "PINHOLE"
    reader_options.camera_params = f"{fx},{fy},{cx},{cy}"

    pycolmap.extract_features(
        database_path=str(database_path),
        image_path=str(images_dir),
        camera_mode=pycolmap.CameraMode.SINGLE,
        reader_options=reader_options,
    )
    enforce_pinhole_camera_model(database_path)

    if matcher == "sequential":
        pycolmap.match_sequential(database_path=str(database_path))
    elif matcher == "exhaustive":
        pycolmap.match_exhaustive(database_path=str(database_path))
    else:
        raise ValueError(f"Unknown matcher: {matcher}")

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
        database_path=str(database_path),
        image_path=str(images_dir),
        output_path=str(output_root),
        options=options,
    )

    print(f"Reconstruction complete. Found {len(maps)} model(s).")
    for i, m in maps.items():
        print(f"  Model {i}: {m.num_images()} images, {m.num_points3D()} 3D points")


def enforce_pinhole_camera_model(database_path: Path) -> None:
    """Guard against pycolmap writing SIMPLE_RADIAL with 4-parameter intrinsics."""
    conn = sqlite3.connect(str(database_path))
    rows = conn.execute("SELECT camera_id, model FROM cameras").fetchall()
    changed = False
    for camera_id, model in rows:
        # COLMAP camera model id: 1 = PINHOLE.
        if int(model) != 1:
            conn.execute("UPDATE cameras SET model = 1 WHERE camera_id = ?", (camera_id,))
            changed = True
    if changed:
        # Existing geometric verifications are invalid if camera model changed.
        conn.execute("DELETE FROM two_view_geometries")
        conn.commit()
        print("Adjusted camera model(s) to PINHOLE and cleared two_view_geometries.")
    else:
        conn.commit()
    conn.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scan-root",
        type=Path,
        default=Path(r"\\apporto.com\dfs\NTHW\Users\yri0347_nthw\2026-04-13--17-46-08"),
        help="Record3D export root containing EXR_RGBD.",
    )
    parser.add_argument(
        "--work-root",
        type=Path,
        default=Path(r"C:\Users\yri0347_nthw\face_scan_fullfps"),
        help="Local working directory for colmap input/output.",
    )
    parser.add_argument(
        "--frame-step",
        type=int,
        default=5,
        help="Frame stride. 5 is recommended for most face scans on shared T4 GPUs.",
    )
    parser.add_argument(
        "--matcher",
        choices=["sequential", "exhaustive"],
        default="sequential",
        help="Sequential is recommended for full video frame sets.",
    )
    args = parser.parse_args()

    rgb_dir = args.scan_root / "EXR_RGBD" / "rgb"
    meta_path = args.scan_root / "EXR_RGBD" / "metadata.json"
    if not rgb_dir.exists():
        raise FileNotFoundError(f"Missing rgb directory: {rgb_dir}")
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing metadata.json: {meta_path}")

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    k = meta["K"]
    fx = float(k[0])
    fy = float(k[4]) if len(k) > 4 else float(k[0])
    cx = float(k[6])
    cy = float(k[7])

    images_dir = args.work_root / "colmap_input" / "images"
    output_root = args.work_root / "colmap_output"
    output_root.mkdir(parents=True, exist_ok=True)

    copied = copy_frames(rgb_dir, images_dir, args.frame_step)
    print(f"Copied {copied} frame(s) to {images_dir}")
    print(
        f"Using intrinsics PINHOLE fx={fx:.3f}, fy={fy:.3f}, "
        f"cx={cx:.3f}, cy={cy:.3f}"
    )

    remove_existing_mapping(output_root)
    run_reconstruction(images_dir, output_root, fx, fy, cx, cy, args.matcher)

    print("\nExpected sparse model output in:")
    print(output_root / "0")


if __name__ == "__main__":
    main()
