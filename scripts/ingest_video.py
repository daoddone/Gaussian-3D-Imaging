#!/usr/bin/env python3
"""T16a — native-camera video ingest: turn an ordinary .mp4/.mov (e.g. a clinician's iPhone
camera-app video) into a standard session layout so the existing chain runs unchanged:

  ingest_video.py --video ref.mov --session sessions/session_<name>
  session_sfm.py --session ...            (two-tier sharpness handles unguided video blur)
  04_metric_anchor.py --session ... --marker-mm 50    (marker-primary: no VIO/LiDAR in plain video)
  stage5 run.py --session ... --config config/pipeline_recommended.yaml

Frames are sampled uniformly in time to --max-frames; intrinsics.json gets a HEURISTIC focal
init (fx = 0.75 * max side — BA refines focal, so init only needs to be sane); metadata marks
the session external_video / no LiDAR so the anchor and branches behave correctly.
"""
import argparse
import json
from pathlib import Path

import cv2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--session", required=True, help="session dir to create")
    ap.add_argument("--max-frames", type=int, default=150)
    args = ap.parse_args()

    vid = Path(args.video)
    sess = Path(args.session)
    rgb = sess / "capture" / "rgb"
    rgb.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(vid))
    if not cap.isOpened():
        raise SystemExit(f"[ingest] cannot open {vid}")
    # iPhone portrait videos are landscape-native + rotation METADATA; make auto-rotation explicit
    # (default in recent OpenCV builds, but a sideways ingest would silently wreck SfM/marker work).
    try:
        cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)
    except Exception:
        pass
    n_src = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[ingest] {vid.name}: {n_src} frames @ {fps:.1f} fps, {W}x{H}")

    # uniform temporal sampling to max-frames (video is dense; SfM needs coverage, not 30 fps)
    keep = set(range(n_src)) if 0 < n_src <= args.max_frames else \
        {int(round(i * (n_src - 1) / (args.max_frames - 1))) for i in range(args.max_frames)} if n_src else None

    written = 0
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if keep is None or idx in keep:
            written += 1
            cv2.imwrite(str(rgb / f"{written:06d}.png"), frame)
        idx += 1
    cap.release()
    if written < 10:
        raise SystemExit(f"[ingest] only {written} frames extracted — video too short/unreadable")

    fx = 0.75 * max(W, H)   # heuristic init; session_sfm BA refines focal
    (sess / "capture" / "intrinsics.json").write_text(json.dumps({
        "K": [[fx, 0.0, W / 2.0], [0.0, fx, H / 2.0], [0.0, 0.0, 1.0]],
        "note": "HEURISTIC focal init from external video (no device intrinsics); BA-refined in SfM",
    }, indent=2))
    (sess / "capture" / "metadata.json").write_text(json.dumps({
        "capture_mode": "external_video", "lidar_enabled": False,
        "source_video": vid.name, "source_fps": fps, "source_frames": n_src,
        "ingested_frames": written, "color_resolution": [W, H],
    }, indent=2))
    print(f"[ingest] wrote {written} frames -> {rgb}")
    print(f"[ingest] next: session_sfm.py --session {sess}  then  "
          f"04_metric_anchor.py --session {sess} --marker-mm <size>")


if __name__ == "__main__":
    main()
