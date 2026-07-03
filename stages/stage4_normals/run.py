#!/usr/bin/env python3
"""Stage 4 entry point: surface-normal prior (StableNormal). OPTIONAL stage.

    python stages/stage4_normals/run.py --session <session_dir> --config <pipeline.yaml>

Predicts a per-frame surface-normal map to gently regularize the reconstruction
in smooth, textureless regions. Writes the Stage 4 output contract
(io_contracts/normals_output.md):

    normals/000001.npy         [H,W,3] unit vectors in [-1,1], CAMERA frame, OpenCV
    normals_weight/000001.npy  [H,W] in [0,1]   (only if confidence-tied weighting)

Runs in its OWN environment (pipeline_stage4_normals; diffusion model, torch).

CRUCIAL convention conversion (docs/COMPONENT_IO_REFERENCE.md):
  StableNormal emits normals in the OpenGL camera frame (+Y up, +Z toward the
  camera). Our contract is OpenCV (+Y down, +Z into the scene). The wrapper
  negates the Y and Z channels (diag(1,-1,-1)) and renormalizes. This makes a
  camera-facing surface have n_z < 0, matching normals_from_depth and the
  orientation self-test. VALIDATE the global sign on one known planar frame
  before a batch run — the source convention is inferred, not documented.

The bias-free swap (normals straight from the metric depth, no learned model)
lives in normals_from_depth.py and is used by Section 8 Experiment B.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[1]
for p in (str(_ROOT), str(_HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
import yaml

from common.file_layout import SessionLayout

# OpenGL camera frame -> OpenCV camera frame (flip Y and Z)
_GL_TO_CV = np.array([1.0, -1.0, -1.0], dtype=np.float32)


def _to_opencv_normals(normals_gl):
    """Convert an OpenGL-frame normal map [H,W,3] to OpenCV and renormalize."""
    n = np.asarray(normals_gl, dtype=np.float32) * _GL_TO_CV[None, None, :]
    norm = np.linalg.norm(n, axis=2, keepdims=True)
    return (n / np.maximum(norm, 1e-9)).astype(np.float32)


def _trust_weight_from_confidence(conf_png_path):
    """Optional per-pixel trust weight tied to the Stage 1 validity mask.

    Start uniform; enable only if Experiment B shows the prior distorting deep
    wounds. Here: weight 1 where the sensor reads valid, downweighted where not.
    """
    from PIL import Image
    m = np.asarray(Image.open(conf_png_path))
    if m.ndim == 3:
        m = m[..., 0]
    return (m >= 128).astype(np.float32)  # 1.0 valid, 0.0 invalid


def run(session_dir, config_path):
    cfg = yaml.safe_load(open(config_path))
    s4 = cfg.get("stage4", {})
    layout = SessionLayout(session_dir)

    rgb_ids = SessionLayout.list_frames(layout.capture_rgb, ".png")
    if not rgb_ids:
        raise SystemExit(f"[stage4] no rgb frames under {layout.capture_rgb}")

    # ---- load StableNormal (requires the stage4 env: torch + diffusers) -----
    import torch
    from PIL import Image

    variant = s4.get("model", "StableNormal_turbo")
    predictor = torch.hub.load("Stable-X/StableNormal", variant, trust_repo=True)

    layout.normals.mkdir(parents=True, exist_ok=True)
    tied = bool(s4.get("confidence_tied_weight", False))
    if tied:
        layout.normals_weight.mkdir(parents=True, exist_ok=True)

    data_type = s4.get("data_type", "indoor")
    first_shape = None
    for fid in rgb_ids:
        img = Image.open(layout.capture_rgb / f"{fid}.png").convert("RGB")
        # Current Stable-X/StableNormal main API: predictor(img, data_type=...)
        # returns a PIL image (8-bit quantized normals), NOT an object with
        # .prediction and NO output_type kwarg. Decode to [-1,1] float. (For an
        # unquantized float path, call predictor.model(img_resized,
        # match_input_resolution=True).prediction[0] instead.)
        pil = predictor(img, data_type=data_type)
        normals_gl = np.asarray(pil).astype(np.float32) / 127.5 - 1.0  # [H,W,3] in ~[-1,1]
        normals_cv = _to_opencv_normals(normals_gl)
        np.save(layout.normals / f"{fid}.npy", normals_cv)
        first_shape = normals_cv.shape

        if tied:
            conf_path = layout.capture_confidence / f"{fid}.png"
            if conf_path.exists():
                w = _trust_weight_from_confidence(conf_path)
                # resample to normal-map resolution if needed
                if w.shape != normals_cv.shape[:2]:
                    from PIL import Image as _Im
                    w = np.asarray(_Im.fromarray((w * 255).astype(np.uint8)).resize(
                        (normals_cv.shape[1], normals_cv.shape[0]))) / 255.0
                np.save(layout.normals_weight / f"{fid}.npy", w.astype(np.float32))

    # folder README noting resolution/convention (contract asks for this if it differs)
    (layout.normals / "README").write_text(
        f"Stage 4 normals. Convention: OpenCV camera frame (n_z<0 faces camera), "
        f"unit vectors in [-1,1]. Resolution: {first_shape[:2] if first_shape else '?'}. "
        f"Model: {variant}. confidence_tied_weight={tied}.\n")

    print(f"[stage4] wrote normals/ for {len(rgb_ids)} frames; model={variant}; "
          f"confidence_tied_weight={tied}")
    return 0


def main():
    ap = argparse.ArgumentParser(description="Stage 4: StableNormal surface-normal prior")
    ap.add_argument("--session", required=True)
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    return run(args.session, args.config)


if __name__ == "__main__":
    sys.exit(main())
