#!/usr/bin/env python3
"""Stage 5 entry point: reconstruction host (MILo, mesh-in-the-loop splatting).

    python stages/stage5_reconstruction/run.py --session <session_dir> --config <pipeline.yaml>

Optimizes surface-aligned Gaussians against the captured images while respecting
the metric depth and (optional) normal prior, and extracts a mesh in the loop.
Produces the Stage 6 outputs (io_contracts/reconstruction_output.md).

Runs in its OWN environment (pipeline_stage5_reconstruction) on the OLDER pinned
toolchain (Python 3.9 / CUDA 11.8 / PyTorch 2.3.1).

=============================================================================
 TWO PARTS OF THIS STAGE ARE NOT ROUTINE SCAFFOLDING. They are flagged, per
 KICKOFF.md, as needing a human engineer working alongside the agent:

   (H1) COMPILING MILo's pinned toolchain from source against the GPU
        (differentiable rasterizer submodules + nvdiffrast + a nearest-neighbor
        helper). From-source CUDA compilation is a common, real failure point.

   (H2) PORTING the depth-and-normal SUPERVISION from DN-Splatter / AGS-Mesh
        into MILo. This is a translation of loss-term CONCEPTS across two
        different frameworks, not a code copy. See README.md "The two hard
        tasks" for the exact loss recipe and the open questions to resolve.

 This run.py does the ROUTINE part it safely can — assembling the MILo dataset
 from the Stage 3 metric outputs — and then HALTS with a clear flag if the
 built+ported host is not present, instead of pretending to reconstruct.
=============================================================================
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[1]
for p in (str(_ROOT), str(_HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

import yaml

from common.file_layout import SessionLayout


def prepare_milo_dataset(layout: SessionLayout, out_dir: Path):
    """Assemble the COLMAP-style dataset MILo ingests, from Stage 3 metric output.

    Layout produced (an identity boundary — MILo reads COLMAP directly):
        <out_dir>/images/000001.png ...     (from capture/rgb, names match images.bin)
        <out_dir>/sparse/0/{cameras,images,points3D}.bin   (metric-locked)

    The metric COLMAP model already carries the baked init point cloud in
    points3D.bin (Stage 3), so MILo initializes from metric geometry even if it
    ignores an external .ply.
    """
    out_dir = Path(out_dir)
    images_dir = out_dir / "images"
    sparse_dir = out_dir / "sparse" / "0"
    images_dir.mkdir(parents=True, exist_ok=True)
    sparse_dir.mkdir(parents=True, exist_ok=True)

    metric_colmap = layout.metric_colmap
    if not (metric_colmap / "images.bin").exists():
        raise SystemExit(f"[stage5] missing metric COLMAP model at {metric_colmap}; run Stage 3 first")
    for f in ("cameras.bin", "images.bin", "points3D.bin"):
        shutil.copy2(metric_colmap / f, sparse_dir / f)

    rgb_ids = SessionLayout.list_frames(layout.capture_rgb, ".png")
    for fid in rgb_ids:
        src = layout.capture_rgb / f"{fid}.png"
        dst = images_dir / f"{fid}.png"
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        try:
            os.symlink(src.resolve(), dst)
        except OSError:
            shutil.copy2(src, dst)

    return {"dataset_dir": str(out_dir), "n_images": len(rgb_ids),
            "images_dir": str(images_dir), "sparse_dir": str(sparse_dir)}


def _host_ready():
    """Return (ready, detail). The built+ported MILo host must be importable and
    the supervision port must be present. Neither exists until H1/H2 are done."""
    try:
        import milo  # noqa: F401  -- the compiled MILo package
    except Exception as e:
        return False, f"MILo not importable ({type(e).__name__}); toolchain compile (H1) not done"
    # H2: the ported supervision module we will add to the MILo host
    ported = _HERE / "supervision" / "ags_depth_normal_losses.py"
    if not ported.exists():
        return False, "ported depth/normal supervision (H2) not present"
    return True, "host ready"


def run(session_dir, config_path):
    cfg = yaml.safe_load(open(config_path))
    s5 = cfg.get("stage5", {})
    layout = SessionLayout(session_dir)

    # ---- gsplat host (default): metric depth-supervised 3DGS + TSDF mesh -----
    # This is the working, disk-smart Stage 5 host (DN-Splatter/AGS-Mesh loss
    # recipe ported onto gsplat; no nerfstudio/MILo toolchain needed).
    host = s5.get("host", "gsplat")
    if host == "gsplat":
        try:
            from . import gsplat_recon
        except ImportError:
            import gsplat_recon
        opts = {"iters": int(s5.get("iterations", 7000)),
                "downscale": float(s5.get("downscale", 2.0)),
                "depth_lambda": float(s5.get("depth_lambda", 0.2)),
                "sh_degree": int(s5.get("sh_degree", 3))}
        gsplat_recon.reconstruct(str(layout.root), opts)
        return 0

    # ---- MILo path (host: milo) — still gated on the flagged H1/H2 tasks -----
    # ---- routine: assemble the MILo dataset from Stage 3 metric outputs -----
    dataset_dir = layout.metric.parent / "reconstruction_input"
    info = prepare_milo_dataset(layout, dataset_dir)
    print(f"[stage5] prepared MILo dataset: {info['n_images']} images at {info['dataset_dir']}")

    # ---- flagged: only proceed if the built+ported host exists --------------
    ready, detail = _host_ready()
    if not ready:
        print("\n" + "=" * 74)
        print("[stage5] HALTING — reconstruction host not ready. This is a FLAG,")
        print("         not a failure. Two human-in-the-loop tasks remain:")
        print(f"           reason: {detail}")
        print("  (H1) compile MILo's pinned toolchain from source against the GPU")
        print("  (H2) port the DN-Splatter/AGS-Mesh depth+normal supervision into MILo")
        print("       (loss-concept translation across frameworks, not a copy)")
        print("  See stages/stage5_reconstruction/README.md 'The two hard tasks'.")
        print("  The dataset is prepared and waiting at:")
        print(f"    {info['dataset_dir']}")
        print("=" * 74)
        return 4  # distinct code: stage prepared but host not built/ported yet

    # ---- when ready: drive MILo with the ported supervision -----------------
    # NOTE: this call shape is a placeholder for the ported entry point; the real
    # signature is finalized during H2 (see README). Kept explicit so the seam is
    # visible rather than hidden.
    from milo_supervised import reconstruct  # provided by the H2 port
    out = reconstruct(
        dataset_dir=info["dataset_dir"],
        capture_dir=str(layout.capture),
        normals_dir=str(layout.normals) if _stage4_enabled(cfg) else None,
        output_dir=str(layout.output),
        options=s5,
    )
    print(f"[stage5] reconstruction complete: {out}")
    return 0


def _stage4_enabled(cfg):
    return bool(cfg.get("stages", {}).get("stage4_normals", {}).get("enabled", False))


def main():
    ap = argparse.ArgumentParser(description="Stage 5: MILo reconstruction host")
    ap.add_argument("--session", required=True)
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    return run(args.session, args.config)


if __name__ == "__main__":
    sys.exit(main())
