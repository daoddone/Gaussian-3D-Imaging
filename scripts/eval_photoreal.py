#!/usr/bin/env python3
"""Photorealism metric harness (T7): PSNR/SSIM (+LPIPS) of a trained Gaussian-splat
reconstruction, rendered against its own source images.

HONEST LABELING: our historical runs train on ALL views (no held-out split), so these
numbers are "train-view reconstruction fidelity (not novel-view generalization)".
The label is embedded in the JSON and the printout.

What it does
  * loads the trained MILo model from <output-dir>/_milo_raw (latest iteration_* under
    point_cloud/ unless --iteration N) and the COLMAP dataset it trained on from
    <output-dir>/_scaled_dataset  — both live in the SCALED frame, so rendering vs the
    dataset images needs no rescaling (poses and gaussians are consistent);
  * renders every Nth training view with the radegs rasterizer (the rasterizer our
    Stage-5 training uses: milo_supervised passes --rasterizer radegs);
  * computes per-view + mean PSNR / SSIM (and LPIPS if available) with EXACTLY the
    conventions of MILo's own metrics.py ([1,3,H,W] tensors in [0,1]; overall-MSE PSNR;
    window-11 SSIM; lpips vgg on [0,1] inputs — the 3DGS-lineage benchmark convention);
  * writes <outdir>/photoreal.json + GT|render|abs-diff comparison PNGs for the
    best / median / worst views, and prints a one-line summary.

How to run (the script is a self-wrapper, same pattern as stage5's milo_supervised.py:
it re-executes itself under the milo conda env with cwd=third_party/MILo/milo and the
CUDA_HOME/PATH/LD_LIBRARY_PATH env vars pointing at ~/miniforge3/envs/milo):

    python3 scripts/eval_photoreal.py --output-dir sessions/<sess>/output_X \
        [--every 8] [--iteration latest] [--max-views 40] \
        [--outdir sessions/<sess>/output_X/review/photoreal]

LPIPS policy: try the pip `lpips` package first; if absent, fall back to MILo's bundled
lpipsPyTorch (vgg — what MILo's metrics.py itself uses; weights already cached in
~/.cache/torch). If neither works, skip gracefully and note "lpips: not installed".
Nothing is ever pip-installed by this script.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
MILO_DIR = _REPO / "third_party" / "MILo" / "milo"
MILO_ENV = Path(os.path.expanduser("~/miniforge3/envs/milo"))
MILO_ENV_PY = MILO_ENV / "bin" / "python"

HONEST_LABEL = "train-view reconstruction fidelity (not novel-view generalization)"
DIFF_GAIN = 4.0  # abs-diff panels are amplified by this factor for visibility


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--output-dir", required=True,
                    help="Stage-5 output dir containing _milo_raw/ and _scaled_dataset/")
    ap.add_argument("--every", type=int, default=8, help="evaluate every Nth training view")
    ap.add_argument("--iteration", default="latest",
                    help="'latest' or an integer iteration under _milo_raw/point_cloud")
    ap.add_argument("--max-views", type=int, default=40, help="cap on evaluated views")
    ap.add_argument("--outdir", default=None,
                    help="where to write photoreal.json + PNGs (default: <output-dir>/review/photoreal)")
    return ap.parse_args(argv)


# --------------------------------------------------------------------------- #
# outer wrapper: re-exec under the milo env (same env recipe as milo_supervised)
# --------------------------------------------------------------------------- #
def relaunch_in_milo_env(args):
    out_dir = Path(args.output_dir).resolve()
    milo_raw = out_dir / "_milo_raw"
    scaled_ds = out_dir / "_scaled_dataset"
    for p, what in ((out_dir, "output dir"), (milo_raw, "_milo_raw (trained MILo model)"),
                    (scaled_ds, "_scaled_dataset (COLMAP dataset it trained on)")):
        if not p.exists():
            sys.exit(f"[photoreal] ERROR: {what} not found: {p}")
    if not MILO_ENV_PY.exists():
        sys.exit(f"[photoreal] ERROR: milo env python not found: {MILO_ENV_PY}")
    outdir = Path(args.outdir).resolve() if args.outdir else out_dir / "review" / "photoreal"

    env = dict(os.environ)
    env["CUDA_HOME"] = str(MILO_ENV)
    env["PATH"] = f"{MILO_ENV}/bin:" + env.get("PATH", "")
    env["LD_LIBRARY_PATH"] = f"{MILO_ENV}/lib:" + env.get("LD_LIBRARY_PATH", "")
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    env["EVAL_PHOTOREAL_INNER"] = "1"

    cmd = [str(MILO_ENV_PY), str(Path(__file__).resolve()),
           "--output-dir", str(out_dir), "--every", str(args.every),
           "--iteration", str(args.iteration), "--max-views", str(args.max_views),
           "--outdir", str(outdir)]
    # cwd MUST be the milo dir (MILo code resolves configs/submodules relative to it)
    r = subprocess.run(cmd, cwd=str(MILO_DIR), env=env)
    sys.exit(r.returncode)


# --------------------------------------------------------------------------- #
# inner: runs inside the milo env, cwd = third_party/MILo/milo
# --------------------------------------------------------------------------- #
def make_comparison_png(gt, render, path, title):
    """GT | render | abs-diff (xDIFF_GAIN) side-by-side with a caption strip."""
    import numpy as np
    from PIL import Image, ImageDraw

    def to_u8(t):  # [3,H,W] in [0,1] -> HxWx3 uint8
        return (t.clamp(0, 1).detach().cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)

    diff = (render - gt).abs().mean(dim=0, keepdim=True).repeat(3, 1, 1) * DIFF_GAIN
    panels = [to_u8(gt), to_u8(render), to_u8(diff)]
    h, w = panels[0].shape[:2]
    gutter, strip = 6, 34
    canvas = np.full((h + strip, 3 * w + 2 * gutter, 3), 255, np.uint8)
    for i, p in enumerate(panels):
        canvas[strip:, i * (w + gutter):i * (w + gutter) + w] = p
    img = Image.fromarray(canvas)
    d = ImageDraw.Draw(img)
    d.text((6, 10), title, fill=(0, 0, 0))
    for i, lbl in enumerate(("GT", "render", f"|diff| x{DIFF_GAIN:g}")):
        d.text((i * (w + gutter) + 6, strip + 6), lbl, fill=(255, 60, 60))
    img.save(path)


def run_inner(args):
    sys.path.insert(0, str(MILO_DIR))
    import torch
    torch.cuda.set_device(torch.device("cuda:0"))

    from argparse import ArgumentParser
    from arguments import ModelParams, PipelineParams, get_combined_args
    from scene import Scene
    from gaussian_renderer import GaussianModel
    from gaussian_renderer.radegs import render_radegs as render_fn
    from utils.loss_utils import ssim as ssim_fn
    from utils.image_utils import psnr as psnr_fn

    out_dir = Path(args.output_dir)
    milo_raw = out_dir / "_milo_raw"
    scaled_ds = out_dir / "_scaled_dataset"
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # --- canonical arg construction (as MILo's render.py / mesh_extract_sdf.py do):
    # ModelParams(sentinel) + cfg_args from the model dir, with -m/-s overridden to
    # our absolute paths. get_combined_args reads sys.argv, so feed it a synthetic one.
    mp = ArgumentParser()
    model = ModelParams(mp, sentinel=True)
    pipeline = PipelineParams(mp)
    _argv = sys.argv
    sys.argv = [sys.argv[0], "-m", str(milo_raw), "-s", str(scaled_ds)]
    try:
        combined = get_combined_args(mp)
    finally:
        sys.argv = _argv
    dataset = model.extract(combined)
    pipe = pipeline.extract(combined)

    it = -1 if str(args.iteration) == "latest" else int(args.iteration)

    lpips_fn, lpips_note = None, "not installed"
    try:
        import lpips as _lpips_pkg  # preferred: the pip package
        _net = _lpips_pkg.LPIPS(net="vgg").cuda()
        lpips_fn = lambda a, b: _net(a, b).item()  # noqa: E731
        lpips_note = "pip lpips (vgg)"
    except Exception:
        try:  # fallback: MILo's bundled lpipsPyTorch — what its own metrics.py uses
            from lpipsPyTorch import lpips as _lpips_milo
            lpips_fn = lambda a, b: _lpips_milo(a, b, net_type="vgg").item()  # noqa: E731
            lpips_note = "lpipsPyTorch vgg (MILo bundled, as metrics.py)"
        except Exception as e:
            lpips_note = f"not installed (pip 'lpips' missing; bundled lpipsPyTorch failed: {e})"
    if lpips_fn is None:
        print("[photoreal] lpips: not installed — skipping LPIPS")

    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=it, shuffle=False)
        loaded_iter = scene.loaded_iter
        n_gauss = int(gaussians.get_xyz.shape[0])
        bg = torch.tensor([1, 1, 1] if dataset.white_background else [0, 0, 0],
                          dtype=torch.float32, device="cuda")
        views = scene.getTrainCameras()
        sel = list(range(0, len(views), max(1, args.every)))
        if len(sel) > args.max_views:  # even subsample so the whole trajectory stays covered
            import numpy as np
            sel = [sel[i] for i in np.linspace(0, len(sel) - 1, args.max_views).round().astype(int)]
        print(f"[photoreal] iter {loaded_iter}, {n_gauss} gaussians; evaluating "
              f"{len(sel)}/{len(views)} train views (every {args.every}, cap {args.max_views})")

        def render_view(view):
            img = render_fn(view, gaussians, pipe, bg,
                            kernel_size=dataset.kernel_size)["render"].clamp(0.0, 1.0)
            gt = view.original_image[0:3].to("cuda").clamp(0.0, 1.0)
            return img, gt

        per_view, failures = [], []
        t0 = time.time()
        for k, i in enumerate(sel):
            view = views[i]
            try:
                img, gt = render_view(view)
                r, g = img.unsqueeze(0), gt.unsqueeze(0)
                rec = {"view": view.image_name,
                       "psnr": round(psnr_fn(r, g).mean().item(), 4),
                       "ssim": round(ssim_fn(r, g).item(), 5)}
                if lpips_fn is not None:
                    rec["lpips"] = round(lpips_fn(r, g), 5)
                per_view.append(rec)
                print(f"[photoreal] {k + 1:3d}/{len(sel)}  {view.image_name:<16} "
                      f"PSNR {rec['psnr']:6.2f}  SSIM {rec['ssim']:.4f}"
                      + (f"  LPIPS {rec['lpips']:.4f}" if "lpips" in rec else ""))
            except Exception as e:  # report faithfully, keep going
                failures.append({"view": view.image_name, "error": str(e)})
                print(f"[photoreal] {k + 1:3d}/{len(sel)}  {view.image_name}: RENDER FAILED: {e}")
        dt = time.time() - t0
        if not per_view:
            sys.exit(f"[photoreal] ERROR: all {len(sel)} renders failed; first: {failures[:1]}")

        # --- comparison PNGs for best / median / worst PSNR views (re-rendered)
        by_name = {v.image_name: v for v in views}
        ranked = sorted(per_view, key=lambda r: r["psnr"])
        picks = {"worst": ranked[0], "median": ranked[len(ranked) // 2], "best": ranked[-1]}
        comparisons = {}
        for tag, rec in picks.items():
            img, gt = render_view(by_name[rec["view"]])
            png = outdir / f"compare_{tag}_{Path(rec['view']).stem}.png"
            title = (f"{tag.upper()}  {rec['view']}   PSNR {rec['psnr']:.2f} dB  "
                     f"SSIM {rec['ssim']:.4f}" + (f"  LPIPS {rec['lpips']:.4f}" if "lpips" in rec else "")
                     + f"   [{HONEST_LABEL}]")
            make_comparison_png(gt, img, png, title)
            comparisons[tag] = str(png)
        h, w = int(views[sel[0]].image_height), int(views[sel[0]].image_width)

    mean = lambda k: sum(r[k] for r in per_view) / len(per_view)  # noqa: E731
    summary = {"psnr_mean": round(mean("psnr"), 3),
               "psnr_min": ranked[0]["psnr"], "psnr_max": ranked[-1]["psnr"],
               "ssim_mean": round(mean("ssim"), 4),
               "ssim_min": min(r["ssim"] for r in per_view),
               "lpips_mean": round(mean("lpips"), 4) if lpips_fn is not None else None}
    result = {
        "task": "T7 photorealism metric harness",
        "label": HONEST_LABEL,
        "output_dir": str(out_dir), "model_dir": str(milo_raw), "dataset_dir": str(scaled_ds),
        "iteration": loaded_iter, "gaussians": n_gauss, "rasterizer": "radegs",
        "render_resolution": [w, h],
        "views_total": len(views), "every": args.every, "views_evaluated": len(per_view),
        "render_failures": failures,
        "lpips": lpips_note,
        "metrics_convention": ("MILo metrics.py conventions: [1,3,H,W] tensors in [0,1]; "
                               "PSNR over all-pixel MSE; SSIM window=11; LPIPS vgg on [0,1] "
                               "inputs (3DGS-lineage benchmark convention)"),
        "note": ("model + dataset are both in MILo's scaled training frame; scale cancels "
                 "for image metrics"),
        "eval_seconds": round(dt, 1),
        "summary": summary,
        "per_view": per_view,
        "comparisons": comparisons,
    }
    (outdir / "photoreal.json").write_text(json.dumps(result, indent=2))

    lp = f"LPIPS {summary['lpips_mean']:.4f}" if summary["lpips_mean"] is not None else "LPIPS n/a"
    print(f"[photoreal] SUMMARY: PSNR {summary['psnr_mean']:.2f} dB | SSIM {summary['ssim_mean']:.4f} | "
          f"{lp} | {len(per_view)} views @ iter {loaded_iter} — {HONEST_LABEL}")
    print(f"[photoreal] wrote {outdir / 'photoreal.json'} + {len(comparisons)} comparison PNGs")
    if failures:
        print(f"[photoreal] WARNING: {len(failures)} view(s) failed to render — see photoreal.json")


if __name__ == "__main__":
    _args = parse_args()
    if os.environ.get("EVAL_PHOTOREAL_INNER") == "1":
        run_inner(_args)
    else:
        relaunch_in_milo_env(_args)
