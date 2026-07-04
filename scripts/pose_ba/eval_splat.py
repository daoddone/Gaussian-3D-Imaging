#!/usr/bin/env python3
"""Render a trained INRIA-format splat over all COLMAP views and score it.

Loads point_cloud.ply (as written by gsplat_recon.export_ply), rebuilds gsplat params,
renders every view at FULL resolution, and reports per-view:
  - Laplacian variance  (sharpness; higher = sharper)
  - PSNR / SSIM vs the source photo  (fidelity)

Same code + same output resolution for every model -> directly comparable. Lets us
compare the half-res baseline, full-res arm A (ARKit) and arm B (seeded-BA) apples to
apples, and against the source photos.

Usage: eval_splat.py <splat.ply> <colmap_sparse0_dir>
CUDA_HOME must be set (gsplat JIT). Run in the pipeline_stage2_frontend env.
"""
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, "/home/paperspace/Documents/VS Code Projects/3D-Gaussian")
from common import colmap_io
from PIL import Image

SPLAT = Path(sys.argv[1])
COLMAP = Path(sys.argv[2])
# derive the session (source RGB) from the COLMAP path, NOT a hardcoded session:
# walk up until we find the dir containing capture/rgb, so cross-session evals compare
# renders against the CORRECT source photos.
SESS = COLMAP.resolve()
while SESS != SESS.parent and not (SESS / "capture" / "rgb").is_dir():
    SESS = SESS.parent
RGB = SESS / "capture/rgb"
DEV = "cuda"
SH_C0 = 0.28209479177387814


def load_splat_ply(path):
    """Invert gsplat_recon.export_ply -> gsplat params on device."""
    import struct
    with open(path, "rb") as fh:
        assert fh.readline().strip() == b"ply"
        names, count = [], 0
        while True:
            ln = fh.readline().decode().strip()
            if ln.startswith("element vertex"):
                count = int(ln.split()[-1])
            elif ln.startswith("property float"):
                names.append(ln.split()[-1])
            elif ln == "end_header":
                break
        data = np.frombuffer(fh.read(count * len(names) * 4), np.float32).reshape(count, len(names))
    col = {n: data[:, i] for i, n in enumerate(names)}
    n = count
    means = torch.tensor(np.stack([col["x"], col["y"], col["z"]], 1), device=DEV)
    f_dc = np.stack([col[f"f_dc_{i}"] for i in range(3)], 1)                     # (N,3)
    rest_cols = sorted((c for c in names if c.startswith("f_rest_")), key=lambda c: int(c.split("_")[-1]))
    f_rest = np.stack([col[c] for c in rest_cols], 1)                            # (N,45) channel-major
    k = f_rest.shape[1] // 3
    shN = f_rest.reshape(n, 3, k).transpose(0, 2, 1)                            # -> (N,15,3)
    sh0 = f_dc[:, None, :]                                                       # (N,1,3)
    opac = col["opacity"]
    scales = np.stack([col[f"scale_{i}"] for i in range(3)], 1)
    quats = np.stack([col[f"rot_{i}"] for i in range(4)], 1)
    return dict(
        means=means,
        sh0=torch.tensor(sh0, device=DEV), shN=torch.tensor(shN, device=DEV),
        opacities=torch.tensor(opac, device=DEV),
        scales=torch.tensor(scales, device=DEV),
        quats=torch.tensor(quats, device=DEV),
    )


def lap_var(gray):  # gray: HxW float tensor
    lap = (-4 * gray[1:-1, 1:-1] + gray[:-2, 1:-1] + gray[2:, 1:-1] + gray[1:-1, :-2] + gray[1:-1, 2:])
    return lap.var().item()


def main():
    from gsplat import rasterization
    p = load_splat_ply(SPLAT)
    cams = colmap_io.read_cameras_binary(COLMAP / "cameras.bin")
    imgs = colmap_io.read_images_binary(COLMAP / "images.bin")

    colors = torch.cat([p["sh0"], p["shN"]], dim=1)
    means, quats = p["means"], F.normalize(p["quats"], dim=-1)
    scales, opac = torch.exp(p["scales"]), torch.sigmoid(p["opacities"])

    lv_r, lv_s, psnr, ssim_l = [], [], [], []
    win = None
    for im in imgs.values():
        cam = cams[im["camera_id"]]
        fx, fy, cx, cy = cam["params"][:4]
        W, H = cam["width"], cam["height"]
        K = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], device=DEV, dtype=torch.float32)
        R = torch.tensor(_q2R(im["qvec"]), device=DEV, dtype=torch.float32)
        t = torch.tensor(im["tvec"], device=DEV, dtype=torch.float32)
        vm = torch.eye(4, device=DEV); vm[:3, :3] = R; vm[:3, 3] = t
        with torch.no_grad():
            rend, _, _ = rasterization(means, quats, scales, opac, colors,
                                       vm[None], K[None], W, H, sh_degree=3,
                                       render_mode="RGB", packed=True)
        img = rend[0, ..., :3].clamp(0, 1)                       # H,W,3
        gray = img.mean(-1)
        lv_r.append(lap_var(gray))

        src_p = RGB / im["name"]
        if src_p.exists():
            src = torch.tensor(np.asarray(Image.open(src_p).convert("RGB"), np.float32) / 255.0, device=DEV)
            if src.shape[:2] != img.shape[:2]:
                src = F.interpolate(src.permute(2, 0, 1)[None], size=img.shape[:2], mode="bilinear",
                                    align_corners=False)[0].permute(1, 2, 0)
            lv_s.append(lap_var(src.mean(-1)))
            mse = ((img - src) ** 2).mean().item()
            psnr.append(-10 * np.log10(max(mse, 1e-10)))
            win, sv = _ssim(img, src, win)
            ssim_l.append(sv)

    def st(a): a = np.array(a); return f"mean {a.mean():8.1f}  median {np.median(a):8.1f}"
    print(f"  views rendered      : {len(lv_r)}  @ {W}x{H}")
    print(f"  render  Laplacian var: {st(lv_r)}")
    if lv_s:
        print(f"  source  Laplacian var: {st(lv_s)}   (render/source = {np.mean(lv_r)/np.mean(lv_s):.3f})")
        print(f"  PSNR vs source (dB)  : mean {np.mean(psnr):.2f}  median {np.median(psnr):.2f}")
        print(f"  SSIM vs source       : mean {np.mean(ssim_l):.4f}  median {np.median(ssim_l):.4f}")


def _q2R(q):
    w, x, y, z = q
    return np.array([[1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
                     [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
                     [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]], np.float32)


def _ssim(a, b, win):
    a = a.permute(2, 0, 1)[None]; b = b.permute(2, 0, 1)[None]
    if win is None:
        ws, sigma, ch = 11, 1.5, 3
        c = torch.arange(ws, device=a.device, dtype=torch.float32) - ws // 2
        g = torch.exp(-(c ** 2) / (2 * sigma ** 2)); g /= g.sum()
        win = (g[:, None] * g[None, :]).expand(ch, 1, ws, ws).contiguous()
    pad, ch = win.shape[-1] // 2, 3
    mu_a = F.conv2d(a, win, padding=pad, groups=ch); mu_b = F.conv2d(b, win, padding=pad, groups=ch)
    sa = F.conv2d(a * a, win, padding=pad, groups=ch) - mu_a ** 2
    sb = F.conv2d(b * b, win, padding=pad, groups=ch) - mu_b ** 2
    sab = F.conv2d(a * b, win, padding=pad, groups=ch) - mu_a * mu_b
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    s = ((2 * mu_a * mu_b + c1) * (2 * sab + c2)) / ((mu_a ** 2 + mu_b ** 2 + c1) * (sa + sb + c2))
    return win, s.mean().item()


if __name__ == "__main__":
    main()
