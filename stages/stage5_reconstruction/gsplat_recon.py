#!/usr/bin/env python3
"""Stage 5 reconstruction: metric depth-supervised 3D Gaussian Splatting on gsplat.

This is the disk-smart Stage 5 host (the DN-Splatter/AGS-Mesh fallback ported onto
gsplat, avoiding a heavy nerfstudio env). It optimizes surface-aligned Gaussians
to reproduce the captured images while being supervised by the metric LiDAR depth
(AGS-Mesh EdgeAwareLogL1), initialized from the Stage 3 metric point cloud, and
extracts a metric TSDF mesh from the rendered depth. Produces the Stage 6
contract: output/point_cloud.ply (Gaussian splat), output/mesh.ply, output/renders/.

Inputs (from the session): metric/colmap/sparse/0 (metric cameras, world_to_camera),
metric/points_metric.ply (init cloud), capture/rgb/*.png, capture/depth/*.npy +
capture/confidence/*.png (metric depth supervision + validity).

Runs in the pipeline_stage2_frontend env (torch 2.5.1+cu124 + gsplat).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[1]
for p in (str(_ROOT), str(_HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from common import colmap_io, plyio
from common import conventions as C
from common.file_layout import SessionLayout

SH_C0 = 0.28209479177387814  # Y_0^0


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
def _load_confidence(path):
    a = np.asarray(Image.open(path))
    if a.ndim == 3:
        a = a[..., 0]
    return a >= 128


def load_dataset(session, downscale, device):
    """Return list of per-frame dicts and a scene-scale estimate."""
    lay = SessionLayout(session)
    cams = colmap_io.read_cameras_binary(lay.metric_colmap / "cameras.bin")
    imgs = colmap_io.read_images_binary(lay.metric_colmap / "images.bin")

    frames = []
    for img_id in sorted(imgs):
        im = imgs[img_id]
        cam = cams[im["camera_id"]]
        fx, fy, cx, cy = cam["params"][:4]
        W0, H0 = cam["width"], cam["height"]

        name = im["name"]                       # e.g. 000001.png
        fid = Path(name).stem
        rgb_path = lay.capture_rgb / name
        if not rgb_path.exists():
            continue
        pil = Image.open(rgb_path).convert("RGB")
        W = int(round(W0 / downscale)); H = int(round(H0 / downscale))
        pil = pil.resize((W, H), Image.BILINEAR)
        rgb = torch.from_numpy(np.asarray(pil, np.float32) / 255.0).to(device)  # H,W,3
        s = W / W0
        K = torch.tensor([[fx * s, 0, cx * s], [0, fy * s, cy * s], [0, 0, 1]],
                         dtype=torch.float32, device=device)

        R = C.quat_to_rotmat(im["qvec"]); t = np.asarray(im["tvec"], float)
        viewmat = torch.eye(4, dtype=torch.float32, device=device)
        viewmat[:3, :3] = torch.from_numpy(R.astype(np.float32))
        viewmat[:3, 3] = torch.from_numpy(t.astype(np.float32))

        # metric sensor depth + validity, resized to render res (same FoV as color)
        depth = np.load(lay.capture_depth / f"{fid}.npy").astype(np.float32)
        valid = np.isfinite(depth)
        conf = lay.capture_confidence / f"{fid}.png"
        if conf.exists():
            valid &= _load_confidence(conf)
        d = np.where(valid, depth, 0.0)
        dt = torch.from_numpy(d)[None, None].to(device)
        mt = torch.from_numpy(valid.astype(np.float32))[None, None].to(device)
        # Mask-normalized resize: average only valid pixels so the 0 hole-fill
        # never bleeds into valid depth targets (which would bias the metric
        # depth loss small in a ring around every hole).
        num = F.interpolate(dt * mt, size=(H, W), mode="bilinear", align_corners=False)
        den = F.interpolate(mt, size=(H, W), mode="bilinear", align_corners=False)
        d_r = (num / den.clamp_min(1e-6))[0, 0]
        m_r = den[0, 0] > 0.5

        # optional Stage 4 normal prior (camera frame, OpenCV), resized to render res
        normal = nmask = None
        npath = lay.normals / f"{fid}.npy"
        if npath.exists():
            nn = np.load(npath).astype(np.float32)      # (Hn,Wn,3)
            ntt = torch.from_numpy(nn).permute(2, 0, 1)[None].to(device)
            n_r = F.interpolate(ntt, size=(H, W), mode="bilinear", align_corners=False)[0].permute(1, 2, 0)
            normal = F.normalize(n_r, dim=-1)
            nmask = normal.norm(dim=-1) > 0.1

        frames.append({"fid": fid, "rgb": rgb, "K": K, "viewmat": viewmat,
                       "depth": d_r, "dmask": m_r, "normal": normal, "nmask": nmask,
                       "W": W, "H": H})

    # scene scale = mean camera-center distance to their centroid
    centers = []
    for f in frames:
        Rt = f["viewmat"].detach().cpu().numpy()
        centers.append(-Rt[:3, :3].T @ Rt[:3, 3])
    centers = np.array(centers)
    scene_scale = float(np.linalg.norm(centers - centers.mean(0), axis=1).mean()) or 1.0
    return frames, scene_scale


def build_gaussians(init_ply, colors_available, device, max_points=200000):
    cloud = plyio.read_ply(init_ply)
    pts = cloud["points"].astype(np.float32)
    cols = cloud.get("colors")
    if pts.shape[0] > max_points:
        sel = np.random.default_rng(0).choice(pts.shape[0], max_points, replace=False)
        pts = pts[sel]; cols = None if cols is None else cols[sel]
    N = pts.shape[0]
    means = torch.tensor(pts, device=device)

    # init scale from mean distance to 3 nearest neighbours, chunked so the
    # pairwise distance matrix never materialises in full (would be ~16 GB).
    with torch.no_grad():
        sub = means[torch.randperm(N, device=device)[:min(N, 15000)]]
        parts = []
        for i in range(0, N, 4096):
            d = torch.cdist(means[i:i + 4096], sub)          # (<=4096, |sub|)
            knn = d.topk(4, largest=False).values[:, 1:]     # drop self/nearest
            parts.append(knn.mean(1))
        dist = torch.cat(parts).clamp_min(1e-6)
    scales = torch.log(dist)[:, None].repeat(1, 3)
    quats = torch.zeros(N, 4, device=device); quats[:, 0] = 1.0
    opacities = torch.logit(torch.full((N,), 0.1, device=device))

    if cols is not None:
        rgb = torch.tensor(cols[:, :3].astype(np.float32) / 255.0, device=device)
    else:
        rgb = torch.full((N, 3), 0.5, device=device)
    sh0 = ((rgb - 0.5) / SH_C0)[:, None, :]              # (N,1,3)
    shN = torch.zeros(N, 15, 3, device=device)           # sh_degree 3 -> 15 rest coeffs

    params = torch.nn.ParameterDict({
        "means": torch.nn.Parameter(means),
        "scales": torch.nn.Parameter(scales),
        "quats": torch.nn.Parameter(quats),
        "opacities": torch.nn.Parameter(opacities),
        "sh0": torch.nn.Parameter(sh0),
        "shN": torch.nn.Parameter(shN),
    }).to(device)
    return params


# --------------------------------------------------------------------------- #
# losses
# --------------------------------------------------------------------------- #
def _gaussian_window(ch, ws=11, sigma=1.5, device="cuda"):
    coords = torch.arange(ws, dtype=torch.float32, device=device) - ws // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2)); g = (g / g.sum())
    w2 = (g[:, None] * g[None, :])
    return w2.expand(ch, 1, ws, ws).contiguous()


def ssim(a, b, window=None):
    # a,b: (1,3,H,W)
    if window is None:
        window = _gaussian_window(a.shape[1], device=a.device)
    pad = window.shape[-1] // 2
    ch = a.shape[1]
    mu_a = F.conv2d(a, window, padding=pad, groups=ch)
    mu_b = F.conv2d(b, window, padding=pad, groups=ch)
    mu_a2, mu_b2, mu_ab = mu_a * mu_a, mu_b * mu_b, mu_a * mu_b
    sa = F.conv2d(a * a, window, padding=pad, groups=ch) - mu_a2
    sb = F.conv2d(b * b, window, padding=pad, groups=ch) - mu_b2
    sab = F.conv2d(a * b, window, padding=pad, groups=ch) - mu_ab
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    s = ((2 * mu_ab + c1) * (2 * sab + c2)) / ((mu_a2 + mu_b2 + c1) * (sa + sb + c2))
    return s.mean()


def edge_aware_logl1(pred_d, gt_d, rgb, mask):
    """AGS-Mesh edge-aware log-L1 depth loss. pred_d,gt_d,mask: (H,W); rgb: (H,W,3)."""
    logl1 = torch.log1p((pred_d - gt_d).abs())
    gx = (rgb[:, :-1] - rgb[:, 1:]).abs().mean(-1)
    gy = (rgb[:-1, :] - rgb[1:, :]).abs().mean(-1)
    lx = torch.exp(-gx) * logl1[:, :-1]
    ly = torch.exp(-gy) * logl1[:-1, :]
    mx = mask[:, :-1]; my = mask[:-1, :]
    loss = 0.0
    if mx.any():
        loss = loss + lx[mx].mean()
    if my.any():
        loss = loss + ly[my].mean()
    return loss


def depth_to_normal(depth, K):
    """Normals (H,W,3) in the camera frame (OpenCV, n_z<0) from a rendered depth
    map, for the normal-supervision loss (DN-Splatter's mono-normal path derives
    the predicted normal from the rendered depth)."""
    H, W = depth.shape
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    vs, us = torch.meshgrid(torch.arange(H, device=depth.device, dtype=torch.float32),
                            torch.arange(W, device=depth.device, dtype=torch.float32),
                            indexing="ij")
    X = (us - cx) / fx * depth
    Y = (vs - cy) / fy * depth
    P = torch.stack([X, Y, depth], dim=-1)
    dx = torch.zeros_like(P); dy = torch.zeros_like(P)
    dx[:, 1:-1] = (P[:, 2:] - P[:, :-2]) * 0.5
    dy[1:-1, :] = (P[2:, :] - P[:-2, :]) * 0.5
    n = F.normalize(torch.cross(dx, dy, dim=-1), dim=-1)
    flip = (n[..., 2] > 0).unsqueeze(-1)
    return torch.where(flip, -n, n)


# --------------------------------------------------------------------------- #
# train
# --------------------------------------------------------------------------- #
def train(session, cfg, iters, downscale, depth_lambda, sh_degree, device,
          normal_lambda=0.0, normal_warmup=1000):
    from gsplat import rasterization
    from gsplat.strategy import DefaultStrategy

    frames, scene_scale = load_dataset(session, downscale, device)
    if not frames:
        raise SystemExit("[stage5] no frames loaded from metric COLMAP + capture/rgb")
    lay = SessionLayout(session)
    params = build_gaussians(lay.metric_points, True, device)
    print(f"[stage5] {len(frames)} views | {params['means'].shape[0]} init gaussians | scene_scale={scene_scale:.3f}")

    lr = {"means": 1.6e-4 * scene_scale, "scales": 5e-3, "quats": 1e-3,
          "opacities": 5e-2, "sh0": 2.5e-3, "shN": 2.5e-3 / 20}
    optimizers = {k: torch.optim.Adam([{"params": params[k], "lr": v, "name": k}], eps=1e-15)
                  for k, v in lr.items()}

    strategy = DefaultStrategy(verbose=False)
    strategy.check_sanity(params, optimizers)
    strategy_state = strategy.initialize_state(scene_scale=scene_scale)

    win = _gaussian_window(3, device=device)
    rng = np.random.default_rng(0)
    order = []
    for step in range(iters):
        if not order:
            order = list(rng.permutation(len(frames)))
        fr = frames[order.pop()]
        cur_sh = min(sh_degree, step // 1000)

        colors = torch.cat([params["sh0"], params["shN"]], dim=1)  # (N,K,3)
        renders, alphas, info = rasterization(
            params["means"], F.normalize(params["quats"], dim=-1),
            torch.exp(params["scales"]), torch.sigmoid(params["opacities"]),
            colors, fr["viewmat"][None], fr["K"][None], fr["W"], fr["H"],
            sh_degree=cur_sh, render_mode="RGB+ED", packed=True, absgrad=True)

        img = renders[0, ..., :3].clamp(0, 1)          # H,W,3
        rd = renders[0, ..., 3]                         # H,W depth (expected)

        strategy.step_pre_backward(params, optimizers, strategy_state, step, info)

        l1 = (img - fr["rgb"]).abs().mean()
        s = ssim(img.permute(2, 0, 1)[None], fr["rgb"].permute(2, 0, 1)[None], win)
        loss = 0.8 * l1 + 0.2 * (1 - s)
        if depth_lambda > 0 and fr["dmask"].any():
            loss = loss + depth_lambda * edge_aware_logl1(rd, fr["depth"], fr["rgb"], fr["dmask"])
        if normal_lambda > 0 and fr.get("normal") is not None and step >= normal_warmup:
            n_pred = depth_to_normal(rd, fr["K"])
            nm = fr["nmask"] & fr["dmask"]
            if nm.any():
                loss = loss + normal_lambda * (n_pred[nm] - fr["normal"][nm]).abs().mean()

        loss.backward()
        for opt in optimizers.values():
            opt.step(); opt.zero_grad(set_to_none=True)
        strategy.step_post_backward(params, optimizers, strategy_state, step, info,
                                    packed=True)

        if step % 500 == 0 or step == iters - 1:
            print(f"[stage5] step {step:5d} loss {loss.item():.4f} l1 {l1.item():.4f} "
                  f"ssim {s.item():.3f} gaussians {params['means'].shape[0]}")

    return params, frames


# --------------------------------------------------------------------------- #
# export (Stage 6)
# --------------------------------------------------------------------------- #
def export_ply(params, path):
    """INRIA/3DGS Gaussian-splat .ply (log-scale, logit-opacity, wxyz quats)."""
    import numpy as np
    n = params["means"].shape[0]
    means = params["means"].detach().cpu().numpy()
    sh0 = params["sh0"].detach().cpu().numpy().reshape(n, -1)          # (N,1,3) -> (N,3) f_dc
    # f_rest must be CHANNEL-major (15 R, 15 G, 15 B), matching INRIA/3DGS save_ply
    # and every external splat viewer; (N,15,3) row-major reshape would interleave
    # by coefficient and scramble the view-dependent color.
    shN = params["shN"].detach().cpu().numpy().transpose(0, 2, 1).reshape(n, -1)
    opac = params["opacities"].detach().cpu().numpy().reshape(n, 1)
    scales = params["scales"].detach().cpu().numpy()
    quats = torch.nn.functional.normalize(params["quats"], dim=-1).detach().cpu().numpy()
    normals = np.zeros((n, 3), np.float32)
    cols = [("x", "f4"), ("y", "f4"), ("z", "f4"), ("nx", "f4"), ("ny", "f4"), ("nz", "f4")]
    cols += [(f"f_dc_{i}", "f4") for i in range(3)]
    cols += [(f"f_rest_{i}", "f4") for i in range(shN.shape[1])]
    cols += [("opacity", "f4")]
    cols += [(f"scale_{i}", "f4") for i in range(3)]
    cols += [(f"rot_{i}", "f4") for i in range(4)]
    data = np.concatenate([means, normals, sh0, shN, opac, scales, quats], axis=1).astype(np.float32)
    arr = np.zeros(n, dtype=cols)
    for i, (nm, _) in enumerate(cols):
        arr[nm] = data[:, i]
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    _write_structured_ply(arr, path)


def _write_structured_ply(arr, path):
    import struct
    names = arr.dtype.names
    header = ["ply", "format binary_little_endian 1.0", f"element vertex {len(arr)}"]
    header += [f"property float {n}" for n in names] + ["end_header"]
    with open(path, "wb") as fh:
        fh.write(("\n".join(header) + "\n").encode())
        fh.write(arr.tobytes())


def export_mesh_and_renders(params, frames, out_dir, sh_degree, device, voxel=0.004, trunc=0.02):
    """TSDF-fuse the rendered depth from the trained Gaussians -> metric mesh."""
    from gsplat import rasterization
    import open3d as o3d
    out_dir = Path(out_dir)
    renders_dir = out_dir / "renders"; renders_dir.mkdir(parents=True, exist_ok=True)
    vol = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel, sdf_trunc=trunc,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)

    colors = torch.cat([params["sh0"], params["shN"]], dim=1)
    for i, fr in enumerate(frames):
        with torch.no_grad():
            renders, _, _ = rasterization(
                params["means"], F.normalize(params["quats"], dim=-1),
                torch.exp(params["scales"]), torch.sigmoid(params["opacities"]),
                colors, fr["viewmat"][None], fr["K"][None], fr["W"], fr["H"],
                sh_degree=sh_degree, render_mode="RGB+ED", packed=True)
        img = (renders[0, ..., :3].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
        depth = renders[0, ..., 3].cpu().numpy().astype(np.float32)
        if i % max(1, len(frames) // 8) == 0:
            Image.fromarray(img).save(renders_dir / f"{fr['fid']}.png")
        o3dcolor = o3d.geometry.Image(np.ascontiguousarray(img))
        o3ddepth = o3d.geometry.Image(np.ascontiguousarray(depth))
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3dcolor, o3ddepth, depth_scale=1.0, depth_trunc=5.0, convert_rgb_to_intensity=False)
        K = fr["K"].cpu().numpy()
        intr = o3d.camera.PinholeCameraIntrinsic(fr["W"], fr["H"], K[0, 0], K[1, 1], K[0, 2], K[1, 2])
        extr = fr["viewmat"].cpu().numpy().astype(np.float64)
        vol.integrate(rgbd, intr, extr)

    mesh = vol.extract_triangle_mesh(); mesh.compute_vertex_normals()
    o3d.io.write_triangle_mesh(str(out_dir / "mesh.ply"), mesh)
    return len(mesh.vertices), len(mesh.triangles)


def reconstruct(session, opts):
    """Run the full Stage 5 -> Stage 6 reconstruction. ``opts`` is a dict with
    iters, downscale, depth_lambda, sh_degree. Returns a provenance dict.
    Importable by the stage run.py entry point."""
    device = "cuda"
    torch.manual_seed(0)
    iters = int(opts.get("iters", 7000))
    downscale = float(opts.get("downscale", 2.0))
    depth_lambda = float(opts.get("depth_lambda", 0.2))
    sh_degree = int(opts.get("sh_degree", 3))
    normal_lambda = float(opts.get("normal_lambda", 0.0))   # >0 uses Stage 4 normals if present

    params, frames = train(session, {}, iters, downscale, depth_lambda, sh_degree, device,
                           normal_lambda=normal_lambda)

    lay = SessionLayout(session)
    lay.output.mkdir(parents=True, exist_ok=True)
    export_ply(params, lay.output_point_cloud)
    nv, nt = export_mesh_and_renders(params, frames, lay.output, sh_degree, device)
    prov = {"stage5_host": "gsplat (depth-supervised)", "gaussians": int(params["means"].shape[0]),
            "views": len(frames), "iters": iters, "downscale": downscale,
            "depth_lambda": depth_lambda, "mesh_vertices": nv, "mesh_triangles": nt}
    (lay.output / "provenance_stage5.json").write_text(json.dumps(prov, indent=2))
    print(f"[stage5] DONE: splat={lay.output_point_cloud} ({params['means'].shape[0]} gaussians), "
          f"mesh={lay.output/'mesh.ply'} ({nv} verts / {nt} tris)")
    return prov


def main():
    ap = argparse.ArgumentParser(description="Stage 5: gsplat metric depth-supervised reconstruction")
    ap.add_argument("--session", required=True)
    ap.add_argument("--config", default="config/pipeline.yaml")
    ap.add_argument("--iters", type=int, default=7000)
    ap.add_argument("--downscale", type=float, default=2.0)
    ap.add_argument("--depth-lambda", type=float, default=0.2)
    ap.add_argument("--normal-lambda", type=float, default=0.0)
    ap.add_argument("--sh-degree", type=int, default=3)
    args = ap.parse_args()
    reconstruct(args.session, {"iters": args.iters, "downscale": args.downscale,
                               "depth_lambda": args.depth_lambda, "sh_degree": args.sh_degree,
                               "normal_lambda": args.normal_lambda})
    return 0


if __name__ == "__main__":
    sys.exit(main())
