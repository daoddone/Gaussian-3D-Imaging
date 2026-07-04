"""Framework-agnostic LiDAR depth + normal supervision for the MILo Stage-5 host (H2 port).

These are the SAME loss concepts the working gsplat host uses (AGS-Mesh edge-aware log-L1
depth + DN-Splatter mono-normal-from-depth), factored out as pure-PyTorch functions so they
can be injected into MILo's training loop (which trains in METRIC coordinates -- verified in
scene/dataset_readers.py getNerfppNorm only scales LR/densification, not geometry -- so the
LiDAR depth in metres matches MILo's rendered z-depth directly, no rescaling).

Used by milo_supervised.py, which patches MILo's train loop to call `depth_normal_loss`.
Kept dependency-light (torch + numpy) so it imports in the milo env.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# losses (ported verbatim in behaviour from gsplat_recon.py)
# --------------------------------------------------------------------------- #
def edge_aware_logl1(pred_d, gt_d, rgb, mask):
    """AGS-Mesh edge-aware log-L1 depth loss.

    pred_d, gt_d, mask: (H,W); rgb: (H,W,3) or (3,H,W). Downweights the depth penalty
    across RGB edges (exp(-|grad rgb|)) so real depth discontinuities are not over-smoothed.
    """
    if rgb.dim() == 3 and rgb.shape[0] == 3:      # (3,H,W) -> (H,W,3)
        rgb = rgb.permute(1, 2, 0)
    # NaN/inf-safe: a non-finite value anywhere (even in masked-out pixels) makes the
    # backward compute 0*NaN=NaN via the masked selection. Sanitize both inputs first.
    pred_d = torch.nan_to_num(pred_d, nan=0.0, posinf=0.0, neginf=0.0)
    gt_d = torch.nan_to_num(gt_d, nan=0.0, posinf=0.0, neginf=0.0)
    logl1 = torch.log1p((pred_d - gt_d).abs())
    gx = (rgb[:, :-1] - rgb[:, 1:]).abs().mean(-1)
    gy = (rgb[:-1, :] - rgb[1:, :]).abs().mean(-1)
    lx = torch.exp(-gx) * logl1[:, :-1]
    ly = torch.exp(-gy) * logl1[:-1, :]
    mx = mask[:, :-1]
    my = mask[:-1, :]
    loss = pred_d.new_zeros(())
    if mx.any():
        loss = loss + lx[mx].mean()
    if my.any():
        loss = loss + ly[my].mean()
    return loss


def depth_to_normal(depth, K):
    """Normals (H,W,3) in the camera frame (OpenCV, n_z<0) from a rendered depth map,
    matching DN-Splatter's mono-normal path (predicted normal derived from rendered depth)."""
    H, W = depth.shape
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    vs, us = torch.meshgrid(torch.arange(H, device=depth.device, dtype=torch.float32),
                            torch.arange(W, device=depth.device, dtype=torch.float32),
                            indexing="ij")
    X = (us - cx) / fx * depth
    Y = (vs - cy) / fy * depth
    P = torch.stack([X, Y, depth], dim=-1)
    dx = torch.zeros_like(P)
    dy = torch.zeros_like(P)
    dx[:, 1:-1] = (P[:, 2:] - P[:, :-2]) * 0.5
    dy[1:-1, :] = (P[2:, :] - P[:-2, :]) * 0.5
    n = F.normalize(torch.cross(dx, dy, dim=-1), dim=-1)
    flip = (n[..., 2] > 0).unsqueeze(-1)
    return torch.where(flip, -n, n)


# --------------------------------------------------------------------------- #
# LiDAR depth + confidence loading (mask-normalized resize, from gsplat_recon.load_dataset)
# --------------------------------------------------------------------------- #
def load_lidar_depth(capture_dir, fid, out_hw, device, conf_thresh=0.5):
    """Load capture/depth/{fid}.npy (+ confidence) resized to out_hw with a validity mask.

    Returns (depth[H,W], mask[H,W] bool). Resize is MASK-NORMALIZED: invalid/zero-hole
    pixels do not bleed into valid targets (num=interp(d*m); den=interp(m); d=num/den).
    Returns (None, None) if no depth file exists for this frame.
    """
    capture_dir = Path(capture_dir)
    dpath = capture_dir / "depth" / f"{fid}.npy"
    if not dpath.exists():
        return None, None
    d = torch.tensor(np.load(dpath).astype(np.float32), device=device)      # (h,w) metres
    # iPhone LiDAR marks no-return pixels as NaN; sanitize BEFORE the mask-normalized
    # resize (d*m would otherwise compute NaN*0 = NaN and poison the whole map).
    d = torch.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0)
    m = (d > 1e-3)
    cpath = capture_dir / "confidence" / f"{fid}.npy"
    if cpath.exists():
        c = torch.tensor(np.load(cpath).astype(np.float32), device=device)
        if c.max() > 1.5:                                                    # 0..255 -> 0..1
            c = c / 255.0
        m = m & (c >= conf_thresh)
    H, W = out_hw
    if d.shape != (H, W):
        dt = d[None, None]
        mt = m[None, None].float()
        num = F.interpolate(dt * mt, size=(H, W), mode="bilinear", align_corners=False)
        den = F.interpolate(mt, size=(H, W), mode="bilinear", align_corners=False)
        d = (num / den.clamp_min(1e-6))[0, 0]
        m = (den[0, 0] > 0.5)
    return d, m


def depth_normal_loss(rendered_depth, cam, rgb, K, depth_lambda=0.2,
                      normal_lambda=0.0, gt_normal=None, normal_mask=None):
    """Combined term for MILo's train loop. `cam` must carry .lidar_depth / .lidar_mask
    (attached by the patched dataset reader); no-ops (returns 0) when depth is absent.

    rendered_depth: (H,W) metric z-depth from MILo's rasterizer. rgb: (3,H,W) or (H,W,3).
    """
    total = rendered_depth.new_zeros(())
    ld = getattr(cam, "lidar_depth", None)
    lm = getattr(cam, "lidar_mask", None)
    if depth_lambda > 0 and ld is not None and lm is not None and lm.any():
        total = total + depth_lambda * edge_aware_logl1(rendered_depth, ld, rgb, lm)
    if normal_lambda > 0 and gt_normal is not None:
        n_pred = depth_to_normal(rendered_depth, K)
        nm = normal_mask if normal_mask is not None else torch.ones_like(rendered_depth, dtype=torch.bool)
        if lm is not None:
            nm = nm & lm
        if nm.any():
            total = total + normal_lambda * (n_pred[nm] - gt_normal[nm]).abs().mean()
    return total
