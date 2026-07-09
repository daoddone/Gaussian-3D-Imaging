#!/usr/bin/env python3
"""Mesh appearance bake: color mesh points from source frames with ROBUST MULTI-VIEW CONSENSUS.

Why not MILo's built-in colors: extraction stores ONE diffuse RGB per vertex AVERAGED over all views
-> blurry ("cartoonish"). Why not naive top-K sharpest: a single sharp-but-shadowed/specular/
misregistered view can poison a point (owner's concern). SOLUTION (multiview_consensus): gather the
top-K best-angle x sharpest visible views per point, take their MEDIAN, reject outliers (shadows,
speculars, bad poses), weighted-average the inliers. Keeps top-K sharpness with total-average
robustness; adapts to whatever view distribution exists (no capture standardization needed).

Reusable by scripts/export_mesh_obj.py (texture bake uses the SAME sampler on texel 3D points).

Runs in pipeline_stage2_frontend env (open3d tensor raycasting + PIL).
Usage: bake_mesh_colors.py --output-dir <output_X> --colmap <sparse/0> --images <rgb dir>
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import open3d as o3d
from PIL import Image

REPO = Path("/home/paperspace/Documents/VS Code Projects/3D-Gaussian")
sys.path.insert(0, str(REPO))
from common import colmap_io  # noqa: E402
from common.conventions import quat_to_rotmat  # noqa: E402


def frame_sharpness(img_gray_small):
    gy, gx = np.gradient(img_gray_small.astype(np.float32))
    return float((gx * gx + gy * gy).mean())


def load_views(colmap_dir, images_dir, max_cams=60):
    """Load COLMAP cameras + source images (evenly subsampled to max_cams) as view dicts."""
    imgs_meta = colmap_io.read_images_binary(Path(colmap_dir) / "images.bin")
    cams_meta = colmap_io.read_cameras_binary(Path(colmap_dir) / "cameras.bin")
    metas = sorted(imgs_meta.values(), key=lambda im: im["name"])
    if len(metas) > max_cams:
        idx = np.linspace(0, len(metas) - 1, max_cams).astype(int)
        metas = [metas[i] for i in idx]
    images_dir = Path(images_dir)
    views = []
    for im in metas:
        p = images_dir / im["name"]
        if not p.exists():
            cand = list(images_dir.glob(Path(im["name"]).stem + ".*"))
            if not cand:
                continue
            p = cand[0]
        cam = cams_meta[im["camera_id"]]
        fx, fy, cx, cy = cam["params"][:4]
        W, H = int(cam["width"]), int(cam["height"])
        pil = Image.open(p).convert("RGB")
        if pil.size != (W, H):
            pil = pil.resize((W, H))
        img = np.asarray(pil, dtype=np.float32) / 255.0
        sharp = frame_sharpness(np.asarray(pil.convert("L").resize((max(W // 8, 1), max(H // 8, 1)))))
        R = quat_to_rotmat(im["qvec"])
        t = np.asarray(im["tvec"], float)
        views.append(dict(R=R, t=t, o=-R.T @ t, fx=fx, fy=fy, cx=cx, cy=cy,
                          W=W, H=H, img=img, sharp=sharp))
    return views


def build_scene(mesh):
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    return scene


def multiview_consensus(points, normals, views, scene, K=6, depth_div=4,
                        cos_floor=0.26, cos_cone=0.70, inlier_thresh=0.16):
    """Robust per-point color via ORTHOGONALITY-GATED consensus (surface-aware, not frontal).

    The best view of a texel is the one looking down the SURFACE NORMAL there (owner's insight:
    a frontal camera is wrong for the side of a curved face). So we score by orthogonality
    (cosang = normal . view_dir) and:
      * hard-exclude grazing views (cosang < cos_floor ~ 75deg): they foreshorten/smear.
      * keep each point's top-K candidates by (cosang^2 * sharpness).
      * ADAPTIVE CONE: if the point has views within cos_cone (~45deg of normal), average ONLY those
        (tight, orthogonal); if the near-orthogonal sample is sparse (none in-cone), fall back to all
        visible views (handles owner's sparse-near-orthogonal worry).
      * MEDIAN-REJECT outliers within the selected set, then weighted-average the inliers -> a single
        sharp bad-lighting/specular/misregistered view cannot poison the color.
    points (N,3), normals (N,3) world; returns colors (N,3) float and covered mask (N,).
    """
    N = len(points)
    best_w = np.zeros((N, K), np.float32)
    best_c = np.zeros((N, K, 3), np.float32)
    best_cos = np.zeros((N, K), np.float32)
    for vw in views:
        R, t, o = vw["R"], vw["t"], vw["o"]
        fx, fy, cx, cy, W, H = vw["fx"], vw["fy"], vw["cx"], vw["cy"], vw["W"], vw["H"]
        pc = (R @ points.T).T + t
        z = pc[:, 2]
        front = z > 1e-6
        u = np.where(front, fx * pc[:, 0] / np.maximum(z, 1e-9) + cx, -1)
        v = np.where(front, fy * pc[:, 1] / np.maximum(z, 1e-9) + cy, -1)
        inb = front & (u >= 0) & (u <= W - 1) & (v >= 0) & (v <= H - 1)
        view = o[None, :] - points
        dist = np.linalg.norm(view, axis=1)
        vdir = view / np.maximum(dist[:, None], 1e-9)
        cosang = (normals * vdir).sum(1)
        ok = inb & (cosang > cos_floor)
        if not ok.any():
            continue
        # occlusion via raycast depth map
        dw, dh = max(W // depth_div, 1), max(H // depth_div, 1)
        Kd = np.array([[fx / depth_div, 0, cx / depth_div],
                       [0, fy / depth_div, cy / depth_div], [0, 0, 1]], np.float64)
        ext = np.eye(4)
        ext[:3, :3] = R
        ext[:3, 3] = t
        rays = o3d.t.geometry.RaycastingScene.create_rays_pinhole(
            o3d.core.Tensor(Kd), o3d.core.Tensor(ext), dw, dh)
        t_hit = scene.cast_rays(rays)["t_hit"].numpy()
        ud = np.clip((u[ok] / depth_div).astype(int), 0, dw - 1)
        vd = np.clip((v[ok] / depth_div).astype(int), 0, dh - 1)
        vis = dist[ok] <= t_hit[vd, ud] * 1.01 + 0.006
        idx = np.where(ok)[0][vis]
        if len(idx) == 0:
            continue
        uu, vv = u[idx], v[idx]
        x0, y0 = uu.astype(int), vv.astype(int)
        x1, y1 = np.minimum(x0 + 1, W - 1), np.minimum(y0 + 1, H - 1)
        ax, ay = (uu - x0)[:, None], (vv - y0)[:, None]
        im = vw["img"]
        col = (im[y0, x0] * (1 - ax) * (1 - ay) + im[y0, x1] * ax * (1 - ay)
               + im[y1, x0] * (1 - ax) * ay + im[y1, x1] * ax * ay)
        ca = cosang[idx]
        w = (ca ** 2) * vw["sharp"]                                  # orthogonality-preferred + sharp
        slot = best_w[idx].argmin(1)
        cur = best_w[idx, slot]
        upd = w > cur
        rows, sl = idx[upd], slot[upd]
        best_w[rows, sl] = w[upd]
        best_c[rows, sl] = col[upd]
        best_cos[rows, sl] = ca[upd]

    valid = best_w > 0
    covered = valid.any(1)
    in_cone = valid & (best_cos > cos_cone)                         # near-orthogonal slots
    has_cone = in_cone.any(1)
    use = np.where(has_cone[:, None], in_cone, valid)               # tight cone, else all visible
    colors = np.zeros((N, 3), np.float32)
    cv = np.where(use[..., None], best_c, np.nan)
    med = np.nanmedian(cv, axis=1)                                   # (N,3) robust center
    dist = np.linalg.norm(np.where(use[..., None], best_c - med[:, None, :], 0), axis=2)
    inlier = use & (dist < inlier_thresh)
    wsel = best_w * inlier
    wsum = wsel.sum(1)
    has = wsum > 0
    colors[has] = (best_c[has] * wsel[has, :, None]).sum(1) / wsum[has, None]
    colors[covered & ~has] = med[covered & ~has]                    # fallback: robust median
    return colors, covered


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--colmap", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--max-cams", type=int, default=60)
    args = ap.parse_args()
    out = Path(args.output_dir)
    mesh = o3d.io.read_triangle_mesh(str(out / "mesh.ply"))
    mesh.compute_vertex_normals()
    V = np.asarray(mesh.vertices)
    N = np.asarray(mesh.vertex_normals)
    print(f"[bake] mesh: {len(V):,} verts")
    views = load_views(args.colmap, args.images, args.max_cams)
    print(f"[bake] {len(views)} views")
    scene = build_scene(mesh)
    colors, covered = multiview_consensus(V, N, views, scene)
    old = np.asarray(mesh.vertex_colors)
    if len(old) == len(V):
        colors[~covered] = old[~covered]
    print(f"[bake] colored {covered.sum():,}/{len(V):,} verts ({100*covered.mean():.1f}%) via robust consensus")
    mesh.vertex_colors = o3d.utility.Vector3dVector(colors.astype(np.float64))
    o3d.io.write_triangle_mesh(str(out / "mesh_textured.ply"), mesh)
    print(f"[bake] wrote {out/'mesh_textured.ply'}")


if __name__ == "__main__":
    main()
