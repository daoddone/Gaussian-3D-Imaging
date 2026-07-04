#!/usr/bin/env python3
"""Re-extract a mesh from an already-trained splat, with de-doubling + finer voxel.

The current pipeline TSDF (voxel 0.004, depth_trunc 5.0, no masking) produces a coarse,
"doubled" mesh: at silhouette edges the expected-depth (ED) render blends fg/bg -> an
intermediate depth -> a spurious second shell between object and background; and the huge
5 m depth_trunc lets stray low-alpha/background depth fuse far away. Fix, before integrate():
  - alpha mask   (only fuse confident interior: alpha > alpha_thr)
  - range mask   (only fuse depth in [dmin, dmax] around the real surface)
  - edge mask    (drop pixels with large local depth gradient = silhouettes/discontinuities)
and integrate at a finer voxel with a tight sdf_trunc. No retraining — renders from the
trained Gaussians and re-fuses. Writes <out> and prints vert/tri counts.

Usage: remesh.py <splat.ply> <colmap_sparse0_dir> <out_mesh.ply>
       [voxel=0.002 alpha_thr=0.5 edge_frac=0.02 pad=0.05]
CUDA_HOME must be set (gsplat JIT); run in pipeline_stage2_frontend env.
"""
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, "/home/paperspace/Documents/VS Code Projects/3D-Gaussian")
from common import colmap_io
# reuse the exact loader from the evaluator
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "eval_splat", "/home/paperspace/Documents/VS Code Projects/3D-Gaussian/scripts/pose_ba/eval_splat.py")
_ev = importlib.util.module_from_spec(_spec)

SPLAT, COLMAP, OUT = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
VOXEL = float(sys.argv[4]) if len(sys.argv) > 4 else 0.002
ALPHA_THR = float(sys.argv[5]) if len(sys.argv) > 5 else 0.5
EDGE_FRAC = float(sys.argv[6]) if len(sys.argv) > 6 else 0.02   # depth-grad > EDGE_FRAC*median_depth -> edge
PAD = float(sys.argv[7]) if len(sys.argv) > 7 else 0.05         # metres beyond point-cloud range to keep
DEV = "cuda"


def q2R(q):
    w, x, y, z = q
    return np.array([[1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
                     [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
                     [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]], np.float32)


def main():
    _spec.loader.exec_module(_ev)
    from gsplat import rasterization
    import open3d as o3d

    p = _ev.load_splat_ply(SPLAT)
    colors = torch.cat([p["sh0"], p["shN"]], dim=1)
    means, quats = p["means"], F.normalize(p["quats"], dim=-1)
    scales, opac = torch.exp(p["scales"]), torch.sigmoid(p["opacities"])

    # scene depth range along the optical axis is unknown per-view; derive a global metric
    # surface band from the point cloud extent projected per-view at integration time.
    cams = colmap_io.read_cameras_binary(COLMAP / "cameras.bin")
    imgs = colmap_io.read_images_binary(COLMAP / "images.bin")

    vol = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=VOXEL, sdf_trunc=max(4 * VOXEL, 0.006),
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)

    kept_frac = []
    for im in imgs.values():
        cam = cams[im["camera_id"]]
        fx, fy, cx, cy = cam["params"][:4]
        W, H = cam["width"], cam["height"]
        K = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], device=DEV, dtype=torch.float32)
        R = torch.tensor(q2R(im["qvec"]), device=DEV, dtype=torch.float32)
        t = torch.tensor(im["tvec"], device=DEV, dtype=torch.float32)
        vm = torch.eye(4, device=DEV); vm[:3, :3] = R; vm[:3, 3] = t
        with torch.no_grad():
            rend, alpha, _ = rasterization(means, quats, scales, opac, colors, vm[None], K[None],
                                           W, H, sh_degree=3, render_mode="RGB+ED", packed=True)
        img = (rend[0, ..., :3].clamp(0, 1) * 255).to(torch.uint8).cpu().numpy()
        depth = rend[0, ..., 3]                                   # H,W expected depth (metres)
        a = alpha[0, ..., 0] if alpha.dim() == 4 else alpha[0]    # H,W

        # --- de-doubling masks ---
        m = a > ALPHA_THR                                          # (1) confident interior only
        dmed = depth[m].median() if m.any() else depth.median()
        # (2) surface band: within the cloud's plausible depth of the median surface
        m = m & (depth > dmed * (1 - 0.5)) & (depth < dmed * (1 + 0.5))
        # (3) edge mask: large local depth gradient = silhouette/discontinuity -> drop
        gx = torch.zeros_like(depth); gy = torch.zeros_like(depth)
        gx[:, 1:-1] = (depth[:, 2:] - depth[:, :-2]).abs() * 0.5
        gy[1:-1, :] = (depth[2:, :] - depth[:-2, :]).abs() * 0.5
        grad = torch.maximum(gx, gy)
        m = m & (grad < EDGE_FRAC * dmed)

        d_masked = torch.where(m, depth, torch.zeros_like(depth)).cpu().numpy().astype(np.float32)
        kept_frac.append(float(m.float().mean()))

        o3dcolor = o3d.geometry.Image(np.ascontiguousarray(img))
        o3ddepth = o3d.geometry.Image(np.ascontiguousarray(d_masked))
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3dcolor, o3ddepth, depth_scale=1.0, depth_trunc=float(dmed * 1.5) + PAD,
            convert_rgb_to_intensity=False)
        intr = o3d.camera.PinholeCameraIntrinsic(W, H, fx, fy, cx, cy)
        vol.integrate(rgbd, intr, vm.cpu().numpy().astype(np.float64))

    mesh = vol.extract_triangle_mesh(); mesh.compute_vertex_normals()
    # keep only the largest connected component (drops any residual floaters)
    labels = np.asarray(mesh.cluster_connected_triangles()[0])
    if labels.size:
        import collections
        big = collections.Counter(labels).most_common(1)[0][0]
        mesh.remove_triangles_by_mask(labels != big); mesh.remove_unreferenced_vertices()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_triangle_mesh(str(OUT), mesh)
    print(f"[remesh] voxel={VOXEL} alpha>{ALPHA_THR} edge<{EDGE_FRAC} | mean kept pixels {np.mean(kept_frac):.2f}")
    print(f"[remesh] {OUT}: {len(mesh.vertices)} verts / {len(mesh.triangles)} tris")


if __name__ == "__main__":
    main()
