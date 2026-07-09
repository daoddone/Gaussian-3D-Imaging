#!/usr/bin/env python3
"""Reusable reconstruction eval/compare harness (run in the pipeline_stage2_frontend env: open3d).

For each session output dir (has point_cloud.ply INRIA gaussians + optional mesh.ply):
  - stats: n_gaussians, cloud extent, object-box (subject) dims; mesh verts/faces, extent, roughness.
  - renders (Open3D offscreen, headless EGL): mesh shaded (orbit views) + cloud colored (orbit views).
  - writes a per-session montage PNG + a stats dict; a top-level montage stacks sessions for A/B.

Roughness = mean dihedral angle between adjacent faces (deg); higher = bumpier. CAVEAT: it is
resolution-sensitive (finer meshes sample more micro-relief) — compare alongside the renders.

Usage:
  eval_recon.py OUT  --labels NAME  [--outdir DIR]
  eval_recon.py OUTA OUTB ... --labels A B ... --outdir DIR   # side-by-side comparison
"""
import argparse
import json
from pathlib import Path

import numpy as np
import open3d as o3d

SH_C0 = 0.28209479177387814


# ---------- IO ----------
def read_inria_ply(path):
    """Return xyz(N,3), rgb(N,3 in [0,1] from SH DC), opacity(N)."""
    with open(path, "rb") as f:
        assert f.readline().strip() == b"ply"
        f.readline()  # format
        names, n = [], 0
        while True:
            ln = f.readline().decode("latin1").strip()
            if ln.startswith("element vertex"):
                n = int(ln.split()[-1])
            elif ln.startswith("property"):
                names.append(ln.split()[-1])
            elif ln == "end_header":
                break
        data = np.frombuffer(f.read(n * len(names) * 4), np.float32).reshape(n, len(names))
    idx = {nm: i for i, nm in enumerate(names)}
    xyz = data[:, [idx["x"], idx["y"], idx["z"]]].astype(np.float64)
    if "f_dc_0" in idx:
        fdc = data[:, [idx["f_dc_0"], idx["f_dc_1"], idx["f_dc_2"]]]
        rgb = np.clip(SH_C0 * fdc + 0.5, 0, 1).astype(np.float64)
    else:
        rgb = np.full((n, 3), 0.6)
    op = 1.0 / (1.0 + np.exp(-data[:, idx["opacity"]])) if "opacity" in idx else np.ones(n)
    return xyz, rgb, op


def object_box(xyz, pct=1.5, pad=0.05):
    lo = np.percentile(xyz, pct, 0)
    hi = np.percentile(xyz, 100 - pct, 0)
    p = pad * (hi - lo)
    return lo - p, hi + p


# ---------- rendering ----------
def _orbit_eyes(center, radius, n=3, elev=0.55):
    eyes = []
    for az in np.linspace(0, 2 * np.pi, n, endpoint=False):
        d = np.array([np.cos(az), np.sin(az), elev], float)
        eyes.append(center + d / np.linalg.norm(d) * radius * 2.3)
    return eyes


def render_geometry(geom, shader, point_size=3.0, W=760, H=620, n_views=3, bg=(1, 1, 1)):
    """Render a geometry from n_views orbit angles; return a horizontally-stacked uint8 image."""
    aabb = geom.get_axis_aligned_bounding_box()
    center = aabb.get_center()
    radius = float(np.linalg.norm(aabb.get_extent())) * 0.5 + 1e-6
    up = [0, 0, 1]
    r = o3d.visualization.rendering.OffscreenRenderer(W, H)
    r.scene.set_background([*bg, 1.0])
    mat = o3d.visualization.rendering.MaterialRecord()
    mat.shader = shader
    if shader == "defaultUnlit":
        mat.point_size = point_size
    r.scene.add_geometry("g", geom, mat)
    r.scene.scene.set_sun_light([0.3, 0.3, -0.8], [1, 1, 1], 90000)
    r.scene.scene.enable_sun_light(True)
    tiles = []
    for eye in _orbit_eyes(center, radius, n_views):
        r.setup_camera(55.0, center.tolist(), eye.tolist(), up)
        tiles.append(np.asarray(r.render_to_image()))
    return np.concatenate(tiles, axis=1)


def mesh_roughness_deg(mesh):
    """Mean dihedral angle between adjacent triangles (deg); higher = bumpier. Vectorized."""
    mesh.compute_triangle_normals()
    tn = np.asarray(mesh.triangle_normals)
    tris = np.asarray(mesh.triangles)
    if len(tris) == 0:
        return float("nan")
    e = np.concatenate([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]], 0)
    e = np.sort(e, axis=1)
    tid = np.tile(np.arange(len(tris)), 3)
    order = np.lexsort((e[:, 1], e[:, 0]))
    e, tid = e[order], tid[order]
    same = (e[1:, 0] == e[:-1, 0]) & (e[1:, 1] == e[:-1, 1])
    p = np.nonzero(same)[0]
    if len(p) == 0:
        return float("nan")
    d = np.clip(np.einsum("ij,ij->i", tn[tid[p]], tn[tid[p + 1]]), -1, 1)
    return float(np.degrees(np.arccos(d)).mean())


# ---------- per-session ----------
def subject_crop_params(xyz, k=1.8):
    """Densest-cluster crop: median center + k * median L1 distance. Robust to floaters."""
    c = np.median(xyz, 0)
    d = np.linalg.norm(xyz - c, 1, axis=1)
    return c, k * np.median(d)


def eval_session(out_dir, label, render_dir):
    out_dir = Path(out_dir)
    stats = {"label": label, "dir": str(out_dir)}
    strips = []
    montage = None
    c = r = None

    cloud_ply = out_dir / "point_cloud.ply"
    if cloud_ply.exists():
        xyz, rgb, op = read_inria_ply(cloud_ply)
        ext = (xyz.max(0) - xyz.min(0)) * 1000
        lo, hi = object_box(xyz)
        obj_dims = (hi - lo) * 1000
        stats["cloud"] = {"n_gaussians": int(len(xyz)),
                          "extent_mm": [round(float(v), 1) for v in ext],
                          "object_box_mm": [round(float(v), 1) for v in obj_dims]}
        m = np.all((xyz >= lo) & (xyz <= hi), 1)
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(xyz[m])
        pc.colors = o3d.utility.Vector3dVector(rgb[m])
        strips.append(("cloud", render_geometry(pc, "defaultUnlit", point_size=3.5)))
        # SUBJECT-centered crop (floater-robust): the densest cluster, not the whole box
        c, r = subject_crop_params(xyz)
        sm = np.linalg.norm(xyz - c, 1, axis=1) < r
        sub_dims = (xyz[sm].max(0) - xyz[sm].min(0)) * 1000 if sm.any() else np.zeros(3)
        stats["subject_cluster"] = {"n_gaussians": int(sm.sum()),
                                    "dims_mm": [round(float(v), 1) for v in sub_dims]}
        spc = o3d.geometry.PointCloud()
        spc.points = o3d.utility.Vector3dVector(xyz[sm])
        spc.colors = o3d.utility.Vector3dVector(rgb[sm])
        strips.append(("cloud-subject", render_geometry(spc, "defaultUnlit", point_size=2.5)))

    mesh_ply = out_dir / "mesh.ply"
    if mesh_ply.exists():
        mesh = o3d.io.read_triangle_mesh(str(mesh_ply))
        mesh.compute_vertex_normals()
        v = np.asarray(mesh.vertices)
        ext = (v.max(0) - v.min(0)) * 1000
        stats["mesh"] = {"vertices": int(len(v)), "faces": int(len(mesh.triangles)),
                         "extent_mm": [round(float(x), 1) for x in ext],
                         "roughness_dihedral_deg": round(mesh_roughness_deg(mesh), 2)}
        strips.append(("mesh", render_geometry(mesh, "defaultLit")))
        if c is not None:
            smesh = o3d.io.read_triangle_mesh(str(mesh_ply))
            sv = np.asarray(smesh.vertices)
            inside = np.linalg.norm(sv - c, 1, axis=1) < r
            fk = inside[np.asarray(smesh.triangles)].all(1)
            smesh.remove_triangles_by_mask(~fk)
            smesh.remove_unreferenced_vertices()
            if len(smesh.vertices) > 100:
                smesh.compute_vertex_normals()
                stats["subject_mesh"] = {"vertices": int(len(smesh.vertices)),
                                         "roughness_dihedral_deg": round(mesh_roughness_deg(smesh), 2)}
                strips.append(("mesh-subject", render_geometry(smesh, "defaultLit")))

    if strips:
        W = max(s.shape[1] for _, s in strips)
        rows = []
        for _, s in strips:
            pad = np.full((s.shape[0], W - s.shape[1], 3), 255, np.uint8) if s.shape[1] < W else None
            rows.append(np.concatenate([s, pad], 1) if pad is not None else s)
        montage = np.concatenate(rows, 0)
        p = Path(render_dir) / f"recon_{label}.png"
        o3d.io.write_image(str(p), o3d.geometry.Image(montage.astype(np.uint8)))
        stats["montage"] = str(p)
    return stats, montage


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("outdirs", nargs="+")
    ap.add_argument("--labels", nargs="+")
    ap.add_argument("--outdir", default=".")
    args = ap.parse_args()
    labels = args.labels or [Path(o).parent.name for o in args.outdirs]
    Path(args.outdir).mkdir(parents=True, exist_ok=True)

    all_stats, montages = [], []
    for od, lab in zip(args.outdirs, labels):
        s, mont = eval_session(od, lab, args.outdir)
        all_stats.append(s)
        if mont is not None:
            montages.append((lab, mont))
        print(f"[eval] {lab}: {json.dumps({k: v for k, v in s.items() if k in ('cloud', 'mesh', 'subject_cluster', 'subject_mesh')})}")

    if len(montages) > 1:
        W = max(m.shape[1] for _, m in montages)
        rows = []
        for lab, m in montages:
            if m.shape[1] < W:
                m = np.concatenate([m, np.full((m.shape[0], W - m.shape[1], 3), 255, np.uint8)], 1)
            rows.append(m)
        comp = np.concatenate(rows, 0)
        cp = Path(args.outdir) / "comparison.png"
        o3d.io.write_image(str(cp), o3d.geometry.Image(comp.astype(np.uint8)))
        print(f"[eval] comparison montage -> {cp}")
    Path(args.outdir, "stats.json").write_text(json.dumps(all_stats, indent=2))
    print(f"[eval] stats -> {Path(args.outdir, 'stats.json')}")


if __name__ == "__main__":
    main()
