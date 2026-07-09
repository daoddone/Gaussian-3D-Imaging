#!/usr/bin/env python3
"""v2 mesh export -> analysis-ready OBJ for the Tepole/Gosain FE pipeline (docs/MESH_EXPORT_SPEC.md).

Produces, from a MILo output dir, an OBJ triangulated surface that is drop-in for their
MeshLab/CloudCompare -> Cubit -> Abaqus chain, and improves on Vectra via a dense source-baked
texture (their stated gap: they lost tattooed-grid strain tracking in humans; dense texture can
restore cross-time correspondence).

Pipeline:
  1. crop to subject ROI (robust cluster), keep largest connected component
  2. clean: remove non-manifold edges, degenerate/duplicated tris, unreferenced verts; consistent normals
  3. decimate to ~target tris (Cubit-ready; cleanliness >> raw count per the spec)
  4. scale to MILLIMETERS (our meshes are metric metres) -- with an explicit scale-provenance sidecar
  5. UV unwrap (xatlas)
  6. bake a texture atlas by sampling SOURCE IMAGES per texel with robust multi-view consensus
     (shared sampler in bake_mesh_colors.multiview_consensus) -> decimated geometry keeps full detail
  7. write mesh.obj + mesh.mtl + mesh_texture.png (mm) + full-res mesh_metric_mm.ply + export_meta.json

Run in pipeline_stage2_frontend env (open3d + PIL + xatlas). Usage:
  export_mesh_obj.py --output-dir <output_X> --colmap <sparse/0> --images <rgb dir>
      [--target-tris 300000] [--texture 2048] [--scale-source lidar_lock|ruler|vio]
      [--scale-confidence "1% MAD (03b LiDAR lock)"]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import open3d as o3d
import xatlas
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from bake_mesh_colors import load_views, build_scene, multiview_consensus  # noqa: E402
from export_review import subject_frame  # noqa: E402


def clean_crop(mesh, center, radius):
    v = np.asarray(mesh.vertices)
    fk = (np.linalg.norm(v - center, 1, axis=1) < radius)[np.asarray(mesh.triangles)].all(1)
    mesh.remove_triangles_by_mask(~fk)
    mesh.remove_unreferenced_vertices()
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()
    # largest connected component (drops detached shards; a skin patch is one sheet)
    labels, counts, _ = mesh.cluster_connected_triangles()
    labels = np.asarray(labels)
    counts = np.asarray(counts)
    if len(counts) > 1:
        keep = labels == int(counts.argmax())
        mesh.remove_triangles_by_mask(~keep)
        mesh.remove_unreferenced_vertices()
    mesh.compute_vertex_normals()
    return mesh


def rasterize_texels(uvs, faces, V3, N3, C3, T):
    """Rasterize each triangle into the TxT atlas; return per-texel (px,py,pos3d,normal3d,basecolor).
    basecolor = barycentric-interpolated per-vertex color C3 -> a dense fallback for texels no view
    sees (prevents black patches; the source-image consensus overlays this where available)."""
    if uvs.max() > 1.5:
        uvs = uvs / max(uvs.max(), 1.0)
    UV = uvs * (T - 1)
    px_list, py_list, pos_list, nrm_list, col_list = [], [], [], [], []
    for fi, f in enumerate(faces):
        if fi % 50000 == 0:
            print(f"[obj]   rasterizing texels {fi:,}/{len(faces):,}", flush=True)
        a, b, c = f
        uv0, uv1, uv2 = UV[a], UV[b], UV[c]
        p0, p1, p2 = V3[a], V3[b], V3[c]
        n0, n1, n2 = N3[a], N3[b], N3[c]
        c0, c1, c2 = C3[a], C3[b], C3[c]
        xmin, ymin = np.floor(np.minimum.reduce([uv0, uv1, uv2])).astype(int)
        xmax, ymax = np.ceil(np.maximum.reduce([uv0, uv1, uv2])).astype(int)
        xmin, ymin = max(xmin, 0), max(ymin, 0)
        xmax, ymax = min(xmax, T - 1), min(ymax, T - 1)
        if xmax < xmin or ymax < ymin:
            continue
        xs, ys = np.meshgrid(np.arange(xmin, xmax + 1), np.arange(ymin, ymax + 1))
        xs = xs.ravel() + 0.5
        ys = ys.ravel() + 0.5
        d = ((uv1[1] - uv2[1]) * (uv0[0] - uv2[0]) + (uv2[0] - uv1[0]) * (uv0[1] - uv2[1]))
        if abs(d) < 1e-9:
            continue
        l0 = ((uv1[1] - uv2[1]) * (xs - uv2[0]) + (uv2[0] - uv1[0]) * (ys - uv2[1])) / d
        l1 = ((uv2[1] - uv0[1]) * (xs - uv2[0]) + (uv0[0] - uv2[0]) * (ys - uv2[1])) / d
        l2 = 1 - l0 - l1
        inside = (l0 >= -1e-4) & (l1 >= -1e-4) & (l2 >= -1e-4)
        if not inside.any():
            continue
        l0, l1, l2 = l0[inside], l1[inside], l2[inside]
        pos = l0[:, None] * p0 + l1[:, None] * p1 + l2[:, None] * p2
        nrm = l0[:, None] * n0 + l1[:, None] * n1 + l2[:, None] * n2
        col = l0[:, None] * c0 + l1[:, None] * c1 + l2[:, None] * c2
        px_list.append((xs[inside] - 0.5).astype(int))
        py_list.append((ys[inside] - 0.5).astype(int))
        pos_list.append(pos)
        nrm_list.append(nrm)
        col_list.append(col)
    return (np.concatenate(px_list), np.concatenate(py_list), np.concatenate(pos_list),
            np.concatenate(nrm_list), np.concatenate(col_list))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--colmap", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--target-tris", type=int, default=200000)
    ap.add_argument("--texture", type=int, default=2048)
    ap.add_argument("--max-cams", type=int, default=60)
    # Scale provenance: auto-discovered from the T1 scale sidecar (scale_sidecar.json, written by
    # scripts/pose_ba/04_metric_anchor.py) found in an ancestor of --colmap; explicit CLI overrides.
    ap.add_argument("--scale-source", default=None)
    ap.add_argument("--scale-confidence", default=None)
    args = ap.parse_args()
    out = Path(args.output_dir)
    exp = out / "export"
    exp.mkdir(exist_ok=True)

    mesh = o3d.io.read_triangle_mesh(str(out / "mesh.ply"))
    mesh.compute_vertex_normals()
    # ROI from the point cloud's robust subject cluster
    from eval_recon import read_inria_ply
    xyz, _, _ = read_inria_ply(out / "point_cloud.ply")
    c, r, _ = subject_frame(xyz)
    mesh = clean_crop(mesh, c, r)
    print(f"[obj] cleaned+cropped: {len(mesh.vertices):,} verts / {len(mesh.triangles):,} tris")

    if len(mesh.triangles) > args.target_tris:
        mesh = mesh.simplify_quadric_decimation(args.target_tris)
        mesh.remove_unreferenced_vertices()
        mesh.compute_vertex_normals()
        print(f"[obj] decimated -> {len(mesh.triangles):,} tris (Cubit-ready)")

    V = np.asarray(mesh.vertices)
    Vmm = (V - c) * 1000.0                       # center + convert metres -> mm
    Nv = np.asarray(mesh.vertex_normals)
    Fv = np.asarray(mesh.triangles)
    ext_mm = (Vmm.max(0) - Vmm.min(0))
    print(f"[obj] subject extent: {ext_mm.round(1).tolist()} mm", flush=True)

    # Robust per-vertex color FIRST (shared sampler) — used both as the atlas base layer (dense
    # fallback: no black texels where a view can't see) and for the reference full-res PLY.
    views = load_views(args.colmap, args.images, args.max_cams)
    scene = build_scene(mesh)
    print(f"[obj] robust vertex-color bake ({len(views)} views)...", flush=True)
    vcol, vcov = multiview_consensus(V, Nv, views, scene)
    base_mesh_col = np.asarray(mesh.vertex_colors)
    if len(base_mesh_col) == len(V):
        vcol[~vcov] = base_mesh_col[~vcov]       # MILo color where nothing sees the vertex
    print(f"[obj] vertex coverage {100*vcov.mean():.1f}%", flush=True)

    # UV unwrap (xatlas chart-packing is the slow step; brute_force off + coarse resolution keeps it
    # to ~1-2 min at a few-hundred-k tris instead of stalling for many minutes)
    print(f"[obj] xatlas unwrap of {len(Fv):,} tris (chart packing)...", flush=True)
    atl = xatlas.Atlas()
    atl.add_mesh(V, Fv)
    copt = xatlas.ChartOptions()
    copt.max_iterations = 1
    popt = xatlas.PackOptions()
    popt.resolution = args.texture
    popt.bruteForce = False
    popt.padding = 2
    atl.generate(chart_options=copt, pack_options=popt)
    vmapping, indices, uvs = atl[0]
    Va, Na, Ca = V[vmapping], Nv[vmapping], vcol[vmapping]   # atlas verts (metres) + base colors
    Vamm = (Va - c) * 1000.0
    print(f"[obj] xatlas: {len(Va):,} atlas verts, {len(indices):,} faces")

    # texture bake: texel -> 3D (world metres) -> robust multi-view sample; barycentric vertex color
    # is the dense base, consensus overlays it where a view sees the texel.
    T = args.texture
    px, py, pos, nrm, base = rasterize_texels(uvs, indices, Va, Na, Ca, T)
    nrm = nrm / np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-9)
    print(f"[obj] rasterized {len(px):,} texels; sampling {len(views)} views...", flush=True)
    cols, covered = multiview_consensus(pos, nrm, views, scene)
    texcol = base.copy()
    texcol[covered] = cols[covered]              # sharp source color where visible, base elsewhere

    atlas = np.zeros((T, T, 3), np.uint8)
    filled = np.zeros((T, T), bool)
    yy = (T - 1) - py                            # OBJ uv origin = bottom-left
    atlas[yy, px] = (np.clip(texcol, 0, 1) * 255).astype(np.uint8)
    filled[yy, px] = True
    # dilate the atlas a few px to kill seams
    from scipy.ndimage import grey_dilation, binary_dilation
    for _ in range(4):
        empty = ~filled
        for ch in range(3):
            d = grey_dilation(atlas[..., ch], size=3)
            atlas[..., ch][empty] = d[empty]
        filled = binary_dilation(filled, iterations=1)
    Image.fromarray(atlas).save(exp / "mesh_texture.png")

    # write OBJ + MTL (positions mm, vt, vn)
    obj = exp / "mesh.obj"
    with open(obj, "w") as fo:
        fo.write("mtllib mesh.mtl\nusemtl skin\n")
        for p in Vamm:
            fo.write(f"v {p[0]:.4f} {p[1]:.4f} {p[2]:.4f}\n")
        for uv in uvs:
            fo.write(f"vt {uv[0]:.6f} {uv[1]:.6f}\n")
        for n in Na:
            fo.write(f"vn {n[0]:.5f} {n[1]:.5f} {n[2]:.5f}\n")
        for f in indices + 1:
            fo.write(f"f {f[0]}/{f[0]}/{f[0]} {f[1]}/{f[1]}/{f[1]} {f[2]}/{f[2]}/{f[2]}\n")
    (exp / "mesh.mtl").write_text(
        "newmtl skin\nKa 0 0 0\nKd 1 1 1\nKs 0 0 0\nmap_Kd mesh_texture.png\n")

    # decimated metric PLY (robust vertex colors, mm) for reference / non-OBJ consumers
    fm = o3d.geometry.TriangleMesh(mesh)
    fm.vertices = o3d.utility.Vector3dVector((V - c) * 1000.0)
    fm.vertex_colors = o3d.utility.Vector3dVector(np.clip(vcol, 0, 1).astype(np.float64))
    fm.compute_vertex_normals()
    o3d.io.write_triangle_mesh(str(exp / "mesh_metric_mm.ply"), fm)

    # Scale provenance: auto-discover the T1 sidecar (scale_sidecar.json from 04_metric_anchor.py)
    # in ancestors of the --colmap dir (…/metric_sfm/colmap/sparse/0 -> …/metric_sfm/). CLI overrides.
    sidecar_summary = None
    scale_source = args.scale_source
    scale_confidence = args.scale_confidence
    _cands = []
    for anc in Path(args.colmap).resolve().parents:
        # sidecar may sit beside the model (metric_sfm/) OR in a sibling metric dir when the
        # model was swapped into metric/colmap for training (the R-run layout)
        _cands += [anc / "scale_sidecar.json", anc / "metric_sfm" / "scale_sidecar.json",
                   anc / "metric" / "scale_sidecar.json"]
    for sc in _cands:
        if sc.exists():
            s = json.loads(sc.read_text())
            sidecar_summary = {k: s.get(k) for k in
                               ("primary_anchor", "scale", "confidence", "anchor_agreement_pct", "notes")}
            scale_source = scale_source or s.get("primary_anchor")
            scale_confidence = scale_confidence or (
                f"{s.get('confidence')} (agreement "
                f"{s.get('anchor_agreement_pct'):.1f}%)" if s.get("anchor_agreement_pct") is not None
                else str(s.get("confidence")))
            print(f"[obj] scale sidecar: {sc} -> {scale_source} / {scale_confidence}")
            break
    scale_source = scale_source or "unknown"
    scale_confidence = scale_confidence or "unverified — no scale sidecar found; verify before FE use"

    meta = {
        "units": "millimeters",
        "coordinate_origin": "subject cluster centroid",
        "topology": "open manifold surface patch (largest connected component); NOT watertight",
        "vertices": int(len(Va)), "triangles": int(len(indices)),
        "subject_extent_mm": [round(float(x), 1) for x in ext_mm],
        "texture": f"mesh_texture.png {T}x{T}, robust multi-view consensus bake",
        "scale_source": scale_source,
        "scale_confidence": scale_confidence,
        "scale_sidecar": sidecar_summary,
        "SCALE_WARNING": ("Absolute scale must be within the collaborator's ~2mm/10cm tolerance. "
                          "If scale_source is not a ruler fiducial, VERIFY before FE use "
                          "(see docs/MESH_EXPORT_SPEC.md)."),
        "spec": "docs/MESH_EXPORT_SPEC.md",
    }
    (exp / "export_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[obj] wrote {exp}/mesh.obj (+mtl,+texture.png), mesh_metric_mm.ply, export_meta.json")
    print(f"[obj] {json.dumps(meta)}")


if __name__ == "__main__":
    main()
