#!/usr/bin/env python3
"""Standardized review/eval exporter (owner request 2026-07-09): for any output dir, produce
subject-centered, canonically-oriented review copies + labeled canonical renders.

Why: (i) manual review — the .ply loads already framed on the subject (no 5-10 min of
translate/rotate/scale), junk periphery stripped; (ii) agent/script evaluation — metrics computed
over the SUBJECT, not thrown off by periphery.

Outputs into <output_dir>/review/:
  subject_cloud.ply / subject_mesh.ply — cropped to the subject cluster, centered at its centroid,
      principal axes aligned to X/Y/Z (largest spread = X). Metric scale preserved.
  views.png — labeled canonical renders (front / three-quarter / top) of BOTH, with the run label,
      subject dims (mm), gaussian/vert counts and subject-mesh roughness burned into the image.
  review.json — the subject-level stats (the numbers scripts should consume).

Run in pipeline_stage2_frontend env (open3d + PIL).
Usage: export_review.py OUTPUT_DIR [--label NAME] [--crop-scale 1.0]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import open3d as o3d
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).parent))
from eval_recon import read_inria_ply, mesh_roughness_deg, render_geometry  # noqa: E402


def subject_frame(xyz, k=1.8):
    """Subject cluster -> (center, radius, rotation) where R aligns principal axes to XYZ."""
    c0 = np.median(xyz, 0)
    d = np.linalg.norm(xyz - c0, 1, axis=1)
    r = k * np.median(d)
    cl = xyz[np.linalg.norm(xyz - c0, 1, axis=1) < r]
    c = cl.mean(0)
    cov = np.cov((cl - c).T)
    w, V = np.linalg.eigh(cov)           # ascending eigenvalues
    R = V[:, ::-1].T                      # rows = principal axes, largest first -> X,Y,Z
    if np.linalg.det(R) < 0:
        R[2] *= -1                        # keep right-handed
    return c, r, R


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("output_dir")
    ap.add_argument("--label", default=None)
    ap.add_argument("--crop-scale", type=float, default=1.0, help="multiplier on the subject radius")
    args = ap.parse_args()
    out = Path(args.output_dir)
    label = args.label or out.parent.name + "/" + out.name
    rev = out / "review"
    rev.mkdir(exist_ok=True)
    stats = {"label": label}

    xyz, rgb, _ = read_inria_ply(out / "point_cloud.ply")
    c, r, R = subject_frame(xyz)
    r *= args.crop_scale
    keep = np.linalg.norm(xyz - c, 1, axis=1) < r
    P = (xyz[keep] - c) @ R.T
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(P)
    pc.colors = o3d.utility.Vector3dVector(rgb[keep])
    o3d.io.write_point_cloud(str(rev / "subject_cloud.ply"), pc)
    dims = (P.max(0) - P.min(0)) * 1000
    stats["subject_cloud"] = {"n_gaussians_subject": int(keep.sum()), "n_gaussians_total": int(len(xyz)),
                              "dims_mm": [round(float(v), 1) for v in dims]}

    mesh_stats_txt = "no mesh"
    smesh = None
    mesh_src = out / ("mesh_textured.ply" if (out / "mesh_textured.ply").exists() else "mesh.ply")
    if mesh_src.exists():
        smesh = o3d.io.read_triangle_mesh(str(mesh_src))
        v = np.asarray(smesh.vertices)
        fk = (np.linalg.norm(v - c, 1, axis=1) < r)[np.asarray(smesh.triangles)].all(1)
        smesh.remove_triangles_by_mask(~fk)
        smesh.remove_unreferenced_vertices()
        smesh.vertices = o3d.utility.Vector3dVector((np.asarray(smesh.vertices) - c) @ R.T)
        smesh.compute_vertex_normals()
        o3d.io.write_triangle_mesh(str(rev / "subject_mesh.ply"), smesh)
        rough = mesh_roughness_deg(smesh)
        stats["subject_mesh"] = {"vertices": int(len(smesh.vertices)),
                                 "faces": int(len(smesh.triangles)),
                                 "roughness_dihedral_deg": round(rough, 2)}
        mesh_stats_txt = f"mesh {len(smesh.vertices):,}v  roughness {rough:.1f} deg"

    # canonical labeled renders: subject is centered/axis-aligned so views are deterministic.
    # ONE OffscreenRenderer per geometry, reused across views (fresh-per-view segfaults EGL).
    def canon_views(geom, shader, ps=3.0):
        ext = geom.get_axis_aligned_bounding_box().get_extent()
        rad = float(np.linalg.norm(ext)) * 0.5
        rr = o3d.visualization.rendering.OffscreenRenderer(760, 640)
        rr.scene.set_background([1, 1, 1, 1])
        m = o3d.visualization.rendering.MaterialRecord()
        m.shader = shader
        if shader == "defaultUnlit":
            m.point_size = ps
        rr.scene.add_geometry("g", geom, m)
        rr.scene.scene.set_sun_light([0.3, 0.3, -0.8], [1, 1, 1], 90000)
        rr.scene.scene.enable_sun_light(True)
        tiles = []
        for name, eye in [("front", (0, -2.2, 0.6)), ("3/4", (1.6, -1.6, 0.9)), ("top", (0, -0.15, 2.4))]:
            rr.setup_camera(55.0, [0, 0, 0], list(np.array(eye) * rad), [0, 0, 1])
            img = np.asarray(rr.render_to_image()).copy()
            pil = Image.fromarray(img)
            ImageDraw.Draw(pil).text((10, 8), name, fill=(180, 30, 30))
            tiles.append(np.asarray(pil))
        return np.concatenate(tiles, 1)

    rows = [canon_views(pc, "defaultUnlit")]
    if smesh is not None and len(smesh.vertices) > 100:
        rows.append(canon_views(smesh, "defaultLit"))
    W = max(rw.shape[1] for rw in rows)
    rows = [np.concatenate([rw, np.full((rw.shape[0], W - rw.shape[1], 3), 255, np.uint8)], 1)
            if rw.shape[1] < W else rw for rw in rows]
    body = np.concatenate(rows, 0)
    header = Image.new("RGB", (W, 56), (245, 245, 245))
    dr = ImageDraw.Draw(header)
    dr.text((10, 6), f"{label}", fill=(0, 0, 0))
    dr.text((10, 24), f"cloud: {keep.sum():,}/{len(xyz):,} subject gaussians | subject dims "
                      f"{dims.round(0).astype(int).tolist()} mm | {mesh_stats_txt}", fill=(60, 60, 60))
    dr.text((10, 40), "row1: point cloud (subject crop) | row2: mesh — canonical views, "
                      "centered + principal-axis aligned", fill=(120, 120, 120))
    full = np.concatenate([np.asarray(header), body], 0)
    Image.fromarray(full).save(rev / "views.png")

    (rev / "review.json").write_text(json.dumps(stats, indent=2))
    print(f"[review] {label}: wrote {rev}/subject_cloud.ply, subject_mesh.ply, views.png, review.json")
    print(f"[review] {json.dumps(stats)}")


if __name__ == "__main__":
    main()
