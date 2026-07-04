"""MILo Stage-5 host (H2): depth-supervised in-loop mesh reconstruction.

Drives the built MILo toolchain (its own `milo` conda env, torch 2.3.1+cu118) as a
subprocess, injecting our LiDAR metric-depth supervision, and returns the standard
Stage-6 output contract (point_cloud.ply + mesh.ply + provenance) in METRIC space.

Key facts handled here (learned during the H1/H2 build):
  * The INRIA-lineage rasterizer overflows on our tiny metric scenes (~0.1 units) with
    `cudaErrorInvalidConfiguration`. Fix: scale the scene to ~unit range (S = 1/nerf_radius)
    for training, and scale the outputs back down by S afterwards. LiDAR depth is scaled by
    S inside the loss (via --lidar_depth_scale) so it matches the scaled render depth.
  * nvdiffrast uses the CUDA context (no EGL) on this headless box (patched in scene/mesh.py).
  * The depth loss piggybacks on MILo's depth render from regularization_from_iter (3000).

reconstruct() signature matches stages/stage5_reconstruction/run.py's call site.
"""
from __future__ import annotations
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent.parent
MILO_DIR = _REPO / "third_party" / "MILo" / "milo"
MILO_ENV_PY = Path(os.path.expanduser("~/miniforge3/envs/milo/bin/python"))
MILO_ENV = Path(os.path.expanduser("~/miniforge3/envs/milo"))

sys.path.insert(0, str(_REPO))
from common import colmap_io  # noqa: E402


# --------------------------------------------------------------------------- #
# structured INRIA .ply read/write (for the scale-down)
# --------------------------------------------------------------------------- #
def _read_inria_ply(path):
    with open(path, "rb") as fh:
        assert fh.readline().strip() == b"ply"
        fmt = fh.readline().strip()
        names, count = [], 0
        while True:
            ln = fh.readline().decode().strip()
            if ln.startswith("element vertex"):
                count = int(ln.split()[-1])
            elif ln.startswith("property"):
                names.append(ln.split()[-1])
            elif ln == "end_header":
                break
        data = np.frombuffer(fh.read(count * len(names) * 4), np.float32).reshape(count, len(names)).copy()
    return names, data


def _write_inria_ply(path, names, data):
    header = ["ply", "format binary_little_endian 1.0", f"element vertex {len(data)}"]
    header += [f"property float {n}" for n in names] + ["end_header"]
    with open(path, "wb") as fh:
        fh.write(("\n".join(header) + "\n").encode())
        fh.write(data.astype(np.float32).tobytes())


def _scale_down_gaussians(in_ply, out_ply, S):
    """MILo saves INRIA-format gaussians in the SCALED frame; bring back to metric:
    positions x,y,z /= S ; log-scales scale_i -= log(S). Colors/rot/opacity unchanged."""
    names, data = _read_inria_ply(in_ply)
    idx = {n: i for i, n in enumerate(names)}
    for c in ("x", "y", "z"):
        data[:, idx[c]] /= S
    for c in ("scale_0", "scale_1", "scale_2"):
        if c in idx:
            data[:, idx[c]] -= np.log(S)
    Path(out_ply).parent.mkdir(parents=True, exist_ok=True)
    _write_inria_ply(out_ply, names, data)
    xyz = data[:, [idx["x"], idx["y"], idx["z"]]].copy()
    return len(data), xyz


def _object_box(xyz, pct=1.0, pad_frac=0.10):
    """Object bounding box from the Gaussian centers: robust [pct, 100-pct] percentile + a
    pad fraction. MILo meshes the whole scene the cameras saw and (via the 9-pivots-per-Gaussian
    Delaunay) reaches slightly past the centers, so its mesh box is 2-3x the object; but the
    OBJECT lives entirely inside the Gaussian cloud, so cropping the mesh to this padded box
    trims background/floater surface WITHOUT touching object fidelity. The pad protects the
    object's true edge (the raw center box would shave a few mm). No largest-component step."""
    lo = np.percentile(xyz, pct, axis=0)
    hi = np.percentile(xyz, 100 - pct, axis=0)
    pad = pad_frac * (hi - lo)
    return lo - pad, hi + pad


def _scale_down_mesh(in_mesh, out_mesh, S, crop_box=None):
    import trimesh
    m = trimesh.load(str(in_mesh), process=False)
    m.vertices = np.asarray(m.vertices, np.float64) / S
    if crop_box is not None:
        lo, hi = crop_box
        v = np.asarray(m.vertices)
        inside = np.all((v >= lo) & (v <= hi), axis=1)
        # keep only faces whose ALL 3 vertices sit inside the object box (drops the
        # background/floater surface + any tets bridging out to it)
        m.update_faces(inside[m.faces].all(axis=1))
        m.remove_unreferenced_vertices()
    Path(out_mesh).parent.mkdir(parents=True, exist_ok=True)
    m.export(str(out_mesh))
    return len(m.vertices), len(m.faces)


# --------------------------------------------------------------------------- #
# scene scale + scaled-dataset assembly
# --------------------------------------------------------------------------- #
def _nerf_radius(sparse_dir):
    """3DGS getNerfppNorm radius from camera centers (world units)."""
    imgs = colmap_io.read_images_binary(Path(sparse_dir) / "images.bin")
    C = []
    for im in imgs.values():
        R = colmap_io_quat_to_R(im["qvec"])
        t = np.asarray(im["tvec"], float)
        C.append(-R.T @ t)
    C = np.array(C)
    center = C.mean(0)
    return float(np.linalg.norm(C - center, axis=1).max() * 1.1)


def colmap_io_quat_to_R(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]], float)


def _write_scaled_dataset(src_ds, dst_ds, S):
    """Copy the COLMAP dataset with world scaled by S (image tvec *= S, points3D *= S)."""
    src_ds, dst_ds = Path(src_ds), Path(dst_ds)
    (dst_ds / "sparse" / "0").mkdir(parents=True, exist_ok=True)
    (dst_ds / "images").mkdir(parents=True, exist_ok=True)
    src_sparse = src_ds / "sparse" / "0"
    cams = colmap_io.read_cameras_binary(src_sparse / "cameras.bin")
    imgs = colmap_io.read_images_binary(src_sparse / "images.bin")
    pts = colmap_io.read_points3D_binary(src_sparse / "points3D.bin")
    for im in imgs.values():
        im["tvec"] = [float(v) * S for v in im["tvec"]]
    for p in pts.values():
        p["xyz"] = [float(v) * S for v in p["xyz"]]
    colmap_io.write_model(dst_ds / "sparse" / "0", cams, imgs, pts)
    for im in imgs.values():
        s = (src_ds / "images" / im["name"]).resolve()
        d = dst_ds / "images" / im["name"]
        if d.exists() or d.is_symlink():
            d.unlink()
        os.symlink(s, d)
    return len(imgs)


# --------------------------------------------------------------------------- #
# entry point (called by run.py)
# --------------------------------------------------------------------------- #
def reconstruct(dataset_dir, capture_dir, normals_dir, output_dir, options):
    # MILo's train.py runs with cwd=milo/, so every path handed to the subprocess MUST be
    # absolute (a relative dataset path makes MILo look under milo/ -> "Could not recognize
    # scene type"). Resolve all inputs here.
    dataset_dir, output_dir = Path(dataset_dir).resolve(), Path(output_dir).resolve()
    capture_dir = str(Path(capture_dir).resolve())
    opt = options or {}
    depth_lambda = float(opt.get("depth_lambda", 0.2))
    imp_metric = opt.get("imp_metric", "indoor")
    resolution = str(opt.get("milo_resolution", 1))          # -r (1 = full res as-loaded)
    dense = bool(opt.get("dense_gaussians", True))           # default ON (recovers thin structure)
    data_device = str(opt.get("data_device", "cpu"))         # "cpu" fits the A4000; "cuda" is faster on the A6000

    # 1) scene scale -> ~unit range for the rasterizer
    radius = _nerf_radius(dataset_dir / "sparse" / "0")
    S = float(opt.get("milo_scale", 0.0)) or max(1.0, 1.0 / max(radius, 1e-6))
    scaled_ds = output_dir / "_scaled_dataset"
    n_imgs = _write_scaled_dataset(dataset_dir, scaled_ds, S)
    raw_out = output_dir / "_milo_raw"
    print(f"[milo] nerf_radius={radius:.4f} -> scale S={S:.3f}; {n_imgs} imgs; "
          f"depth_lambda={depth_lambda}; dense={dense}; data_device={data_device}")

    env = dict(os.environ)
    env["CUDA_HOME"] = str(MILO_ENV)
    env["PATH"] = f"{MILO_ENV}/bin:" + env.get("PATH", "")
    env["LD_LIBRARY_PATH"] = f"{MILO_ENV}/lib:" + env.get("LD_LIBRARY_PATH", "")
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    env["PYTHONPATH"] = str(_HERE / "supervision") + ":" + env.get("PYTHONPATH", "")

    # 2) train (depth-supervised) then extract mesh, in the milo env
    train_cmd = [str(MILO_ENV_PY), "train.py", "-s", str(scaled_ds), "-m", str(raw_out),
                 "--imp_metric", imp_metric, "--rasterizer", "radegs", "-r", resolution, "--quiet",
                 "--lidar_depth_dir", str(capture_dir), "--lidar_depth_lambda", str(depth_lambda),
                 "--lidar_depth_scale", str(S), "--data_device", data_device]
    if dense:
        # --dense_gaussians recovers thin structure (glasses frame, edges) the base densifier drops.
        # Tradeoff: it slightly roughens flat regions (redistribution, not a strict win). REVISIT on the
        # A6000 — tune MILo's regularizers + add feature-preserving smoothing (docs/EXPERIMENTS_BACKLOG.md).
        train_cmd.append("--dense_gaussians")
    subprocess.run(train_cmd, cwd=str(MILO_DIR), env=env, check=True)
    subprocess.run([str(MILO_ENV_PY), "mesh_extract_sdf.py", "-s", str(scaled_ds), "-m", str(raw_out),
                    "--rasterizer", "radegs"], cwd=str(MILO_DIR), env=env, check=True)

    # 3) scale outputs back to METRIC + write the Stage-6 contract
    it_dirs = sorted((raw_out / "point_cloud").glob("iteration_*"), key=lambda p: int(p.name.split("_")[1]))
    gz_in = it_dirs[-1] / "point_cloud.ply"
    n_g, gxyz = _scale_down_gaussians(gz_in, output_dir / "point_cloud.ply", S)
    # crop the mesh to the object (Gaussian-cloud) box + pad so it comes out object-tight,
    # not enclosing the whole scene. Disable with options milo_crop_pad < 0.
    pad = float(opt.get("milo_crop_pad", 0.10))
    crop = _object_box(gxyz, pad_frac=pad) if pad >= 0 else None
    nv, nf = _scale_down_mesh(raw_out / "mesh_learnable_sdf.ply", output_dir / "mesh.ply", S, crop_box=crop)

    prov = {"stage5_host": "MILo (in-loop mesh, depth-supervised)",
            "gaussians": n_g, "views": n_imgs, "depth_lambda": depth_lambda,
            "milo_scale": S, "nerf_radius": radius, "imp_metric": imp_metric,
            "mesh_vertices": nv, "mesh_triangles": nf, "mesh_cropped_to_object_pad": pad if crop else None,
            "rasterizer": "radegs", "note": "trained scaled x1/radius; outputs re-metriced /S; mesh cropped to Gaussian box+pad"}
    (output_dir / "provenance_stage5.json").write_text(json.dumps(prov, indent=2))
    print(f"[milo] DONE: {output_dir}/point_cloud.ply ({n_g} gaussians), mesh.ply ({nv} verts / {nf} tris)")
    return prov
