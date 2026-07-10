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
    # dense_gaussians supports "auto" (resolved after schedule selection below): the two PROVEN
    # recipes used OPPOSITE dense settings — feet flagship = fast + dense TRUE (its 7x budget on a
    # sparse init came from dense; at fast, dense never had a VRAM problem), face v3 = quality_mid +
    # dense FALSE (the VRAM/distillation retirement was a QUALITY-schedule finding). Unification
    # design error (owner-caught 2026-07-09): freezing dense to one side produced an untested hybrid
    # (fast+nondense) on weak captures. "auto" couples the setting to the branch's proven recipe.
    dense_raw = opt.get("dense_gaussians", True)
    dense = dense_raw if isinstance(dense_raw, bool) else (str(dense_raw).lower() != "false")
    data_device = str(opt.get("data_device", "cpu"))         # "cpu" fits the A4000; "cuda" is faster on the A6000
    mesh_reg = bool(opt.get("mesh_regularization", True))    # in-loop mesh (MILo's core); off = cloud-only

    # nvdiffrast's CUDA rasterizer caps the in-loop MESH render at 2048 px/side. A capture whose color
    # resolution exceeds that (the HQ-Depth 4032x3024 path) triggers a CUDA-700 illegal address in
    # nvdiffrast's fineRasterKernel at the first mesh build (iter 8001) — reproducible, pose- AND
    # density-independent; ARKit's 1920x1440 is under the cap and unaffected. (v0.3.3's >2048 auto-tiling
    # fails on this compiled build.) This is a HARD rasterizer limit, not a quality knob: cap the training
    # resolution so max(w,h) <= 2048 whenever mesh regularization is on. Logged for transparency. As a
    # side benefit it equalizes render resolution vs the ARKit path (2016 vs 1920) for a fair comparison.
    if mesh_reg:
        _cams = colmap_io.read_cameras_binary(Path(dataset_dir) / "sparse" / "0" / "cameras.bin")
        _max_side = max(max(int(c["width"]), int(c["height"])) for c in _cams.values())
        _r = int(resolution)
        while _max_side / _r > 2048 and _r < 8:
            _r *= 2
        if _r != int(resolution):
            print(f"[milo] nvdiffrast 2048 mesh-raster cap: capture {_max_side}px at -r {resolution} exceeds "
                  f"2048 -> using -r {_r} (~{_max_side // _r}px) so the in-loop mesh rasterizes")
            resolution = str(_r)

    # AUTO capacity selection (owner mandate: any capture processed to its best capacity without
    # per-capture tuning). THE LAW from the quality campaign: capacity must match input quality —
    # excess capacity on weak inputs manufactures junk detail. Capture strength is measured by
    # view count AND SfM/init point support (AND, because ARKit models carry 100k BAKED DA3 points
    # that would otherwise inflate the score). Thresholds calibrated on the campaign anchors:
    # face v1 172v/45k pts (strong), face v3 362v/117k (strong), feet 47-57v (weak).
    if str(opt.get("milo_schedule", "fast")) == "auto":
        _imgs_n = len(colmap_io.read_images_binary(dataset_dir / "sparse" / "0" / "images.bin"))
        _pts_n = len(colmap_io.read_points3D_binary(dataset_dir / "sparse" / "0" / "points3D.bin"))
        _strong = _imgs_n >= 120 and _pts_n >= 40_000
        opt = dict(opt)
        opt["milo_schedule"] = "quality_mid" if _strong else "fast"
        if _strong and str(opt.get("mesh_config", "default")) == "default":
            opt["mesh_config"] = "quality_mid"
        print(f"[milo] AUTO capacity: views={_imgs_n} init_pts={_pts_n} -> "
              f"{'STRONG' if _strong else 'WEAK'} capture -> schedule={opt['milo_schedule']}")

    # Branch-coupled dense (see comment at dense_raw): weak/fast -> TRUE (flagship recipe: aggressive
    # growth compensates the sparse init); strong/quality -> FALSE (v3 recipe + the real VRAM limit).
    if str(dense_raw).lower() == "auto":
        dense = str(opt.get("milo_schedule", "fast")) == "fast"
        print(f"[milo] AUTO dense_gaussians -> {dense} (branch-coupled to schedule)")

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
    mesh_config_name = str(opt.get("mesh_config", "default"))
    if mesh_config_name and mesh_config_name != "default":
        train_cmd += ["--mesh_config", mesh_config_name]   # MILo mesh tet-grid preset: verylowres..veryhighres
    # Training SCHEDULE preset (configs/<name>). Default "fast" = Mini-Splatting2 accelerated: densify
    # stops at iter 3k, simp at 3k/8k, 18k iters -> small gaussian budgets (~50-400k) that CAP mesh
    # granularity. "quality" = MILo/3DGS stock: 30k iters, densify until 15k, simp 15k/20k, mesh reg
    # 20k->30k (use with mesh_config "quality"). NOTE: config-file values OVERRIDE CLI args in MILo.
    schedule = str(opt.get("milo_schedule", "fast"))
    if schedule and schedule != "fast":
        train_cmd += ["--config_path", f"./configs/{schedule}"]
    # Distillation retention percentile (T8): the DIRECT final gaussian/vertex-count control
    # (0.99 = stock; e.g. 0.995 keeps ~2x more through simp1/simp2 for strong captures).
    _ret = float(opt.get("simp_retention", 0.99))
    if _ret != 0.99:
        train_cmd += ["--simp_retention", str(_ret)]
        print(f"[milo] simp retention = {_ret}")

    # Edge-aware flatness prior (textureless-surface wobble, e.g. table between the feet).
    _flat = float(opt.get("flatness_lambda", 0.0))
    if _flat > 0:
        train_cmd += ["--flatness_lambda", str(_flat),
                      "--flatness_edge_beta", str(opt.get("flatness_edge_beta", 8.0))]
        print(f"[milo] flatness prior ON: lambda={_flat}")
    # Subject isolation (stage 2): photometric mask + out-of-mask opacity penalty. Masks are made by
    # scripts/make_subject_masks.py into <session>/subject_masks (session root = capture_dir's parent).
    if bool(opt.get("subject_isolation", False)):
        mask_dir = Path(capture_dir).parent / "subject_masks"
        if (mask_dir / "box.json").exists():
            train_cmd += ["--subject_mask_dir", str(mask_dir)]
            print(f"[milo] subject isolation ON: masks from {mask_dir}")
        else:
            print(f"[milo] WARNING: subject_isolation requested but {mask_dir} missing — run "
                  f"scripts/make_subject_masks.py first; proceeding WITHOUT isolation")
    # Subject 3D box-prune — the designed "mop-up" half of isolation (photometric masks stop background
    # BIRTH; hull-interior background keeps real-pixel gradients and needs REMOVAL — the Andrew block).
    # Gated independently of the mask so mask-only / prune-only / both can be A/B'd. box.json is in
    # METRIC units; training runs in the S-scaled dataset, so scale the box by S.
    if bool(opt.get("subject_box_prune", False)):
        box_json = Path(capture_dir).parent / "subject_masks" / "box.json"
        if box_json.exists():
            _bj = json.loads(box_json.read_text())
            _bp = [v * S for v in _bj["box_lo"]] + [v * S for v in _bj["box_hi"]]
            # equals-form: the value can start with a minus sign, which argparse would
            # otherwise parse as a new flag ("expected one argument")
            train_cmd += ["--subject_box_prune=" + ",".join(f"{v:.6f}" for v in _bp)]
            print(f"[milo] subject box-prune ON: metric box.json scaled xS={S:.3f}")
        else:
            print(f"[milo] WARNING: subject_box_prune requested but {box_json} missing — skipping")
    if dense:
        # --dense_gaussians recovers thin structure (glasses frame, edges) the base densifier drops.
        # Tradeoff: it slightly roughens flat regions (redistribution, not a strict win). REVISIT on the
        # A6000 — tune MILo's regularizers + add feature-preserving smoothing (docs/EXPERIMENTS_BACKLOG.md).
        train_cmd.append("--dense_gaussians")
    # In-loop mesh regularization is MILo's core feature, but its nvdiffrast rasterization crashes
    # (CUDA 700) reproducibly for the pose-free (DA3-estimated-pose) path at the simplification step —
    # the mesh degenerates during training. Disable it there to still get the gaussian cloud (the
    # primary output); no learnable SDF is produced, so mesh extraction/scale-down is skipped.
    mesh_reg = bool(opt.get("mesh_regularization", True))
    if not mesh_reg:
        train_cmd.append("--no_mesh_regularization")
    subprocess.run(train_cmd, cwd=str(MILO_DIR), env=env, check=True)
    # Extraction must target the iteration training ACTUALLY saved (schedule-dependent: fast=18k,
    # quality=30k). mesh_extract_sdf's own default is a hardcoded 18000 -> pass the latest found.
    it_dirs = sorted((raw_out / "point_cloud").glob("iteration_*"), key=lambda p: int(p.name.split("_")[1]))
    last_iter = int(it_dirs[-1].name.split("_")[1])
    if mesh_reg:
        subprocess.run([str(MILO_ENV_PY), "mesh_extract_sdf.py", "-s", str(scaled_ds), "-m", str(raw_out),
                        "--rasterizer", "radegs", "--iteration", str(last_iter)],
                       cwd=str(MILO_DIR), env=env, check=True)

    # 3) scale outputs back to METRIC + write the Stage-6 contract
    gz_in = it_dirs[-1] / "point_cloud.ply"
    n_g, gxyz = _scale_down_gaussians(gz_in, output_dir / "point_cloud.ply", S)
    nv, nf, pad, crop = 0, 0, float(opt.get("milo_crop_pad", 0.10)), None
    if mesh_reg:
        # crop the mesh to the object (Gaussian-cloud) box + pad so it comes out object-tight,
        # not enclosing the whole scene. Disable with options milo_crop_pad < 0.
        crop = _object_box(gxyz, pad_frac=pad) if pad >= 0 else None
        nv, nf = _scale_down_mesh(raw_out / "mesh_learnable_sdf.ply", output_dir / "mesh.ply", S, crop_box=crop)

    # Metric-scale provenance (task T1): auto-discover the scale sidecar written by
    # scripts/pose_ba/04_metric_anchor.py in ancestors of the dataset dir (e.g. metric_sfm/), so
    # every reconstruction records WHERE its absolute scale came from and how confident it is.
    metric_scale = None
    try:
        _cands = []
        for _anc in Path(dataset_dir).resolve().parents:
            _cands += [_anc / "scale_sidecar.json", _anc / "metric_sfm" / "scale_sidecar.json",
                       _anc / "metric" / "scale_sidecar.json"]
        for _sc in _cands:
            if _sc.exists():
                _s = json.loads(_sc.read_text())
                metric_scale = {k: _s.get(k) for k in
                                ("primary_anchor", "scale", "confidence", "anchor_agreement_pct")}
                break
    except Exception:  # noqa: BLE001 — provenance enrichment must never fail the run
        pass

    prov = {"stage5_host": "MILo (in-loop mesh, depth-supervised)" if mesh_reg else "MILo (splatting only, mesh reg off)",
            "milo_schedule": schedule, "mesh_config": mesh_config_name,
            "gaussians": n_g, "views": n_imgs, "depth_lambda": depth_lambda, "mesh_regularization": mesh_reg,
            "milo_scale": S, "nerf_radius": radius, "imp_metric": imp_metric,
            "metric_scale_anchor": metric_scale,
            "mesh_vertices": nv, "mesh_triangles": nf, "mesh_cropped_to_object_pad": pad if crop else None,
            "rasterizer": "radegs", "note": "trained scaled x1/radius; outputs re-metriced /S; mesh cropped to Gaussian box+pad"}
    (output_dir / "provenance_stage5.json").write_text(json.dumps(prov, indent=2))

    # Mesh appearance bake (the "cartoonish" fix): re-color vertices from the TOP-3 sharpest
    # best-angle source views (visibility-tested) instead of MILo's view-averaged estimate.
    # Writes mesh_textured.ply alongside mesh.ply. Best-effort — never fails the reconstruction.
    try:
        _evalpy = Path(os.path.expanduser("~/miniforge3/envs/pipeline_stage2_frontend/bin/python"))
        if _evalpy.exists() and mesh_reg and (output_dir / "mesh.ply").exists():
            subprocess.run([str(_evalpy), str(_REPO / "scripts" / "bake_mesh_colors.py"),
                            "--output-dir", str(output_dir),
                            "--colmap", str(Path(dataset_dir) / "sparse" / "0"),
                            "--images", str(Path(dataset_dir) / "images")],
                           timeout=1800, check=False,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if (output_dir / "mesh_textured.ply").exists():
                print(f"[milo] sharp color bake -> {output_dir}/mesh_textured.ply")
    except Exception as _e:  # noqa: BLE001
        print(f"[milo] color bake skipped: {_e}")

    # Analysis-ready OBJ export (task T6): clean/decimate/UV-unwrap/texture-bake -> export/mesh.obj
    # (+MTL+PNG, mm) + mesh_metric_mm.ply + export_meta.json with auto-discovered scale provenance
    # (T1 sidecar). Default ON so every reconstruction emits the downstream deliverable; disable with
    # options export_obj: false. Best-effort — never fails the reconstruction. ~5-15 min extra.
    try:
        _evalpy = Path(os.path.expanduser("~/miniforge3/envs/pipeline_stage2_frontend/bin/python"))
        if (_evalpy.exists() and mesh_reg and (output_dir / "mesh.ply").exists()
                and bool(opt.get("export_obj", True))):
            subprocess.run([str(_evalpy), str(_REPO / "scripts" / "export_mesh_obj.py"),
                            "--output-dir", str(output_dir),
                            "--colmap", str(Path(dataset_dir) / "sparse" / "0"),
                            "--images", str(Path(dataset_dir) / "images")],
                           timeout=2400, check=False,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if (output_dir / "export" / "mesh.obj").exists():
                print(f"[milo] OBJ export -> {output_dir}/export/ (mesh.obj + texture + export_meta.json)")
            else:
                print("[milo] OBJ export did not produce mesh.obj — run scripts/export_mesh_obj.py "
                      "manually to see the error")
    except Exception as _e:  # noqa: BLE001
        print(f"[milo] OBJ export skipped: {_e}")

    # Standardized review set (owner request): subject-centered/axis-aligned .ply copies + labeled
    # canonical renders + subject-level stats, so every output is instantly reviewable by human or
    # script without manual reorientation. Best-effort — never fails the reconstruction.
    try:
        _evalpy = Path(os.path.expanduser("~/miniforge3/envs/pipeline_stage2_frontend/bin/python"))
        if _evalpy.exists():
            subprocess.run([str(_evalpy), str(_REPO / "scripts" / "export_review.py"), str(output_dir),
                            "--label", f"{Path(output_dir).parent.name}/{Path(output_dir).name}"],
                           timeout=900, check=False,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print(f"[milo] review set -> {output_dir}/review/ (subject-centered ply + views.png + review.json)")
    except Exception as _e:  # noqa: BLE001
        print(f"[milo] review export skipped: {_e}")
    if mesh_reg:
        print(f"[milo] DONE: {output_dir}/point_cloud.ply ({n_g} gaussians), mesh.ply ({nv} verts / {nf} tris)")
    else:
        print(f"[milo] DONE: {output_dir}/point_cloud.ply ({n_g} gaussians), mesh reg OFF (cloud only)")
    return prov
