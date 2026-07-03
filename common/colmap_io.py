"""Read and write the COLMAP sparse-model binary format (stdlib + numpy).

The reconstruction host (Stage 5, MILo) ingests a COLMAP sparse model, so
Stages 2 and 3 must emit ``cameras.bin``, ``images.bin`` and ``points3D.bin``.
This module reimplements just enough of COLMAP's ``read_write_model.py`` to
write (and read back, for tests) those three files, using only ``struct`` and
``numpy`` so it can live in the dependency-light ``common/`` package.

Binary layout (little-endian), mirroring COLMAP exactly:

cameras.bin
    uint64 num_cameras
    per camera:  int32 camera_id, int32 model_id, uint64 width, uint64 height,
                 float64 * num_params  (params)

images.bin
    uint64 num_images
    per image:   int32 image_id,
                 float64 qw, qx, qy, qz,       # world-to-camera rotation, w-first
                 float64 tx, ty, tz,           # world-to-camera translation
                 int32 camera_id,
                 char* name (null-terminated),
                 uint64 num_points2D,
                 per point2D: float64 x, float64 y, int64 point3D_id

points3D.bin
    uint64 num_points
    per point:   uint64 point3D_id, float64 x,y,z, uint8 r,g,b, float64 error,
                 uint64 track_length,
                 per track elem: int32 image_id, int32 point2D_idx

Poses are world-to-camera; quaternions are (w, x, y, z). See
``io_contracts/frontend_output.md`` and ``metric_output.md``.
"""
from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

from common.conventions import rotmat_to_quat, quat_to_rotmat

# COLMAP camera model ids we use. (id, name, num_params, param order)
CAMERA_MODELS = {
    "SIMPLE_PINHOLE": (0, 3),  # f, cx, cy
    "PINHOLE": (1, 4),  # fx, fy, cx, cy
    "SIMPLE_RADIAL": (2, 4),  # f, cx, cy, k
    "RADIAL": (3, 5),
    "OPENCV": (4, 8),  # fx, fy, cx, cy, k1, k2, p1, p2
}
_MODEL_BY_ID = {v[0]: (k, v[1]) for k, v in CAMERA_MODELS.items()}


# --------------------------------------------------------------------------- #
# low-level helpers
# --------------------------------------------------------------------------- #
def _write(fh, fmt, *vals):
    fh.write(struct.pack("<" + fmt, *vals))


def _read(fh, fmt):
    fmt = "<" + fmt
    size = struct.calcsize(fmt)
    return struct.unpack(fmt, fh.read(size))


# --------------------------------------------------------------------------- #
# writers
# --------------------------------------------------------------------------- #
def write_cameras_binary(path, cameras):
    """cameras: dict camera_id -> {"model", "width", "height", "params"(seq)}."""
    with open(path, "wb") as fh:
        _write(fh, "Q", len(cameras))
        for cam_id, cam in sorted(cameras.items()):
            model_id, nparams = CAMERA_MODELS[cam["model"]]
            params = list(cam["params"])
            if len(params) != nparams:
                raise ValueError(
                    f"camera {cam_id} model {cam['model']} expects {nparams} params, got {len(params)}"
                )
            _write(fh, "iiQQ", int(cam_id), int(model_id), int(cam["width"]), int(cam["height"]))
            _write(fh, "d" * nparams, *[float(p) for p in params])


def write_images_binary(path, images):
    """images: dict image_id -> {"qvec"(w,x,y,z), "tvec"(3), "camera_id", "name",
    optional "xys"(N,2), "point3D_ids"(N,)}. Poses are world-to-camera."""
    with open(path, "wb") as fh:
        _write(fh, "Q", len(images))
        for img_id, img in sorted(images.items()):
            q = [float(v) for v in img["qvec"]]
            t = [float(v) for v in img["tvec"]]
            _write(fh, "idddddddi", int(img_id), q[0], q[1], q[2], q[3], t[0], t[1], t[2], int(img["camera_id"]))
            fh.write(img["name"].encode("utf-8") + b"\x00")
            xys = img.get("xys")
            p3d = img.get("point3D_ids")
            npt = 0 if xys is None else len(xys)
            _write(fh, "Q", npt)
            for k in range(npt):
                pid = -1 if p3d is None else int(p3d[k])
                _write(fh, "ddq", float(xys[k][0]), float(xys[k][1]), pid)


def write_points3D_binary(path, points3D):
    """points3D: dict point3D_id -> {"xyz"(3), "rgb"(3 uint8), "error"(float),
    optional "track"[(image_id, point2D_idx), ...]}. May be empty."""
    with open(path, "wb") as fh:
        _write(fh, "Q", len(points3D))
        for pid, p in sorted(points3D.items()):
            xyz = [float(v) for v in p["xyz"]]
            rgb = [int(v) for v in p.get("rgb", (128, 128, 128))]
            err = float(p.get("error", 0.0))
            _write(fh, "Q", int(pid))
            _write(fh, "ddd", *xyz)
            _write(fh, "BBB", *rgb)
            _write(fh, "d", err)
            track = p.get("track", [])
            _write(fh, "Q", len(track))
            for image_id, p2d_idx in track:
                _write(fh, "ii", int(image_id), int(p2d_idx))


def write_model(out_dir, cameras, images, points3D):
    """Write all three files into ``out_dir`` (created if needed)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_cameras_binary(out_dir / "cameras.bin", cameras)
    write_images_binary(out_dir / "images.bin", images)
    write_points3D_binary(out_dir / "points3D.bin", points3D)


# --------------------------------------------------------------------------- #
# readers (used by contract tests / round-trip checks)
# --------------------------------------------------------------------------- #
def read_cameras_binary(path):
    cameras = {}
    with open(path, "rb") as fh:
        (num,) = _read(fh, "Q")
        for _ in range(num):
            cam_id, model_id, width, height = _read(fh, "iiQQ")
            model_name, nparams = _MODEL_BY_ID[model_id]
            params = _read(fh, "d" * nparams)
            cameras[cam_id] = {
                "model": model_name,
                "width": width,
                "height": height,
                "params": list(params),
            }
    return cameras


def read_images_binary(path):
    images = {}
    with open(path, "rb") as fh:
        (num,) = _read(fh, "Q")
        for _ in range(num):
            rec = _read(fh, "idddddddi")
            img_id = rec[0]
            qvec = np.array(rec[1:5])
            tvec = np.array(rec[5:8])
            cam_id = rec[8]
            name = b""
            while True:
                c = fh.read(1)
                if c == b"\x00":
                    break
                name += c
            (npt,) = _read(fh, "Q")
            xys = np.zeros((npt, 2))
            p3d = np.zeros(npt, dtype=np.int64)
            for k in range(npt):
                x, y, pid = _read(fh, "ddq")
                xys[k] = (x, y)
                p3d[k] = pid
            images[img_id] = {
                "qvec": qvec,
                "tvec": tvec,
                "camera_id": cam_id,
                "name": name.decode("utf-8"),
                "xys": xys,
                "point3D_ids": p3d,
            }
    return images


def read_points3D_binary(path):
    points = {}
    with open(path, "rb") as fh:
        (num,) = _read(fh, "Q")
        for _ in range(num):
            (pid,) = _read(fh, "Q")
            xyz = np.array(_read(fh, "ddd"))
            rgb = np.array(_read(fh, "BBB"))
            (err,) = _read(fh, "d")
            (tlen,) = _read(fh, "Q")
            track = [tuple(_read(fh, "ii")) for _ in range(tlen)]
            points[pid] = {"xyz": xyz, "rgb": rgb, "error": err, "track": track}
    return points


# --------------------------------------------------------------------------- #
# convenience: build a COLMAP model from per-frame intrinsics + extrinsics
# --------------------------------------------------------------------------- #
def build_pinhole_model(frame_ids, K_by_frame, Rt_w2c_by_frame, names_by_frame,
                        resolution, shared_intrinsics=False):
    """Assemble (cameras, images) dicts for a PINHOLE model.

    frame_ids        : ordered list of frame id strings.
    K_by_frame       : {frame_id: 3x3 intrinsic matrix}.
    Rt_w2c_by_frame  : {frame_id: (R, t)} world-to-camera.
    names_by_frame   : {frame_id: image filename written in images.bin}.
    resolution       : (width, height) the intrinsics apply to.
    shared_intrinsics: if True, one camera is shared by all images (assumes K
                       is identical across frames); otherwise one camera/frame.

    image_id and camera_id are assigned as 1-based indices in frame order,
    matching COLMAP's 1-based convention. points3D is left to the caller.
    """
    width, height = int(resolution[0]), int(resolution[1])
    cameras = {}
    images = {}

    def K_to_params(K):
        return [float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])]

    if shared_intrinsics:
        K0 = np.asarray(K_by_frame[frame_ids[0]], dtype=float)
        cameras[1] = {"model": "PINHOLE", "width": width, "height": height, "params": K_to_params(K0)}

    for i, fid in enumerate(frame_ids, start=1):
        R, t = Rt_w2c_by_frame[fid]
        qvec = rotmat_to_quat(R)
        if shared_intrinsics:
            cam_id = 1
        else:
            cam_id = i
            K = np.asarray(K_by_frame[fid], dtype=float)
            cameras[cam_id] = {"model": "PINHOLE", "width": width, "height": height, "params": K_to_params(K)}
        images[i] = {
            "qvec": qvec,
            "tvec": np.asarray(t, dtype=float).reshape(3),
            "camera_id": cam_id,
            "name": names_by_frame[fid],
        }
    return cameras, images
