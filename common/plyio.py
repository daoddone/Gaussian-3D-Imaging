"""Minimal, dependency-light PLY point-cloud reader/writer (stdlib + numpy).

Supports the ``vertex`` element with x,y,z positions and optional colours
(red/green/blue, uint8) and normals (nx,ny,nz, float). Handles ``ascii`` and
``binary_little_endian`` formats. This exists so every stage can read/write
plain point clouds (Stage 2 ``points.ply``, Stage 3 ``points_metric.ply``)
without pulling a heavy dependency into ``common/``; Open3D is used only where
richer geometry ops (e.g. ICP) are actually needed.

This is a plain point cloud writer, not a Gaussian-splat ``.ply`` writer
(Stage 6 output carries per-Gaussian attributes and is produced by Stage 5).
"""
from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

# PLY property type -> (struct char, numpy dtype, byte size)
_PLY_TYPES = {
    "char": ("b", np.int8),
    "int8": ("b", np.int8),
    "uchar": ("B", np.uint8),
    "uint8": ("B", np.uint8),
    "short": ("h", np.int16),
    "int16": ("h", np.int16),
    "ushort": ("H", np.uint16),
    "uint16": ("H", np.uint16),
    "int": ("i", np.int32),
    "int32": ("i", np.int32),
    "uint": ("I", np.uint32),
    "uint32": ("I", np.uint32),
    "float": ("f", np.float32),
    "float32": ("f", np.float32),
    "double": ("d", np.float64),
    "float64": ("d", np.float64),
}


def write_ply(path, points, colors=None, normals=None, binary=True):
    """Write a point cloud.

    points  : (N,3) float array, in meters.
    colors  : optional (N,3) uint8 (0..255).
    normals : optional (N,3) float unit vectors.
    """
    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must be (N,3)")
    n = points.shape[0]

    props = [("x", "float"), ("y", "float"), ("z", "float")]
    columns = [points]
    if normals is not None:
        normals = np.asarray(normals, dtype=np.float32).reshape(n, 3)
        props += [("nx", "float"), ("ny", "float"), ("nz", "float")]
        columns.append(normals)
    if colors is not None:
        colors = np.asarray(colors).reshape(n, 3).astype(np.uint8)
        props += [("red", "uchar"), ("green", "uchar"), ("blue", "uchar")]
        columns.append(colors)

    fmt = "binary_little_endian" if binary else "ascii"
    header = ["ply", f"format {fmt} 1.0", f"element vertex {n}"]
    header += [f"property {t} {name}" for name, t in props]
    header.append("end_header")
    header_bytes = ("\n".join(header) + "\n").encode("ascii")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if binary:
        with open(path, "wb") as fh:
            fh.write(header_bytes)
            # One little-endian struct record per vertex. Cast each field to the
            # Python type struct expects (int for integer props, float otherwise),
            # because the interleaved array is float64 after concatenation.
            struct_fmt = "<" + "".join(_PLY_TYPES[t][0] for _, t in props)
            packer = struct.Struct(struct_fmt)
            casters = [int if _PLY_TYPES[t][0] in "bBhHiI" else float for _, t in props]
            flat = np.concatenate([c for c in columns], axis=1)
            for row in flat:
                fh.write(packer.pack(*[cast(v) for cast, v in zip(casters, row.tolist())]))
    else:
        with open(path, "w") as fh:
            fh.write(header_bytes.decode("ascii"))
            flat = np.concatenate([c for c in columns], axis=1)
            for row in flat:
                fh.write(" ".join(_fmt_ascii(v, t) for v, (_, t) in zip(row.tolist(), props)) + "\n")


def _fmt_ascii(value, ply_type):
    if ply_type in ("uchar", "uint8", "char", "int8", "int", "int32", "uint", "uint32", "short", "ushort", "int16", "uint16"):
        return str(int(value))
    return repr(float(value))


def read_ply(path):
    """Read a point cloud written by :func:`write_ply` (or a compatible file).

    Returns a dict with ``points`` (N,3 float32) and, when present, ``colors``
    (N,3 uint8) and ``normals`` (N,3 float32).
    """
    path = Path(path)
    with open(path, "rb") as fh:
        magic = fh.readline().strip()
        if magic != b"ply":
            raise ValueError(f"{path}: not a PLY file")
        fmt = None
        count = 0
        props = []  # list of (name, type)
        in_vertex = False
        while True:
            line = fh.readline()
            if not line:
                raise ValueError(f"{path}: unexpected EOF in header")
            tok = line.split()
            if not tok:
                continue
            key = tok[0]
            if key == b"format":
                fmt = tok[1].decode()
            elif key == b"element":
                in_vertex = tok[1] == b"vertex"
                if in_vertex:
                    count = int(tok[2])
            elif key == b"property" and in_vertex:
                props.append((tok[2].decode(), tok[1].decode()))
            elif key == b"end_header":
                break

        names = [p[0] for p in props]
        types = [p[1] for p in props]

        if fmt == "ascii":
            data = np.zeros((count, len(props)), dtype=np.float64)
            for i in range(count):
                vals = fh.readline().split()
                data[i] = [float(v) for v in vals[: len(props)]]
        elif fmt == "binary_little_endian":
            struct_fmt = "<" + "".join(_PLY_TYPES[t][0] for t in types)
            packer = struct.Struct(struct_fmt)
            data = np.zeros((count, len(props)), dtype=np.float64)
            for i in range(count):
                rec = fh.read(packer.size)
                data[i] = packer.unpack(rec)
        else:
            raise ValueError(f"{path}: unsupported PLY format {fmt!r}")

    out = {}
    idx = {name: k for k, name in enumerate(names)}
    out["points"] = data[:, [idx["x"], idx["y"], idx["z"]]].astype(np.float32)
    if all(c in idx for c in ("nx", "ny", "nz")):
        out["normals"] = data[:, [idx["nx"], idx["ny"], idx["nz"]]].astype(np.float32)
    if all(c in idx for c in ("red", "green", "blue")):
        out["colors"] = data[:, [idx["red"], idx["green"], idx["blue"]]].astype(np.uint8)
    return out
