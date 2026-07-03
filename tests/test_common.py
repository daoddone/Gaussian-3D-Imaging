"""Unit tests for the common/ package. Runnable with pytest, or directly via
tests/run_all.py (no pytest dependency required)."""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np

from common import conventions as C
from common import plyio, colmap_io


def test_pose_inverse_and_center():
    rng = np.random.default_rng(0)
    ang = 0.7
    R = np.array([[np.cos(ang), -np.sin(ang), 0], [np.sin(ang), np.cos(ang), 0], [0, 0, 1]])
    Cc = np.array([0.3, -0.2, 1.5])
    R_w2c, t_w2c = C.invert_pose(R, Cc)          # from c2w to w2c
    center = C.camera_center(R_w2c, t_w2c, C.WORLD_TO_CAMERA)
    assert np.allclose(center, Cc, atol=1e-9)
    center2 = C.camera_center(R, Cc, C.CAMERA_TO_WORLD)
    assert np.allclose(center2, Cc, atol=1e-9)


def test_quat_roundtrip():
    for ang in (0.0, 0.3, 1.2, 2.9):
        axis = np.array([0.2, 0.9, -0.3]); axis /= np.linalg.norm(axis)
        K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
        R = np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * (K @ K)
        q = C.rotmat_to_quat(R)
        assert abs(np.linalg.norm(q) - 1.0) < 1e-9
        assert np.allclose(C.quat_to_rotmat(q), R, atol=1e-8)


def test_arkit_conversion():
    ang = 0.5
    R = np.array([[np.cos(ang), 0, np.sin(ang)], [0, 1, 0], [-np.sin(ang), 0, np.cos(ang)]])
    T = np.eye(4); T[:3, :3] = R; T[:3, 3] = [1, 2, 3]
    R_cv, t_cv = C.opencv_c2w_from_arkit(T)
    assert np.allclose(R_cv, R @ np.diag([1, -1, -1]), atol=1e-12)
    assert np.allclose(t_cv, [1, 2, 3])


def test_ply_roundtrip():
    rng = np.random.default_rng(1)
    pts = rng.normal(size=(200, 3)).astype(np.float32)
    cols = rng.integers(0, 256, size=(200, 3)).astype(np.uint8)
    nrm = rng.normal(size=(200, 3)); nrm /= np.linalg.norm(nrm, axis=1, keepdims=True)
    nrm = nrm.astype(np.float32)
    for binary in (True, False):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "cloud.ply"
            plyio.write_ply(p, pts, colors=cols, normals=nrm, binary=binary)
            back = plyio.read_ply(p)
            assert np.allclose(back["points"], pts, atol=1e-5), f"points binary={binary}"
            assert np.array_equal(back["colors"], cols), f"colors binary={binary}"
            assert np.allclose(back["normals"], nrm, atol=1e-5), f"normals binary={binary}"


def test_colmap_roundtrip():
    ang = 0.9
    R = np.array([[np.cos(ang), -np.sin(ang), 0], [np.sin(ang), np.cos(ang), 0], [0, 0, 1.0]])
    t = np.array([0.1, 0.2, 0.3])
    frame_ids = ["000001", "000002"]
    K = np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1.0]])
    cams, imgs = colmap_io.build_pinhole_model(
        frame_ids, {f: K for f in frame_ids},
        {f: (R, t) for f in frame_ids},
        {f: f"{f}.png" for f in frame_ids},
        resolution=(640, 480), shared_intrinsics=True)
    pts3d = {1: {"xyz": np.array([1.0, 2.0, 3.0]), "rgb": (10, 20, 30),
                 "error": 0.5, "track": [(1, 0), (2, 3)]}}
    with tempfile.TemporaryDirectory() as td:
        colmap_io.write_model(td, cams, imgs, pts3d)
        rc = colmap_io.read_cameras_binary(Path(td) / "cameras.bin")
        ri = colmap_io.read_images_binary(Path(td) / "images.bin")
        rp = colmap_io.read_points3D_binary(Path(td) / "points3D.bin")
    assert rc[1]["model"] == "PINHOLE"
    assert rc[1]["width"] == 640 and rc[1]["height"] == 480
    assert np.allclose(rc[1]["params"], [500, 500, 320, 240])
    # recover rotation from stored quaternion
    R_back = C.quat_to_rotmat(ri[1]["qvec"])
    assert np.allclose(R_back, R, atol=1e-8)
    assert np.allclose(ri[1]["tvec"], t, atol=1e-9)
    assert ri[1]["name"] == "000001.png"
    assert np.allclose(rp[1]["xyz"], [1, 2, 3])
    assert tuple(int(x) for x in rp[1]["rgb"]) == (10, 20, 30)
    assert rp[1]["track"] == [(1, 0), (2, 3)]


def test_align_umeyama_recovers_similarity():
    from stages.stage3_metric.align import umeyama, apply_similarity_points
    rng = np.random.default_rng(3)
    src = rng.normal(size=(50, 3))
    ang = 0.8
    R = np.array([[np.cos(ang), 0, np.sin(ang)], [0, 1, 0], [-np.sin(ang), 0, np.cos(ang)]])
    s_true = 1.37
    t_true = np.array([0.5, -1.0, 2.0])
    dst = apply_similarity_points(src, s_true, R, t_true)
    s, Rr, tr = umeyama(src, dst, with_scale=True)
    assert abs(s - s_true) < 1e-6, f"scale {s} vs {s_true}"
    assert np.allclose(Rr, R, atol=1e-6)
    assert np.allclose(tr, t_true, atol=1e-6)


def test_robust_depth_fit_recovers_scale():
    from stages.stage3_metric.align import robust_depth_fit
    rng = np.random.default_rng(4)
    front = rng.uniform(0.3, 1.0, size=5000)
    s_true, b_true = 1.08, 0.0
    sensor = s_true * front + b_true
    # add 10% gross outliers
    idx = rng.choice(front.size, size=500, replace=False)
    sensor[idx] += rng.uniform(0.2, 0.5, size=idx.size)
    fit = robust_depth_fit(front, sensor, fit_offset=False, rng=rng)
    assert abs(fit["scale"] - s_true) < 0.01, f"scale {fit['scale']} vs {s_true}"
    assert fit["inlier_fraction"] > 0.85


ALL_TESTS = [
    test_pose_inverse_and_center,
    test_quat_roundtrip,
    test_arkit_conversion,
    test_ply_roundtrip,
    test_colmap_roundtrip,
    test_align_umeyama_recovers_similarity,
    test_robust_depth_fit_recovers_scale,
]
