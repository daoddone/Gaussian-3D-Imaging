"""Orientation self-test (mandatory, Section 5 of the build specification).

Renders a known synthetic object (a sphere) under a known camera pose in the
OpenCV convention and confirms the two invariants that a coordinate-convention
mistake would break:

  1. **Depth increases away from the camera.** The rendered depth of the sphere
     is smallest at the point nearest the camera (the near pole) and grows
     toward the silhouette.
  2. **Surface normals point outward.** On the visible surface the outward
     normal points back toward the camera; expressed in the camera frame its
     component along the viewing ray is negative (the normal opposes the ray).

It also round-trips a non-identity pose through the ``conventions`` helpers so a
regression in pose handling is caught. Run this whenever a convention crosses a
stage boundary (the pipeline's classic silent failure).

numpy-only. Usage:  python -m common.orientation_selftest
"""
from __future__ import annotations

import sys

import numpy as np

from common import conventions as C


# --------------------------------------------------------------------------- #
# synthetic geometry
# --------------------------------------------------------------------------- #
def make_sphere(center, radius, n_theta=60, n_phi=120):
    """Return (points_world Nx3, normals_world Nx3) on a sphere, normals outward."""
    center = np.asarray(center, dtype=float)
    thetas = np.linspace(0.05, np.pi - 0.05, n_theta)  # polar
    phis = np.linspace(0.0, 2.0 * np.pi, n_phi, endpoint=False)
    th, ph = np.meshgrid(thetas, phis, indexing="ij")
    th = th.ravel()
    ph = ph.ravel()
    dirs = np.stack(
        [np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)], axis=1
    )
    pts = center[None, :] + radius * dirs
    normals = dirs  # unit, outward (radial)
    return pts, normals


def pinhole_K(fx=600.0, fy=600.0, cx=320.0, cy=240.0):
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])


def project(points_world, R_w2c, t_w2c, K):
    """Project world points into a camera. Returns (uv Nx2, depth N, pts_cam Nx3)."""
    pts_cam = (R_w2c @ points_world.T).T + t_w2c[None, :]
    depth = pts_cam[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        x = pts_cam[:, 0] / pts_cam[:, 2]
        y = pts_cam[:, 1] / pts_cam[:, 2]
    u = K[0, 0] * x + K[0, 2]
    v = K[1, 1] * y + K[1, 2]
    return np.stack([u, v], axis=1), depth, pts_cam


def render_depth_normal(points_world, normals_world, R_w2c, t_w2c, K, width, height):
    """Nearest-point z-buffer render. Returns (depth_img, normal_img, mask)."""
    uv, depth, pts_cam = project(points_world, R_w2c, t_w2c, K)
    normals_cam = (R_w2c @ normals_world.T).T

    # keep points in front of camera and facing it (normal opposes view ray)
    ray = pts_cam / np.linalg.norm(pts_cam, axis=1, keepdims=True)
    facing = np.einsum("ij,ij->i", normals_cam, ray) < 0.0
    inb = (
        (depth > 0)
        & facing
        & (uv[:, 0] >= 0)
        & (uv[:, 0] < width)
        & (uv[:, 1] >= 0)
        & (uv[:, 1] < height)
    )

    depth_img = np.full((height, width), np.nan, dtype=np.float32)
    normal_img = np.zeros((height, width, 3), dtype=np.float32)
    mask = np.zeros((height, width), dtype=bool)

    px = np.round(uv[inb, 0]).astype(int)
    py = np.round(uv[inb, 1]).astype(int)
    dz = depth[inb]
    nn = normals_cam[inb]
    order = np.argsort(-dz)  # write far first so nearest wins
    for i in order:
        xx, yy = px[i], py[i]
        if 0 <= xx < width and 0 <= yy < height:
            if not mask[yy, xx] or dz[i] < depth_img[yy, xx]:
                depth_img[yy, xx] = dz[i]
                normal_img[yy, xx] = nn[i]
                mask[yy, xx] = True
    return depth_img, normal_img, mask


# --------------------------------------------------------------------------- #
# the test
# --------------------------------------------------------------------------- #
def run_selftest(verbose=False):
    report = {"checks": {}, "ok": True}

    def check(name, ok, detail=""):
        report["checks"][name] = {"ok": bool(ok), "detail": detail}
        report["ok"] = report["ok"] and bool(ok)
        if verbose:
            print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")

    width, height = 640, 480
    K = pinhole_K(600, 600, width / 2, height / 2)

    # Camera at world origin, looking down +z (identity world_to_camera).
    R = np.eye(3)
    t = np.zeros(3)

    D = 1.0  # sphere center 1 m in front
    r = 0.25
    pts, nrm = make_sphere(center=[0, 0, D], radius=r)

    # --- outward-normal construction sanity (world frame) ---
    radial = pts - np.array([0, 0, D])[None, :]
    outward = np.einsum("ij,ij->i", nrm, radial) > 0
    check("normals_outward_world", np.all(outward),
          f"{int(outward.sum())}/{len(pts)} normals point radially outward")

    depth_img, normal_img, mask = render_depth_normal(pts, nrm, R, t, K, width, height)
    nvalid = int(mask.sum())
    check("render_nonempty", nvalid > 100, f"{nvalid} valid pixels")

    # --- invariant 1: depth increases away from camera ---
    valid_depths = depth_img[mask]
    near = float(np.nanmin(valid_depths))
    far = float(np.nanmax(valid_depths))
    # near pole geometric depth = D - r; limb depth ~ D
    check("depth_positive", near > 0, f"min depth {near:.4f} m")
    check("depth_near_pole", abs(near - (D - r)) < 0.03,
          f"near depth {near:.4f} m vs expected {D - r:.4f} m")
    check("depth_range_forward", far <= D + 1e-3 and far > near,
          f"depth spans [{near:.4f}, {far:.4f}] m")

    # depth increases from the image center outward along the central scanline
    cy = height // 2
    row = depth_img[cy]
    cols = np.where(np.isfinite(row))[0]
    if cols.size > 5:
        cx = width / 2
        rad = np.abs(cols - cx)
        dvals = row[cols]
        # positive correlation between distance-from-center and depth
        corr = float(np.corrcoef(rad, dvals)[0, 1])
        check("depth_increases_outward", corr > 0.5,
              f"corr(radius, depth) = {corr:.3f} along central scanline")
    else:
        check("depth_increases_outward", False, "too few pixels on scanline")

    # --- invariant 2: visible normals point back toward camera (n_z < 0) ---
    vis_normals = normal_img[mask]
    nz = vis_normals[:, 2]
    frac_toward = float(np.mean(nz < 0))
    check("normals_face_camera", frac_toward > 0.98,
          f"{100 * frac_toward:.1f}% of visible normals have n_z < 0 (toward camera)")

    # --- convention round-trip through a non-identity pose ---
    # camera placed at world (0.3, -0.2, -0.5), some rotation
    ang = 0.4
    Rc = np.array([[np.cos(ang), 0, np.sin(ang)],
                   [0, 1, 0],
                   [-np.sin(ang), 0, np.cos(ang)]])
    center_world = np.array([0.3, -0.2, -0.5])
    # build camera_to_world (Rc, center_world); derive world_to_camera
    R_w2c, t_w2c = C.invert_pose(Rc, center_world)
    recovered_center = C.camera_center(R_w2c, t_w2c, C.WORLD_TO_CAMERA)
    check("camera_center_roundtrip",
          np.allclose(recovered_center, center_world, atol=1e-9),
          f"recovered center {recovered_center.round(4).tolist()}")

    # camera_to_world <-> world_to_camera inverse consistency
    R_c2w, t_c2w = C.to_camera_to_world(R_w2c, t_w2c, C.WORLD_TO_CAMERA)
    check("pose_inverse_consistency",
          np.allclose(R_c2w, Rc, atol=1e-9) and np.allclose(t_c2w, center_world, atol=1e-9),
          "world_to_camera inverted back to camera_to_world")

    # quaternion round-trip
    q = C.rotmat_to_quat(Rc)
    Rq = C.quat_to_rotmat(q)
    check("quat_roundtrip", np.allclose(Rq, Rc, atol=1e-8),
          f"||R - R(quat(R))|| = {np.linalg.norm(Rq - Rc):.2e}")

    # ARKit(OpenGL) -> OpenCV conversion flips y,z of camera axes
    T = np.eye(4)
    T[:3, :3] = Rc
    T[:3, 3] = center_world
    R_cv, t_cv = C.opencv_c2w_from_arkit(T)
    expected = Rc @ np.diag([1.0, -1.0, -1.0])
    check("arkit_to_opencv", np.allclose(R_cv, expected, atol=1e-9) and np.allclose(t_cv, center_world),
          "ARKit camera_to_world converted to OpenCV")

    return report["ok"], report


def main():
    ok, report = run_selftest(verbose=True)
    print()
    print("ORIENTATION SELF-TEST:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
