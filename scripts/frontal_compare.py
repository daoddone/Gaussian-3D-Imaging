#!/usr/bin/env python3
"""Frontal-view mesh comparison (owner-directed methodology): render every arm from the SAME
real capture camera, chosen as the most FACE-FRONTAL source frame — so arms are compared looking
at the anatomy head-on, and floaters behind the subject can't masquerade as reliable surface.

Frame choice: haar frontal-face detection over the registered source frames (tried at 4 rotations
— app captures are often sensor-rotated); score = face area, most-centered wins ties. Fallback
(no faces, e.g. feet): the middle registered frame. The winning rotation is applied to the OUTPUT
images only (display upright); the render itself uses the exact COLMAP pose + intrinsics.

Usage:
  frontal_compare.py --sparse <sparse/0 in OUTPUT space> --images <source frames dir>
                     --arm "LABEL=/path/to/mesh.ply" [--arm ...] --out cmp.jpg [--max-side 1100]
"""
import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path("/home/paperspace/Documents/VS Code Projects/3D-Gaussian")
sys.path.insert(0, str(REPO))
from common import colmap_io  # noqa: E402


def quat_to_rotmat(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]], float)


def pick_frontal_frame(imgs, images_dir):
    import cv2
    casc = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    best = None  # (score, name, rot_k)
    for im in imgs.values():
        p = images_dir / im["name"]
        if not p.exists():
            continue
        g = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if g is None:
            continue
        scale = 640.0 / max(g.shape)
        g = cv2.resize(g, (int(g.shape[1] * scale), int(g.shape[0] * scale)))
        for k in range(4):
            gk = np.rot90(g, k)
            faces = casc.detectMultiScale(np.ascontiguousarray(gk), 1.2, 5, minSize=(60, 60))
            for (x, y, w, h) in faces:
                cx, cy = x + w / 2, y + h / 2
                H, W = gk.shape
                centered = 1.0 - (abs(cx - W / 2) / W + abs(cy - H / 2) / H)
                score = w * h * (0.5 + centered)
                if best is None or score > best[0]:
                    best = (score, im["name"], k)
    if best:
        print(f"[frontal] most-frontal frame: {best[1]} (rot90 x{best[2]}, score {best[0]:.0f})")
        return best[1], best[2]
    mid = sorted(i["name"] for i in imgs.values())[len(imgs) // 2]
    print(f"[frontal] no face detected anywhere - fallback to middle frame {mid}")
    return mid, 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sparse", required=True, help="COLMAP sparse/0 in the OUTPUT (metric) space")
    ap.add_argument("--images", required=True, help="source frames dir (for the frontal pick)")
    ap.add_argument("--arm", action="append", required=True, help="LABEL=/path/to/mesh.ply")
    ap.add_argument("--out", required=True)
    ap.add_argument("--frame", help="override: use this exact frame name")
    ap.add_argument("--max-side", type=int, default=1100, help="render resolution cap")
    args = ap.parse_args()

    import open3d as o3d
    from PIL import Image, ImageDraw

    sparse = Path(args.sparse)
    imgs = colmap_io.read_images_binary(sparse / "images.bin")
    cams = colmap_io.read_cameras_binary(sparse / "cameras.bin")

    if args.frame:
        name, rot_k = args.frame, 0
    else:
        name, rot_k = pick_frontal_frame(imgs, Path(args.images))
    im = next(i for i in imgs.values() if i["name"] == name)
    cam = cams[im["camera_id"]]
    fx, fy, cx, cy = [float(v) for v in cam["params"][:4]]
    W, H = int(cam["width"]), int(cam["height"])
    s = min(1.0, args.max_side / max(W, H))
    W, H, fx, fy, cx, cy = int(W * s), int(H * s), fx * s, fy * s, cx * s, cy * s
    ext = np.eye(4)
    ext[:3, :3] = quat_to_rotmat(np.asarray(im["qvec"], float))
    ext[:3, 3] = np.asarray(im["tvec"], float)
    intr = o3d.camera.PinholeCameraIntrinsic(W, H, fx, fy, cx, cy)

    panels = []
    for spec in args.arm:
        label, mesh_path = spec.split("=", 1)
        m = o3d.io.read_triangle_mesh(mesh_path)
        m.compute_vertex_normals()
        row = []
        for shader, paint in [("defaultUnlit", False), ("defaultLit", True)]:
            r = o3d.visualization.rendering.OffscreenRenderer(W, H)
            r.scene.set_background([1.0, 1.0, 1.0, 1.0])
            mat = o3d.visualization.rendering.MaterialRecord()
            mat.shader = shader
            mm = o3d.geometry.TriangleMesh(m)
            if paint:
                mm.paint_uniform_color([0.76, 0.76, 0.79])
            r.scene.add_geometry("m", mm, mat)
            r.setup_camera(intr, ext)
            img = np.asarray(r.render_to_image())
            row.append(np.rot90(img, rot_k))
            del r
        panels.append((label, np.hstack(row)))

    ims = [(t, Image.fromarray(p)) for t, p in panels]
    PW = max(i.width for _, i in ims)
    LBL = 46
    canvas = Image.new("RGB", (PW, sum(i.height + LBL for _, i in ims)), (18, 18, 18))
    d = ImageDraw.Draw(canvas)
    y = 0
    for t, i in ims:
        d.text((14, y + 14), f"{t}   |   frontal capture view: {name}   |   left=baked colors, right=shaded geometry",
               fill=(255, 220, 120))
        y += LBL
        canvas.paste(i, (0, y))
        y += i.height
    canvas.save(args.out, quality=90)
    print(f"[frontal] wrote {args.out}")


if __name__ == "__main__":
    main()
