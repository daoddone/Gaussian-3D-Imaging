#!/usr/bin/env python3
"""Pose-BA step 2 (env gs-ba): triangulate real tracks at the FIXED ARKit poses,
then joint bundle adjustment with per-frame extrinsics FREE (removes the drift).

Intrinsics are kept FIXED: on a ~25 cm short-baseline face orbit the focal length
is nearly degenerate with depth/global scale, so a free focal would let BA bury
residual pose error in a bogus focal — the exact soft-splat/doubled-mesh signature
we are removing. The single shared sensor is gauge-redundant -> refine_sensor_from_rig=False.
"""
import os
from pathlib import Path
from hloc import triangulation
import pycolmap

REPO = Path("/home/paperspace/Documents/VS Code Projects/3D-Gaussian")
SESS = REPO / os.environ.get("PBA_SESS", "sessions/session_20260703_145121")
IMAGES = SESS / "capture/rgb"
REF = SESS / "metric/colmap/sparse/0"
WORK = SESS / "pose_ba"
pairs = WORK / "pairs.txt"; feats = WORK / "feats.h5"; matches = WORK / "matches.h5"


def reproj(rec):
    for m in ("compute_mean_reprojection_error", "compute_mean_reproj_error"):
        if hasattr(rec, m):
            try:
                return getattr(rec, m)()
            except Exception:
                pass
    return float("nan")


# --- triangulate at fixed ARKit poses (real SuperPoint tracks; poses not moved) ---
triangulation.main(sfm_dir=WORK / "triangulated", reference_model=REF, image_dir=IMAGES,
                   pairs=pairs, features=feats, matches=matches,
                   skip_geometric_verification=False, estimate_two_view_geometries=False,
                   verbose=False)
rec = pycolmap.Reconstruction(str(WORK / "triangulated"))
print(f"[02] triangulated: {rec.num_points3D()} pts, {len(rec.reg_image_ids())} reg images, "
      f"reproj {reproj(rec):.3f} px")

# --- joint BA: free per-frame extrinsics, fixed intrinsics ---
o = pycolmap.BundleAdjustmentOptions()
o.refine_focal_length = False
o.refine_principal_point = False
o.refine_extra_params = False
o.refine_rig_from_world = True       # per-frame extrinsics -> removes ARKit drift
o.refine_sensor_from_rig = False     # single shared sensor is gauge-redundant
o.refine_points3D = True
o.print_summary = True
try:
    pycolmap.bundle_adjustment(rec, o)
except Exception as e:
    print("[02] pycolmap.bundle_adjustment(rec, o) failed:", type(e).__name__, e)
    print("[02] fallback: BundleAdjustmentConfig + default adjuster")
    cfg = pycolmap.BundleAdjustmentConfig()
    for iid in rec.reg_image_ids():
        cfg.add_image(iid)
    for cid in rec.cameras:
        cfg.set_constant_cam_intrinsics(cid)
    try:
        cfg.fix_gauge(pycolmap.BundleAdjustmentGauge.TWO_CAMS_FROM_WORLD)
    except Exception:
        pass
    pycolmap.create_default_bundle_adjuster(o, cfg, rec).solve()

(WORK / "refined").mkdir(parents=True, exist_ok=True)
rec.write(str(WORK / "refined"))
print(f"[02] AFTER BA: {rec.num_points3D()} pts, {len(rec.reg_image_ids())} reg images, "
      f"reproj {reproj(rec):.3f} px -> wrote {WORK / 'refined'}")
