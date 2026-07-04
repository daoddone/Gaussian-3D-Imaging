#!/usr/bin/env python3
"""Run the whole verifiable slice with no pytest dependency:

  1. common/ unit tests (conventions, PLY, COLMAP round-trips, align math)
  2. the mandatory orientation self-test
  3. generate a synthetic session with a KNOWN scale
  4. run Stage 3 end-to-end on it
  5. assert Stage 3 recovered the scale, passed, and wrote valid outputs
  6. run Experiment A and confirm a uniform single-scale residual

Usage:  python tests/run_all.py
Exit code 0 iff everything passes.
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np

PY = sys.executable
results = []


def record(name, ok, detail=""):
    results.append((name, ok, detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))


def run(cmd):
    p = subprocess.run([PY] + cmd, cwd=str(_ROOT), capture_output=True, text=True)
    return p.returncode, p.stdout, p.stderr


def section(title):
    print(f"\n===== {title} =====")


def main():
    # --- 1. unit tests ----------------------------------------------------
    section("common/ unit tests")
    from tests import test_common
    for fn in test_common.ALL_TESTS:
        try:
            fn()
            record(f"unit:{fn.__name__}", True)
        except Exception as e:
            record(f"unit:{fn.__name__}", False, f"{type(e).__name__}: {e}")

    # --- 2. orientation self-test ----------------------------------------
    section("orientation self-test")
    from common.orientation_selftest import run_selftest
    ok, rep = run_selftest(verbose=False)
    record("orientation_selftest", ok,
           detail="" if ok else json.dumps({k: v for k, v in rep["checks"].items() if not v["ok"]}))

    # --- 3/4. synthetic session + Stage 3 --------------------------------
    section("Stage 3 end-to-end on synthetic session")
    TRUE_SCALE = 1.05
    with tempfile.TemporaryDirectory() as td:
        sess = Path(td) / "synthetic_demo"
        rc, out, err = run(["tests/make_synthetic_session.py", "--out", str(sess),
                            "--scale", str(TRUE_SCALE), "--frames", "8"])
        record("generate_synthetic_session", rc == 0, detail=err.strip()[-300:] if rc else "")
        if rc != 0:
            return _finish()

        rc, out, err = run(["stages/stage3_metric/run.py", "--session", str(sess),
                            "--config", "config/pipeline.yaml"])
        # exit 0 = pass (synthetic anchors agree). 3 would mean flagged.
        record("stage3_run_exit0", rc == 0, detail=(err.strip()[-400:] if rc not in (0, 3) else f"exit={rc}"))

        report_path = sess / "metric" / "scale_report.json"
        if not report_path.exists():
            record("scale_report_exists", False, "no scale_report.json")
            return _finish()
        record("scale_report_exists", True)
        report = json.loads(report_path.read_text())

        applied = report["applied_scale"]
        record("recovered_scale", abs(applied - TRUE_SCALE) < 0.01,
               f"applied_scale={applied:.5f} vs TRUE_SCALE={TRUE_SCALE}")
        record("status_pass", report["status"] == "pass", f"status={report['status']} flags={report['flags']}")

        a = report["anchors"]
        record("depth_anchor_available", a["sensor_depth"]["available"],
               f"scale={a['sensor_depth']['scale_estimate']}")
        record("camera_anchor_available", a["camera_path"]["available"],
               f"scale={a['camera_path']['scale_estimate']}")
        if a["sensor_depth"]["available"]:
            record("depth_scale_correct", abs(a["sensor_depth"]["scale_estimate"] - TRUE_SCALE) < 0.02)
        if a["camera_path"]["available"]:
            record("camera_scale_correct", abs(a["camera_path"]["scale_estimate"] - TRUE_SCALE) < 0.01)

        fr = report.get("final_residual_meters")
        record("final_residual_small", fr is not None and fr < 0.005,
               f"final_residual={fr} m")

        # metric point cloud recovers the true metric geometry
        from common import plyio
        from tests.make_synthetic_session import sphere_points
        P_m, _ = sphere_points(np.array([0.0, 0.0, 0.0]), radius=0.09)
        metric = plyio.read_ply(sess / "metric" / "points_metric.ply")["points"]
        record("metric_points_count", metric.shape[0] == P_m.shape[0],
               f"{metric.shape[0]} vs {P_m.shape[0]}")
        if metric.shape[0] == P_m.shape[0]:
            med = float(np.median(np.linalg.norm(metric - P_m, axis=1)))
            record("metric_points_match_truth", med < 0.002, f"median dist {1000*med:.3f} mm")

        # COLMAP model round-trips and has baked points
        from common import colmap_io
        cdir = sess / "metric" / "colmap" / "sparse" / "0"
        imgs = colmap_io.read_images_binary(cdir / "images.bin")
        cams = colmap_io.read_cameras_binary(cdir / "cameras.bin")
        p3d = colmap_io.read_points3D_binary(cdir / "points3D.bin")
        record("colmap_images", len(imgs) == 8, f"{len(imgs)} images")
        record("colmap_cameras", len(cams) >= 1, f"{len(cams)} cameras")
        record("colmap_points_baked", len(p3d) > 0, f"{len(p3d)} points3D baked")

        # per-frame device K: the synthetic capture carries K_per_frame, so Stage 3 must give each
        # image its own camera (not one shared) and log the DA3-vs-device intrinsics comparison.
        colmap_out = report.get("outputs", {}).get("colmap", {})
        record("intrinsics_per_frame", colmap_out.get("intrinsics_source") == "capture_per_frame",
               f"source={colmap_out.get('intrinsics_source')}")
        record("colmap_per_frame_cameras", len(cams) == 8, f"{len(cams)} cameras (expect 8)")
        record("da3_vs_device_logged", colmap_out.get("da3_vs_device_K") is not None,
               detail="" if colmap_out.get("da3_vs_device_K") else "no comparison in report")

        # --- 6. Experiment A ---------------------------------------------
        section("Experiment A on synthetic session")
        outdir = Path(td) / "expA"
        rc, out, err = run(["experiments/experiment_a_single_scale.py", "--sessions", str(sess),
                            "--config", "config/pipeline.yaml", "--out", str(outdir)])
        record("experiment_a_runs", rc == 0, detail=err.strip()[-300:] if rc else "")
        res_json = outdir / "results.json"
        if res_json.exists():
            expa = json.loads(res_json.read_text())
            verdict = expa[0].get("drift", {}).get("verdict")
            record("experiment_a_uniform", verdict == "uniform", f"verdict={verdict}")

    return _finish()


def _finish():
    section("SUMMARY")
    n = len(results)
    passed = sum(1 for _, ok, _ in results if ok)
    for name, ok, detail in results:
        if not ok:
            print(f"  FAILED: {name} {('— ' + detail) if detail else ''}")
    print(f"\n{passed}/{n} checks passed.")
    return 0 if passed == n else 1


if __name__ == "__main__":
    sys.exit(main())
