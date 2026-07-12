#!/usr/bin/env python3
"""T5 — capture-mode benchmark harness (roadmap task T5).

Takes N sessions of the SAME subject captured in different modes (arkit1080 / arkit4K / hqStills,
LiDAR on/off) and produces the standardized comparison: per-session SfM -> metric anchor ->
capture-quality score -> stage-5 reconstruction (recommended auto config) -> cross-mode eval
(equal-footing stats + FRONTAL renders, per owner comparison methodology) + a metric-scale table
from the sidecars. Idempotent: each stage is skipped when its output already exists, so the
harness can be re-run as sessions arrive.

Usage:
  t5_mode_benchmark.py --session sessions/<S1> --session sessions/<S2> [...]
                       [--labels 1080p 4K ...] [--no-recon] [--outdir sessions/_sweep_eval/t5_modes]

Capture protocol for the dataset (owner): same subject, same lighting, same slow fill-frame orbit,
one session per mode; ArUco sheet in frame for the scale row.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO = Path("/home/paperspace/Documents/VS Code Projects/3D-Gaussian")
GS_BA = Path.home() / "miniforge3/envs/gs-ba/bin/python"
FRONTEND = Path.home() / "miniforge3/envs/pipeline_stage2_frontend/bin/python"


def run(cmd, **kw):
    print(f"[t5] $ {' '.join(str(c) for c in cmd)}")
    return subprocess.run([str(c) for c in cmd], cwd=str(REPO), **kw).returncode


def ensure_chain(sess: Path, do_recon: bool) -> dict:
    """SfM -> anchor -> quality -> stage5 (each idempotent). Returns collected facts."""
    facts = {"session": sess.name}
    meta = json.loads((sess / "capture" / "metadata.json").read_text()) if (sess / "capture" / "metadata.json").exists() else {}
    facts["capture_mode"] = meta.get("capture_mode", "?")
    facts["lidar"] = meta.get("lidar_enabled", "?")

    if not (sess / "pose_ba" / "sfm_noseed" / "images.bin").exists():
        if run([GS_BA, "scripts/session_sfm.py", "--session", sess]) != 0:
            facts["error"] = "sfm failed"
            return facts
    if not (sess / "metric_sfm" / "scale_sidecar.json").exists():
        run([GS_BA, "scripts/pose_ba/04_metric_anchor.py", "--session", sess])
    qj = sess / "capture_quality.json"
    if not qj.exists():
        run([GS_BA, "scripts/capture_quality.py", "--session", sess, "--out", qj])

    sidecar = sess / "metric_sfm" / "scale_sidecar.json"
    if sidecar.exists():
        sc = json.loads(sidecar.read_text())
        facts["scale"] = sc.get("scale")
        facts["primary"] = sc.get("primary_anchor")
        facts["agreement_pct"] = sc.get("anchor_agreement_pct")
        if facts["agreement_pct"] is None:
            facts["agreement_pct"] = sc.get("marker_agreement_pct")
        facts["confidence"] = sc.get("confidence")
        rv = sc.get("reference_validation")
        if rv:
            facts["marker_scale_err_pct"] = rv.get("scale_error_pct")
    if qj.exists():
        q = json.loads(qj.read_text())
        facts["quality"] = {k: q.get(k) for k in ("verdict", "n_frames", "sharpness_median", "coverage_deg") if k in q}

    out = sess / "output_t5"
    if do_recon and not (out / "point_cloud.ply").exists():
        if run(["python3", "stages/stage5_reconstruction/run.py", "--session", sess,
                "--config", "config/pipeline_recommended.yaml"]) == 0 and (sess / "output").exists():
            (sess / "output").rename(out)
    facts["reconstructed"] = (out / "point_cloud.ply").exists()
    return facts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", action="append", required=True)
    ap.add_argument("--labels", nargs="*")
    ap.add_argument("--no-recon", action="store_true")
    ap.add_argument("--outdir", default="sessions/_sweep_eval/t5_modes")
    args = ap.parse_args()

    sessions = [(REPO / s) if not Path(s).is_absolute() else Path(s) for s in args.session]
    labels = args.labels or [s.name[-8:] for s in sessions]
    outdir = REPO / args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    rows = []
    for s in sessions:
        rows.append(ensure_chain(s, do_recon=not args.no_recon))

    recon = [(l, s / "output_t5") for l, s in zip(labels, sessions) if (s / "output_t5" / "point_cloud.ply").exists()]
    if len(recon) >= 2:
        run([FRONTEND, "scripts/eval_recon.py", *[d for _, d in recon],
             "--labels", *[l for l, _ in recon], "--outdir", outdir])
        first = sessions[0]
        arm_args = []
        for l, d in recon:
            mesh = d / "mesh_textured.ply"
            if not mesh.exists():
                mesh = d / "mesh.ply"
            arm_args += ["--arm", f"{l}={mesh}"]
        run([FRONTEND, "scripts/frontal_compare.py",
             "--sparse", first / "reconstruction_input" / "sparse" / "0",
             "--images", first / "reconstruction_input" / "images",
             *arm_args, "--out", outdir / "t5_frontal_compare.jpg"])

    lines = ["# T5 capture-mode benchmark", "",
             "| session | mode | lidar | frames | sharp med | scale primary | VIO/LiDAR agree % | marker err % | confidence | reconstructed |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        q = r.get("quality", {})
        lines.append(f"| {r['session'][-20:]} | {r.get('capture_mode')} | {r.get('lidar')} | "
                     f"{q.get('n_frames', '?')} | {q.get('sharpness_median', '?')} | {r.get('primary', '?')} | "
                     f"{r.get('agreement_pct', '?')} | {r.get('marker_scale_err_pct', '—')} | "
                     f"{r.get('confidence', '?')} | {r.get('reconstructed')} |")
    lines += ["", "Reconstruction stats: see `stats.json` + `comparison.png`; anatomy-frontal views: `t5_frontal_compare.jpg`.",
              "Judge modes on: scale confidence, frontal surface quality (equal-footing), still-success rate, capture ergonomics."]
    (outdir / "T5_REPORT.md").write_text("\n".join(lines))
    print(f"[t5] report -> {outdir}/T5_REPORT.md")
    print(json.dumps(rows, indent=2, default=str))


if __name__ == "__main__":
    main()
