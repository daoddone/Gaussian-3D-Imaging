#!/usr/bin/env python3
"""Experiment A — does a single global scale make the reconstruction metric?

Section 8 of the build specification, and the week-one headline result.

For each existing scan (a session folder that already has Stage 1 `capture/` and
Stage 2 `frontend/` outputs), fit ONE global scale factor (and offset) against
the sensor depth, apply it, and measure whether the residual error is UNIFORM
across frames or DRIFTS frame to frame. A uniform residual confirms the easy
single-scale case and validates the whole metric approach. The residual is
recorded per scan.

This drives Stage 3's depth anchor directly (that fit *is* Experiment A) and
adds a per-frame drift analysis on top.

Usage:
    python experiments/experiment_a_single_scale.py \
        --sessions sessions/scanA sessions/scanB ... \
        --config config/pipeline.yaml \
        --out experiments/results/experiment_a
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import yaml

from common.file_layout import SessionLayout
from stages.stage3_metric import anchors


def drift_analysis(per_frame_residual, cv_threshold=0.5, drift_threshold=0.5):
    """Classify a per-frame residual series as 'uniform' or 'drifts'.

    per_frame_residual : dict frame_id -> residual (meters).
    Returns metrics + verdict. Thresholds are heuristic; the numbers and the
    plot are what a human reads.
    """
    fids = sorted(per_frame_residual.keys())
    r = np.array([per_frame_residual[f] for f in fids], dtype=float)
    n = r.size
    if n < 2:
        return {"n_frames": int(n), "verdict": "insufficient_frames"}

    mean_r = float(r.mean())
    std_r = float(r.std())
    cv = float(std_r / mean_r) if mean_r > 0 else float("inf")

    # linear trend of residual vs normalized frame index (0..1)
    x = np.linspace(0.0, 1.0, n)
    A = np.stack([x, np.ones_like(x)], axis=1)
    slope, intercept = np.linalg.lstsq(A, r, rcond=None)[0]
    span = float(abs(slope))  # meters of drift across the whole scan
    normalized_drift = float(span / mean_r) if mean_r > 0 else float("inf")

    uniform = (cv <= cv_threshold) and (normalized_drift <= drift_threshold)
    return {
        "n_frames": int(n),
        "mean_residual_meters": mean_r,
        "std_residual_meters": std_r,
        "coefficient_of_variation": cv,
        "drift_slope_meters_per_scan": float(slope),
        "normalized_drift": normalized_drift,
        "verdict": "uniform" if uniform else "drifts",
        "frame_ids": fids,
        "per_frame_residual_meters": [float(v) for v in r],
    }


def run_one(session_dir, cfg):
    layout = SessionLayout(session_dir)
    depth = anchors.depth_anchor(layout, cfg)
    result = {
        "session": str(session_dir),
        "depth_anchor_available": bool(depth.get("available")),
    }
    if not depth.get("available"):
        result["note"] = depth.get("note", "depth anchor unavailable")
        return result
    result.update({
        "global_scale": depth["scale_estimate"],
        "global_offset_meters": depth.get("offset_meters", 0.0),
        "inlier_fraction": depth["inlier_fraction"],
        "points_used": depth["points_used"],
        "frames_used": depth["frames_used"],
        "global_residual_meters": depth["residual_meters"],
    })
    result["drift"] = drift_analysis(depth["per_frame_residual_meters"])
    return result


def _maybe_plot(results, out_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return False
    for res in results:
        d = res.get("drift")
        if not d or "per_frame_residual_meters" not in d:
            continue
        fig, ax = plt.subplots(figsize=(7, 3.5))
        y = np.array(d["per_frame_residual_meters"]) * 1000.0  # mm
        ax.plot(range(len(y)), y, marker="o", ms=3, lw=1)
        ax.axhline(np.mean(y), color="gray", ls="--", lw=1, label=f"mean {np.mean(y):.2f} mm")
        ax.set_xlabel("frame index")
        ax.set_ylabel("residual (mm)")
        ax.set_title(f"{Path(res['session']).name}: single-scale residual "
                     f"({d['verdict']}, scale={res.get('global_scale', float('nan')):.4f})")
        ax.legend()
        fig.tight_layout()
        fig.savefig(Path(out_dir) / f"residual_{Path(res['session']).name}.png", dpi=120)
        plt.close(fig)
    return True


def main():
    ap = argparse.ArgumentParser(description="Experiment A: single global scale")
    ap.add_argument("--sessions", nargs="+", required=True, help="session folders (existing scans)")
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", default="experiments/results/experiment_a")
    args = ap.parse_args()

    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = [run_one(s, cfg) for s in args.sessions]

    # summary table
    lines = ["session,depth_available,global_scale,global_residual_mm,verdict,cv,normalized_drift"]
    for r in results:
        if r.get("depth_anchor_available"):
            d = r["drift"]
            lines.append(
                f"{Path(r['session']).name},True,{r['global_scale']:.5f},"
                f"{1000.0 * r['global_residual_meters']:.3f},{d.get('verdict')},"
                f"{d.get('coefficient_of_variation', float('nan')):.3f},"
                f"{d.get('normalized_drift', float('nan')):.3f}")
        else:
            lines.append(f"{Path(r['session']).name},False,,,,,")

    (out_dir / "summary.csv").write_text("\n".join(lines) + "\n")
    (out_dir / "results.json").write_text(json.dumps(results, indent=2))
    plotted = _maybe_plot(results, out_dir)

    print("Experiment A — single global scale")
    print("\n".join(lines))
    print(f"\nWrote {out_dir/'summary.csv'}, {out_dir/'results.json'}"
          + ("" if plotted else " (matplotlib absent: no plots)"))
    n_uniform = sum(1 for r in results if r.get("drift", {}).get("verdict") == "uniform")
    n_ok = sum(1 for r in results if r.get("depth_anchor_available"))
    print(f"\nVerdict: {n_uniform}/{n_ok} scans show a UNIFORM single-scale residual "
          f"(uniform => the single-scale metric assumption holds).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
