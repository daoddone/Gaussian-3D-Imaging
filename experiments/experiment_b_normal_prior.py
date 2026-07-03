#!/usr/bin/env python3
"""Experiment B — does the learned normal prior actually help, and does it hurt?

Section 8 of the build specification. On the same existing scans, reconstruct
under three conditions and measure surface accuracy against the reference:

    (1) no normal prior at all,
    (2) normals computed directly from the sensor depth (no learned model),
    (3) StableNormal at a fixed modest weight.

Read-off rule (from the spec):
  - condition 3 clearly improves smooth regions without distorting wounds -> keep
    Stage 4 with no gating;
  - condition 3 distorts deep wounds -> add the single confidence-tied weight;
  - no clear help anywhere -> remove Stage 4 entirely.

IMPORTANT — SCOPE / DEPENDENCY (flagged to the team):
  The DEFINITIVE verdict requires a *reconstruction* pass per condition plus a
  surface reference, i.e. Stage 5 (or the DN-Splatter fallback host) and a
  Canfield Vectra / ground-truth mesh. Those are NOT available in week one. This
  harness therefore does two things:

    A) Prepares the three normal sources per frame (condition 2 is computed here;
       condition 3 calls Stage 4 / StableNormal when its environment exists;
       condition 1 is the empty set), so the reconstruction step is ready to run
       the moment Stage 5 is compiled.
    B) Runs the achievable *proxy* now: it compares each condition's normals
       against a geometry reference (normals derived from the metric sensor
       depth) via mean angular error, per region, so we get an early read on
       where a learned prior diverges from the sensor — especially in
       low-confidence (wound-like) regions where a wrong flat prior is dangerous.

  The proxy is NOT the surface-accuracy number the spec asks for; it is an early
  indicator. The reconstruction-and-measure step is marked PENDING until Stage 5.

Usage:
    python experiments/experiment_b_normal_prior.py \
        --sessions sessions/scanA ... --config config/pipeline.yaml \
        --out experiments/results/experiment_b
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
from stages.stage4_normals.normals_from_depth import normals_from_depth, angular_error_degrees

CONDITIONS = ("none", "from_depth", "stablenormal")


def _reference_normals(depth, K, valid):
    """Geometry reference: normals from the (metric) sensor depth."""
    return normals_from_depth(depth, K, valid=valid)


def proxy_evaluate(session_dir, cfg):
    """Compare each condition's normals to the sensor-depth reference, split by
    high- vs low-confidence regions (low confidence ~ wound-like)."""
    layout = SessionLayout(session_dir)
    if not layout.capture_intrinsics.exists():
        return {"session": str(session_dir), "note": "no capture intrinsics"}
    cap = anchors.load_capture_intrinsics(layout.capture_intrinsics)
    K = cap["K_sensor"]

    frames = SessionLayout.list_frames(layout.capture_depth, ".npy")
    stable_dir = layout.normals  # condition 3 outputs (if Stage 4 was run)

    per_cond = {c: {"high_conf_deg": [], "low_conf_deg": []} for c in ("from_depth", "stablenormal")}
    frames_used = 0
    for fid in frames:
        depth = np.load(layout.capture_depth / f"{fid}.npy").astype(float)
        valid = np.isfinite(depth) & (depth > 0)
        conf_path = layout.capture_confidence / f"{fid}.png"
        high = valid.copy()
        low = ~valid
        if conf_path.exists():
            hv = anchors._load_confidence(conf_path)
            high = valid & hv
            low = valid & ~hv
        ref = _reference_normals(depth, K, valid)

        # condition 2: normals from depth (compared to the same reference -> ~0,
        # a sanity floor; the interesting comparison is condition 3)
        nd = normals_from_depth(depth, K, valid=valid)
        e_hi, _ = angular_error_degrees(nd, ref, valid=high)
        per_cond["from_depth"]["high_conf_deg"].append(e_hi)

        # condition 3: StableNormal output, if present
        sp = stable_dir / f"{fid}.npy"
        if sp.exists():
            ns = np.load(sp).astype(float)
            if ns.shape[:2] == ref.shape[:2]:
                e_hi3, _ = angular_error_degrees(ns, ref, valid=high)
                e_lo3, nlo = angular_error_degrees(ns, ref, valid=low)
                per_cond["stablenormal"]["high_conf_deg"].append(e_hi3)
                if nlo > 0:
                    per_cond["stablenormal"]["low_conf_deg"].append(e_lo3)
        frames_used += 1

    def _mean(xs):
        xs = [v for v in xs if v == v]  # drop nan
        return float(np.mean(xs)) if xs else None

    summary = {"session": str(session_dir), "frames_used": frames_used, "conditions": {}}
    for c, blk in per_cond.items():
        summary["conditions"][c] = {
            "mean_angular_error_high_conf_deg": _mean(blk["high_conf_deg"]),
            "mean_angular_error_low_conf_deg": _mean(blk["low_conf_deg"]),
        }
    summary["conditions"]["none"] = {"note": "no normal prior; nothing to compare"}
    summary["stablenormal_present"] = any(
        (stable_dir / f"{fid}.npy").exists() for fid in frames)
    return summary


def main():
    ap = argparse.ArgumentParser(description="Experiment B: normal-prior study (proxy + prep)")
    ap.add_argument("--sessions", nargs="+", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", default="experiments/results/experiment_b")
    args = ap.parse_args()

    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = [proxy_evaluate(s, cfg) for s in args.sessions]
    (out_dir / "proxy_results.json").write_text(json.dumps(results, indent=2))

    print("Experiment B — normal-prior proxy (angular error vs sensor-depth reference)")
    for r in results:
        cond = r.get("conditions", {})
        sn = cond.get("stablenormal", {})
        print(f"  {Path(r['session']).name}: "
              f"stablenormal_hi={sn.get('mean_angular_error_high_conf_deg')} deg, "
              f"stablenormal_lo={sn.get('mean_angular_error_low_conf_deg')} deg, "
              f"stablenormal_present={r.get('stablenormal_present')}")
    print(f"\nWrote {out_dir/'proxy_results.json'}")
    print("\n*** RECONSTRUCTION-AND-MEASURE STEP: PENDING ***")
    print("The definitive Experiment B verdict (keep / gate / remove Stage 4) needs a")
    print("reconstruction per condition (Stage 5 or DN-Splatter) + a surface reference.")
    print("The proxy above is an early indicator only, not the surface-accuracy number.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
