#!/usr/bin/env python3
"""Top-level coordinator (Section 6 of the build specification).

Runs each enabled stage as a SUBPROCESS inside that stage's own conda
environment, passing data only by file path. It never imports any stage's code,
which is what keeps the conflicting environments isolated.

    python orchestrate.py --session sessions/<id> --config config/pipeline.yaml

Behavior:
  * runs the enabled stages in order (2 -> 3 -> 4 -> 5);
  * HALTS and reports if Stage 3 raises a scale-disagreement flag (exit code 3)
    and stage3.flag_halts_pipeline is true;
  * treats Stage 5 exit code 4 (host prepared but not yet built/ported — tasks
    H1/H2) as a clean, expected stop, not a crash;
  * records component versions / model identifiers into output/provenance.json.

Flags:
  --only <stage_key>   run just one stage (e.g. stage3_metric)
  --no-conda           run `python <entry> ...` directly (for local testing on a
                       stage whose deps are already importable, e.g. Stage 3)
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import yaml

# canonical run order; Stage 1 (capture app) is not orchestrated here
STAGE_ORDER = ["stage2_frontend", "stage3_metric", "stage4_normals", "stage5_reconstruction"]

# meaningful non-zero exit codes a stage may return
EXIT_STAGE3_FLAG = 3      # scale disagreement
EXIT_STAGE5_NOT_READY = 4  # dataset prepared, host not built/ported (H1/H2)


def _build_cmd(stage_cfg, session, config, no_conda):
    entry = stage_cfg["entry"]
    if no_conda:
        return [sys.executable, entry, "--session", str(session), "--config", str(config)]
    env = stage_cfg["env"]
    return ["conda", "run", "-n", env, "python", entry,
            "--session", str(session), "--config", str(config)]


def write_provenance(session, cfg):
    """Assemble output/provenance.json (reconstruction_output.md schema)."""
    session = Path(session)
    out_dir = session / "output"
    out_dir.mkdir(parents=True, exist_ok=True)

    scale_report = {}
    sr_path = session / "metric" / "scale_report.json"
    if sr_path.exists():
        full = json.loads(sr_path.read_text())
        scale_report = {"final_residual_meters": full.get("final_residual_meters"),
                        "status": full.get("status"), "applied_scale": full.get("applied_scale")}

    n_rgb = len(list((session / "capture" / "rgb").glob("*.png"))) if (session / "capture" / "rgb").is_dir() else 0
    stage4_on = bool(cfg.get("stages", {}).get("stage4_normals", {}).get("enabled", False))

    prov = {
        "session_id": session.name,
        "generated": datetime.now().strftime("%m-%d-%Y %H:%M") + " local",
        "components": {
            "front_end": {"name": cfg.get("stage2", {}).get("model", "DA3NESTED-GIANT-LARGE"), "version": "..."},
            "normal_prior": {"name": cfg.get("stage4", {}).get("model", "StableNormal"),
                             "version": "...", "enabled": stage4_on},
            "reconstruction_host": {"name": cfg.get("stage5", {}).get("host", "MILo"),
                                    "version": "...",
                                    "densification": cfg.get("stage5", {}).get("densification", "mini-splatting2")},
        },
        "scale_report": scale_report,
        "capture": {"frame_count": n_rgb,
                    "video_seconds": cfg.get("capture", {}).get("video_seconds", 20),
                    "device": "..."},
    }
    (out_dir / "provenance.json").write_text(json.dumps(prov, indent=2))
    return out_dir / "provenance.json"


def run(session, config_path, only=None, no_conda=False):
    cfg = yaml.safe_load(open(config_path))
    stages_cfg = cfg.get("stages", {})
    s3_cfg = cfg.get("stage3", {})

    to_run = [only] if only else STAGE_ORDER
    for key in to_run:
        sc = stages_cfg.get(key)
        if sc is None:
            print(f"[orchestrate] no config for {key}; skipping")
            continue
        if not sc.get("enabled", False) and not only:
            print(f"[orchestrate] {key} disabled; skipping")
            continue

        cmd = _build_cmd(sc, session, config_path, no_conda)
        print(f"\n[orchestrate] === {key} ===\n[orchestrate] $ {' '.join(cmd)}")
        proc = subprocess.run(cmd)
        rc = proc.returncode

        if key == "stage3_metric" and rc == EXIT_STAGE3_FLAG:
            if bool(s3_cfg.get("flag_halts_pipeline", True)):
                print("\n[orchestrate] HALT: Stage 3 raised a scale-disagreement flag "
                      "(stage3.flag_halts_pipeline=true). Inspect metric/scale_report.json.")
                write_provenance(session, cfg)
                return EXIT_STAGE3_FLAG
            print("[orchestrate] Stage 3 flagged but flag_halts_pipeline=false; continuing.")
            continue

        if key == "stage5_reconstruction" and rc == EXIT_STAGE5_NOT_READY:
            print("[orchestrate] Stage 5 prepared the dataset but the host is not built/ported "
                  "yet (tasks H1/H2). Stopping cleanly.")
            write_provenance(session, cfg)
            return EXIT_STAGE5_NOT_READY

        if rc != 0:
            print(f"[orchestrate] {key} failed with exit code {rc}; halting.")
            return rc

    prov = write_provenance(session, cfg)
    print(f"\n[orchestrate] done. Wrote {prov}")
    return 0


def main():
    ap = argparse.ArgumentParser(description="Pipeline coordinator")
    ap.add_argument("--session", required=True)
    ap.add_argument("--config", default="config/pipeline.yaml")
    ap.add_argument("--only", choices=STAGE_ORDER, help="run just one stage")
    ap.add_argument("--no-conda", action="store_true",
                    help="run entry points directly (local testing; no conda run)")
    args = ap.parse_args()
    return run(args.session, args.config, only=args.only, no_conda=args.no_conda)


if __name__ == "__main__":
    sys.exit(main())
