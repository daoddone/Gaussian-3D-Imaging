#!/bin/bash
# Follow-on arm: subject isolation A/B (fast + λ0 + isolation vs the proven-best fast + λ0).
# Waits for the overnight matrix runner to finish, then runs one arm + eval. Idempotent.
set -u
REPO="/home/paperspace/Documents/VS Code Projects/3D-Gaussian"
cd "$REPO" || exit 1
export PYTHONPATH="$REPO"
EVALPY="$HOME/miniforge3/envs/pipeline_stage2_frontend/bin/python"
ONITE="$REPO/sessions/_sweep_eval/overnight"
STATUS="$ONITE/status.log"
S_HQ="sessions/session_20260704_143324"
LABEL="hq_isolated_fast_l0"
log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$STATUS"; }

[ -d "$S_HQ/output_$LABEL" ] && { log "[ISO] already done — exit"; exit 0; }

# wait for the matrix runner (lockfile) AND any MILo process to clear
while [ -f "$ONITE/runner.lock" ] && kill -0 "$(cat "$ONITE/runner.lock" 2>/dev/null)" 2>/dev/null; do sleep 300; done
while pgrep -f "envs/milo/bin/python" >/dev/null 2>&1; do sleep 300; done
sleep 120

log "[ISO] launching subject-isolation arm (fast + lambda0 + masks)"
rm -rf "$S_HQ/output"
python3 stages/stage5_reconstruction/run.py --session "$S_HQ" --config config/pipeline_a6000_depth0_isolated.yaml \
  > "$ONITE/ISO.log" 2>&1
if [ -f "$S_HQ/output/provenance_stage5.json" ]; then
  mv "$S_HQ/output" "$S_HQ/output_$LABEL"
  log "[ISO] relabeled -> output_$LABEL"
  "$EVALPY" scripts/eval_recon.py "$S_HQ/output_hq_depth0" "$S_HQ/output_$LABEL" \
    --labels HQ-fast-l0 HQ-fast-l0-ISOLATED --outdir sessions/_sweep_eval/hq_isolation 2>&1 \
    | grep -aE "^\[eval\]" | tee -a "$STATUS"
  log "[ISO] complete — eval at sessions/_sweep_eval/hq_isolation"
else
  log "[ISO] FAILED — see $ONITE/ISO.log"
  tail -c 500 "$ONITE/ISO.log" | tr '\r' '\n' | tail -4 | tee -a "$STATUS"
  rm -rf "$S_HQ/output"
fi
