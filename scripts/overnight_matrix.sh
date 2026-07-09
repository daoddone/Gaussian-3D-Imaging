#!/bin/bash
# Overnight autonomous experiment runner (2026-07-07). See docs/PIPELINE_JOURNAL.md for the full plan.
#
# Runs the remaining quality-matrix + refinement arms STRICTLY SERIALLY (measured: quality-schedule
# runs must own the GPU through their densification peaks — two concurrent = OOM). Idempotent and
# self-healing: each arm is skipped if its labeled output exists; if an unlabeled completed output
# exists (provenance_stage5.json present — e.g. an in-session chain died after training but before
# relabeling), it is relabeled + evaluated rather than retrained; if a MILo train is still running,
# the runner waits for it. Safe to re-run at any time. Launch session-independent:
#   nohup bash scripts/overnight_matrix.sh >> "sessions/_sweep_eval/overnight/runner.log" 2>&1 & disown
set -u
REPO="/home/paperspace/Documents/VS Code Projects/3D-Gaussian"
cd "$REPO" || exit 1
export PYTHONPATH="$REPO"
EVALPY="$HOME/miniforge3/envs/pipeline_stage2_frontend/bin/python"
ONITE="$REPO/sessions/_sweep_eval/overnight"
mkdir -p "$ONITE"
STATUS="$ONITE/status.log"
LOCK="$ONITE/runner.lock"

log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$STATUS"; }

# single instance
if [ -f "$LOCK" ] && kill -0 "$(cat "$LOCK" 2>/dev/null)" 2>/dev/null; then
  log "another runner (pid $(cat "$LOCK")) is alive — exiting"; exit 0
fi
echo $$ > "$LOCK"
log "=== OVERNIGHT RUNNER START (pid $$) ==="

wait_for_gpu() {
  # wait until no MILo trainer/extractor is running; then a grace period (lets any in-session
  # chain finish its relabel+eval); then confirm still clear.
  while true; do
    while pgrep -f "envs/milo/bin/python" >/dev/null 2>&1; do sleep 120; done
    sleep 300
    pgrep -f "envs/milo/bin/python" >/dev/null 2>&1 || break
  done
}

eval_pair() { # eval_pair OUT_A LABEL_A OUT_B LABEL_B EVAL_SUBDIR
  "$EVALPY" "$REPO/scripts/eval_recon.py" "$1" "$3" --labels "$2" "$4" \
    --outdir "$REPO/sessions/_sweep_eval/$5" 2>&1 | grep -aE "^\[eval\]" | tee -a "$STATUS"
}

run_arm() { # run_arm SESSION CONFIG LABEL BASELINE_DIR BASELINE_LABEL EVAL_SUBDIR ARM_LABEL
  local SESSION="$1" CONFIG="$2" LABEL="$3" BASE_DIR="$4" BASE_LAB="$5" EVSUB="$6" ARM="$7"
  local OUTL="$SESSION/output_$LABEL"
  if [ -d "$OUTL" ]; then log "[$ARM] already done ($OUTL) — skip"; return 0; fi
  wait_for_gpu
  if [ -d "$OUTL" ]; then log "[$ARM] appeared while waiting — skip"; return 0; fi
  if [ -f "$SESSION/output/provenance_stage5.json" ]; then
    log "[$ARM] found completed unlabeled output — self-healing relabel"
  else
    log "[$ARM] launching: $CONFIG -> $LABEL"
    rm -rf "$SESSION/output"
    python3 stages/stage5_reconstruction/run.py --session "$SESSION" --config "$CONFIG" \
      > "$ONITE/${ARM}.log" 2>&1
    local rc=$?
    if [ $rc -ne 0 ] || [ ! -f "$SESSION/output/provenance_stage5.json" ]; then
      log "[$ARM] FAILED (rc=$rc) — last lines:"; tail -c 600 "$ONITE/${ARM}.log" | tr '\r' '\n' | tail -4 | tee -a "$STATUS"
      rm -rf "$SESSION/output"; return 1
    fi
  fi
  mv "$SESSION/output" "$OUTL" && log "[$ARM] relabeled -> $OUTL"
  eval_pair "$BASE_DIR" "$BASE_LAB" "$OUTL" "$LABEL" "$EVSUB"
  log "[$ARM] complete"
}

S_ARKIT="sessions/session_20260704_143210"
S_HQ="sessions/session_20260704_143324"

# --- Arm 0 (self-heal only): R2' HQ qualmid λ0.2 — may already be handled by the in-session chain
run_arm "$S_HQ" "config/pipeline_a6000_quality.yaml" "hq_R2_qualmid" \
        "$S_HQ/output_hq_dense" "HQ-fast-dense-l02" "hq_schedule" "R2"

# --- Arm 1: R4' HQ qualmid λ0 — THE decisive λ pair at capacity (vs R2')
run_arm "$S_HQ" "config/pipeline_a6000_quality_depth0.yaml" "hq_R4_qualmid_l0" \
        "$S_HQ/output_hq_R2_qualmid" "HQ-R2-qualmid-l02" "hq_lambda_capacity" "R4"

# --- Arm 2: R3' ARKit qualmid λ0 — λ pair on the VIO path (vs R1'')
run_arm "$S_ARKIT" "config/pipeline_a6000_quality_depth0.yaml" "arkit_R3_qualmid_l0" \
        "$S_ARKIT/output_arkit_R1_qualitymid" "ARKit-R1-qualmid-l02" "arkit_lambda_capacity" "R3"

# --- Arm 3: R6a HQ qualmid λ0.05 — the "whisper" (floater control without noise-stamping; vs R4')
run_arm "$S_HQ" "config/pipeline_a6000_quality_l005.yaml" "hq_R6_qualmid_l005" \
        "$S_HQ/output_hq_R4_qualmid_l0" "HQ-R4-qualmid-l0" "hq_whisper" "R6a"

# --- Arm 4: R6b best-face combo — v3 sharp-362 images + quality_mid schedule (demo artifact)
FACE="sessions/face_depthfree_test"
if [ -d "$FACE/output_v3_quality_mid" ]; then
  log "[R6b] already done — skip"
else
  wait_for_gpu
  log "[R6b] launching face v3 @ quality_mid"
  python3 scripts/face_depthfree_test.py --stage milo --schedule quality_mid --variant v3 \
    > "$ONITE/R6b.log" 2>&1
  if [ -f "$FACE/output_v3_quality_mid/provenance_stage5.json" ]; then
    eval_pair "$FACE/output_v3" "Face-v3-fast" "$FACE/output_v3_quality_mid" "Face-v3-qualmid" "face_best"
    log "[R6b] complete"
  else
    log "[R6b] FAILED — see $ONITE/R6b.log"; tail -c 600 "$ONITE/R6b.log" | tr '\r' '\n' | tail -4 | tee -a "$STATUS"
  fi
fi

# --- Morning summary
{
  echo "# Overnight matrix — morning summary ($(date))"
  echo
  echo "## Status log"; echo '```'; cat "$STATUS"; echo '```'
  echo
  echo "## All eval stats"
  for f in "$REPO"/sessions/_sweep_eval/*/stats.json; do
    echo "### $f"; echo '```json'; cat "$f"; echo '```'
  done
} > "$ONITE/MORNING_SUMMARY.md"
log "=== RUNNER DONE — summary at $ONITE/MORNING_SUMMARY.md ==="
rm -f "$LOCK"
