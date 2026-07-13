#!/bin/bash
# T15 sleep-side: power off the box ONLY when demonstrably idle (owner rule: never cut off work).
# Runs from systemd timer every 5 min. Interlocks — ANY one blocks shutdown:
#   1. established inbound SSH (covers terminal ssh, VS Code remote, scp/rsync in flight)
#   2. browser-desktop input idle < DWELL (xprintidle; if it errors -> BLOCK, conservative)
#   3. pipeline/agent processes alive (training, SfM, exports, watchers, Claude Code)
#   4. GPU utilization > 5%
#   5. manual hold file (pipeline-hold / pipeline-release)
#   6. recent sessions/ activity (a transmission just landed)
# Plus a dwell: ALL interlocks must have been clear for DWELL_SECS continuously.
# Dry-run: idle_shutdown_check.sh --dry-run   (prints verdict, never powers off)

DWELL_SECS=${DWELL_SECS:-1800}
STATE=/var/tmp/pipeline_idle_state
REPO="/home/paperspace/Documents/VS Code Projects/3D-Gaussian"
DRY=0; [ "$1" = "--dry-run" ] && DRY=1

busy=""
# 1. inbound ssh (established on :22)
n_ssh=$(ss -Hnt state established '( sport = :22 )' 2>/dev/null | wc -l)
[ "$n_ssh" -gt 0 ] && busy="ssh($n_ssh)"
# 2. desktop input idle
idle_ms=$(sudo -u paperspace DISPLAY=:0 XAUTHORITY=/home/paperspace/.Xauthority xprintidle 2>/dev/null)
if [ -z "$idle_ms" ]; then busy="$busy xprintidle-failed(conservative-block)";
elif [ "$idle_ms" -lt $((DWELL_SECS * 1000)) ]; then busy="$busy desktop-active($((idle_ms / 60000))min)"; fi
# 3. pipeline / agent processes
PROC_RE='train\.py|stage5_reconstruction/run\.py|session_sfm|04_metric_anchor|eval_recon|export_mesh_obj|ingest_video|mesh_extract|make_subject_masks|frontal_compare|t5_mode_benchmark|anthropic\.claude-code'
n_proc=$(pgrep -fc "$PROC_RE" 2>/dev/null || echo 0)
[ "$n_proc" -gt 0 ] && busy="$busy pipeline-procs($n_proc)"
# 4. GPU
gpu=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -1)
[ -n "$gpu" ] && [ "$gpu" -gt 5 ] && busy="$busy gpu(${gpu}%)"
# 5. hold file
[ -f /var/tmp/pipeline-keepalive ] && busy="$busy HOLD-FILE"
# 6. recent sessions/ writes (transmission dwell)
newest=$(find "$REPO/sessions" -maxdepth 3 -newermt "-${DWELL_SECS} seconds" -print -quit 2>/dev/null)
[ -n "$newest" ] && busy="$busy recent-session-files"

now=$(date +%s)
if [ -n "$busy" ]; then
  echo "$now" > "$STATE"
  echo "$(date -Is) BUSY: $busy"
  exit 0
fi
last_busy=$(cat "$STATE" 2>/dev/null || echo "$now")
clear_for=$((now - last_busy))
if [ "$clear_for" -lt "$DWELL_SECS" ]; then
  echo "$(date -Is) idle, dwell $clear_for/${DWELL_SECS}s"
  exit 0
fi
echo "$(date -Is) IDLE past dwell -> POWEROFF (dry=$DRY)"
[ "$DRY" -eq 1 ] && exit 0
/usr/bin/systemctl poweroff
