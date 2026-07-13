#!/bin/bash
# Owner's Mac-side control for the pipeline box (T15). Copy this file to your Mac, chmod +x.
# One-time setup on the Mac:  echo 'PAPERSPACE_API_KEY=<your key>' > ~/.pipeline_box.conf
#   (key: Paperspace console -> Team settings -> API keys. Machine ID is baked in below.)
# Usage:
#   pipeline_box.sh on                # start the machine, wait for SSH
#   pipeline_box.sh off               # safe power-off over SSH (refuses if pipeline busy)
#   pipeline_box.sh status            # is it reachable?
#   pipeline_box.sh send <files...>   # wake if needed, then rsync into sessions/
set -e
MACHINE_ID="pszf0re6sq9b"
HOST="paperspace@184.105.3.239"
DEST="/home/paperspace/Documents/VS Code Projects/3D-Gaussian/sessions/"
[ -f ~/.pipeline_box.conf ] && source ~/.pipeline_box.conf

ssh_up() { ssh -o ConnectTimeout=5 -o BatchMode=yes "$HOST" true 2>/dev/null; }

wake() {
  if ssh_up; then echo "box already up"; return 0; fi
  echo "box is off — sending start via Paperspace API..."
  if command -v pspace >/dev/null; then
    pspace machine start "$MACHINE_ID"
  elif [ -n "$PAPERSPACE_API_KEY" ]; then
    # Core-machines API; if your account uses the new API, replace with:
    #   curl -s -X PATCH https://api.paperspace.com/v1/machines/$MACHINE_ID/start -H "Authorization: Bearer $PAPERSPACE_API_KEY"
    curl -s -X POST "https://api.paperspace.io/machines/$MACHINE_ID/start" \
         -H "x-api-key: $PAPERSPACE_API_KEY" -H "Content-Type: application/json" && echo
  else
    echo "no pspace CLI and no PAPERSPACE_API_KEY in ~/.pipeline_box.conf"; exit 1
  fi
  echo -n "waiting for SSH"
  for _ in $(seq 1 60); do
    if ssh_up; then echo " — UP"; return 0; fi
    echo -n "."; sleep 5
  done
  echo " — timed out after 5 min (check the Paperspace console)"; exit 1
}

case "$1" in
  on) wake ;;
  off) ssh -t "$HOST" pipeline-off ;;
  status) ssh_up && echo "UP" || echo "DOWN/unreachable" ;;
  send)
    shift; [ $# -ge 1 ] || { echo "usage: pipeline_box.sh send <files...>"; exit 1; }
    wake
    rsync -avh --progress "$@" "$HOST:$DEST"
    echo "delivered to sessions/ — box will auto-sleep ~30 min after all activity stops"
    ;;
  *) echo "usage: pipeline_box.sh {on|off|status|send <files...>}"; exit 1 ;;
esac
