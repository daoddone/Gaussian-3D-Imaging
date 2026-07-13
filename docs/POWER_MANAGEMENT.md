# Power management (T15) — sleep-when-idle + wake from your Mac

Billing stops while the machine is off; storage persists. Wake takes ~1–3 min.

## Sleep side (installed + enabled on the box)
`pipeline-idle.timer` runs every 5 min; powers off ONLY when ALL of these have been clear for a
continuous 30 min:
1. no established inbound SSH (terminal, VS Code remote, scp/rsync all count as busy)
2. browser-desktop input idle ≥ 30 min (xprintidle; errors block conservatively)
3. no pipeline/agent processes (training, SfM, exports, watchers, Claude Code)
4. GPU ≤ 5% utilization
5. no manual hold (`pipeline-hold`)
6. no sessions/ file activity in the last 30 min (a transmission resets the clock)

Box commands: `pipeline-hold` / `pipeline-release` (block/allow auto-sleep),
`pipeline-off` (manual safe shutdown; refuses while pipeline busy, `--force` overrides),
`sudo pipeline-idle-check --dry-run` (prints the current verdict + which interlocks are active).
Logs: `journalctl -u pipeline-idle.service`.

## Wake side (your Mac)
Copy `tools/mac/pipeline_box.sh` to your Mac, `chmod +x`, one-time setup:
`echo 'PAPERSPACE_API_KEY=<key>' > ~/.pipeline_box.conf` (console → API keys), or install the
`pspace` CLI and log in. Machine ID `pszf0re6sq9b` is baked in (it's the hostname).
- `pipeline_box.sh on` — start + wait for SSH
- `pipeline_box.sh send P1_*.zip video.mov` — wake if needed, rsync into `sessions/`
- `pipeline_box.sh off` / `status`
Fallback: the Paperspace web console start/stop buttons always work.

## First-use validation (owner, ~5 min — the one thing not yet tested end-to-end)
Powering off from inside this agent session would kill the session, so the first real cycle is
yours: when done for the day, run `pipeline-off` (or let idle-sleep fire); next morning run
`pipeline_box.sh on` from the Mac and confirm SSH + `systemctl status anatomycapture-upload`
(enabled at boot). If the curl API variant 401s, use the pspace CLI line or the console, and note
which worked — the script's comment shows the alternate endpoint.

## Notes
- The upload receiver autostarts at boot, so app Transmit works as soon as the box is up.
- Nothing auto-processes on arrival (unchanged); processing starts when you/I kick it off.
- The 30-min dwell + interlocks make "forced off while coding" impossible by construction; if you
  ever want the box pinned up overnight for a long unattended run WITHOUT any pipeline process
  (rare), use `pipeline-hold`.
