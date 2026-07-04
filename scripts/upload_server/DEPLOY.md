# Deploying the upload receiver (always-on, PIN-gated)

Goal: tap **Transmit** on the phone → enter a short **PIN** → the capture lands in `sessions/`.
No manual port juggling, no URL/token typing.

## 1. Secrets file (not committed)

Pick a token and a short PIN, and put them where only root/the service reads them:

```bash
sudo tee /etc/anatomycapture-upload.env >/dev/null <<'EOF'
UPLOAD_TOKEN=<a long random secret>
UPLOAD_PIN=6250
EOF
sudo chmod 600 /etc/anatomycapture-upload.env
```

- `UPLOAD_TOKEN` — baked into the app build (below); rarely changes.
- `UPLOAD_PIN` — the 4-digit code the clinician types at transmit time (**6250**; the app shows the
  hint "Paul's Locker Combination", never the value). The server requires **both** the token AND the
  PIN (`X-Upload-Pin` header), so the baked token alone can't upload — a small extra gate. (It's not
  brute-force-proof on its own; keep the server behind the network you trust and rotate as needed.)

## 2. Install the service

```bash
cd "/home/paperspace/Documents/VS Code Projects/3D-Gaussian"
sudo cp scripts/upload_server/anatomycapture-upload.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now anatomycapture-upload
systemctl status anatomycapture-upload --no-pager      # should be active (running)
```

It now auto-starts on boot and restarts on failure. Logs: `journalctl -u anatomycapture-upload -f`.
Health check from the phone browser: `http://184.105.3.239:8902/health`.

## 3. Bake the config into the app (no Settings entry)

`UploadConfig` reads Info.plist keys `UPLOAD_BASE_URL` / `UPLOAD_TOKEN` first, then falls back to the
Settings sheet. To bake them for your personal build, add to `stages/stage1_capture/project.yml`
under the target's `settings.base` (do NOT commit the real token):

```yaml
    INFOPLIST_KEY_UPLOAD_BASE_URL: http://184.105.3.239:8902
    INFOPLIST_KEY_UPLOAD_TOKEN: <same as UPLOAD_TOKEN above>
```

Then `xcodegen generate` + rebuild. (Or just enter them once in the in-app Settings — it persists in
UserDefaults, so it's a one-time entry either way.) The **PIN is never baked** — it's typed each
transmit, so it isn't sitting in the binary or in git.

## Stop / change

```bash
sudo systemctl restart anatomycapture-upload    # after editing the env file
sudo systemctl disable --now anatomycapture-upload
```
