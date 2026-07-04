#!/usr/bin/env python3
"""Minimal capture-upload receiver for AnatomyCapture (iOS) -> Linux.

Receives a zipped capture session from the phone and stores it under sessions/.
DELIBERATELY does NOT auto-process (during the A/B experimental stage we choose the
processing per-comparison by hand). Metadata (description, framework used, orientation,
timestamps) travels inside the zip as capture/metadata.json, written by the app; this
server just unzips and logs it.

Design (per the capture-design research):
  - iOS side sends a URLSession *background* uploadTask(fromFile: zip): the HTTP body IS
    the raw zip bytes (not multipart), which is the most robust path over clinic wifi/cellular.
  - Auth: a single shared bearer token (env UPLOAD_TOKEN), trivially revocable, not shell-capable.
  - Binds to 127.0.0.1 by default. PUBLIC EXPOSURE (TLS via Caddy + open port) is a change to
    the box's network surface and is intentionally NOT done here -- stand that up only after
    the user OKs it (see scripts/upload_server/README on how).

Usage:  UPLOAD_TOKEN=<secret> python3 server.py [--host 127.0.0.1] [--port 8000] [--dest sessions]
"""
import argparse
import hmac
import io
import json
import os
import re
import zipfile
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs

REPO = Path(__file__).resolve().parents[2]
TOKEN = os.environ.get("UPLOAD_TOKEN", "")
# Optional short PIN (e.g. 4 digits) the clinician enters on the phone at transmit time — a small
# extra gate so a baked-in app token alone can't upload. Only enforced when set (X-Upload-Pin header).
PIN = os.environ.get("UPLOAD_PIN", "")
DEST = REPO / "sessions"
MAX_BYTES = 500 * 1024 * 1024          # 500 MB per upload cap
_SAFE = re.compile(r"[^A-Za-z0-9_.-]")


def _clean(name: str) -> str:
    """Sanitize a session id to a safe single path component."""
    name = _SAFE.sub("_", (name or "").strip())
    return name[:80] or "session_unnamed"


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _json(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _authed(self) -> bool:
        if not TOKEN:                                   # server misconfigured -> refuse all
            return False
        got = self.headers.get("Authorization", "")
        want = f"Bearer {TOKEN}"
        return hmac.compare_digest(got, want)

    def do_GET(self):
        if urlparse(self.path).path == "/health":
            return self._json(200, {"ok": True, "dest": str(DEST)})
        return self._json(404, {"error": "not found"})

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path != "/upload":
            return self._json(404, {"error": "not found"})
        if not self._authed():
            return self._json(401, {"error": "unauthorized"})
        if PIN and not hmac.compare_digest(self.headers.get("X-Upload-Pin", ""), PIN):
            return self._json(401, {"error": "bad pin"})

        session = _clean(parse_qs(parsed.query).get("session", [""])[0])
        length = int(self.headers.get("Content-Length", 0))
        if length <= 0 or length > MAX_BYTES:
            return self._json(413, {"error": f"bad length {length} (cap {MAX_BYTES})"})

        # read the raw zip body (URLSession uploadTask fromFile sends the file as the body)
        buf = bytearray()
        remaining = length
        while remaining > 0:
            chunk = self.rfile.read(min(1 << 20, remaining))
            if not chunk:
                break
            buf.extend(chunk)
            remaining -= len(chunk)

        try:
            zf = zipfile.ZipFile(io.BytesIO(bytes(buf)))
        except zipfile.BadZipFile:
            return self._json(400, {"error": "body is not a valid zip"})

        # unique destination; never overwrite an existing session
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")
        out = DEST / f"{session}"
        if out.exists():
            out = DEST / f"{session}__{stamp}"
        out.mkdir(parents=True, exist_ok=True)

        # safe extraction: reject absolute paths / .. traversal
        for member in zf.namelist():
            target = (out / member).resolve()
            if not str(target).startswith(str(out.resolve())):
                return self._json(400, {"error": f"unsafe path in zip: {member}"})
        zf.extractall(out)

        meta = {}
        for cand in (out / "capture" / "metadata.json", out / "metadata.json"):
            if cand.exists():
                try:
                    meta = json.loads(cand.read_text())
                except Exception:
                    meta = {"_note": "metadata.json present but unparseable"}
                break
        n_files = sum(1 for _ in out.rglob("*") if _.is_file())
        print(f"[upload] {out.name}: {n_files} files, {length/1e6:.1f} MB | meta={meta}", flush=True)
        return self._json(200, {"ok": True, "session": out.name, "files": n_files,
                                "bytes": length, "metadata": meta})

    def log_message(self, *a):
        pass  # quiet default access log; we print our own line per upload


def main():
    global DEST
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--dest", default=str(DEST))
    args = ap.parse_args()
    DEST = Path(args.dest)
    DEST.mkdir(parents=True, exist_ok=True)
    if not TOKEN:
        print("WARNING: UPLOAD_TOKEN not set -> all uploads will be rejected. "
              "Set UPLOAD_TOKEN=<secret> before serving.", flush=True)
    print(f"[upload-server] listening on http://{args.host}:{args.port}  dest={DEST}", flush=True)
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
