#!/usr/bin/env bash
set -e
REPO="/home/paperspace/Documents/VS Code Projects/3D-Gaussian"
ENV="$HOME/miniforge3/envs/milo"
cd "$REPO"
"$ENV/bin/python" - <<'PY'
import sys, shutil, os
sys.path.insert(0, ".")
sys.path.insert(0, "stages/stage5_reconstruction")
from pathlib import Path
from common import colmap_io
S = Path("sessions/session_20260703_203728")
ds = S / "reconstruction_input"
(ds / "sparse/0").mkdir(parents=True, exist_ok=True)
(ds / "images").mkdir(parents=True, exist_ok=True)
shutil.copy2(S/"metric_ba/colmap/sparse/0/cameras.bin", ds/"sparse/0/")
shutil.copy2(S/"metric_ba/colmap/sparse/0/images.bin", ds/"sparse/0/")
shutil.copy2(S/"metric/colmap/sparse/0/points3D.bin", ds/"sparse/0/")
imgs = colmap_io.read_images_binary(str(ds/"sparse/0/images.bin"))
for im in imgs.values():
    d = ds/"images"/im["name"]
    if d.exists() or d.is_symlink(): d.unlink()
    os.symlink((S/"capture/rgb"/im["name"]).resolve(), d)
print(f"[hand] assembled reconstruction_input: {len(imgs)} images")
import milo_supervised as M
prov = M.reconstruct(dataset_dir=str(ds), capture_dir=str(S/"capture"),
                     normals_dir=None, output_dir=str(S/"output_milo_full"),
                     options={"depth_lambda": 0.2, "imp_metric": "indoor"})
print("[hand] provenance:", prov)
PY
echo "###### MILO HAND DONE ######"
