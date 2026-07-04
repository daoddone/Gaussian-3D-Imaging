#!/usr/bin/env python3
"""Laplacian-variance sharpness over a folder of PNGs (higher = sharper).

Usage: sharpness.py LABEL:dir [LABEL:dir ...]
Reports mean/median Laplacian variance per folder. Compare renders vs source photos
at MATCHED resolution (variance scales with resolution, so only compare like sizes).
"""
import sys
import glob
import numpy as np
from PIL import Image


def laplacian_var(path):
    g = np.asarray(Image.open(path).convert("L"), np.float32)
    lap = (-4 * g[1:-1, 1:-1] + g[:-2, 1:-1] + g[2:, 1:-1] + g[1:-1, :-2] + g[1:-1, 2:])
    return float(lap.var()), g.shape


for spec in sys.argv[1:]:
    label, d = spec.split(":", 1)
    pngs = sorted(glob.glob(d + "/*.png"))
    if not pngs:
        print(f"{label:22s} (no PNGs in {d})")
        continue
    vs, shp = [], None
    for p in pngs:
        v, shp = laplacian_var(p)
        vs.append(v)
    vs = np.array(vs)
    print(f"{label:22s} n={len(vs):3d}  {shp[1]}x{shp[0]}  "
          f"lap-var mean {vs.mean():8.1f}  median {np.median(vs):8.1f}  min {vs.min():7.1f}")
