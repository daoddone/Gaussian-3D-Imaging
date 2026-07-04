#!/usr/bin/env bash
set -e
cd "/home/paperspace/Documents/VS Code Projects/3D-Gaussian"
export PBA_SESS="sessions/session_20260703_203728"
GSBA="$HOME/miniforge3/envs/gs-ba/bin/python"
S2="$HOME/miniforge3/envs/pipeline_stage2_frontend/bin/python"
S="sessions/session_20260703_203728"
rm -rf "$S/pose_ba/triangulated" "$S/pose_ba/refined" "$S/pose_ba/sfm_noseed"
echo "### 01 match (SuperPoint + exhaustive LightGlue) ###"; "$GSBA" scripts/pose_ba/01_match.py 2>&1 | grep -aE "\[01\]|Error"
echo "### 02 seeded BA ###"; "$GSBA" scripts/pose_ba/02_triangulate_ba.py 2>&1 | grep -aE "\[02\]|Error"
echo "### 02b unseeded SfM ###"; "$GSBA" scripts/pose_ba/02b_sfm_noseed.py 2>&1 | grep -aE "\[02b\]|Error"
echo "### 03 relock seeded -> metric_ba ###"; "$S2" scripts/pose_ba/03_relock.py --model pose_ba/refined --out metric_ba 2>&1 | grep -aE "\[03\]|Error"
echo "### 03 relock unseeded -> metric_ba_noseed ###"; "$S2" scripts/pose_ba/03_relock.py --model pose_ba/sfm_noseed --out metric_ba_noseed 2>&1 | grep -aE "\[03\]|Error"
echo "### 3-method pose deltas vs ARKit ###"
"$GSBA" - <<'PY'
import sys,numpy as np; sys.path.insert(0,".")
from common import colmap_io
def load(p):
    imgs=colmap_io.read_images_binary(p+"/images.bin");o={}
    for v in imgs.values():
        q=np.asarray(v["qvec"],float);w,x,y,z=q
        R=np.array([[1-2*(y*y+z*z),2*(x*y-w*z),2*(x*z+w*y)],[2*(x*y+w*z),1-2*(x*x+z*z),2*(y*z-w*x)],[2*(x*z-w*y),2*(y*z+w*x),1-2*(x*x+y*y)]])
        t=np.asarray(v["tvec"],float);o[v["name"]]=(R,-R.T@t)
    return o
S="sessions/session_20260703_203728"
ark=load(S+"/metric/colmap/sparse/0")
for tag,d in [("seeded BA  ",S+"/metric_ba/colmap/sparse/0"),("unseeded SfM",S+"/metric_ba_noseed/colmap/sparse/0")]:
    m=load(d);common=sorted(set(ark)&set(m));rot=[];cen=[]
    for n in common:
        Ra,Ca=ark[n];Rb,Cb=m[n]
        rot.append(np.degrees(np.arccos(np.clip((np.trace(Ra@Rb.T)-1)/2,-1,1))));cen.append(np.linalg.norm(Ca-Cb)*1000)
    rot=np.array(rot);cen=np.array(cen)
    print(f"  {tag} vs ARKit ({len(common)} frames): rot median {np.median(rot):.3f}deg (max {rot.max():.3f}) | center median {np.median(cen):.3f}mm (max {cen.max():.3f})")
PY
echo "###### HAND POSEBA DONE ######"
