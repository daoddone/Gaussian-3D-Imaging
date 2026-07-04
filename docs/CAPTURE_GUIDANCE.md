# Capture guidance (Stage 1, AnatomyCapture app)

Answers to the on-device questions, and how to get the best capture for the pipeline.

## "valid depth %" — what it means and how to raise it

The readout is the **share of depth pixels the sensor trusts** on the current frame. It is shown
live in preview (a framing aid) and during recording. The two backends compute it differently, so
they read differently — this is expected, not a bug:

| Backend | "valid" rule | Why it reads the way it does |
|---|---|---|
| **ARKit** | depth pixel has `ARConfidenceLevel >= medium` | ARKit **temporally fills + smooths** its depth and then confidence-thresholds it, so the map has few holes. It reads **high at ~1–2 ft** and drops when too close (confidence falls) or too far. |
| **HQ-Depth** | depth pixel is **finite** (raw LiDAR returned a value) | Raw, unfiltered LiDAR: no-return pixels arrive as **NaN** (holes). Closer than the LiDAR near-field (~25 cm), on dark/specular/edge regions, holes dominate → **low %**. It needs the subject **past ~25–30 cm** and filling the frame. |

So ARKit's % is inherently higher (it's a filled map); HQ's % is lower (it's raw). **Don't compare
the two numbers directly** — compare each to its own "is the subject well-covered?" bar.

**To raise it:** fill the frame with the subject, hold ARKit at ~1–2 ft / HQ at ~30 cm+, keep the
surface roughly fronto-parallel (grazing angles and thin edges drop out), and avoid dark/shiny
regions. The guidance capsule under the shutter tells you which way to move when the % is low.

## ARKit coverage: point cloud, not the room-scale mesh

The post-record **3D inspector now shows a LiDAR coverage point cloud**, not ARKit's fused mesh.
ARKit's scene-reconstruction mesh is tuned for **room mapping** and looks coarse/blobby on a
close-range subject (a foot, a wound) — that's a limitation of the fused mesh, not your capture.
The point cloud is far denser and truer to a close subject, and (unlike the mesh) is available for
the **HQ-Depth** path too:

- **ARKit**: the cloud is **fused across the orbit** (world-space, using the live pose).
- **HQ-Depth**: no live pose, so the cloud is a **single-view (2.5-D)** snapshot of the most recent
  frame — enough to confirm "did I get the region," not a full 3-D surface.

The live overlay (mesh wireframe) is unchanged; a **live accumulated point-cloud overlay** is a
follow-up (needs a Metal/ARSCNView path — deferred to avoid destabilizing the working capture).

The mesh no longer includes geometry seen **before** you pressed Record: recording now clears the
scene mesh at the start (`.resetSceneReconstruction`), so the coverage reflects only the recording.

## Which backend to use

- **ARKit** — has a metric camera path (poses), so the pipeline is most accurate; best default.
- **HQ-Depth** — raw absolute LiDAR + high-res color, no pose (pipeline recovers pose via unseeded
  SfM/DA3). Use for close-up detail where you want the unfiltered depth.

## Focus

Continuous autofocus is on by default (HQ path); **tap the preview to focus a specific region**,
and use the AF/Lock toggle to pin focus for a static close-up. Note the wide camera has a hardware
minimum focus distance (~15 cm) — closer than that can't be brought into focus on that lens
regardless of software; per-frame intrinsics keep whatever focus you get metrically valid.

## Transmission

Two ways to get a capture to the Linux box:
1. **AirDrop → Files → Mac → ssh** (what you've been doing): the capture dir carries everything.
2. **In-app Transmit**: set the receiver **Base URL + token** in **Settings** (gear, top-left)
   first — otherwise Transmit reports "set server URL + token in Settings first". The description
   you type (setup or review screen) is now written into `metadata.json` on disk either way.
