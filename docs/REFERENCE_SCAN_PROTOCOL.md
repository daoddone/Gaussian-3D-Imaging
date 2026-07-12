# Reference-scan validation protocol (T13) — the Vectra day, turnkey

Purpose: the one validation we cannot self-serve — SURFACE accuracy of our meshes against a
clinical-grade reference scanner (Canfield Vectra, via Tepole's group). Scale accuracy is already
validated internally (ArUco chain, sub-1%); this day answers "is the *surface* right to ~1 mm."

## Before the day (us)
- [ ] Print + ruler-verify two ArUco sheets (100 mm check bar within 0.5 mm).
- [ ] App on the capture phone; Transmit reachable (token + PIN); test one throwaway upload.
- [ ] Battery banks; second phone as backup; this protocol printed.
- [ ] Confirm subjects/consent per their IRB (porcine TE = their existing workflow; humans need IRB).

## Per subject/site (target 15–20 min each)
1. **Vectra scan first** (their standard workflow) — export OBJ + their scale metadata.
2. **Our captures, same pose/session, minimal delay** (subject motion between scans = the main confound):
   - arkit4K, LiDAR on, slow fill-frame orbit 30–60 s, ArUco sheet flat near (not on) the anatomy.
   - Repeat ×2 (repeatability arm), plus one arkit1080 capture (mode robustness arm).
3. Log: subject/site ID, lighting, standoff, anything unusual (motion, sweat/specular skin, hair).

## Analysis (scripted, back home)
- Pipeline: standard chain -> metric OBJ per capture (`config/pipeline_recommended.yaml`).
- Rigid-align ours -> Vectra (ICP after coarse landmark init; NO scale in the alignment — scale is
  ours and must stand on its own; report the residual scale factor as a finding).
- Metrics (per T13 spec, Wound3DAssist-style):
  - Chamfer + Hausdorff distance (mm), normal consistency, per-vertex deviation heatmap.
  - Geodesic length/width of 2–3 marked features vs Vectra's own measurements.
  - Our repeatability: capture-1 vs capture-2 (same metrics) — bounds our noise floor.
- Deliverable: per-subject table + heatmaps; verdict vs the ~1 mm target and the collaborator's
  2 mm / 10 cm tolerance; failure modes cataloged (specular skin, hair, thin edges).

## Success criteria
- Median surface deviation ≤ 1 mm on skin regions (excl. hair); Hausdorff tail explained.
- Scale residual vs Vectra ≤ 1% (consistent with our marker validation).
- Repeatability ≤ half the Vectra deviation (else our noise dominates the comparison).

## Failure planning
- If deviation clusters at specular/wet regions: capture-side (cross-polarization backlog item),
  not reconstruction-side — document, don't tune on the day.
- If global scale off >1%: check print bar photo, sidecar confidence, near-field LiDAR note; the
  three-way sidecar (VIO/LiDAR/marker) localizes which anchor drifted.
