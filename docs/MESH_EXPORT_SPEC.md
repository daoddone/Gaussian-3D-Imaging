# Mesh export spec — Tepole/Gosain tissue-expansion pipeline compatibility

Requirements derived from the collaborator's published methods (research 2026-07-09), so our mesh is
drop-in for their downstream geometric analysis / isogeometric FE, and improves on Vectra where it can.

## Target papers (their workflow)
- Laudo et al., "Predictive Modeling of Human Skin Deformation and Growth During Tissue Expansion in
  Postmastectomy Breast Reconstruction," J Biomech Eng 147(7):071002 (2025). PMC12147933.
- Laudo et al., "Development and calibration of digital twins for human skin growth in tissue
  expansion," Acta Biomaterialia 198:267-280 (2025). PMC12117169.
- Porcine lineage (analysis math + capture origin): PMC4520804 (2015), PMC6004345 (2018).

## Their chain (what our OBJ feeds) — CORRECTED against the 4 papers (adversarially verified 2026-07-09)
Vectra H2 3D photo (or MVS from many photos in the older papers) → align/register in MeshLab (area) +
CloudCompare (surface-to-surface distance); **exact registration algorithm NOT stated in any paper —
do NOT assume ICP** → discretize to FE (breast2025: 35,319 **hexahedral C3D8** in Cubit, skin extruded
to **1.55 mm**; jbiomech2018 scalp: 141,497 **TETRAHEDRAL** elements + a 6807-node skull, 3.6 mm —
element TYPE and thickness differ per study) → Abaqus growth sim → Bayesian/GP calibration against
surface distance + area-growth. **Accuracy actually stated:** breast2025 mean surface-distance error
**1.6 mm (range 0–7 mm, "<2 mm")**; porcine MVS reconstruction error **0.6–2%** (vs the ruler). The
"1.05–1.79 mm" and "23k–58k hex" figures in earlier drafts were NOT in these papers — removed. They
REMESH regardless, so CLEANLINESS + absolute SCALE are the real constraints, not raw density.

## REQUIREMENTS (the exporter must satisfy)
1. **Format:** OBJ (+ MTL + texture PNG) AND PLY. Load in MeshLab/CloudCompare/Cubit/Abaqus. NOTE: the
   exact on-disk format they ingest is INFERRED (OBJ/PLY), never stated in the papers — confirm with them.
2. **Units:** MILLIMETERS, true absolute scale. ⚠️ MAKE-OR-BREAK — the human FE work is in mm and the
   stated surface error is ~1.6 mm (unqualified mean; the papers do NOT state the ROI length, so the
   "over ~10 cm" framing was a cross-paper conflation — dropped). Our ~12% LiDAR-vs-VIO scale ambiguity
   dwarfs a ~mm budget → scale MUST be anchored (sensor lock and/or ruler) before FE use. Export writes
   the scale source + confidence (+ MAD) into a sidecar; flag if unverified.
3. **Topology:** open single-surface manifold PATCH (outer skin sheet), consistent OUTWARD normals. NOT
   watertight — they extrude/offset it to a thin solid; self-intersections/non-manifold edges break Cubit.
   We assign NO thickness (that is theirs).
4. **Cleanliness (binding constraint):** no holes inside the ROI, no non-manifold edges, no
   self-intersections, minimal noise, largest-connected-component only. Must survive their registration
   (software: MeshLab/CloudCompare; algorithm unstated) + Cubit extrusion.
5. **Density:** provide the full-res mesh AND a decimated ~100k-500k-tri variant (well-conditioned
   for Cubit). Cleanliness ≫ count (their FE meshes are ~35k–140k elements after their own remeshing).
6. **Texture:** retained (OBJ+MTL+PNG). Currently OPTIONAL to their math but our #1 strategic lever
   (below).

## OPPORTUNITIES (exceed Vectra)
- **Restore dense material-point correspondence** — their stated #1 clinical limitation: they lost the
  porcine tattooed-grid strain tracking in humans and are reduced to *overall* area/distance metrics.
  Our dense photometric texture enables cross-time texture/splat correspondence → full-field strain +
  growth maps WITHOUT tattooing patients. Highest-value differentiator (future work).
- Many-view splat beats Vectra's 2-shot stereo on coverage/holes/noise (esp. breast/IMF recesses) — IF
  we hit mm scale.
- Automation: emit analysis-ready ROI-cropped, hole-free, mm-scaled OBJ + a Cubit-ready decimated
  surface → removes their manual MeshLab/CloudCompare cleaning.
- Accepted-envelope precedent: iPhone-X vs Vectra M1 agreed 0.57-1.85 mm (PMC10320691) — our path is in
  range if scale is controlled.

## Markers/fiducials — from the papers, adversarially verified 2026-07-09
- **Scale in the human breast study is from the CALIBRATED VECTRA H2 hardware — NO ruler, NO scale bar,
  NO markers described for scale** (breast2025). The two LED beams only set camera standoff.
- **The older MVS papers DID use a physical RULER for scale:** porcine2015 fits a cubic spline to a
  ruler's 1 cm marks (reconstruction error 0.6–2%); jbiomech2018 shows a yellow ruler on the drape.
  iso2020 instead uses the known 10×10 cm tattooed-grid dimension for scale. → **A ruler is well within
  their MVS heritage; our using one for validation is precedented.** (Correction: earlier draft wrongly
  said pigs get scale "from the grid" — the GRID is correspondence; the RULER is scale.)
- **Tattooed grid (porcine only):** a 10×10 cm, 121-point (11×11) grid gives DENSE material-point
  correspondence → the F/Fgrowth/Fstretch strain heat-maps. Humans are NOT grid-tattooed.
- **Human limitation (breast2025, verbatim):** "we are limited to overall distance error and overall
  area growth… New registration methods with built-in tattooing could be used to circumvent the lack of
  the grid." → THE GAP our dense texture could fill (UNPROVEN — validation required).
- **The white strip in breast2025 Fig 1(a):** located "on the chest"; its PURPOSE IS NOT STATED in the
  paper. The "base of breast / IMF landmark" reading is an OWNER RECOLLECTION, not a paper fact — treat
  as an open question, not confirmed.
- **Two distinct marker roles (never conflate):**
  1. SCALE (ruler / sensor) — absolute mm. For us: sensor (LiDAR/VIO) anchor is the markerless path;
     ruler is VALIDATION-grade cross-check. Needed on accuracy captures.
  2. CORRESPONDENCE/REGISTRATION (grid, landmarks) — align timepoints + track strain. LONGITUDINAL only;
     our texture/splat correspondence is the candidate replacement (must be validated vs a grid).
- Single-timepoint geometry (near-term) needs only trustworthy absolute SCALE (sensor, ruler-validated).

## Open items to confirm with the collaborator
- Exact on-disk format they ingest (inferred OBJ/PLY; not stated in papers).
- Whether they want the decimated Cubit-ready variant or will remesh from full-res themselves.
- Whether they'd accept computational (texture-based) correspondence in place of physical grids/marks.
