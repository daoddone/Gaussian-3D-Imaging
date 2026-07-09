# DOMAIN LAYER — application context & rationale (Opus-4.8-owned)

> **AGENT ROUTING:** This document and the tasks it governs are handled by **Opus 4.8**. It records the
> real-world application purpose behind the domain-neutral TECH_ROADMAP.md. It is kept separate so the
> engineering roadmap can be worked on by a technology-scoped agent without domain framing. **Do not
> route this document's tasks to the technology-scoped agent.** The engineering pipeline is identical
> either way — this layer only supplies *meaning* and *acceptance criteria*, not different code.

## 1. What the "target subject" actually is
The generic "target subject / object" in TECH_ROADMAP.md is, in reality, **patient anatomy** captured for
clinical documentation and downstream biomechanical modeling. The pipeline's purpose is to produce
**metric, photorealistic 3D records of anatomy** from a short iPhone capture — a low-cost, high-automation
alternative to a dedicated clinical 3D-photo system.

## 2. The load-bearing domain constraint (never lose this)
**"Faithful to the abnormal."** This is a clinical tool for documenting anatomy that may be pathological.
Therefore the reconstruction must **NOT** apply any healthy-anatomy prior, statistical face/body model,
or aggressive smoothing that could regularize away genuine abnormal structure. This is WHY:
- we rejected surface-supervision smoothing that "looked cleaner" — over-smoothing is dangerous here;
- the flatness prior is gated to hard-protect the subject and is BACKLOGGED until proven non-destructive;
- "fidelity" beats "prettiness" in every A/B tie.
A general-purpose 3D pipeline would happily smooth; ours must not. Any agent optimizing "quality" must be
told, in domain-neutral terms, that **genuine fine/irregular surface detail is signal, not noise.**

## 3. What "downstream engineering / FEA analysis" means
The mesh consumer is the **Tepole (Purdue) / Gosain (Lurie Children's/Northwestern) tissue-expansion
biomechanics group.** They ingest surface meshes into an isogeometric / finite-element pipeline
(CloudCompare/MeshLab → Cubit → Abaqus) that models **skin deformation and growth during tissue
expansion** (e.g., postmastectomy breast reconstruction, pediatric scalp). Requirements verified from
their papers are in MESH_EXPORT_SPEC.md (the domain-neutral version); the anatomy specifics:
- The "open surface patch" = a **skin surface** they extrude to a thin solid and grow via FE.
- The "absolute mm scale, ~2 mm tolerance" = their skin-surface-distance + area-growth error budget.
- The "reference measurement system" = a **Canfield Vectra** stereophotogrammetry unit (their current
  clinical capture device; we aim to match or exceed it with a phone).
- Their stated unmet need = **dense material-point correspondence for full-field strain** (they have it
  in tattooed-grid porcine studies, lack it in un-tattooed human patients). Our dense texture could
  restore it WITHOUT marking patients — the flagship differentiator (unproven; validate before claiming).

## 4. Domain meaning of the generic tasks
| TECH_ROADMAP task | Domain meaning / acceptance criterion |
|---|---|
| Metric scale (T1–T3) | Clinical measurement accuracy; ~2 mm over a lesion/region; enables the FEA collaborator's mm budget |
| Reference-object validation (T2) | Phantom/ruler-vs-Vectra accuracy test; the path to a defensible "~1 mm vs Vectra" claim |
| Capture-mode benchmark (T5) | Which iPhone capture protocol a clinician should use; usability (time) matters at the bedside |
| Photoreal metric (T7) | Clinical documentation fidelity of the splat (what clinicians view) |
| Segmentation refine (T9) | Isolating anatomy from clinical-scene background (drapes, table, other body parts) |
| Flatness prior (T10) | Off-anatomy surface cleanup ONLY; must never touch the anatomy — hence backlogged + subject-gated |

## 5. Markers / fiducials (domain nuance)
- **Ruler in-frame:** we keep it as a VALIDATION cross-check of markerless sensor scale — and it matches
  their MVS heritage (their porcine/scalp papers used a ruler at 0.6–2%). The project GOAL is markerless
  (a usability/access improvement over their marker-based capture), with the ruler proving the accuracy.
- **Anatomical landmarks / grids** (clavicle/sternum marks, quadrant letters, tattooed grids): these are
  for their LONGITUDINAL registration + strain, not for our single-timepoint geometry. Our candidate
  replacement is texture/splat correspondence across timepoints (future, unproven). Their 2025 human
  paper is markerless for registration (no grid); the white tape in their figure is of unstated purpose.

## 6. Tasks that MUST stay with Opus 4.8 (domain-bound)
- Enforcing "faithful to the abnormal" in any quality/smoothing decision (the anatomy fidelity guardrail).
- The reference-scanner (Vectra) capture-day protocol and the clinical-accuracy claims.
- Collaborator (Tepole/Gosain) coordination, correspondence-for-strain validation, marker protocol.
- Any dataset containing real patient anatomy; any clinical acceptance criterion.
- Interpreting whether recovered surface detail is "genuine abnormal structure" vs artifact.

## 7. Tasks safe for the technology-scoped agent (domain-neutral)
Everything in TECH_ROADMAP.md §5 framed generically: sensor metric-scale estimation + validation,
capture-mode implementation/benchmarking, SfM/splat/mesh engineering, OBJ/PLY export, photoreal metrics,
compute/perf. These are ordinary 3D-reconstruction engineering on a generic "subject/object."

---
*This layer changes NO code. It ensures the domain purpose and its non-negotiable fidelity constraint are
never lost while the engineering is advanced in domain-neutral terms.*
