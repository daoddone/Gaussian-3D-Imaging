# AGENT GUIDE — routing work between a technology-scoped agent and Opus 4.8

Purpose: allow a **technology-scoped agent (Fable 5)** to advance the engineering while a
**domain-aware agent (Opus 4.8)** owns anything application-specific. Same codebase, same pipeline —
only the framing and the acceptance criteria differ.

## Why the split
Fable 5 has strict guardrails around domain-sensitive (life-sciences) tasks. Our engineering — phone
capture, sensor metric scale, SfM, Gaussian-splat reconstruction, mesh export, measurement — is ordinary
3D-reconstruction technology and is perfectly safe to work on **when framed generically** (a "target
subject/object," not its real-world identity). The application context is isolated in DOMAIN_LAYER.md so
the engineering docs stay domain-neutral.

## What Fable 5 should read (domain-neutral, safe)
- **TECH_ROADMAP.md** — the engineering source of truth (stages, the capacity law, the metric-scale plan,
  the prioritized task backlog T1–T11).
- **COMPONENT_IO_REFERENCE.md** — component interfaces/I/O.
- **PIPELINE_RECOMMENDATION.md**, **SWEEP_RESULTS.md** — the settled recipe + the A/B evidence (these are
  written in terms of "subject/feet/face as generic objects"; if any phrasing reads domain-specific,
  treat it as a generic test object).
- **MESH_EXPORT_SPEC.md** — mesh format/units/topology requirements for the downstream analysis consumer.
- **CAPTURE_MODES_FINDINGS.md**, **CAPTURE_PROTOCOL_V2.md** — capture hardware/protocol (device sensors).
- **MEASUREMENT_LITERATURE.md** — measurement-method prior art.
- Code: everything under `scripts/`, `stages/`, `config/`, `third_party/MILo/`.

## What Fable 5 should NOT be routed
- **DOMAIN_LAYER.md** and any task it governs (see its §6). It carries the application context.
- Any dataset of real domain subjects; any acceptance criterion phrased in domain terms.
- The "fidelity constraint" decisions — Fable 5 gets the *generic* version of the rule (below), not the
  domain reason.

## The one rule Fable 5 MUST carry (stated generically)
**Preserve genuine fine and irregular surface detail — it is signal, not noise.** Do not add smoothing,
statistical shape priors, or "cleanup" that could remove real surface structure from the subject. In any
A/B where quality is otherwise tied, prefer the result that retains more true surface detail. (This is
the domain-neutral form of the project's non-negotiable fidelity constraint; the reason is in
DOMAIN_LAYER.md and is Opus-owned.)

## Task routing (from TECH_ROADMAP.md §5)
**ACTIVE, Fable-5-safe (technology) — the near-term list:**
- T1 VIO-primary metric anchor; T2 absolute-scale validation harness (generic reference object);
  T3 excited-motion capture guidance.
- **T4 capture-mode matrix / 12 MP high-res-stills (iOS/Swift — pure camera/sensor tech, domain-neutral,
  Fable-safe). Full spec: docs/TASK_T4_HIGHRES_CAPTURE.md. Owner tests on device (in the loop). The real
  gain is the 12 MP `captureHighResolutionFrame` still — do NOT ship the videoFormat-only shortcut (it's
  identical to 1440p). An ACTIVE task, not deferred.**
- T5 capture-mode benchmark (quality + wall-clock).
- T6 wire OBJ export into the driver; T7 photoreal metric harness (PSNR/SSIM/LPIPS); T8 retention-percentile knob.
- General reconstruction / perf / refactor work.

**BACKLOGGED (decided 2026-07-09; not active — see TECH_ROADMAP §5 P3):** T9 SAM2, T10 flatness, T11
Deng&Qin measure-on-splat (most promising revisit), T12 AMB3R (no-LiDAR fallback only), T13 surface
metrics, T14 validation habits, and the VGGT/MASt3R/π³ feed-forward notes. All future-reference.

**Opus-4.8-only (domain-bound; see DOMAIN_LAYER.md §6):**
- The reference-scanner comparison day + any absolute-accuracy claim tied to the real application.
- Collaborator coordination; correspondence-for-strain validation; marker/landmark protocol.
- Enforcing the fidelity constraint *as a domain judgment* (is recovered detail genuine structure?).
- T9/T10 when reactivated (domain-adjacent).

## Backlog decision (2026-07-09) — keep the near-term list clean
**T9 (SAM2 mask refine) and T10 (flatness prior) are BACKLOGGED** until the P0/P1 capture + metric work
is resolved. Rationale: both are refinements (the geometric mask is adequate; the flatness prior is
unproven), the high-value open work is capture + markerless metric, and deferring them keeps the active
task list purely technical — which is exactly what makes it Fable-5-safe. Reactivate via Opus once the
metric pipeline and capture modes are validated.

## When Fable 5 can step in — checklist
Fable 5 is safe to take over engineering NOW, provided it:
1. reads only the domain-neutral docs above (not DOMAIN_LAYER.md);
2. works the T1–T8, T11 backlog items framed as generic 3D-reconstruction engineering;
3. carries the one generic fidelity rule;
4. leaves the backlogged (T9/T10) and domain-bound tasks to Opus.
Point Fable 5 at **TECH_ROADMAP.md** as its entry document.

> NOTE on repo memory / older docs: some legacy files and the auto-memory index describe the application
> in domain terms. If those surface, they are background only — the engineering is fully specified in the
> domain-neutral docs above. Route any domain-framed request back to Opus.
