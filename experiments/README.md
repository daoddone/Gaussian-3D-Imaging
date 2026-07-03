# Week-one experiments (Section 8)

Run on **existing scans** before building the full pipeline. They prove the
risky parts first (Section 7) and need no graphics-card compilation, capture
app, or model porting.

Each "scan" is a `sessions/<id>/` folder that already has Stage 1 `capture/`
and Stage 2 `frontend/` outputs (from existing capture data + a Depth Anything 3
inference pass).

## Experiment A — does a single global scale make the reconstruction metric?

```bash
python experiments/experiment_a_single_scale.py \
    --sessions sessions/scanA sessions/scanB ... \
    --config config/pipeline.yaml \
    --out experiments/results/experiment_a
```

Fits one global scale (+offset) against the sensor depth per scan, then reports
whether the per-frame residual is **uniform** (single-scale assumption holds —
validates the metric approach) or **drifts**. Emits `summary.csv`,
`results.json`, and per-scan residual plots (if matplotlib is present).

## Experiment B — does the learned normal prior help, and does it ever hurt?

```bash
python experiments/experiment_b_normal_prior.py \
    --sessions sessions/scanA ... --config config/pipeline.yaml \
    --out experiments/results/experiment_b
```

Compares three normal sources (none / from-depth / StableNormal).

> **Scope flag:** the *definitive* verdict needs a reconstruction pass per
> condition (Stage 5 or the DN-Splatter fallback host) plus a surface reference
> (Canfield Vectra / ground truth), which do not exist in week one. This harness
> prepares the three normal sources and runs an **angular-error proxy** vs the
> sensor-depth geometry now, and prints `RECONSTRUCTION-AND-MEASURE STEP:
> PENDING`. It is an early indicator, not the surface-accuracy number the spec
> asks for.

## Dependencies to reproduce

Experiment A needs Stage 2 (`frontend/`) outputs; run Depth Anything 3 inference
on the existing scans first (that is a minimal Stage 2 inference — no COLMAP
productionization needed for the experiment). See the top-level README for the
week-one data checklist.
