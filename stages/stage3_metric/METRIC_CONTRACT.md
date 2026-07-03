# Stage 3 Metric-Module Interface Contract

> The interface contract the reconstruction engineers build against, so their work can proceed
> in parallel with the capture and front-end work against a frozen boundary. It produces
> exactly the output defined in `io_contracts/metric_output.md`.
>
> Notation: an abbreviation is written out in full the first time with the short form in
> parentheses. File extensions and code symbols are reproduced exactly.

## Purpose

Guarantee that the reconstruction is at true physical scale, and produce a single residual-error number that becomes the pipeline's headline accuracy figure. The front-end model already outputs geometry in meters, but that scale is a learned estimate, so this module cross-checks it against physical measurements and, if they disagree, corrects to the physical measurements and raises a flag.

## The concept, in plain terms

A reconstruction can be correct in shape but off by a single overall size factor (and a single offset). Think of a scale model whose proportions are all right but whose overall size is unknown. This module solves for that size factor by aligning the reconstruction to real-world measurements, then applies it, at which point the reconstruction is metric. It does this against up to three independent physical measurements and checks whether they agree, because agreement among independent measurements is what makes a metric claim credible.

Two standard tools are used, and both are worth naming:

- A **similarity transform** is the mathematical operation that scales, rotates, and shifts one set of points to best match another. It has seven numbers: one for scale, three for rotation, three for the shift. Solving for the best similarity transform between two matched point sets has a known closed-form solution (the Umeyama method).
- **Iterative Closest Point** is a refinement that repeatedly pairs each point in one cloud with its nearest point in the other and finds the rigid movement that best aligns them, repeating until it settles. It sharpens an alignment that is already roughly correct.
- Where individual measurements are noisy, a **robust fit** is used, meaning a fitting method that ignores outliers rather than being dragged by them. One common form, Random Sample Consensus (RANSAC), repeatedly fits to small random subsets and keeps the fit that the most measurements agree with.

## The three physical anchors

1. **Sensor depth.** The depth sensor's readings from Stage 1, used only where the depth is valid. For each frame, compare the front-end reconstruction's depth to the sensor's depth at the valid pixels and fit the single scale factor that best matches them, robustly.
2. **Camera path.** The metric camera positions from Stage 1's tracking stream, matched against the camera positions the front end estimated. Fitting the best similarity transform between the two sets of camera positions recovers the scale directly, and this anchor does not depend on the noisy depth at all.
3. **Physical ruler (optional).** A physical object of known size placed in the frame. Measuring its reconstructed size and comparing to its true size gives a scale factor, and this is the only anchor that is a genuine physical ground truth rather than an estimate.

## Inputs (read from the session folder)

- From Stage 2: `frontend/points.ply` (the dense reconstruction), the per-frame poses under `frontend/poses/`, and `frontend/depth/*.npy` (the front end's own per-frame depth).
- From Stage 1: `capture/depth/*.npy`, `capture/confidence/*.png`, and `capture/poses.json`.
- The `README` in the capture folder, to learn whether the camera-path anchor is present.
- Numeric thresholds from `config/pipeline.yaml` (see below).

## Outputs (written to the session folder, this is the frozen contract)

- `metric/points_metric.ply` ... the reconstruction after the chosen scale and alignment have been applied, now at true metric scale.
- `metric/colmap/sparse/0/` ... the metric-locked camera model in the camera format the reconstruction host reads (`cameras.bin`, `images.bin`, `points3D.bin`).
- `metric/scale_report.json` ... the full record described below.

## The processing steps

1. Load the front-end reconstruction and poses, and the Stage 1 sensor depth, confidence, and camera path.
2. For each available anchor, estimate the single scale factor that best maps the front-end geometry onto that physical measurement, using a robust fit for the depth anchor and the closed-form similarity fit for the camera-path anchor.
3. Compare the scale factors from the available anchors to each other, and to the scale the front-end model itself claims (its output is already in meters, so its claimed factor is one).
4. If the anchors agree within the configured threshold, apply a consensus scale (the median of the available anchor estimates) and record a pass.
5. If they disagree beyond the threshold, prefer the physical anchors (the sensor depth and the ruler) over the model's own claimed scale, apply that, and record a flag. Physical measurement outranks a learned estimate.
6. Apply the chosen similarity transform to both the point cloud and the camera poses.
7. Optionally refine the alignment between the scaled reconstruction and the point cloud formed from the sensor depth using Iterative Closest Point.
8. Compute the final residual error, meaning the average remaining distance between the scaled reconstruction and the physical measurements, and write all of it to `scale_report.json`.

## The `scale_report.json` schema (frozen fields)

```json
{
  "session_id": "example_2026_07_03_A",
  "generated": "07-03-2026 14:22 local",
  "front_end_model": "DA3NESTED-GIANT-LARGE",
  "front_end_claims_metric": true,
  "anchors": {
    "sensor_depth": {
      "available": true,
      "scale_estimate": 1.021,
      "points_used": 45200,
      "inlier_fraction": 0.93,
      "residual_meters": 0.0011
    },
    "camera_path": {
      "available": true,
      "scale_estimate": 1.008,
      "frames_used": 118,
      "residual_meters": 0.0009
    },
    "physical_ruler": {
      "available": false,
      "scale_estimate": null,
      "known_size_meters": null,
      "measured_size_meters": null
    }
  },
  "agreement": {
    "max_pairwise_scale_difference_percent": 1.3,
    "threshold_percent": 3.0,
    "status": "pass"
  },
  "applied_scale": 1.015,
  "applied_scale_source": "median_of_available_anchors",
  "final_residual_meters": 0.0010,
  "status": "pass",
  "flags": []
}
```

Field meanings, briefly:

- Each anchor block records whether it was available, its estimated scale factor, how much data it used, and its own residual. An unavailable anchor is marked and does not contribute.
- `agreement` records how far apart the anchor estimates were, the threshold they were compared against, and whether they passed.
- `applied_scale` and `applied_scale_source` record the scale actually used and why (consensus, or physical-priority when there was disagreement).
- `final_residual_meters` is the headline accuracy number reported in the paper and compared against the Canfield Vectra reference.
- `status` is `pass` or `flag`. `flags` lists reasons if flagged (for example, `anchors_disagree` or `only_one_anchor_available`).

## Configurable thresholds (from `config/pipeline.yaml`)

- The agreement threshold as a percentage (the default suggestion is three percent), meaning how far apart the anchor scale estimates may be before the session is flagged.
- The minimum inlier fraction for the depth anchor below which the depth anchor is treated as unavailable rather than trusted.
- Whether a flag halts the pipeline or merely annotates the output (the coordinator reads this).

## Command-line signature

```
conda run -n pipeline_stage3_metric python stages/stage3_metric/run.py --session sessions/<session_id> --config config/pipeline.yaml
```

## Design constraints (so the module stays swappable)

- The module must not assume which front end produced the reconstruction. It reads only the frozen file contract, so it works unchanged if Stage 2 is replaced.
- All coordinate handling uses the computer-vision convention, matching the rest of the pipeline. Convert once if any input arrives in another convention, and confirm with the orientation self-test.
- The module reports disagreement rather than hiding it. A flagged session is a useful signal that something in capture or geometry is off, and it should surface, not be averaged away.
