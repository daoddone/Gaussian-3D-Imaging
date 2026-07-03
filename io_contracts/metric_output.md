# Stage 3 output contract: metric alignment and validation

Path: `sessions/<session_id>/metric/`

## Files

- `points_metric.ply` ...
  The dense point cloud after the chosen scale and alignment are applied, now at true metric
  scale, in meters.

- `colmap/sparse/0/cameras.bin`, `images.bin`, `points3D.bin` ...
  The metric-locked camera model in COLMAP sparse-model format, world-to-camera, for the
  reconstruction host.

- `scale_report.json` ...
  The full record of the scaling decision and the accuracy figure. Its fields are specified in
  the metric-module contract (`stages/stage3_metric/METRIC_CONTRACT.md`). The single most
  important field is `final_residual_meters`, which is the pipeline's headline accuracy number,
  and `status`, which is `pass` or `flag`.
