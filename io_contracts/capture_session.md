# Stage 1 output contract: capture session

Path: `sessions/<session_id>/capture/`

## Files

- `rgb/000001.png` ...
  Color frames at the highest practical resolution, lossless. Record the resolution in the
  folder README.

- `depth/000001.npy` ...
  Metric depth, 32-bit floating point, shape `[height, width]`, units of meters. Pixels with no
  reliable reading are stored as not-a-number (the value "NaN", meaning "no valid measurement
  here"). Stored at the depth sensor's native resolution (roughly 320 by 240 for continuous
  capture); do not upsample it to the color resolution.

- `confidence/000001.png` ...
  A per-pixel validity mask: value 255 where the depth reading is valid, value 0 where it is
  invalid (that is, where the depth is not-a-number). This is a validity mask, not a graded
  confidence. Graded confidence (low, medium, high) is an optional upgrade, not required.

- `intrinsics.json` ...
  The camera's internal optical parameters. Schema:

  ```json
  {
    "convention": "OpenCV",
    "color_resolution": [1920, 1440],
    "depth_resolution": [320, 240],
    "intrinsic_matrix_applies_to": "color",
    "K": [[1450.0, 0.0, 960.0], [0.0, 1450.0, 720.0], [0.0, 0.0, 1.0]],
    "K_per_frame": {
      "000001": [[1450.0, 0.0, 960.0], [0.0, 1450.0, 720.0], [0.0, 0.0, 1.0]],
      "000002": [[1451.3, 0.0, 960.1], [0.0, 1451.3, 719.8], [0.0, 0.0, 1.0]]
    }
  }
  ```

  The matrix `K` is scaled to the color resolution it applies to. Depth pixels are related to
  color pixels by a later stage using the two resolutions.

  `K_per_frame` is **optional**: a `{frame_id: 3x3}` map of the TRUE per-frame device intrinsics
  (they drift within a session from autofocus breathing / OIS). When present, Stage 3 gives each
  image its own COLMAP camera (per-frame K) instead of assuming the single top-level `K` for all
  frames; the single `K` (first frame) is retained so older single-K readers keep working. Frame
  ids match the `rgb/`, `depth/`, `poses.json`, `timestamps.json` keys.

- `poses.json` ...
  Per-frame camera pose from the parallel tracking stream, in meters, already converted to the
  computer-vision convention. Schema:

  ```json
  {
    "convention": "OpenCV",
    "pose_type": "camera_to_world",
    "poses": {
      "000001": { "R": [[1,0,0],[0,1,0],[0,0,1]], "t": [0.0, 0.0, 0.0] }
    }
  }
  ```

  If the tracking stream was not run (the fallback in the capture specification), omit this
  file and state its absence in the folder README, so Stage 3 knows the camera-path anchor is
  unavailable.

- `timestamps.json` ...
  Per-frame timestamps used to align the streams. Schema:

  ```json
  {
    "unit": "seconds",
    "timestamps": { "000001": 0.000, "000002": 0.033 }
  }
  ```

- `README` ...
  States the coordinate convention, the color and depth resolutions, and whether the pose
  stream is present.
