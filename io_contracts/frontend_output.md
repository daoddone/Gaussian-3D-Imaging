# Stage 2 output contract: front end (poses and dense geometry)

Path: `sessions/<session_id>/frontend/`

## Files

- `poses.json` ...
  Per-frame camera pose from the front-end model. Schema:

  ```json
  {
    "convention": "OpenCV",
    "pose_type": "world_to_camera",
    "poses": {
      "000001": { "R": [[1,0,0],[0,1,0],[0,0,1]], "t": [0.0, 0.0, 0.0] }
    }
  }
  ```

- `intrinsics.json` ...
  Per-frame camera intrinsics from the front-end model (the model may report a slightly
  different focal length per frame). Schema:

  ```json
  {
    "convention": "OpenCV",
    "resolution": [512, 384],
    "K": { "000001": [[400.0,0.0,256.0],[0.0,400.0,192.0],[0.0,0.0,1.0]] }
  }
  ```

- `depth/000001.npy` ...
  The front-end model's own predicted depth, 32-bit floating point, shape `[height, width]`,
  meters.

- `conf/000001.npy` ...
  The front-end model's own confidence, 32-bit floating point, shape `[height, width]`, values
  from 0 to 1. This is a genuine graded confidence from the model, distinct from the Stage 1
  validity mask.

- `points.ply` ...
  The fused dense point cloud, in meters. A plain point cloud (positions, and color if
  available), not a Gaussian-splat file.

- `colmap/sparse/0/cameras.bin`, `images.bin`, `points3D.bin` ...
  The same camera data written in the COLMAP sparse-model format that the reconstruction host
  reads. Poses are world-to-camera; the camera model is a pinhole model built from the
  intrinsics. Populate `points3D.bin` from the dense cloud, or leave it minimal if the host is
  given the initialization point cloud separately.

## Note on scale

The front-end model reports geometry already in meters. Stage 3 verifies this against physical
measurements and corrects it if needed; do not assume it is exact.
