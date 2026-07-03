# Stage 4 output contract: surface-normal prior (optional)

Path: `sessions/<session_id>/normals/`

This stage is optional and may be absent if the normal prior is turned off. If present:

## Files

- `000001.npy` ...
  Per-frame surface-normal map, 32-bit floating point, shape `[height, width, 3]`. Each pixel is
  a unit vector (length one) whose three values lie between negative one and one, expressed in
  the camera frame in the computer-vision convention. Resolution matches the color frames;
  state it in a folder README if it differs.

## Optional files

- `normals_weight/000001.npy` ...
  Per-frame per-pixel trust weight, 32-bit floating point, shape `[height, width]`, values from
  0 to 1. Present only if the confidence-tied weighting is enabled (see the build specification,
  Stage 4). Absent when a fixed uniform weight is used.
