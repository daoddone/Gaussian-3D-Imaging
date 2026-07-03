# Input and output contracts

Every stage reads and writes files following these contracts. The files on disk are the
interface between stages. Conventions common to all contracts are listed here once; the
per-stage files in this folder assume them.

## Conventions common to all stages

- Coordinate convention: the computer-vision convention (often called OpenCV), in which the
  camera looks down its positive z axis, with x pointing right and y pointing down.
- Camera poses: every pose file states its own `pose_type`, either `camera_to_world` or
  `world_to_camera`, so there is never any ambiguity. Convert once at a boundary if needed,
  using the shared helper in `common/conventions.py`.
- Units: all depth and geometry are in meters, stored as 32-bit floating point.
- Per-frame numeric arrays: stored as `.npy` (lossless). Color images: stored as `.png`
  (lossless).
- Frame numbering: zero-padded to six digits (for example `000001`), shared across every
  per-frame folder, so the same index refers to the same instant everywhere.
- Every session folder contains a short `README` stating the convention in use and the color
  and depth resolutions.
- The orientation self-test (a rendered known object whose normals must point outward and
  whose depth must increase away from the camera) must pass before any output is trusted.

## Session working directory

All files for one capture live under `sessions/<session_id>/`, with one subfolder per stage:
`capture/`, `frontend/`, `metric/`, `normals/`, `output/`.

## The per-stage contracts in this folder

- `capture_session.md` ... Stage 1 output.
- `frontend_output.md` ... Stage 2 output.
- `metric_output.md` ... Stage 3 output.
- `normals_output.md` ... Stage 4 output (optional stage).
- `reconstruction_output.md` ... Stage 6 output.
