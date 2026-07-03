# `common/` — dependency-light shared helpers

Small, pure helpers that every stage needs. **Hard rule (Section 3.5):** this
package depends on nothing beyond the Python standard library and `numpy`, so it
installs cleanly into every stage's otherwise-conflicting environment. Anything
heavier than `numpy` belongs inside the stage that needs it, never here.

Install into a stage environment as a local editable package:

```bash
pip install -e ./common
```

After that, `import common.colmap_io` (etc.) works in that environment. When run
straight from the source tree, each stage's `run.py` and every test also add the
repo root to `sys.path`, so imports resolve without installation too.

## Modules

| Module | Purpose |
| --- | --- |
| `conventions.py` | OpenCV coordinate convention; pose-type normalisation (`world_to_camera` ↔ `camera_to_world`); camera centers; rotation↔quaternion (COLMAP `w,x,y,z`); ARKit→OpenCV pose conversion; `poses.json` I/O. |
| `plyio.py` | Minimal PLY point-cloud read/write (ascii + binary little-endian), positions + optional colors/normals. Plain clouds only, not Gaussian-splat `.ply`. |
| `colmap_io.py` | COLMAP sparse-model binary read/write (`cameras.bin`, `images.bin`, `points3D.bin`); `build_pinhole_model` from per-frame K + world-to-camera poses. |
| `file_layout.py` | `SessionLayout` — canonical paths inside `sessions/<session_id>/`; 6-digit `frame_id`; frame listing. |
| `orientation_selftest.py` | The mandatory render-a-known-object test (Section 5): depth increases away from camera, normals point outward, pose/quaternion round-trips. Run when any convention crosses a boundary. |

## Coordinate convention (the load-bearing rule)

OpenCV **everywhere**: camera looks down **+z**, **x** right, **y** down; depth
increases along +z. Every pose file declares its own `pose_type`. Convert
exactly once at a boundary using `conventions.py`, then re-run the orientation
self-test.

```bash
python -m common.orientation_selftest
```
