# Stage 6 output contract: final deliverables

Path: `sessions/<session_id>/output/`

## Files

- `point_cloud.ply` ...
  The refined Gaussian splat: the optimized, surface-aligned Gaussians that form the
  photorealistic, view-dependent radiance field. This is a Gaussian-splat `.ply`, meaning it
  carries per-Gaussian attributes (position, scale, orientation, opacity, and view-dependent
  color coefficients), not a plain point cloud. In meters.

- `mesh.ply` (and optionally `mesh.obj`) ...
  The metric surface mesh extracted during optimization, in meters.

- `renders/` ...
  Preview images rendered from the reconstruction, for quality inspection.

- `provenance.json` ...
  A record for reproducibility and the eventual regulatory pathway. Schema:

  ```json
  {
    "session_id": "example_2026_07_03_A",
    "generated": "07-03-2026 14:22 local",
    "components": {
      "front_end": { "name": "DA3NESTED-GIANT-LARGE", "version": "..." },
      "normal_prior": { "name": "StableNormal", "version": "...", "enabled": true },
      "reconstruction_host": { "name": "MILo", "version": "...", "densification": "mini-splatting2" }
    },
    "scale_report": { "final_residual_meters": 0.0010, "status": "pass" },
    "capture": { "frame_count": 48, "video_seconds": 20, "device": "..." }
  }
  ```
