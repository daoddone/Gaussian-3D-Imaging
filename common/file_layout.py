"""Session-folder path helpers.

All files for one capture live under ``sessions/<session_id>/`` with one
subfolder per stage: ``capture/``, ``frontend/``, ``metric/``, ``normals/``,
``output/`` (see ``io_contracts/README.md``). This module centralises those
paths so no stage hard-codes them. Frame numbering is zero-padded to six digits
and shared across every per-frame folder.

numpy-only by the dependency-light rule (numpy is not even needed here, but the
package rule still applies).
"""
from __future__ import annotations

from pathlib import Path


def frame_id(i) -> str:
    """Zero-padded 6-digit frame id, e.g. frame_id(1) -> '000001'."""
    return f"{int(i):06d}"


class SessionLayout:
    """Resolve the canonical paths inside one ``sessions/<session_id>/`` folder."""

    def __init__(self, session_dir):
        self.root = Path(session_dir)

    # -- stage directories --------------------------------------------------
    @property
    def capture(self) -> Path:
        return self.root / "capture"

    @property
    def frontend(self) -> Path:
        return self.root / "frontend"

    @property
    def metric(self) -> Path:
        return self.root / "metric"

    @property
    def normals(self) -> Path:
        return self.root / "normals"

    @property
    def output(self) -> Path:
        return self.root / "output"

    # -- stage 1 (capture) files -------------------------------------------
    @property
    def capture_rgb(self) -> Path:
        return self.capture / "rgb"

    @property
    def capture_depth(self) -> Path:
        return self.capture / "depth"

    @property
    def capture_confidence(self) -> Path:
        return self.capture / "confidence"

    @property
    def capture_intrinsics(self) -> Path:
        return self.capture / "intrinsics.json"

    @property
    def capture_poses(self) -> Path:
        return self.capture / "poses.json"

    @property
    def capture_timestamps(self) -> Path:
        return self.capture / "timestamps.json"

    @property
    def capture_readme(self) -> Path:
        return self.capture / "README"

    # -- stage 2 (frontend) files ------------------------------------------
    @property
    def frontend_poses(self) -> Path:
        return self.frontend / "poses.json"

    @property
    def frontend_intrinsics(self) -> Path:
        return self.frontend / "intrinsics.json"

    @property
    def frontend_depth(self) -> Path:
        return self.frontend / "depth"

    @property
    def frontend_conf(self) -> Path:
        return self.frontend / "conf"

    @property
    def frontend_points(self) -> Path:
        return self.frontend / "points.ply"

    @property
    def frontend_colmap(self) -> Path:
        return self.frontend / "colmap" / "sparse" / "0"

    # -- stage 3 (metric) files --------------------------------------------
    @property
    def metric_points(self) -> Path:
        return self.metric / "points_metric.ply"

    @property
    def metric_colmap(self) -> Path:
        # Two metric layouts exist: legacy Stage-3 `metric/` and the T1/T16 anchor's `metric_sfm/`
        # (session_sfm -> 04_metric_anchor flow, incl. referral videos). Prefer whichever has a model.
        legacy = self.metric / "colmap" / "sparse" / "0"
        if (legacy / "images.bin").exists():
            return legacy
        anchor = self.root / "metric_sfm" / "colmap" / "sparse" / "0"
        if (anchor / "images.bin").exists():
            return anchor
        return legacy

    @property
    def metric_scale_report(self) -> Path:
        return self.metric / "scale_report.json"

    # -- stage 4 (normals) files -------------------------------------------
    @property
    def normals_weight(self) -> Path:
        return self.normals / "normals_weight"

    # -- stage 6 (output) files --------------------------------------------
    @property
    def output_point_cloud(self) -> Path:
        return self.output / "point_cloud.ply"

    @property
    def output_mesh(self) -> Path:
        return self.output / "mesh.ply"

    @property
    def output_renders(self) -> Path:
        return self.output / "renders"

    @property
    def output_provenance(self) -> Path:
        return self.output / "provenance.json"

    # -- frame listing helpers ---------------------------------------------
    @staticmethod
    def list_frames(folder, suffix):
        """Sorted list of frame-id strings present in ``folder`` with ``suffix``.

        e.g. list_frames(capture_depth, '.npy') -> ['000001', '000002', ...].
        """
        folder = Path(folder)
        if not folder.is_dir():
            return []
        ids = []
        for p in folder.iterdir():
            if p.is_file() and p.name.endswith(suffix):
                stem = p.name[: -len(suffix)]
                if stem.isdigit():
                    ids.append(stem)
        return sorted(ids)

    def ensure_stage_dirs(self, stage):
        """Create the output directory tree for a stage. Returns the stage dir."""
        d = getattr(self, stage)
        d.mkdir(parents=True, exist_ok=True)
        return d
