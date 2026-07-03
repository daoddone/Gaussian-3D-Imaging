"""Build the frozen ``scale_report.json`` and decide which scale to apply.

Schema and decision rules are defined in METRIC_CONTRACT.md ("The
`scale_report.json` schema" and "The processing steps"):

  * If the available anchors agree within the configured threshold, apply the
    consensus scale (median of the available anchor estimates) and record a
    pass.
  * If they disagree beyond the threshold, prefer the physical anchors (sensor
    depth and ruler) over the model's own claimed scale, apply that, and record
    a flag. Physical measurement outranks a learned estimate.

Reporting disagreement (rather than silently averaging it away) is a design
requirement: a flagged session is a useful signal that capture or geometry is
off.
"""
from __future__ import annotations

import numpy as np

# names considered "physical ground truth" and thus preferred on disagreement
PHYSICAL_ANCHORS = ("physical_ruler", "sensor_depth")


def decide_scale(anchor_scales, threshold_percent):
    """Choose the applied scale from the available anchor estimates.

    anchor_scales : dict name -> scale_estimate (float) for AVAILABLE anchors
                    only (caller has already dropped unavailable / low-inlier
                    anchors).
    Returns dict: applied_scale, applied_scale_source, status ('pass'|'flag'),
    flags (list), max_pairwise_scale_difference_percent, threshold_percent.
    """
    names = [n for n, v in anchor_scales.items() if v is not None]
    vals = np.array([anchor_scales[n] for n in names], dtype=float)

    result = {
        "threshold_percent": float(threshold_percent),
        "max_pairwise_scale_difference_percent": None,
        "flags": [],
    }

    if len(vals) == 0:
        result.update({
            "applied_scale": 1.0,
            "applied_scale_source": "none",
            "status": "flag",
            "flags": ["no_anchor_available"],
            "max_pairwise_scale_difference_percent": None,
        })
        return result

    if len(vals) == 1:
        only = names[0]
        result.update({
            "applied_scale": float(vals[0]),
            "applied_scale_source": f"single_anchor:{only}",
            "status": "flag",
            "flags": ["only_one_anchor_available"],
            "max_pairwise_scale_difference_percent": 0.0,
        })
        return result

    # pairwise percent differences relative to the pair mean
    max_pct = 0.0
    for i in range(len(vals)):
        for j in range(i + 1, len(vals)):
            a, b = vals[i], vals[j]
            pct = 100.0 * abs(a - b) / ((a + b) / 2.0)
            max_pct = max(max_pct, pct)
    result["max_pairwise_scale_difference_percent"] = float(max_pct)

    if max_pct <= threshold_percent:
        result.update({
            "applied_scale": float(np.median(vals)),
            "applied_scale_source": "median_of_available_anchors",
            "status": "pass",
        })
    else:
        # disagreement: prefer physical anchors over the model's claimed scale
        phys = [anchor_scales[n] for n in names if n in PHYSICAL_ANCHORS]
        if phys:
            result.update({
                "applied_scale": float(np.median(phys)),
                "applied_scale_source": "physical_priority",
                "status": "flag",
                "flags": ["anchors_disagree"],
            })
        else:
            result.update({
                "applied_scale": float(np.median(vals)),
                "applied_scale_source": "median_no_physical_anchor",
                "status": "flag",
                "flags": ["anchors_disagree", "no_physical_anchor"],
            })
    return result


def build_report(session_id, front_end_model, generated, depth, camera, ruler,
                 decision, final_residual_meters):
    """Assemble the full scale_report.json dict per METRIC_CONTRACT.md."""
    report = {
        "session_id": session_id,
        "generated": generated,
        "front_end_model": front_end_model,
        "front_end_claims_metric": True,
        "anchors": {
            "sensor_depth": {
                "available": bool(depth.get("available")),
                "scale_estimate": depth.get("scale_estimate"),
                "points_used": depth.get("points_used", 0),
                "inlier_fraction": depth.get("inlier_fraction", 0.0),
                "residual_meters": depth.get("residual_meters"),
            },
            "camera_path": {
                "available": bool(camera.get("available")),
                "scale_estimate": camera.get("scale_estimate"),
                "frames_used": camera.get("frames_used", 0),
                "residual_meters": camera.get("residual_meters"),
            },
            "physical_ruler": {
                "available": bool(ruler.get("available")),
                "scale_estimate": ruler.get("scale_estimate"),
                "known_size_meters": ruler.get("known_size_meters"),
                "measured_size_meters": ruler.get("measured_size_meters"),
            },
        },
        "agreement": {
            "max_pairwise_scale_difference_percent": decision["max_pairwise_scale_difference_percent"],
            "threshold_percent": decision["threshold_percent"],
            "status": decision["status"],
        },
        "applied_scale": decision["applied_scale"],
        "applied_scale_source": decision["applied_scale_source"],
        "final_residual_meters": final_residual_meters,
        "status": decision["status"],
        "flags": decision["flags"],
    }
    # carry through notes from unavailable anchors for debugging
    for key, blk in (("sensor_depth", depth), ("camera_path", camera), ("physical_ruler", ruler)):
        if "note" in blk:
            report["anchors"][key]["note"] = blk["note"]
    return report
