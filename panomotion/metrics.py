"""Spherical error metrics, seam continuity and polar diagnostics."""

from __future__ import annotations

import numpy as np

from . import sphere


def angular_error_deg(est_dir, gt_dir):
    return np.rad2deg(sphere.geodesic_angle(est_dir, gt_dir))


def region_masks(lon, lat, seam_band_deg: float = 8.0,
                 polar_deg: float = 75.0):
    abs_lon = np.abs(sphere.wrap_lon(lon))
    return {
        "polar": np.abs(lat) >= np.deg2rad(polar_deg),
        "seam": abs_lon >= np.pi - np.deg2rad(seam_band_deg),
        "mid": (np.abs(lat) < np.deg2rad(polar_deg))
        & (abs_lon < np.pi - np.deg2rad(seam_band_deg)),
    }


def error_summary(err_deg, mask=None):
    e = np.asarray(err_deg, float)
    if mask is not None:
        e = e[np.asarray(mask, bool)]
    if e.size == 0:
        return {"n": 0, "median": None, "mean": None, "p95": None, "max": None}
    return {
        "n": int(e.size),
        "median": round(float(np.median(e)), 4),
        "mean": round(float(e.mean()), 4),
        "p95": round(float(np.percentile(e, 95)), 4),
        "max": round(float(e.max()), 4),
    }


def seam_continuity(src, target, omega, credible, cols: int, rows: int,
                    gt_target=None):
    """Per-row continuity across the longitude seam.

    ``src/target`` are flattened ``(rows*cols, 3)`` arrays; ``omega`` may be
    flattened or a rows-x-cols grid.  All comparisons use world-frame vectors
    and geodesic arc lengths, never raw longitude subtraction.
    """
    src = np.asarray(src).reshape(rows, cols, 3)
    target = np.asarray(target).reshape(rows, cols, 3)
    omega = np.asarray(omega)
    if omega.ndim == 2:
        omega = omega.reshape(rows * cols, 3)
    omega = omega.reshape(rows, cols, 3)
    credible = np.asarray(credible, bool).reshape(rows, cols)
    gt = None if gt_target is None else np.asarray(gt_target).reshape(rows, cols, 3)

    table = []
    for j in range(rows):
        a = cols - 1  # +pi side
        b = 0         # -pi side
        sa, sb = src[j, a], src[j, b]
        ta, tb = target[j, a], target[j, b]
        oa, ob = omega[j, a], omega[j, b]
        src_gap = float(sphere.geodesic_angle(sa, sb))
        tgt_gap = float(sphere.geodesic_angle(ta, tb))
        omega_diff = float(np.linalg.norm(oa - ob))
        # Move neighbour b with a's motion and compare with b's measured target.
        xpred_b = sphere.rodrigues(oa, sb)
        cross_dis = float(sphere.geodesic_angle(xpred_b, tb))
        xpred_a = sphere.rodrigues(ob, sa)
        cross_dis = 0.5 * (cross_dis
                          + float(sphere.geodesic_angle(xpred_a, ta)))
        lon_r, _ = sphere.direction_to_lonlat(sa)
        lon_l, _ = sphere.direction_to_lonlat(sb)
        naive_gap = float(abs(lon_l - lon_r))
        both = bool(credible[j, a] and credible[j, b])
        row = {
            "row": int(j),
            "lat_deg": round(float(np.rad2deg(sphere.direction_to_lonlat(sa)[1])), 3),
            "src_gap_deg": round(np.rad2deg(src_gap), 4),
            "target_gap_deg": round(np.rad2deg(tgt_gap), 4),
            "naive_lon_gap_deg": round(np.rad2deg(naive_gap), 3),
            "omega_diff_deg": round(np.rad2deg(omega_diff), 4),
            "cross_prediction_deg": round(np.rad2deg(cross_dis), 4),
            "both_credible": both,
        }
        if gt is not None:
            row["err_a_deg"] = round(float(np.rad2deg(
                sphere.geodesic_angle(ta, gt[j, a]))), 4)
            row["err_b_deg"] = round(float(np.rad2deg(
                sphere.geodesic_angle(tb, gt[j, b]))), 4)
        table.append(row)

    cred_rows = [r for r in table if r["both_credible"]]
    if cred_rows:
        agg = {
            "rows_both_credible": len(cred_rows),
            "median_omega_diff_deg": round(float(np.median(
                [r["omega_diff_deg"] for r in cred_rows])), 4),
            "max_cross_prediction_deg": round(float(np.max(
                [r["cross_prediction_deg"] for r in cred_rows])), 4),
            "median_cross_prediction_deg": round(float(np.median(
                [r["cross_prediction_deg"] for r in cred_rows])), 4),
        }
    else:
        agg = {"rows_both_credible": 0}
    return table, agg


def field_neighbor_discontinuity(omega_grid, credible):
    """World-frame omega differences between adjacent credible grid nodes."""
    g = np.asarray(omega_grid)
    rows, cols = g.shape[:2]
    cred = np.asarray(credible, bool).reshape(rows, cols)
    vals = []
    spots = []
    # Horizontal (wrap at seam, which is exactly what must stay continuous).
    for j in range(rows):
        for i in range(cols):
            i2 = (i + 1) % cols
            if cred[j, i] and cred[j, i2]:
                d = float(np.linalg.norm(g[j, i] - g[j, i2]))
                vals.append(d)
                if d > np.deg2rad(1.5):
                    spots.append({"j": j, "i": i, "kind": "horizontal",
                                 "omega_diff_deg": round(np.rad2deg(d), 3)})
    # Vertical.
    for j in range(rows - 1):
        for i in range(cols):
            if cred[j, i] and cred[j + 1, i]:
                d = float(np.linalg.norm(g[j, i] - g[j + 1, i]))
                vals.append(d)
    if not vals:
        return {"median_deg": None, "p95_deg": None, "max_deg": None}, spots
    v = np.array(vals)
    return {
        "median_deg": round(float(np.rad2deg(np.median(v))), 4),
        "p95_deg": round(float(np.rad2deg(np.percentile(v, 95))), 4),
        "max_deg": round(float(np.rad2deg(v.max())), 4),
    }, spots


def build_anomalies(pair_index: int, seam_agg: dict, polar_stats: dict,
                    support_fraction: float, reproj: dict | None,
                    gt_available: bool):
    """Rule-based, actually-computed anomaly flags surfaced by the GUI."""
    out = []
    if seam_agg.get("rows_both_credible", 0) >= 4:
        mx = seam_agg["max_cross_prediction_deg"]
        if mx is not None and mx > 1.5:
            out.append({
                "level": "warning", "type": "seam_jump", "pair": pair_index,
                "message": f"经度接缝两侧最大相互预测偏差 {mx:.2f}°，连续性异常",
                "value_deg": mx,
            })
    pf = polar_stats.get("credible_fraction")
    if pf is not None and pf < 0.30:
        out.append({
            "level": "info", "type": "polar_low_support", "pair": pair_index,
            "message": f"极区可信运动占比仅 {pf * 100:.0f}%（已切换局部坐标基处理）",
            "value": round(pf, 3),
        })
    if gt_available and polar_stats.get("error"):
        med = polar_stats["error"].get("median")
        if med is not None and med > 1.5:
            out.append({
                "level": "warning", "type": "polar_artifact", "pair": pair_index,
                "message": f"极区角度误差中位 {med:.2f}°，存在极区伪影",
                "value_deg": med,
            })
    if support_fraction < 0.30:
        out.append({
            "level": "info", "type": "low_support", "pair": pair_index,
            "message": f"全图仅 {support_fraction * 100:.0f}% 网格获得受支持匹配",
            "value": round(support_fraction, 3),
        })
    if reproj is not None and reproj.get("psnr") is not None \
            and np.isfinite(reproj["psnr"]) and reproj["psnr"] < 27.0:
        out.append({
            "level": "warning", "type": "reprojection", "pair": pair_index,
            "message": f"重投影 PSNR {reproj['psnr']:.1f} dB 偏低",
            "value": round(reproj["psnr"], 2),
        })
    return out
