"""End-to-end spherical motion pipeline: estimate -> validate -> metrics ->
reprojection -> artifacts/JSON/videos consumed by the graphical player.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import metrics as M
from . import sphere
from .baseline import estimate_pair_planar
from .estimate import MatchConfig, estimate_pair, fit_global_rotation, match_batch, luminance
from . import media
from .tracking import build_seed_tracks, compute_tracks
from .warp import (fill_missing_omega, photometric_error, reproject_frame,
                   to_float)


@dataclass
class PipelineConfig:
    name: str = "demo"
    cols: int = 64
    rows: int = 32
    n_frames: int = 12
    width: int = 960
    height: int = 480
    run_baseline: bool = True
    fps: float = 5.0
    ncc_gate: float = 0.78
    texture_gate: float = 0.028


def _json_default(o):
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    raise TypeError(type(o))


def write_json(obj, path):
    Path(path).write_text(json.dumps(obj, default=_json_default), encoding="utf-8")


def node_labels(seq, src, t, width, height):
    u, v = sphere.direction_to_pixel(src, width, height)
    ui = np.clip(np.rint(u).astype(int) % width, 0, width - 1)
    vi = np.clip(np.rint(v).astype(int), 0, height - 1)
    return seq.labels[t][vi, ui]


def metric_patch_purity(seq, src, t, radius_deg: float = 3.2):
    """True angular-disc label homogeneity (UV boxes are distorted near poles)."""
    b1, b2, _ = sphere.safe_tangent_basis(src)
    r = np.deg2rad(radius_deg)
    angles = np.linspace(0, 2 * np.pi, 10, endpoint=False)
    same = np.ones(len(src), dtype=bool)
    center_lab = node_labels(seq, src, t, seq.width, seq.height)
    for a in angles:
        sample = sphere.exp_map(src, r * np.cos(a), r * np.sin(a), b1, b2)
        lab = node_labels(seq, sample, t, seq.width, seq.height)
        same &= lab == center_lab
    return same


def classify_with_backward(res, lum_a, lum_b, pcfg: PipelineConfig,
                           mcfg: MatchConfig):
    src = res["src"]
    target = res["target"]
    ncc = res["ncc"]
    tex = res["texture"]
    strict = (ncc >= pcfg.ncc_gate) & (tex >= pcfg.texture_gate)

    R, _, gres = fit_global_rotation(src, target, confidence=np.clip(ncc, 0, 1),
                                     support=strict)
    gpred = sphere.rotate_dirs(R, src)
    gres = sphere.geodesic_angle(gpred, target)
    med = float(np.median(gres[strict])) if strict.any() else 1.0
    gthr = max(np.deg2rad(0.8), 4.0 * med)
    is_global = strict & (gres < gthr)

    cls = np.zeros(len(src), np.uint8)
    cls[is_global] = 1

    cand_idx = np.where(strict & ~is_global)[0]
    fb_err = np.full(len(src), np.nan)
    if len(cand_idx):
        back = match_batch(lum_b, lum_a, target[cand_idx],
                           res["width"], res["height"], mcfg)
        fb_e = sphere.geodesic_angle(back["target"], src[cand_idx])
        fb_err[cand_idx] = np.rad2deg(fb_e)
        ok = (fb_e < np.deg2rad(0.8)) & (back["ncc"] >= 0.55)
        cls[cand_idx[ok]] = 2

    credible = cls > 0
    return {
        "cls": cls, "credible": credible, "R": R,
        "global_omega": sphere.rotation_matrix_to_vector(R),
        "global_residual": gres, "global_threshold_deg": np.rad2deg(gthr),
        "fb_error_deg": fb_err,
    }


def _panel(img, title, w=470, h=236):
    from PIL import Image, ImageDraw
    im = Image.fromarray(img).resize((w, h), Image.BILINEAR).convert("RGB")
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, w, 24], fill=(12, 16, 26))
    d.text((8, 5), title, fill=(235, 240, 255), font=media.get_font(15))
    return np.array(im)


def run_pipeline(frames, run_dir, seq=None, pcfg: PipelineConfig | None = None,
                 progress=None):
    pcfg = pcfg or PipelineConfig()
    run_dir = Path(run_dir)
    for sub in ["frames", "reproj", "error", "data", "videos"]:
        (run_dir / sub).mkdir(parents=True, exist_ok=True)

    def emit(stage, frac, msg):
        if progress:
            progress(stage, frac, msg)

    n = len(frames)
    h, w = frames[0].shape[:2]
    mcfg = MatchConfig()
    has_gt = seq is not None

    # Persist playback frames.
    for t, f in enumerate(frames):
        media.save_jpg(f, run_dir / "frames" / f"frame_{t:03d}.jpg", quality=88)

    global_Rs = []
    pair_reports = []
    all_anomalies = []
    planar_rows = []
    composite_frames = []
    gt_motion_frames = []
    est_motion_frames = []
    reproj_psnrs, gt_psnrs = [], []

    for t in range(n - 1):
        t_start = time.time()
        emit("estimate", t / (n - 1), f"分析帧对 {t + 1}/{n - 1}（球面切平面匹配）")
        res = estimate_pair(frames[t], frames[t + 1], mcfg,
                            cols=pcfg.cols, rows=pcfg.rows)
        lum_a = luminance(frames[t])
        lum_b = luminance(frames[t + 1])
        verdict = classify_with_backward(res, lum_a, lum_b, pcfg, mcfg)
        cls = verdict["cls"]
        credible = verdict["credible"]
        R = verdict["R"]
        global_Rs.append(R)

        src = res["src"]
        target = res["target"]
        omega_meas = res["omega"]
        labels = node_labels(seq, src, t, w, h) if has_gt else None
        gt_dirs = seq.gt_targets(src, t, labels=labels) if has_gt else None
        err = M.angular_error_deg(target, gt_dirs) if has_gt else None

        # Final regularised world-frame omega grid.
        omega_grid = omega_meas.copy()
        cls_grid = cls.reshape(pcfg.rows, pcfg.cols)
        omega_grid[cls_grid == 1] = verdict["global_omega"]
        credible_grid = credible.reshape(pcfg.rows, pcfg.cols)
        omega_filled = fill_missing_omega(
            omega_grid, credible_grid, verdict["global_omega"], passes=30)

        warped = reproject_frame(frames[t], omega_filled)
        rp_metrics, diff = photometric_error(frames[t + 1], warped)
        reproj_psnrs.append(rp_metrics["psnr"])
        media.save_jpg(warped, run_dir / "reproj" / f"reproj_{t:03d}.jpg", 88)
        heat = media.error_heatmap(diff, scale=5.0)
        media.save_jpg(heat, run_dir / "error" / f"error_{t:03d}.jpg", 85)

        gt_reproj_psnr = None
        if has_gt:
            gt_omega = sphere.rotation_vector_between(src, gt_dirs) \
                .reshape(pcfg.rows, pcfg.cols, 3)
            gt_warped = reproject_frame(frames[t], gt_omega)
            gt_rp, _ = photometric_error(frames[t + 1], gt_warped)
            gt_reproj_psnr = gt_rp["psnr"]
            gt_psnrs.append(gt_reproj_psnr)

        # Regions.
        lon, lat = sphere.direction_to_lonlat(src)
        regions = M.region_masks(lon, lat)
        pure = metric_patch_purity(seq, src, t) if has_gt else np.ones(len(src), bool)
        err_stats = {}
        if has_gt:
            err_stats["all_credible"] = M.error_summary(err, credible)
            err_stats["pure_credible"] = M.error_summary(err, credible & pure)
            for rname in ["polar", "seam", "mid"]:
                err_stats[rname] = M.error_summary(
                    err, credible & pure & regions[rname])
            err_stats["global"] = M.error_summary(err, (cls == 1) & pure)
            err_stats["local_objects"] = M.error_summary(err, cls == 2)
            for lab, nm in [(1, "polar_crosser"), (2, "seam_crosser")]:
                err_stats[nm] = M.error_summary(err, labels == lab)

        polar_stats = {
            "nodes": int(regions["polar"].sum()),
            "credible_fraction": round(float(
                credible[regions["polar"]].mean()), 4)
            if regions["polar"].any() else None,
            "basis_switched_nodes": int(res["polar_basis"].sum()),
        }
        if has_gt:
            polar_stats["error"] = M.error_summary(
                err, credible & pure & regions["polar"])

        seam_table, seam_agg = M.seam_continuity(
            src, target, omega_meas.reshape(-1, 3), credible,
            pcfg.cols, pcfg.rows, gt_target=gt_dirs)
        neigh, _spots = M.field_neighbor_discontinuity(omega_meas, credible)

        # Naive planar baseline for quantitative contrast.
        planar_stats = None
        if pcfg.run_baseline and has_gt:
            pl = estimate_pair_planar(frames[t], frames[t + 1])
            pl_labels = node_labels(seq, pl["src"], t, w, h)
            pl_gt = seq.gt_targets(pl["src"], t, labels=pl_labels)
            pl_err = M.angular_error_deg(pl["target"], pl_gt)
            pl_lon, pl_lat = sphere.direction_to_lonlat(pl["src"])
            pl_reg = M.region_masks(pl_lon, pl_lat)
            planar_stats = {
                "all_median": round(float(np.median(pl_err)), 3),
                "mid_median": round(float(np.median(pl_err[pl_reg["mid"]])), 3),
                "polar_median": round(float(np.median(pl_err[pl_reg["polar"]])), 3),
                "seam_median": round(float(np.median(pl_err[pl_reg["seam"]])), 3),
            }
            planar_rows.append(planar_stats)

        support_fraction = float(credible.mean())
        anomalies = M.build_anomalies(
            t, seam_agg, polar_stats, support_fraction, rp_metrics, has_gt)
        all_anomalies.extend(anomalies)

        # -- per-pair JSON for the player ----------------------------------
        tlon, tlat = sphere.direction_to_lonlat(target)
        pair_json = {
            "pair": t,
            "cols": pcfg.cols, "rows": pcfg.rows, "width": w, "height": h,
            "target_lon": np.round(np.rad2deg(tlon), 3),
            "target_lat": np.round(np.rad2deg(tlat), 3),
            "arc_deg": np.round(np.rad2deg(res["arc"]), 3),
            "ncc": np.round(res["ncc"], 3),
            "texture": np.round(res["texture"], 4),
            "cls": cls.tolist(),
            "pure": pure.astype(int).tolist(),
            "seam": seam_table,
            "seam_agg": seam_agg,
            "neighbor": neigh,
            "global_rotation_deg": round(float(np.linalg.norm(
                verdict["global_omega"]) * 180 / np.pi), 4),
            "global_threshold_deg": verdict["global_threshold_deg"],
            "support_fraction": round(support_fraction, 4),
            "reprojection": {k: (round(v, 3) if np.isfinite(v) else None)
                             for k, v in rp_metrics.items()},
            "anomalies": anomalies,
        }
        if has_gt:
            glon, glat = sphere.direction_to_lonlat(gt_dirs)
            pair_json["gt_lon"] = np.round(np.rad2deg(glon), 3)
            pair_json["gt_lat"] = np.round(np.rad2deg(glat), 3)
            pair_json["error_deg"] = np.round(err, 3)
            pair_json["error_stats"] = err_stats
            pair_json["gt_reproj_psnr"] = round(gt_reproj_psnr, 2)
        write_json(pair_json, run_dir / "data" / f"pair_{t:03d}.json")

        pair_reports.append({
            "pair": t,
            "support_fraction": pair_json["support_fraction"],
            "global_rotation_deg": pair_json["global_rotation_deg"],
            "reprojection_psnr": pair_json["reprojection"]["psnr"],
            "gt_reproj_psnr": pair_json.get("gt_reproj_psnr"),
            "seam": seam_agg,
            "polar_credible_fraction": polar_stats["credible_fraction"],
            "polar_error_median": (polar_stats.get("error") or {}).get("median"),
            "error_credible_median": (err_stats.get("pure_credible") or {}).get("median"),
            "error_credible_p95": (err_stats.get("pure_credible") or {}).get("p95"),
            "planar": planar_stats,
            "anomaly_count": len(anomalies),
            "seconds": round(time.time() - t_start, 1),
        })

        # -- videos ----------------------------------------------------------
        composite_frames.append(_build_check_panel(
            frames[t], warped, frames[t + 1], heat, t, rp_metrics))
        gt_motion_frames.append(_annotate_motion(
            frames[t], src, gt_dirs if has_gt else target, None, t,
            "真实球面运动 GT（测地线箭头）", seam=True, cols=pcfg.cols)
            if has_gt else None)
        est_motion_frames.append(_annotate_motion(
            frames[t], src, target, cls, t,
            "估计球面运动（蓝=相机旋转 橙=局部物体）", seam=True,
            cols=pcfg.cols))

        emit("pair", (t + 1) / (n - 1),
             f"完成帧对 {t + 1}/{n - 1}，可信占比 {support_fraction * 100:.0f}%")

    # Tracks.
    emit("tracks", 1.0, "计算球面轨迹")
    seeds = build_seed_tracks(seq)
    tracks = compute_tracks(frames, seeds, global_Rs, seq, mcfg)
    write_json(tracks, run_dir / "tracks.json")

    # Videos.
    emit("video", 1.0, "写出视频")
    media.write_mp4(frames, run_dir / "videos" / "original.mp4", pcfg.fps)
    media.write_mp4([f for f in est_motion_frames if f is not None],
                    run_dir / "videos" / "estimated_spherical_motion.mp4", pcfg.fps)
    media.write_mp4(composite_frames,
                    run_dir / "videos" / "reprojection_check.mp4", pcfg.fps)
    if has_gt:
        media.write_mp4([f for f in gt_motion_frames if f is not None],
                        run_dir / "videos" / "ground_truth_motion.mp4", pcfg.fps)
    try:
        small_gif = [Image_resize_width(f, 720) for f in composite_frames]
        media.write_gif(small_gif, run_dir / "videos" / "reprojection_check.gif",
                        fps=pcfg.fps)
    except Exception as e:  # GIF is a convenience fallback.
        print("gif skipped:", e)

    # Aggregate report.
    def agg(key_path_rows):
        vals = [r for r in key_path_rows if r is not None]
        return vals

    planar_summary = None
    if planar_rows:
        planar_summary = {
            k: round(float(np.mean([r[k] for r in planar_rows])), 3)
            for k in planar_rows[0]
        }
    err_med = [r["error_credible_median"] for r in pair_reports
               if r["error_credible_median"] is not None]
    err_p95 = [r["error_credible_p95"] for r in pair_reports
               if r["error_credible_p95"] is not None]
    report = {
        "run": pcfg.name,
        "has_ground_truth": has_gt,
        "n_frames": n,
        "width": w, "height": h,
        "cols": pcfg.cols, "rows": pcfg.rows,
        "fps": pcfg.fps,
        "summary": {
            "mean_credible_fraction": round(float(np.mean(
                [r["support_fraction"] for r in pair_reports])), 4),
            "mean_angle_error_median_deg": round(float(np.mean(err_med)), 4)
            if err_med else None,
            "worst_pair_p95_error_deg": round(float(np.max(err_p95)), 4)
            if err_p95 else None,
            "mean_reprojection_psnr": round(float(np.mean(reproj_psnrs)), 2),
            "mean_gt_reproj_psnr": round(float(np.mean(gt_psnrs)), 2)
            if gt_psnrs else None,
            "planar_baseline_mean": planar_summary,
            "total_anomalies": len(all_anomalies),
        },
        "pairs": pair_reports,
        "anomalies": all_anomalies,
        "track_count": len(tracks),
    }
    write_json(report, run_dir / "report.json")
    meta = {
        "name": pcfg.name, "width": w, "height": h,
        "n_frames": n, "cols": pcfg.cols, "rows": pcfg.rows,
        "fps": pcfg.fps, "has_ground_truth": has_gt,
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "graphical_entry": "python -m panomotion gui",
    }
    write_json(meta, run_dir / "meta.json")
    emit("done", 1.0, f"完成：{run_dir}")
    return report


def Image_resize_width(img, width):
    from PIL import Image
    im = Image.fromarray(img)
    h = int(im.height * width / im.width)
    return np.array(im.resize((width, h), Image.BILINEAR))


def _build_check_panel(frame_t, warped, frame_t1, heat, t, metrics):
    p1 = _panel(frame_t, f"t={t} 原始帧 A")
    p2 = _panel(warped, "A 按估计球面运动重投影到 B")
    p3 = _panel(frame_t1, f"t={t + 1} 真实下一帧 B")
    p4 = _panel(heat, f"放大残差  MAE={metrics['mae']:.3f} PSNR={metrics['psnr']:.1f}dB")
    return np.concatenate([p1, p2, p3, p4], axis=1)


def _annotate_motion(frame, src, tgt, cls, pair_idx, title, seam=True,
                     cols=64):
    from PIL import Image, ImageDraw
    canvas = Image.fromarray(frame.copy()).convert("RGB")
    w, h = canvas.size
    overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    od = ImageDraw.Draw(overlay)
    # Seam markers and polar caps.
    if seam:
        for x in (1, w - 2):
            od.line([(x, 0), (x, h)], fill=(255, 60, 220, 230), width=3)
        for lat_deg in (75, -75):
            v = (np.pi / 2 - np.deg2rad(lat_deg)) / np.pi * h - 0.5
            for x in range(0, w, 18):
                od.line([(x, v), (x + 9, v)], fill=(255, 255, 255, 150), width=1)
    canvas = Image.alpha_composite(canvas.convert("RGBA"), overlay).convert("RGB")

    for k in range(len(src)):
        i = k % cols
        j = k // cols
        if i % 2 or j % 2:
            continue
        if cls is not None:
            if cls[k] == 1:
                color = (90, 200, 255)
            elif cls[k] == 2:
                color = (255, 176, 40)
            else:
                continue
        else:
            color = (120, 255, 170)
        media.draw_spherical_arrow(canvas, src[k], tgt[k], color=color,
                                   width_px=1, head=5)
    d = ImageDraw.Draw(canvas)
    d.rectangle([0, 0, w, 26], fill=(12, 16, 26))
    d.text((10, 5), f"{title}  ·  pair {pair_idx}",
           fill=(235, 240, 255), font=media.get_font(16))
    # Seam labels.
    d.text((6, h - 20), "经度接缝 -180°", fill=(255, 120, 230),
           font=media.get_font(14))
    tw = d.textlength("+180° 接缝", font=media.get_font(14))
    d.text((w - tw - 6, h - 20), "+180° 接缝", fill=(255, 120, 230),
           font=media.get_font(14))
    return np.array(canvas)
