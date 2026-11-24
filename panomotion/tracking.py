"""Persistent spherical tracks across frames."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from . import sphere
from .estimate import MatchConfig, match_batch, luminance


def fibonacci_directions(count: int):
    i = np.arange(count)
    golden = np.pi * (3.0 - np.sqrt(5.0))
    y = 1.0 - 2.0 * (i + 0.5) / count
    r = np.sqrt(1.0 - y * y)
    lon = golden * i
    return np.stack([r * np.cos(lon), y, r * np.sin(lon)], axis=1)


def build_seed_tracks(seq=None, n_background: int = 16):
    """Seeds: the two moving objects, seam-adjacent and polar samples, generic."""
    seeds = []
    if seq is not None:
        for obj in seq.objects:
            cam_dir0 = seq.Rs[0].T @ obj.centers[0]
            seeds.append({
                "name": obj.name, "label": obj.label,
                "dir": sphere.normalize(cam_dir0), "stress": "object",
            })
    for lon_deg, lat_deg in [(176.5, 0), (-176.5, 25), (178.0, -35),
                             (179.0, 55), (-178.5, -60)]:
        seeds.append({
            "name": f"seam {lat_deg:+d}", "label": 0,
            "dir": sphere.lonlat_to_direction(np.deg2rad(lon_deg),
                                               np.deg2rad(lat_deg)),
            "stress": "seam",
        })
    for lon_deg, lat_deg in [(20, 86), (-140, 88), (95, 84)]:
        seeds.append({
            "name": f"polar {lat_deg}", "label": 0,
            "dir": sphere.lonlat_to_direction(np.deg2rad(lon_deg),
                                               np.deg2rad(lat_deg)),
            "stress": "polar",
        })
    for d in fibonacci_directions(n_background):
        seeds.append({"name": "bg", "label": 0, "dir": d, "stress": "grid"})
    return seeds


def _gt_chain(seed_dir, label, seq, n_frames):
    if seq is None:
        return None
    chain = [np.asarray(seed_dir, float)]
    for t in range(n_frames - 1):
        M = seq.background_matrix(t) if label == 0 \
            else seq.object_matrix(t, seq.objects[label - 1])
        chain.append(sphere.rotate_dirs(M, chain[-1]))
    return np.stack(chain)


def compute_tracks(frames, seeds, global_Rs, seq=None,
                   cfg: MatchConfig | None = None):
    """Track every seed through all frame pairs."""
    base_cfg = cfg or MatchConfig()
    # Sparse seeds can afford a finer candidate grid than the dense estimator;
    # a 1.6-2 deg grid can miss a narrow NCC peak and permanently lose a track.
    cfg = replace(base_cfg, search_step_deg=1.0, search_radius_deg=10.0)
    n_frames = len(frames)
    n_tr = len(seeds)
    dirs = np.stack([s["dir"] for s in seeds])

    est = np.zeros((n_tr, n_frames, 3))
    est[:, 0] = dirs
    accepted = np.zeros((n_tr, n_frames - 1), bool)
    inferred = np.zeros((n_tr, n_frames - 1), bool)
    ncc = np.zeros((n_tr, n_frames - 1))
    arc = np.zeros((n_tr, n_frames - 1))
    gt = None
    if seq is not None:
        gt = np.stack([_gt_chain(s["dir"], s["label"], seq, n_frames)
                       for s in seeds])

    for t in range(n_frames - 1):
        lum_a, lum_b = luminance(frames[t]), luminance(frames[t + 1])
        h, w = lum_a.shape
        m = match_batch(lum_a, lum_b, dirs, w, h, cfg)
        quality = (m["ncc"] >= 0.68) & (m["texture"] >= 0.018)
        R = global_Rs[t] if global_Rs is not None else np.eye(3)
        predicted = sphere.rotate_dirs(R, dirs)
        global_err = sphere.geodesic_angle(m["target"], predicted)

        # Forward-backward check for the two local-object tracks, whose motion
        # deliberately disagrees with the dominant camera rotation.
        obj_idx = [k for k, s in enumerate(seeds) if s["label"] > 0]
        fb_ok = np.zeros(n_tr, dtype=bool)
        if obj_idx:
            idx = np.array(obj_idx)
            back = match_batch(lum_b, lum_a, m["target"][idx], w, h, cfg)
            fb = sphere.geodesic_angle(back["target"], dirs[idx])
            fb_ok[idx] = fb < np.deg2rad(0.9)

        for k in range(n_tr):
            ncc[k, t] = m["ncc"][k]
            arc[k, t] = np.rad2deg(m["arc"][k])
            global_consistent = global_err[k] < np.deg2rad(1.2)
            if seeds[k]["label"] == 0:
                verified = quality[k] and global_consistent
            else:
                verified = quality[k] and (fb_ok[k] or global_consistent)
            if verified:
                dirs[k] = m["target"][k]
                accepted[k, t] = True
            elif seeds[k]["label"] > 0 and m["ncc"][k] >= 0.5:
                # Local object tracks may legitimately disagree with the global
                # camera rotation; keep a weak measured follow.
                dirs[k] = m["target"][k]
            else:
                # Background/seam/polar seeds must obey the dominant rotation.
                # A high-NCC but globally inconsistent shift is an aperture or
                # repetitive-texture false match: propagate with R and flag it.
                dirs[k] = predicted[k]
                inferred[k, t] = True
        est[:, t + 1] = dirs

    tracks = []
    for k, s in enumerate(seeds):
        lo, la = sphere.direction_to_lonlat(est[k])
        item = {
            "id": k,
            "name": s["name"],
            "label": int(s["label"]),
            "stress": s["stress"],
            "est_lon": np.rad2deg(sphere.wrap_lon(lo)).round(4).tolist(),
            "est_lat": np.rad2deg(la).round(4).tolist(),
            "accepted": accepted[k].tolist(),
            "inferred": inferred[k].tolist(),
            "ncc": ncc[k].round(3).tolist(),
            "arc_deg": arc[k].round(4).tolist(),
        }
        if gt is not None:
            glo, gla = sphere.direction_to_lonlat(gt[k])
            item["gt_lon"] = np.rad2deg(sphere.wrap_lon(glo)).round(4).tolist()
            item["gt_lat"] = np.rad2deg(gla).round(4).tolist()
            errs = np.rad2deg(sphere.geodesic_angle(est[k, 1:], gt[k, 1:]))
            item["segment_error_deg"] = errs.round(4).tolist()
            item["median_error_deg"] = round(float(np.median(errs)), 4)
            item["max_error_deg"] = round(float(errs.max()), 4)

            # Coordinate-singularity showcase: NAIVE wrapped-lon jump vs the
            # true geodesic step.  Across the seam the naive jump approaches
            # 360 degrees; over a pole it flips by ~180 degrees.
            naive_jump = np.abs(np.diff(glo))
            lon_u = np.unwrap(glo)
            real_lon_step = np.abs(np.diff(lon_u))
            steps = np.rad2deg(sphere.geodesic_angle(gt[k, :-1], gt[k, 1:]))
            j = int(np.argmax(naive_jump))
            item["max_lon_jump_deg"] = round(float(np.rad2deg(naive_jump[j])), 2)
            item["unwrapped_lon_step_deg"] = round(
                float(np.rad2deg(real_lon_step[j])), 3)
            item["arc_at_jump_deg"] = round(float(steps[j]), 3)
            # A longitude jump at high latitude is the pole singularity, not a
            # seam crossing; keep the two badges mutually distinguishable.
            lat_at_jump = float(np.max(np.abs(gla[max(0, j - 1):j + 2])))
            jump_is_adjacency = bool(naive_jump[j] > np.deg2rad(120)
                                     and steps[j] < 15.0)
            item["crosses_pole"] = bool(np.max(np.abs(gla)) > np.deg2rad(85)
                                        and jump_is_adjacency)
            item["crosses_seam"] = bool(jump_is_adjacency
                                        and lat_at_jump < np.deg2rad(70))
        tracks.append(item)
    return tracks
