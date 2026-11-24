"""Spherical motion estimation.

Every sample is a unit direction on the sphere.  For each grid node we open a
local tangent plane (switching to a non-degenerate basis near poles), extract
a geodesic patch, search candidate *directions* by NCC over geodesic offsets
and refine with batched Lucas-Kanade alignment.  Matches are returned as
direction pairs and world-frame rotation vectors: an actual directional
correspondence rather than an unexplained 2-D pixel shift.

The heavy matching is fully batched over nodes so a dense panorama grid runs
with a small number of vectorised NumPy operations.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import sphere
from .warp import sample_equirect, to_float


def luminance(img: np.ndarray) -> np.ndarray:
    f = to_float(img)
    if f.ndim == 2:
        return f
    return 0.2126 * f[..., 0] + 0.7152 * f[..., 1] + 0.0722 * f[..., 2]


@dataclass
class MatchConfig:
    patch: int = 21
    angular_step: float | None = None  # radians per tangent sample
    search_radius_deg: float = 8.0
    search_step_deg: float = 1.6

    @property
    def offsets(self):
        r = np.deg2rad(self.search_radius_deg)
        s = np.deg2rad(self.search_step_deg)
        vals = np.arange(-r, r + 0.5 * s, s)
        ox, oy = np.meshgrid(vals, vals)
        return np.stack([ox.ravel(), oy.ravel()], axis=1)


def tangent_plane_coords(patch: int, angular_step: float):
    k = (np.arange(patch) - (patch - 1) / 2.0) * angular_step
    x, y = np.meshgrid(k, k)
    return x, y


def _batched_dirs(c, b1, b2, x, y, sx, sy):
    """Exp-map directions for N centers and PxP plane coords plus shifts.

    c,b1,b2 are ``(N,3)``; x,y are ``(P,P)``; sx,sy are scalars or ``(N,)``.
    Returns ``(N,P,P,3)``.
    """
    if np.isscalar(sx) or (isinstance(sx, np.ndarray) and sx.ndim == 0):
        tx = x[None, ...] + float(sx)
        ty = y[None, ...] + float(sy)
    else:
        sx = np.asarray(sx, dtype=np.float64).reshape(-1, 1, 1)
        sy = np.asarray(sy, dtype=np.float64).reshape(-1, 1, 1)
        tx = x[None, ...] + sx
        ty = y[None, ...] + sy
    theta = np.hypot(tx, ty)[..., None]
    move = (b1[:, None, None, :] * tx[..., None]
            + b2[:, None, None, :] * ty[..., None])
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    out = c[:, None, None, :] * cos_t + move * sin_t / np.maximum(theta, 1e-12)
    small = theta < 1e-9
    small_out = sphere.normalize(c[:, None, None, :] + move)
    out = np.where(small, small_out, sphere.normalize(out))
    return out


def _ncc_all(template_c, template_norm, patches):
    """NCC between N templates and N candidate patches -> (N,)."""
    n = patches.shape[0]
    flat = patches.reshape(n, -1)
    centered = flat - flat.mean(axis=1, keepdims=True)
    num = np.einsum("ij,ij->i", centered, template_c.reshape(n, -1))
    den_i = np.sqrt((centered * centered).sum(axis=1))
    return num / np.maximum(template_norm * den_i, 1e-12)


def match_batch(lum_a, lum_b, centers, width: int, height: int,
                cfg: MatchConfig | None = None, progress=None):
    """Match one tangent patch per center; returns arrays keyed by center."""
    cfg = cfg or MatchConfig()
    centers = np.asarray(centers, dtype=np.float64).reshape(-1, 3)
    n = len(centers)
    step = cfg.angular_step or (sphere.TWO_PI / width)
    x, y = tangent_plane_coords(cfg.patch, step)
    b1, b2, polar = sphere.safe_tangent_basis(centers)

    base_dirs = _batched_dirs(centers, b1, b2, x, y, 0.0, 0.0)
    uu, vv = sphere.direction_to_pixel(base_dirs, width, height)
    template = sample_equirect(lum_a, uu, vv)  # (N,P,P)
    t_mean = template.mean(axis=(1, 2), keepdims=True)
    template_c = template - t_mean
    template_norm = np.sqrt((template_c ** 2).sum(axis=(1, 2)))
    template_std = template.reshape(n, -1).std(axis=1)

    offsets = cfg.offsets  # (K,2)
    k_count = len(offsets)
    scores = np.empty((n, k_count))
    for k in range(k_count):
        dirs = _batched_dirs(centers, b1, b2, x, y,
                             float(offsets[k, 0]), float(offsets[k, 1]))
        uu, vv = sphere.direction_to_pixel(dirs, width, height)
        patches = sample_equirect(lum_b, uu, vv)
        scores[:, k] = _ncc_all(template_c, template_norm, patches)
        if progress is not None and (k + 1) % 16 == 0:
            progress(k + 1, k_count)

    best_k = scores.argmax(axis=1)
    rows = np.arange(n)
    best_score = scores[rows, best_k]
    # Second best distinct candidate for the confidence margin.
    tmp = scores.copy()
    tmp[rows, best_k] = -np.inf
    second_score = tmp.max(axis=1)
    shift_x = offsets[best_k, 0].copy()
    shift_y = offsets[best_k, 1].copy()

    # Batched additive Lucas-Kanade refinement.
    lk_ok = np.ones(n, dtype=bool)
    for _ in range(8):
        dirs = _batched_dirs(centers, b1, b2, x, y, shift_x, shift_y)
        uu, vv = sphere.direction_to_pixel(dirs, width, height)
        img = sample_equirect(lum_b, uu, vv)
        gx = np.gradient(img, step, axis=2)
        gy = np.gradient(img, step, axis=1)
        err = template - img
        sxx = (gx * gx).sum(axis=(1, 2))
        sxy = (gx * gy).sum(axis=(1, 2))
        syy = (gy * gy).sum(axis=(1, 2))
        rhsx = (gx * err).sum(axis=(1, 2))
        rhsy = (gy * err).sum(axis=(1, 2))
        det = sxx * syy - sxy * sxy
        good = det > 1e-9
        dx = np.zeros(n)
        dy = np.zeros(n)
        dx[good] = (syy[good] * rhsx[good] - sxy[good] * rhsy[good]) / det[good]
        dy[good] = (sxx[good] * rhsy[good] - sxy[good] * rhsx[good]) / det[good]
        lk_ok &= good
        mag = np.hypot(dx, dy)
        cap = np.deg2rad(1.0)
        big = mag > cap
        if np.any(big):
            dx[big] *= cap / mag[big]
            dy[big] *= cap / mag[big]
        shift_x += dx
        shift_y += dy
        if np.nanmax(np.hypot(dx, dy)) < 1e-5:
            break

    final_dirs = _batched_dirs(centers, b1, b2, x, y, shift_x, shift_y)
    uu, vv = sphere.direction_to_pixel(final_dirs, width, height)
    final_patch = sample_equirect(lum_b, uu, vv)
    final_c = final_patch - final_patch.mean(axis=(1, 2), keepdims=True)
    final_norm = np.sqrt((final_c ** 2).sum(axis=(1, 2)))
    ncc_final = np.einsum("nij,nij->n", template_c, final_c) / np.maximum(
        template_norm * final_norm, 1e-12)

    gx_px = np.gradient(final_patch, axis=2)
    gy_px = np.gradient(final_patch, axis=1)
    texture = np.sqrt((gx_px ** 2 + gy_px ** 2).mean(axis=(1, 2)))

    target = sphere.exp_map(centers, shift_x, shift_y, b1, b2)
    arc = sphere.geodesic_angle(centers, target)
    total_shift = np.hypot(shift_x, shift_y)
    supported = (
        (ncc_final >= 0.55)
        & (texture >= 0.012)
        & lk_ok
        & (template_std > 1e-4)
        & (total_shift <= np.deg2rad(11.0))
    )

    return {
        "center": centers,
        "target": target,
        "shift_x": shift_x,
        "shift_y": shift_y,
        "arc": arc,
        "ncc": ncc_final,
        "ncc_coarse": best_score,
        "margin": ncc_final - second_score,
        "texture": texture,
        "template_std": template_std,
        "supported": supported,
        "polar_basis": polar,
        "b1": b1,
        "b2": b2,
    }


def match_direction(lum_a, lum_b, c, width: int, height: int,
                    cfg: MatchConfig | None = None):
    """Single-direction convenience wrapper used by persistent tracks."""
    out = match_batch(lum_a, lum_b, np.asarray(c, dtype=np.float64)[None, :],
                      width, height, cfg)
    return {k: (v[0] if isinstance(v, np.ndarray) and v.shape[0] == 1
                and v.ndim >= 1 and k not in ("b1", "b2") else v)
            for k, v in out.items()}


# ---------------------------------------------------------------------------
# Global rotation fitting (Wahba / Kabsch on direction correspondences)
# ---------------------------------------------------------------------------

def kabsch_rotation(src: np.ndarray, tgt: np.ndarray, weights=None):
    src = np.asarray(src, dtype=np.float64)
    tgt = np.asarray(tgt, dtype=np.float64)
    if weights is None:
        w = np.ones(len(src))
    else:
        w = np.asarray(weights, dtype=np.float64)
    w = w / max(w.sum(), 1e-12)
    H = src.T @ (w[:, None] * tgt)
    u_svd, s, vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(vt.T @ u_svd.T))
    D = np.diag([1.0, 1.0, d])
    R = vt.T @ D @ u_svd.T
    return R


def fit_global_rotation(src, tgt, confidence=None, support=None,
                        iterations: int = 3):
    """Robustly fit the dominant rotation; returns R, inliers, residuals."""
    n = len(src)
    if support is None:
        mask = np.ones(n, dtype=bool)
    else:
        mask = np.asarray(support, dtype=bool).copy()
    if confidence is not None:
        mask &= np.asarray(confidence) > 0.3
    if mask.sum() < 8:
        return np.eye(3), np.zeros(n, bool), np.full(n, np.inf)

    conf = np.clip(confidence, 1e-3, 1.0) if confidence is not None else None
    R = kabsch_rotation(src[mask], tgt[mask],
                        conf[mask] if conf is not None else None)
    residual = sphere.geodesic_angle(sphere.rotate_dirs(R, src), tgt)
    inlier = mask.copy()
    for _ in range(iterations):
        med = np.median(residual[mask]) if mask.any() else 0.0
        thr = max(np.deg2rad(1.0), 5.0 * med)
        new_inlier = mask & (residual < thr)
        if new_inlier.sum() < 8:
            break
        R = kabsch_rotation(src[new_inlier], tgt[new_inlier],
                            conf[new_inlier] if conf is not None else None)
        residual = sphere.geodesic_angle(sphere.rotate_dirs(R, src), tgt)
        if np.array_equal(new_inlier, inlier):
            inlier = new_inlier
            break
        inlier = new_inlier
    return R, inlier, residual


# ---------------------------------------------------------------------------
# Whole-grid pair estimation
# ---------------------------------------------------------------------------

def estimate_pair(img_a, img_b, cfg: MatchConfig | None = None,
                  cols: int = 64, rows: int = 32, progress=None):
    cfg = cfg or MatchConfig()
    lum_a = luminance(img_a)
    lum_b = luminance(img_b)
    h, w = lum_a.shape

    grid = sphere.make_grid(w, h, cols, rows)
    dirs = grid["dir"].reshape(-1, 3)
    m = match_batch(lum_a, lum_b, dirs, w, h, cfg, progress=progress)

    R, inlier, global_res = fit_global_rotation(
        dirs, m["target"], confidence=np.clip(m["ncc"], 0, 1),
        support=m["supported"])
    global_omega = sphere.rotation_matrix_to_vector(R)
    omega = sphere.rotation_vector_between(dirs, m["target"])

    return {
        "width": w,
        "height": h,
        "cols": cols,
        "rows": rows,
        "grid": grid,
        "src": dirs,
        "target": m["target"],
        "omega": omega.reshape(rows, cols, 3),
        "arc": m["arc"],
        "shift_x": m["shift_x"],
        "shift_y": m["shift_y"],
        "ncc": m["ncc"],
        "texture": m["texture"],
        "supported": m["supported"],
        "polar_basis": m["polar_basis"],
        "global_R": R,
        "global_omega": global_omega,
        "global_inlier": inlier,
        "global_residual": global_res,
        "match": m,
    }
