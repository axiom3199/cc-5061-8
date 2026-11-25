"""Naive planar equirectangular block-matching baseline.

This is the "apply ordinary 2-D flow to the unfolded image" approach, kept
deliberately naive: pixel windows, a translation-only UV model and clamped
borders (no seam adjacency, no pole-aware geometry).  It exists to quantify
why planar displacement is not a valid spherical motion representation near
the poles and across the longitude seam.  The production estimator lives in
:mod:`panomotion.estimate`.
"""

from __future__ import annotations

import numpy as np

from . import sphere
from .estimate import MatchConfig, luminance
from .sampling import to_float


def sample_uv_clamp(img, u, v):
    """Ordinary bilinear UV sampling with clamped borders (no seam wrap)."""
    h, w = img.shape[:2]
    u = np.asarray(u, float)
    v = np.asarray(v, float)
    x0 = np.floor(u).astype(np.int64)
    y0 = np.floor(v).astype(np.int64)
    fx = u - x0
    fy = v - y0
    x0c = np.clip(x0, 0, w - 1)
    x1c = np.clip(x0 + 1, 0, w - 1)
    y0c = np.clip(y0, 0, h - 1)
    y1c = np.clip(y0 + 1, 0, h - 1)
    if img.ndim == 3:
        fx, fy = fx[..., None], fy[..., None]
    Ia = img[y0c, x0c]
    Ib = img[y0c, x1c]
    Ic = img[y1c, x0c]
    Id = img[y1c, x1c]
    return (Ia * (1 - fx) + Ib * fx) * (1 - fy) \
        + (Ic * (1 - fx) + Id * fx) * fy


def estimate_pair_planar(img_a, img_b, cols: int = 48, rows: int = 24,
                         patch: int = 21):
    lum_a = luminance(img_a)
    lum_b = luminance(img_b)
    h, w = lum_a.shape
    grid = sphere.make_grid(w, h, cols, rows)
    uu = grid["u"].reshape(-1)
    vv = grid["v"].reshape(-1)
    n = len(uu)

    k = (np.arange(patch) - (patch - 1) / 2.0)
    kx, ky = np.meshgrid(k, k)

    ta = sample_uv_clamp(lum_a, uu[:, None, None] + kx[None],
                         vv[:, None, None] + ky[None])
    tac = ta - ta.mean((1, 2), keepdims=True)
    tan = np.sqrt((tac ** 2).sum((1, 2)))

    radius_px = np.deg2rad(7.0) * w / sphere.TWO_PI
    step_px = np.deg2rad(2.0) * w / sphere.TWO_PI
    vals = np.arange(-radius_px, radius_px + 0.5 * step_px, step_px)
    ox, oy = np.meshgrid(vals, vals)
    offsets = np.stack([ox.ravel(), oy.ravel()], 1)

    scores = np.empty((n, len(offsets)))
    for q, (dx, dy) in enumerate(offsets):
        tb = sample_uv_clamp(lum_b, uu[:, None, None] + kx[None] + dx,
                             vv[:, None, None] + ky[None] + dy)
        tbc = tb - tb.mean((1, 2), keepdims=True)
        num = np.einsum("nij,nij->n", tac, tbc)
        den = tan * np.sqrt((tbc ** 2).sum((1, 2)))
        scores[:, q] = num / np.maximum(den, 1e-12)

    best = scores.argmax(1)
    dx = offsets[best, 0].copy()
    dy = offsets[best, 1].copy()

    # Pixel-unit LK refinement.
    for _ in range(8):
        tb = sample_uv_clamp(lum_b, uu[:, None, None] + kx[None] + dx[:, None, None],
                             vv[:, None, None] + ky[None] + dy[:, None, None])
        gx = np.gradient(tb, axis=2)
        gy = np.gradient(tb, axis=1)
        err = ta - tb
        sxx = (gx * gx).sum((1, 2))
        sxy = (gx * gy).sum((1, 2))
        syy = (gy * gy).sum((1, 2))
        rhsx = (gx * err).sum((1, 2))
        rhsy = (gy * err).sum((1, 2))
        det = sxx * syy - sxy * sxy
        good = det > 1e-9
        ex = np.zeros(n)
        ey = np.zeros(n)
        ex[good] = (syy[good] * rhsx[good] - sxy[good] * rhsy[good]) / det[good]
        ey[good] = (sxx[good] * rhsy[good] - sxy[good] * rhsx[good]) / det[good]
        mag = np.hypot(ex, ey)
        big = mag > 1.5
        ex[big] *= 1.5 / mag[big]
        ey[big] *= 1.5 / mag[big]
        dx += ex
        dy += ey
        if np.nanmax(np.hypot(ex, ey)) < 1e-3:
            break

    tgt_u = uu + dx
    tgt_v = vv + dy
    # Naive interpretation of the 2-D offset as equirect pixels.  Longitude is
    # periodic but latitude is fed straight through, so polar geometry is wrong.
    target_dir = sphere.pixel_to_direction(tgt_u, tgt_v, w, h)
    src_dir = grid["dir"].reshape(-1, 3)
    return {
        "src": src_dir,
        "target": target_dir,
        "u": uu, "v": vv,
        "du": dx, "dv": dy,
        "cols": cols, "rows": rows,
    }
