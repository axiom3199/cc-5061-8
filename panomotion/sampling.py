"""Equirectangular image sampling.

Pixel access only.  The longitude-seam wraparound and the pole reflection are
owned by :mod:`panomotion.topology`; this sampler applies those rules and does
bilinear interpolation.  It holds no motion state.
"""

from __future__ import annotations

import numpy as np

from . import sphere, topology


def to_float(img: np.ndarray) -> np.ndarray:
    if img.dtype == np.uint8:
        return img.astype(np.float64) / 255.0
    return img.astype(np.float64)


def to_uint8(img: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(img * 255.0), 0, 255).astype(np.uint8)


def sample_equirect(img: np.ndarray, u, v) -> np.ndarray:
    """Bilinear sample ``img`` at pixel coords ``u, v`` with seam/pole rules.

    ``img`` may be ``H x W`` or ``H x W x C``.  Output shape is
    ``u.shape + (C,)`` (or ``u.shape`` for 2-D images).
    """
    img = np.asarray(img)
    h, w = img.shape[:2]
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    u, v = topology.reflect_polar_pixel(u, v, w, h)

    x0 = np.floor(u).astype(np.int64)
    y0 = np.floor(v).astype(np.int64)
    fx = u - x0
    fy = v - y0

    x0_mod = topology.wrap_pixel_u(x0, w).astype(np.int64)
    x1_mod = topology.wrap_pixel_u(x0 + 1, w).astype(np.int64)  # seam adjacency
    y0c = np.clip(y0, 0, h - 1)
    y1c = np.clip(y0 + 1, 0, h - 1)

    if img.ndim == 3:
        fx = fx[..., None]
        fy = fy[..., None]
        Ia = img[y0c, x0_mod]
        Ib = img[y0c, x1_mod]
        Ic = img[y1c, x0_mod]
        Id = img[y1c, x1_mod]
        top = Ia * (1.0 - fx) + Ib * fx
        bot = Ic * (1.0 - fx) + Id * fx
        return top * (1.0 - fy) + bot * fy

    Ia = img[y0c, x0_mod]
    Ib = img[y0c, x1_mod]
    Ic = img[y1c, x0_mod]
    Id = img[y1c, x1_mod]
    top = Ia * (1.0 - fx) + Ib * fx
    bot = Ic * (1.0 - fx) + Id * fx
    return top * (1.0 - fy) + bot * fy


def sample_directions(img: np.ndarray, dirs: np.ndarray) -> np.ndarray:
    """Sample an equirect image for an array of unit directions."""
    h, w = img.shape[:2]
    u, v = sphere.direction_to_pixel(dirs, w, h)
    return sample_equirect(img, u, v)
