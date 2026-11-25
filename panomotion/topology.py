"""Spherical topology: longitude wraparound and pole basis changes.

This module owns the two coordinate-singularity rules of the equirectangular
plate and nothing else:

* the longitude seam at ``lon = +/- pi`` is an *adjacency* (modulo ``2*pi``),
  not a 360-degree gap;
* walking past a pole reflects the row coordinate and shifts longitude by pi,
  and tangent frames near a pole must switch to a non-degenerate reference
  axis.

Image sampling lives in :mod:`panomotion.sampling`; it consumes these rules
instead of re-implementing them.  Motion transport uses world-frame direction
vectors and never depends on either rule directly.
"""

from __future__ import annotations

import numpy as np

from . import sphere

TWO_PI = sphere.TWO_PI


# ---------------------------------------------------------------------------
# Longitude seam (wraparound adjacency)
# ---------------------------------------------------------------------------

def wrap_longitude(lon):
    """Wrap longitude(s) into ``[-pi, pi)``."""
    lon = np.asarray(lon, dtype=np.float64)
    return (lon + np.pi) % TWO_PI - np.pi


def wrap_pixel_u(u, width: int):
    """Modular horizontal pixel index across the longitude seam."""
    u = np.asarray(u, dtype=np.float64)
    return np.mod(u, width)


def longitude_step(lon_a, lon_b):
    """Signed shortest angular step from ``lon_a`` to ``lon_b`` (radians).

    A step across the seam returns a small signed value instead of the naive
    ~2*pi difference.
    """
    return wrap_longitude(np.asarray(lon_b) - np.asarray(lon_a))


def crosses_seam_lonlat(lon_a, lon_b, jump_rad: float = np.pi):
    """True where consecutive longitudes meet only through the seam.

    A raw jump of ~2*pi (default threshold ``pi``) means the two points are
    neighbours across the seam rather than physically far apart.
    """
    return np.abs(np.asarray(lon_b) - np.asarray(lon_a)) > jump_rad


# ---------------------------------------------------------------------------
# Pole crossing (equirect plate edges)
# ---------------------------------------------------------------------------

def reflect_polar_pixel(u, v, width: int, height: int):
    """Reflect sample coordinates that cross a pole.

    Walking past the north/south edge is not clamping: the ray continues on
    the opposite longitude (``u += width/2``) and the row coordinate is
    mirrored.  Longitude wrapping itself is handled by modular x indexing in
    the sampler; this function only owns the pole rule.
    """
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)

    north = v < -0.5
    if np.any(north):
        v = np.where(north, -v - 1.0, v)
        u = np.where(north, u + width / 2.0, u)

    south = v > height - 0.5
    if np.any(south):
        v = np.where(south, 2.0 * height - v - 1.0, v)
        u = np.where(south, u + width / 2.0, u)

    # One reflection puts coordinates back inside; guard extreme footprints.
    v = np.clip(v, 0.0, height - 1.0)
    return u, v


def crosses_pole_lonlat(lon, lat, polar_lat_deg: float = 85.0,
                        jump_rad: float = np.deg2rad(120.0)):
    """Mark a trajectory step that goes over a pole.

    The signature of a pole crossing is a near-180-degree naive longitude flip
    while both points sit at high latitude; the geodesic step itself stays
    small.  Seam crossings happen at low/mid latitude and are excluded here.
    """
    lon = np.asarray(lon, dtype=np.float64)
    lat = np.asarray(lat, dtype=np.float64)
    jumps = np.abs(np.diff(lon))
    high = np.maximum(np.abs(lat[:-1]), np.abs(lat[1:])) > np.deg2rad(polar_lat_deg)
    return np.concatenate([[False], (jumps > jump_rad) & high])


def polar_basis_mask(d, polar_threshold: float = 0.85):
    """Boolean mask of directions whose east/north frame is degenerate."""
    d = np.asarray(d, dtype=np.float64)
    return np.abs(d[..., 1]) > polar_threshold


def tangent_basis(d, polar_threshold: float = 0.85):
    """Orthonormal tangent basis with an explicit pole basis change.

    Returns ``(b1, b2, polar)``.  Away from the poles the geographic
    east/north frame is used; inside the polar cap the frame switches to a
    world-axis reference so the basis never collapses.  Motion results are
    converted back to world directions immediately, so the choice of frame
    never leaks into stored motion.
    """
    return sphere.safe_tangent_basis(d, polar_threshold=polar_threshold)


# ---------------------------------------------------------------------------
# Seam adjacency on a regular analysis grid
# ---------------------------------------------------------------------------

def seam_neighbour_pairs(cols: int, rows: int):
    """Flattened ``(right_border, left_border)`` node index pairs per row."""
    return sphere.seam_pairs(cols, rows)
