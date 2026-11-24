"""Spherical geometry for equirectangular panorama motion estimation.

Coordinate convention
---------------------
* Longitude ``lon`` in radians, range ``[-pi, pi)``; the longitude seam is at
  ``lon = +/- pi`` and corresponds to the left/right image border.
* Latitude ``lat`` in radians, range ``[-pi/2, pi/2]``; ``+pi/2`` is north.
* Unit direction vector::

      d = (cos(lat) cos(lon), sin(lat), cos(lat) sin(lon))

  i.e. y is the up axis.  All motion results are ultimately stored as pairs of
  unit directions or as world-frame rotation vectors, never as unexplained
  pixel offsets.

The equirectangular pixel convention treats integer coordinates as pixel
centers.  For an image of size ``W x H``::

      lon = (u + 0.5) / W * 2*pi - pi
      lat = pi/2 - (v + 0.5) / H * pi
"""

from __future__ import annotations

import numpy as np

TWO_PI = 2.0 * np.pi


# ---------------------------------------------------------------------------
# Equirectangular <-> direction conversions
# ---------------------------------------------------------------------------

def pixel_to_lonlat(u, v, width: int, height: int):
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    lon = (u + 0.5) / width * TWO_PI - np.pi
    lat = np.pi / 2.0 - (v + 0.5) / height * np.pi
    return lon, lat


def lonlat_to_pixel(lon, lat, width: int, height: int):
    lon = np.asarray(lon, dtype=np.float64)
    lat = np.asarray(lat, dtype=np.float64)
    u = (lon + np.pi) / TWO_PI * width - 0.5
    v = (np.pi / 2.0 - lat) / np.pi * height - 0.5
    return u, v


def lonlat_to_direction(lon, lat):
    lon = np.asarray(lon, dtype=np.float64)
    lat = np.asarray(lat, dtype=np.float64)
    cl = np.cos(lat)
    return np.stack([cl * np.cos(lon), np.sin(lat), cl * np.sin(lon)], axis=-1)


def pixel_to_direction(u, v, width: int, height: int):
    lon, lat = pixel_to_lonlat(u, v, width, height)
    return lonlat_to_direction(lon, lat)


def direction_to_lonlat(d):
    d = np.asarray(d, dtype=np.float64)
    x, y, z = d[..., 0], d[..., 1], d[..., 2]
    lat = np.arcsin(np.clip(y, -1.0, 1.0))
    lon = np.arctan2(z, x)
    return lon, lat


def direction_to_pixel(d, width: int, height: int):
    lon, lat = direction_to_lonlat(d)
    return lonlat_to_pixel(lon, lat, width, height)


def wrap_lon(lon):
    """Wrap longitude into [-pi, pi)."""
    lon = np.asarray(lon, dtype=np.float64)
    return (lon + np.pi) % TWO_PI - np.pi


# ---------------------------------------------------------------------------
# Basis frames
# ---------------------------------------------------------------------------

def east_north_basis(d):
    """East and north unit tangent vectors.

    Degenerate at the poles where east/north are undefined.  Callers that need
    a frame close to a pole should use :func:`safe_tangent_basis`.
    """
    d = np.asarray(d, dtype=np.float64)
    single = d.ndim == 1
    if single:
        d = d[None, :]
    up = np.array([0.0, 1.0, 0.0])
    east = np.cross(up[None, :], d)
    en = np.linalg.norm(east, axis=-1, keepdims=True)
    # Away from the poles this is sin(lat-complement); guard the division.
    east = east / np.maximum(en, 1e-12)
    north = np.cross(d, east)
    north = north / np.maximum(np.linalg.norm(north, axis=-1, keepdims=True), 1e-12)
    if single:
        return east[0], north[0]
    return east, north


def safe_tangent_basis(d, polar_threshold: float = 0.85):
    """Return an orthonormal tangent basis that stays valid at the poles.

    Near the poles the geographic east/north frame is singular, so we switch to
    an axis-reference frame.  The returned basis is arbitrary but continuous
    enough for local patch extraction; results are converted back to world
    directions immediately, so the basis choice never leaks into motion data.

    Returns ``(b1, b2, polar)`` where ``polar`` marks the switched frames.
    """
    d = np.asarray(d, dtype=np.float64)
    single = d.ndim == 1
    if single:
        d = d[None, :]

    poleness = np.abs(d[..., 1])
    polar = poleness > polar_threshold
    # Reference axis: world y away from poles, world x near the poles.
    ref = np.zeros_like(d)
    use_y = ~polar
    ref[use_y, 1] = 1.0
    ref[polar] = np.array([1.0, 0.0, 0.0])

    b1 = np.cross(ref, d)
    n = np.linalg.norm(b1, axis=-1, keepdims=True)
    # Extremely unlikely second degeneracy (d parallel to chosen ref): swap ref.
    tiny = (n[..., 0] < 1e-9)
    if np.any(tiny):
        ref[tiny] = np.array([0.0, 0.0, 1.0])
        b1[tiny] = np.cross(ref[tiny], d[tiny])
        n = np.linalg.norm(b1, axis=-1, keepdims=True)
    b1 = b1 / np.maximum(n, 1e-12)
    b2 = np.cross(d, b1)
    b2 = b2 / np.maximum(np.linalg.norm(b2, axis=-1, keepdims=True), 1e-12)

    if single:
        return b1[0], b2[0], bool(polar[0])
    return b1, b2, polar


# ---------------------------------------------------------------------------
# Geodesic / exponential maps and rotation vectors
# ---------------------------------------------------------------------------

def normalize(x, axis: int = -1, eps: float = 1e-12):
    n = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / np.maximum(n, eps)


def geodesic_angle(d1, d2):
    """Shortest angular distance in radians between unit direction arrays."""
    d1 = np.asarray(d1, dtype=np.float64)
    d2 = np.asarray(d2, dtype=np.float64)
    dot = np.sum(d1 * d2, axis=-1)
    return np.arccos(np.clip(dot, -1.0, 1.0))


def exp_map(d, tx, ty, b1, b2):
    """Geodesic exponential: move from ``d`` by tangent components.

    ``target = d cos(theta) + (b1*tx + b2*ty) / theta sin(theta)`` with
    ``theta = hypot(tx, ty)``.  Unlike naively adding lon/lat offsets this is
    well defined when the path crosses a pole.
    """
    d = np.asarray(d, dtype=np.float64)
    move = b1 * np.asarray(tx)[..., None] + b2 * np.asarray(ty)[..., None]
    theta = np.hypot(tx, ty)
    th = np.asarray(theta, dtype=np.float64)
    shape = th.shape + (1,)
    th_b = th[..., None]
    small = th_b < 1e-9
    unit_move = move / np.maximum(th_b, 1e-12)
    out = d * np.cos(th_b) + unit_move * np.sin(th_b)
    # Small-angle branch: first-order expansion, then normalize.
    small_out = normalize(d + move)
    out = np.where(np.broadcast_to(small, out.shape), small_out, out)
    return normalize(out)


def tangent_components(d0, d1, b1, b2):
    """Log map expressed in basis ``(b1, b2)`` plus arc length."""
    d0 = np.asarray(d0, dtype=np.float64)
    d1 = np.asarray(d1, dtype=np.float64)
    theta = geodesic_angle(d0, d1)
    axis_sin = np.cross(d0, d1)
    s = np.sin(theta)
    valid = s > 1e-9
    # Unit geodesic tangent direction at d0 toward d1.
    tdir = np.zeros_like(d0)
    if np.any(valid):
        tdir[valid] = axis_sin[valid] / s[valid, None]
    # axis_sin / sin(theta) = unit rotation axis; tangent direction is axis x d0.
    tdir = np.cross(tdir, d0)
    tx = np.sum(tdir * b1, axis=-1) * theta
    ty = np.sum(tdir * b2, axis=-1) * theta
    return tx, ty, theta


def rotation_vector_between(d0, d1):
    """World-frame axis-angle vector mapping ``d0`` to ``d1``.

    ``Rodrigues(omega) d0 == d1`` (up to numerical precision).  Storing motion
    this way is independent of the equirectangular coordinate singularity and
    of whichever local tangent basis was used.
    """
    d0 = np.asarray(d0, dtype=np.float64)
    d1 = np.asarray(d1, dtype=np.float64)
    theta = geodesic_angle(d0, d1)
    cross = np.cross(d0, d1)
    s = np.sin(theta)
    axis = np.zeros_like(d0)
    big = s > 1e-9
    axis[big] = cross[big] / s[big, None]
    return axis * theta[..., None]


def rodrigues(omega, d):
    """Rotate direction(s) ``d`` by world-frame rotation vector(s) ``omega``.

    ``omega`` and ``d`` are broadcast against each other over leading dims.
    """
    omega = np.asarray(omega, dtype=np.float64)
    d = np.asarray(d, dtype=np.float64)
    theta = np.linalg.norm(omega, axis=-1, keepdims=True)
    axis = omega / np.maximum(theta, 1e-12)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    # Rodrigues: d cos + (k x d) sin + k (k . d)(1 - cos)
    kd = np.sum(axis * d, axis=-1, keepdims=True)
    out = d * cos_t + np.cross(axis, d) * sin_t + axis * kd * (1.0 - cos_t)
    return normalize(out)


def rotation_matrix_to_vector(R):
    """Log map of a proper rotation matrix to a rotation vector."""
    cos_theta = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    theta = np.arccos(cos_theta)
    if theta < 1e-9:
        return np.zeros(3)
    skew = (R - R.T) / (2.0 * np.sin(theta))
    return theta * np.array([skew[2, 1], skew[0, 2], skew[1, 0]])


def rotation_yxz(yaw: float, pitch: float = 0.0, roll: float = 0.0):
    """Rotation with yaw about y, pitch about x, roll about z (intrinsic-ish).

    The composition ``R = Rz(roll) @ Rx(pitch) @ Ry(yaw)`` maps local frame
    directions to world directions.  Only the composition matters for the
    estimator and ground truth.
    """
    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cr, sr = np.cos(roll), np.sin(roll)
    Ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])
    Rx = np.array([[1.0, 0.0, 0.0], [0.0, cp, -sp], [0.0, sp, cp]])
    Rz = np.array([[cr, -sr, 0.0], [sr, cr, 0.0], [0.0, 0.0, 1.0]])
    return Rz @ Rx @ Ry


def rotate_dirs(R, d):
    """Apply rotation matrix ``R`` to direction array (...,3)."""
    return normalize(d @ R.T)


# ---------------------------------------------------------------------------
# Grid construction
# ---------------------------------------------------------------------------

def make_grid(width: int, height: int, cols: int, rows: int):
    """Node pixel coords / lonlat / directions for a regular analysis grid."""
    us = (np.arange(cols) + 0.5) * width / cols
    vs = (np.arange(rows) + 0.5) * height / rows
    uu, vv = np.meshgrid(us, vs)
    lon, lat = pixel_to_lonlat(uu, vv, width, height)
    d = lonlat_to_direction(lon, lat)
    return {
        "u": uu,
        "v": vv,
        "lon": lon,
        "lat": lat,
        "dir": d,
        "i": np.arange(cols)[None, :].repeat(rows, axis=0),
        "j": np.arange(rows)[:, None].repeat(cols, axis=1),
    }


# ---------------------------------------------------------------------------
# Seam / pole diagnostics (pure geometry helpers)
# ---------------------------------------------------------------------------

def seam_pairs(cols: int, rows: int):
    """Indices of node pairs that are physical neighbours across the lon seam.

    Returns array ``(K, 2)`` of flattened node indices ``(left_border,
    right_border)`` for every grid row.  Flattening is row-major ``j*cols+i``.
    """
    js = np.arange(rows)
    right = js * cols + (cols - 1)  # lon close to +pi
    left = js * cols + 0            # lon close to -pi
    return np.stack([right, left], axis=1)


def naive_lon_gap(lon_right, lon_left):
    """The misleading ~2*pi gap obtained by subtracting longitudes directly."""
    return lon_left - lon_right
