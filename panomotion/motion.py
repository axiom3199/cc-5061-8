"""Spherical motion transport and the frozen external motion interface.

Design boundary
---------------
Motion is exchanged as *space direction vectors* and world-frame rotation
vectors, never as viewport tangent coordinates.  A per-viewport tangent basis
is only constructed at the sampling boundary (see :mod:`panomotion.topology`
and :mod:`panomotion.estimate`); converting a result to another viewport does
not change the transported direction.

:class:`SphereMotion` is the frozen external interface:

* ``advance`` maps a unit direction at frame ``t`` to its direction at
  ``t + 1`` with a Rodrigues rotation, so hotspot anchors stay glued to the
  panorama frame by frame (including across the seam and the poles);
* :class:`SphereMotionTrajectory` chains pair motions and reports boundary
  velocities with forward/backward finite differences at longitude-seam
  crossings and polar basis changes;
* the display layer reads motion through these objects but cannot mutate the
  solver state (all fields are derived, no in-place update API).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import sphere, topology


@dataclass(frozen=True)
class SphereMotion:
    """One frame pair's world-frame motion field.

    ``omega_grid`` is a ``rows x cols x 3`` rotation-vector field on the
    analysis grid.  The grid mapping itself is a projection concern; this
    class only evaluates motion for unit directions.
    """

    omega_grid: np.ndarray
    width: int
    height: int
    global_omega: np.ndarray

    def __post_init__(self):
        object.__setattr__(self, "omega_grid",
                           np.asarray(self.omega_grid, dtype=np.float64))
        object.__setattr__(self, "global_omega",
                           np.asarray(self.global_omega, dtype=np.float64))

    @property
    def rows(self):
        return self.omega_grid.shape[0]

    @property
    def cols(self):
        return self.omega_grid.shape[1]

    def omega_at(self, dirs):
        """Bilinearly sample the world omega field for unit directions.

        Horizontal interpolation wraps at the longitude seam; vertical
        coordinates clamp to the nearest node row (world vectors are finite at
        the poles).  This is the only place a direction touches the grid.
        """
        from .sampling import sample_equirect
        dirs = np.asarray(dirs, dtype=np.float64)
        rows, cols = self.omega_grid.shape[:2]
        u, v = sphere.direction_to_pixel(dirs, self.width, self.height)
        fi = (u + 0.5) * cols / self.width - 0.5
        fj = (v + 0.5) * rows / self.height - 0.5
        fi0 = np.floor(fi).astype(np.int64)
        tx = fi - fi0
        i0 = np.mod(fi0, cols)
        i1 = np.mod(fi0 + 1, cols)
        fj_c = np.clip(fj, 0.0, rows - 1.000001)
        fj0 = np.floor(fj_c).astype(np.int64)
        ty = fj_c - fj0
        j0 = np.clip(fj0, 0, rows - 1)
        j1 = np.clip(fj0 + 1, 0, rows - 1)
        w00 = ((1 - tx) * (1 - ty))[..., None]
        w10 = (tx * (1 - ty))[..., None]
        w01 = ((1 - tx) * ty)[..., None]
        w11 = (tx * ty)[..., None]
        g = self.omega_grid
        return (g[j0, i0] * w00 + g[j0, i1] * w10
                + g[j1, i0] * w01 + g[j1, i1] * w11)

    def advance(self, dirs):
        """Transport unit directions from frame t to frame t+1."""
        dirs = np.asarray(dirs, dtype=np.float64)
        omega = self.omega_at(dirs)
        return sphere.rodrigues(omega, dirs)

    def advance_global(self, dirs):
        """Transport using only the dominant camera rotation."""
        return sphere.rodrigues(self.global_omega, np.asarray(dirs, float))

    def arc_deg(self, dirs):
        """Geodesic displacement per direction (degrees), seam/pole safe."""
        return np.rad2deg(sphere.geodesic_angle(
            np.asarray(dirs, float), self.advance(dirs)))


# ---------------------------------------------------------------------------
# Chained trajectory driven by pair motions
# ---------------------------------------------------------------------------

@dataclass
class BoundaryVelocity:
    pair: int
    arc_deg: float                 # true geodesic step
    lon_step_deg: float            # wrapped (shortest) longitude step
    naive_lon_jump_deg: float      # raw |delta lon|, ~360/180 at boundaries
    crosses_seam: bool
    crosses_pole: bool


class SphereMotionTrajectory:
    """A per-frame hotspot anchor chain driven by SphereMotion pair fields.

    The anchor is always a unit direction.  Projections split it for display;
    this object never does, so a seam-crossing or pole-crossing hotspot stays
    one identity for its whole life.  A pre-measured ``chain`` (for example a
    sparse object-specific match chain) can be supplied directly; otherwise the
    chain is advanced pair by pair through the dense motion fields.
    """

    def __init__(self, anchor, motions, name: str = "hotspot", chain=None):
        self.name = name
        self.motions = list(motions)
        if chain is not None:
            self.chain = sphere.normalize(np.asarray(chain, dtype=np.float64))
        else:
            chain = [sphere.normalize(np.asarray(anchor, dtype=np.float64))]
            for m in self.motions:
                chain.append(
                    sphere.normalize(m.advance(chain[-1][None, :])[0]))
            self.chain = np.stack(chain)

    def __len__(self):
        return len(self.chain)

    def anchor(self, frame: int):
        return self.chain[frame]

    def lonlat(self):
        return sphere.direction_to_lonlat(self.chain)

    def boundary_velocities(self):
        """Forward/backward finite differences with explicit boundary flags."""
        lon, lat = self.lonlat()
        out = []
        for t in range(len(self.chain) - 1):
            d0, d1 = self.chain[t], self.chain[t + 1]
            arc = float(np.rad2deg(sphere.geodesic_angle(d0, d1)))
            wrapped = topology.longitude_step(lon[t], lon[t + 1])
            naive = float(abs(lon[t + 1] - lon[t]))
            pole = bool(topology.crosses_pole_lonlat(
                lon[t:t + 2], lat[t:t + 2])[1])
            seam = bool(topology.crosses_seam_lonlat(lon[t], lon[t + 1])) \
                and not pole
            out.append(BoundaryVelocity(
                pair=t, arc_deg=arc,
                lon_step_deg=float(np.rad2deg(wrapped)),
                naive_lon_jump_deg=float(np.rad2deg(naive)),
                crosses_seam=seam, crosses_pole=pole))
        return out

    def to_dict(self):
        lon, lat = self.lonlat()
        return {
            "name": self.name,
            "lon_deg": np.rad2deg(topology.wrap_longitude(lon)).round(5).tolist(),
            "lat_deg": np.rad2deg(lat).round(5).tolist(),
            "dir": self.chain.round(8).tolist(),
            "boundary": [vars(v) for v in self.boundary_velocities()],
        }


# ---------------------------------------------------------------------------
# Tangent-basis conversion (viewport boundary only)
# ---------------------------------------------------------------------------

def tangent_transport(motion: SphereMotion, dirs, polar_threshold: float = 0.85):
    """Return the same motion expressed in each direction's local basis.

    This is intentionally a *view* on the world motion: callers that need
    tangent-plane coordinates (block matching, per-viewport diagnostics) get
    ``(b1, b2, tx, ty, polar)`` here, while the underlying transport in
    :meth:`SphereMotion.advance` stays basis-independent.
    """
    dirs = np.asarray(dirs, dtype=np.float64)
    target = motion.advance(dirs)
    b1, b2, polar = topology.tangent_basis(dirs, polar_threshold)
    tx, ty, theta = sphere.tangent_components(dirs, target, b1, b2)
    return {"b1": b1, "b2": b2, "polar": polar,
            "tx": tx, "ty": ty, "arc": theta, "target": target}


def angular_equivalence(dirs_a, dirs_b, valid_mask=None,
                        max_err_deg: float = 1e-6):
    """Spherical-angle equivalence gate between two direction results.

    Returns ``(ok, max_err_deg_observed, valid_fraction)`` so callers can
    attach real numbers to an equivalence claim instead of asserting it.
    """
    a = np.asarray(dirs_a, dtype=np.float64)
    b = np.asarray(dirs_b, dtype=np.float64)
    err = np.rad2deg(sphere.geodesic_angle(a, b))
    if valid_mask is None:
        valid_mask = np.ones(len(err), bool)
    valid_mask = np.asarray(valid_mask, bool)
    if valid_mask.sum() == 0:
        return False, float(np.max(err)) if len(err) else 0.0, 0.0
    observed = float(err[valid_mask].max())
    fraction = float(valid_mask.mean())
    return observed <= max_err_deg, observed, fraction
