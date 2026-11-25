"""Projection boundary: every 2-D outlet implements one spherical interface.

Two projections are supported:

* :class:`EquirectProjection` - the unfolded plate, longitude seam splits a
  spherical polyline into multiple 2-D segments;
* :class:`PerspectiveProjection` - the player's sphere view, points behind
  the camera are culled.

Both consume the *same* unit-direction inputs and return the same fragment
contract (:class:`ProjectedFragment`), so a hotspot that breaks into two
pieces at the seam is still one spherical object: every fragment carries the
caller's ``object_id``.  Region clipping is defined once on the sphere
(:func:`clip_spherical_region`) and every projection clips that same object,
which is what guarantees the two outlets cut out the same physical thing and
never spawn an extra hotspot at the seam.

The display layer may project freely; nothing here mutates motion state.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import sphere, topology


@dataclass(frozen=True)
class ProjectedPoint:
    x: float
    y: float
    visible: bool


@dataclass(frozen=True)
class ProjectedFragment:
    """One connected 2-D piece of a single spherical object.

    ``object_id`` is preserved across fragments so the hotspot editor can
    recognise seam-split pieces as the same hotspot.
    """

    object_id: int
    points: tuple  # tuple[(x, y), ...]
    sphere_dirs: tuple  # tuple[np.ndarray(3,), ...] parallel to points


@dataclass(frozen=True)
class SphericalRegion:
    """A geodesic cap (and optional inner exclusion ring) on the sphere."""

    center: np.ndarray
    radius_rad: float
    inner_radius_rad: float = 0.0

    def contains(self, dirs) -> np.ndarray:
        dirs = np.asarray(dirs, dtype=np.float64)
        ang = sphere.geodesic_angle(np.broadcast_to(
            np.asarray(self.center, dtype=np.float64), dirs.shape), dirs)
        inside = ang <= self.radius_rad
        if self.inner_radius_rad > 0.0:
            inside &= ang >= self.inner_radius_rad
        return inside


# ---------------------------------------------------------------------------
# Spherical-object primitives (projection independent)
# ---------------------------------------------------------------------------

def geodesic_polyline(d0, d1, samples: int = 16):
    """Densified shortest-path unit directions between two directions."""
    d0 = np.asarray(d0, dtype=np.float64)
    d1 = np.asarray(d1, dtype=np.float64)
    ts = np.linspace(0.0, 1.0, max(2, samples))
    omega = sphere.rotation_vector_between(d0, d1)
    return np.stack([sphere.rodrigues(omega * t, d0) for t in ts])


def clip_spherical_region(dirs, region: SphericalRegion):
    """Return the in-region direction runs of one spherical object.

    The output is a list of *runs* (each a ``(M, 3)`` array).  On the sphere
    there is one object; a run only becomes multiple 2-D fragments when a
    projection breaks it at a seam or behind a camera.
    """
    dirs = np.asarray(dirs, dtype=np.float64)
    mask = region.contains(dirs)
    runs = []
    start = None
    for k, ok in enumerate(mask):
        if ok and start is None:
            start = k
        if not ok and start is not None:
            runs.append(dirs[start:k])
            start = None
    if start is not None:
        runs.append(dirs[start:])
    return runs


# ---------------------------------------------------------------------------
# Projection boundary
# ---------------------------------------------------------------------------

class SphericalProjection:
    """Common outlet contract.  Subclasses implement the 2-D mapping only."""

    def project(self, d) -> ProjectedPoint:
        raise NotImplementedError

    def unproject(self, x, y):
        """Inverse map from an outlet point to a unit direction."""
        raise NotImplementedError

    # -- shared fragment machinery ----------------------------------------
    def project_polyline(self, dirs, object_id: int):
        """Project a spherical polyline, splitting only on outlet breaks.

        Returns a list of :class:`ProjectedFragment`.  A break is decided by
        the concrete projection (:meth:`breaks_between`); every fragment keeps
        the same ``object_id``.
        """
        dirs = np.asarray(dirs, dtype=np.float64)
        fragments: list[ProjectedFragment] = []
        pts: list = []
        sdirs: list = []
        prev = None
        prev_d = None
        for d in dirs:
            p = self.project(d)
            if not p.visible or (prev is not None
                                 and self.breaks_between(prev, p, prev_d, d)):
                if len(pts) >= 2:
                    fragments.append(ProjectedFragment(
                        object_id, tuple(pts), tuple(sdirs)))
                pts, sdirs = [], []
                if not p.visible:
                    prev, prev_d = None, None
                    continue
            pts.append((p.x, p.y))
            sdirs.append(np.asarray(d, dtype=np.float64))
            prev, prev_d = p, d
        if len(pts) >= 2:
            fragments.append(ProjectedFragment(
                object_id, tuple(pts), tuple(sdirs)))
        return fragments

    def clip_region(self, dirs, region: SphericalRegion, object_id: int):
        """Clip one spherical object's polyline through a region, then project.

        The spherical clip happens first and is shared by every projection;
        only the resulting fragments are outlet-specific.
        """
        out = []
        for run in clip_spherical_region(dirs, region):
            out.extend(self.project_polyline(run, object_id))
        return out

    def breaks_between(self, p0: ProjectedPoint, p1: ProjectedPoint,
                       d0, d1) -> bool:
        return False


class EquirectProjection(SphericalProjection):
    """Equirectangular plate outlet with the longitude seam as the break."""

    def __init__(self, width: int, height: int):
        self.width = int(width)
        self.height = int(height)

    def project(self, d):
        u, v = sphere.direction_to_pixel(
            np.asarray(d, dtype=np.float64), self.width, self.height)
        return ProjectedPoint(float(u), float(v), True)

    def unproject(self, x, y):
        return sphere.pixel_to_direction(x, y, self.width, self.height)

    def breaks_between(self, p0, p1, d0, d1):
        # A jump of half the plate means the geodesic crossed the seam.
        return abs(p1.x - p0.x) > self.width / 2.0

    def clip_region_pixels(self, dirs, region: SphericalRegion, object_id: int):
        """Convenience outlet: seam-split ``(pixel_segments, object_id)``."""
        frags = self.clip_region(dirs, region, object_id)
        return [(np.asarray(f.points), f.object_id) for f in frags]


class PerspectiveProjection(SphericalProjection):
    """Pinhole outlet used by the interactive sphere view.

    The orientation is held as the view-to-world rotation ``R`` (world = R v).
    Yaw/pitch construction keeps the player's ``Rx(pitch) Ry(yaw)`` convention;
    :meth:`look_at` builds the frame directly with an orthonormal basis so no
    angle inversion can hit a gimbal branch.  ``fov`` is the vertical field of
    view in radians.
    """

    def __init__(self, width: int, height: int, yaw: float | None = None,
                 pitch: float | None = None, fov_rad: float = np.deg2rad(68.0),
                 near: float = 1e-4, R: np.ndarray | None = None):
        self.width = int(width)
        self.height = int(height)
        self.fov_rad = float(fov_rad)
        self.near = float(near)
        if R is None:
            yaw = 0.0 if yaw is None else float(yaw)
            pitch = 0.0 if pitch is None else float(pitch)
            self.yaw, self.pitch = yaw, pitch
            self.R = sphere.rotation_yxz(yaw, pitch)
        else:
            self.R = np.asarray(R, dtype=np.float64)
            self.yaw = self.pitch = None
        self.focal = (self.height / 2.0) / np.tan(self.fov_rad / 2.0)
        self.cx = self.width / 2.0
        self.cy = self.height / 2.0

    def with_view(self, yaw=None, pitch=None, fov_rad=None):
        return PerspectiveProjection(
            self.width, self.height,
            self.yaw if yaw is None else yaw,
            self.pitch if pitch is None else pitch,
            self.fov_rad if fov_rad is None else fov_rad,
            near=self.near)

    @classmethod
    def look_at(cls, width, height, center_dir, fov_rad, near: float = 1e-4,
                world_up=None):
        """Build a perspective outlet whose view forward hits ``center_dir``.

        The camera frame is constructed directly on the sphere (forward =
        target direction, up aligned to the world y axis) so the target lands
        exactly at the image centre with no angle-inversion singularity.
        """
        forward = sphere.normalize(np.asarray(center_dir, dtype=np.float64))
        up_world = np.array([0.0, 1.0, 0.0]) if world_up is None \
            else sphere.normalize(np.asarray(world_up, dtype=np.float64))
        right = sphere.normalize(np.cross(up_world, forward))
        if np.linalg.norm(right) < 1e-9:  # target at the up axis
            right = sphere.normalize(np.cross(np.array([1.0, 0.0, 0.0]),
                                               forward))
        up = np.cross(forward, right)
        # Columns are the world directions of the local x/y/z camera axes.
        R = np.stack([right, up, forward], axis=1)
        return cls(width, height, fov_rad=fov_rad, near=near, R=R)

    def _world_to_view(self, d):
        d = np.asarray(d, dtype=np.float64)
        return sphere.rotate_dirs(self.R.T, d[None, :])[0]

    def project(self, d):
        v = self._world_to_view(d)
        if v[2] <= self.near:
            return ProjectedPoint(0.0, 0.0, False)
        return ProjectedPoint(
            float(self.cx + self.focal * v[0] / v[2]),
            float(self.cy - self.focal * v[1] / v[2]), True)

    def unproject(self, x, y):
        sx = (x + 0.5 - self.cx) / self.focal
        sy = (self.cy - (y + 0.5)) / self.focal
        v = sphere.normalize(np.array([sx, sy, 1.0]))
        return sphere.rotate_dirs(self.R, v[None, :])[0]

    def breaks_between(self, p0, p1, d0, d1):
        # Two visible points whose connecting geodesic dips behind the camera.
        v0 = self._world_to_view(d0)
        v1 = self._world_to_view(d1)
        return (v0[2] > self.near) and (v1[2] > self.near) and \
            float(v0 @ v1) < -0.2

    def visible_mask(self, dirs):
        dirs = np.asarray(dirs, dtype=np.float64)
        views = sphere.rotate_dirs(self.R.T, dirs)
        return views[..., 2] > self.near

# ---------------------------------------------------------------------------
# Cross-projection equivalence helpers
# ---------------------------------------------------------------------------

def region_coverage(region: SphericalRegion, dirs) -> float:
    """Fraction of sampled directions inside the region (valid-area metric)."""
    dirs = np.asarray(dirs, dtype=np.float64)
    if len(dirs) == 0:
        return 0.0
    return float(region.contains(dirs).mean())


def fragment_object_ids(fragments) -> set:
    """Distinct spherical objects represented by a fragment collection."""
    return {f.object_id for f in fragments}


def same_spherical_objects(frags_a, frags_b) -> bool:
    """Two outlets cut the same objects iff their object-id sets match."""
    return fragment_object_ids(frags_a) == fragment_object_ids(frags_b)
