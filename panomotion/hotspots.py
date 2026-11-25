"""Panorama hotspots: spherical anchors with projection-split identity.

A hotspot is anchored by a unit direction and a geodesic radius, not by pixel
coordinates.  The two projection outlets may split its outline into several
2-D fragments (longitude seam, camera back hemisphere); every fragment keeps
the hotspot's id, so the editor recognises them as one spherical object and
never creates a second hotspot at a seam.

Motion-field driven per-frame stability comes from
:class:`panomotion.motion.SphereMotionTrajectory`; this module owns identity,
clipping and the editor rules.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import motion as motion_mod
from . import projection as proj
from . import sphere
from .topology import tangent_basis


@dataclass
class Hotspot:
    hotspot_id: int
    name: str
    anchor: np.ndarray            # unit direction at frame 0
    radius_rad: float
    trajectory: motion_mod.SphereMotionTrajectory | None = None

    def __post_init__(self):
        self.anchor = sphere.normalize(np.asarray(self.anchor, dtype=np.float64))

    def direction_at(self, frame: int):
        if self.trajectory is not None:
            return self.trajectory.anchor(frame)
        return self.anchor

    def region_at(self, frame: int) -> proj.SphericalRegion:
        return proj.SphericalRegion(self.direction_at(frame), self.radius_rad)

    def outline_dirs(self, frame: int, samples: int = 72):
        """Dense geodesic circle of unit directions around the anchor."""
        center = self.direction_at(frame)
        b1, b2, _ = tangent_basis(center[None, :])
        b1, b2 = b1[0], b2[0]
        ang = np.linspace(0.0, 2.0 * np.pi, samples, endpoint=False)
        return sphere.exp_map(
            np.broadcast_to(center, (samples, 3)),
            self.radius_rad * np.cos(ang),
            self.radius_rad * np.sin(ang),
            np.broadcast_to(b1, (samples, 3)),
            np.broadcast_to(b2, (samples, 3)))

    def to_dict(self, frame_count: int | None = None):
        item = {
            "hotspot_id": self.hotspot_id,
            "name": self.name,
            "radius_deg": float(np.rad2deg(self.radius_rad)),
            "anchor_dir": self.anchor.round(8).tolist(),
        }
        if self.trajectory is not None:
            item["trajectory"] = self.trajectory.to_dict()
            item["frames"] = {
                str(t): {
                    "dir": self.direction_at(t).round(8).tolist(),
                    "lonlat": [round(float(x), 5) for x in
                               np.rad2deg(sphere.direction_to_lonlat(
                                   self.direction_at(t)))],
                }
                for t in range(len(self.trajectory))
            }
        return item


class HotspotEditor:
    """Create/select/clip hotspots through spherical semantics only.

    The editor is projection-agnostic: picking happens with a unit direction
    and clipping is delegated to :mod:`panomotion.projection`.  Consequently
    the equirect and perspective outlets always resolve to the same hotspot
    for the same physical object.
    """

    def __init__(self, motions=None, pick_tol_deg: float = 6.0):
        self.motions = list(motions or [])
        self.pick_tol_rad = np.deg2rad(pick_tol_deg)
        self._hotspots: list[Hotspot] = []
        self._next_id = 1

    # -- identity ----------------------------------------------------------
    def add(self, anchor_dir, name: str | None = None,
            radius_deg: float = 5.0, chain=None) -> Hotspot:
        hid = self._next_id
        self._next_id += 1
        traj = motion_mod.SphereMotionTrajectory(
            anchor_dir, self.motions, name=name or f"hotspot-{hid}",
            chain=chain) if (self.motions or chain is not None) else None
        hs = Hotspot(hid, name or f"hotspot-{hid}",
                     np.asarray(anchor_dir, float),
                     np.deg2rad(radius_deg), trajectory=traj)
        self._hotspots.append(hs)
        return hs

    def remove(self, hotspot_id: int):
        self._hotspots = [h for h in self._hotspots if h.hotspot_id != hotspot_id]

    def clear(self):
        self._hotspots = []
        self._next_id = 1

    @property
    def hotspots(self):
        return tuple(self._hotspots)

    def pick(self, direction, frame: int = 0):
        """Nearest hotspot whose region contains the clicked direction.

        Seam-split clicks resolve by angular distance on the sphere, so a
        click on either side of the seam returns the same hotspot.
        """
        direction = sphere.normalize(np.asarray(direction, dtype=np.float64))
        best, best_ang = None, self.pick_tol_rad
        for hs in self._hotspots:
            ang = float(sphere.geodesic_angle(
                direction[None, :], hs.direction_at(frame)[None, :])[0])
            if ang <= hs.radius_rad and ang < best_ang + hs.radius_rad:
                if best is None or ang < best_ang:
                    best, best_ang = hs, ang
        return best

    # -- projection outlets ------------------------------------------------
    def render(self, projection: proj.SphericalProjection, frame: int = 0):
        """Project every hotspot; returns fragments tagged with hotspot id."""
        fragments = []
        for hs in self._hotspots:
            outline = hs.outline_dirs(frame)
            # Close the circle on the sphere before outlet splitting.
            outline = np.concatenate([outline, outline[:1]])
            fragments.extend(projection.project_polyline(outline, hs.hotspot_id))
        return fragments

    def clip_hotspot(self, hotspot_id: int,
                     projections, frame: int = 0):
        """Clip one hotspot through every projection outlet.

        Returns ``{projection_name: [fragments]}`` plus an equivalence flag:
        all outlets must represent exactly the same set of hotspot ids, which
        is the boundary condition that forbids an extra seam hotspot.
        """
        hs = next((h for h in self._hotspots
                   if h.hotspot_id == hotspot_id), None)
        if hs is None:
            raise KeyError(hotspot_id)
        outline = np.concatenate([hs.outline_dirs(frame),
                                  hs.outline_dirs(frame)[:1]])
        out = {}
        id_sets = []
        for name, p in projections.items():
            frags = p.project_polyline(outline, hs.hotspot_id)
            out[name] = frags
            id_sets.append(proj.fragment_object_ids(frags))
        equivalent = all(s == id_sets[0] for s in id_sets) and \
            id_sets[0] == {hs.hotspot_id}
        return {"fragments": out, "equivalent": equivalent,
                "object_ids": sorted(id_sets[0])}

    def to_list(self):
        return [hs.to_dict() for hs in self._hotspots]
