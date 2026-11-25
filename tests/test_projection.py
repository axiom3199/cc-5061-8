"""Projection boundary tests.

The two outlets (equirect plate and perspective view) must implement one
spherical interface: seam/pole splits produce multiple 2-D fragments that keep
a single object id, and clipping a hotspot through both outlets cuts the same
spherical object without ever spawning an extra hotspot at the seam.
"""

import numpy as np
import pytest

from panomotion import sphere
from panomotion.projection import (EquirectProjection, PerspectiveProjection,
                                   SphericalRegion, clip_spherical_region,
                                   geodesic_polyline, fragment_object_ids,
                                   same_spherical_objects)


def test_equirect_splits_seam_polyline_but_keeps_one_object_id():
    W, H = 480, 240
    proj = EquirectProjection(W, H)
    # Geodesic that walks across the +pi/-pi seam near the equator.
    d0 = sphere.lonlat_to_direction(np.pi - 0.15, 0.0)
    d1 = sphere.lonlat_to_direction(-np.pi + 0.15, 0.0)
    line = geodesic_polyline(d0, d1, samples=40)
    frags = proj.project_polyline(line, object_id=7)
    assert len(frags) >= 2  # the seam physically breaks the 2-D drawing
    assert fragment_object_ids(frags) == {7}
    # Every fragment stays inside the plate; none wraps into a stray point.
    for f in frags:
        xs = [p[0] for p in f.points]
        assert min(xs) >= -1.0 and max(xs) <= W


def test_pole_crossing_does_not_create_extra_object():
    W, H = 480, 240
    proj = EquirectProjection(W, H)
    d0 = sphere.lonlat_to_direction(0.3, 1.45)
    d1 = sphere.lonlat_to_direction(0.3 + np.pi, 1.45)  # ~180 lon flip
    line = geodesic_polyline(d0, d1, samples=48)
    frags = proj.project_polyline(line, object_id=3)
    assert fragment_object_ids(frags) == {3}
    # The true geodesic step is small even though the longitude flip is huge.
    assert sphere.geodesic_angle(d0[None], d1[None])[0] < np.deg2rad(20.0)


def test_perspective_visible_mask_and_center_look_at():
    d = sphere.lonlat_to_direction(0.7, -0.3)
    proj = PerspectiveProjection.look_at(200, 100, d, np.deg2rad(90.0))
    pp = proj.project(d)
    assert pp.visible
    assert np.hypot(pp.x - proj.cx, pp.y - proj.cy) < 1e-6
    # A point on the opposite hemisphere is culled, not drawn behind.
    back = -d
    assert not proj.project(back).visible
    assert not proj.visible_mask(back[None, :])[0]


def test_clip_region_runs_are_spherical_not_pixel_boxes():
    center = sphere.lonlat_to_direction(0.0, 0.0)
    region = SphericalRegion(center, np.deg2rad(10.0))
    ring_angles = np.linspace(0, 2 * np.pi, 60, endpoint=False)
    b1, b2, _ = sphere.safe_tangent_basis(center[None, :])
    inside = sphere.exp_map(np.broadcast_to(center, (60, 3)),
                            np.deg2rad(5) * np.cos(ring_angles),
                            np.deg2rad(5) * np.sin(ring_angles),
                            np.broadcast_to(b1[0], (60, 3)),
                            np.broadcast_to(b2[0], (60, 3)))
    assert all(len(run) == 60 for run in clip_spherical_region(inside, region))


def test_two_outlets_clip_the_same_hotspot_across_the_seam():
    # Hotspot centred on the seam so the equirect outlet must split it.
    center = sphere.lonlat_to_direction(np.pi - 0.02, 0.2)
    radius = np.deg2rad(6.0)
    region = SphericalRegion(center, radius)
    b1, b2, _ = sphere.safe_tangent_basis(center[None, :])
    ang = np.linspace(0, 2 * np.pi, 96, endpoint=False)
    outline = sphere.exp_map(np.broadcast_to(center, (96, 3)),
                             radius * np.cos(ang), radius * np.sin(ang),
                             np.broadcast_to(b1[0], (96, 3)),
                             np.broadcast_to(b2[0], (96, 3)))
    outline = np.concatenate([outline, outline[:1]])

    eq = EquirectProjection(480, 240)
    persp = PerspectiveProjection.look_at(240, 120, center, np.deg2rad(120.0))

    eq_frags = eq.project_polyline(outline, object_id=5)
    pe_frags = persp.project_polyline(outline, object_id=5)

    # Equirect genuinely splits at the seam, perspective sees one cap; both
    # resolve to the same single spherical object.
    assert len(eq_frags) >= 2
    assert fragment_object_ids(eq_frags) == {5}
    assert fragment_object_ids(pe_frags) == {5}
    assert same_spherical_objects(eq_frags, pe_frags)


def test_clip_through_region_then_both_outlets_agrees_on_ids():
    center = sphere.lonlat_to_direction(-2.9, 0.4)
    region = SphericalRegion(center, np.deg2rad(8.0))
    line = geodesic_polyline(
        sphere.lonlat_to_direction(-3.05, 0.4),
        sphere.lonlat_to_direction(-2.75, 0.4), samples=40)
    eq = EquirectProjection(480, 240)
    persp = PerspectiveProjection.look_at(240, 120, center, np.deg2rad(110.0))
    a = eq.clip_region(line, region, object_id=9)
    b = persp.clip_region(line, region, object_id=9)
    assert fragment_object_ids(a) == fragment_object_ids(b) == {9}


def test_perspective_roundtrip_is_accurate_inside_valid_region():
    rng = np.random.default_rng(4)
    proj = PerspectiveProjection(320, 240, yaw=0.4, pitch=-0.2,
                                 fov_rad=np.deg2rad(75.0))
    errs = []
    for _ in range(60):
        d = sphere.normalize(rng.normal(size=3))
        pp = proj.project(d)
        if not pp.visible:
            continue
        back = proj.unproject(pp.x, pp.y)
        errs.append(np.rad2deg(sphere.geodesic_angle(d[None], back[None]))[0])
    # Half-pixel sampling convention; well under one degree at this FOV.
    assert max(errs) < 0.8
