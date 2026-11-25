"""Spherical motion interface tests.

Motion is exchanged as world directions/rotation vectors.  These tests pin
the external behaviour: direction transport is basis-independent and stable
across the seam and poles, boundary velocities expose the longitude/pole
finite differences, and the tangent view is a pure read-only conversion.
"""

import numpy as np
import pytest

from panomotion import sphere, topology
from panomotion.motion import (SphereMotion, SphereMotionTrajectory,
                               angular_equivalence, tangent_transport)


def _constant_motion(omega, width=240, height=120, cols=12, rows=6):
    grid = np.broadcast_to(omega, (rows, cols, 3)).copy()
    return SphereMotion(grid, width, height, omega)


def test_advance_matches_rodrigues_and_is_basis_independent():
    R = sphere.rotation_yxz(np.deg2rad(4.0), np.deg2rad(-2.5), 0.0)
    omega = sphere.rotation_matrix_to_vector(R)
    m = _constant_motion(omega)
    rng = np.random.default_rng(0)
    dirs = sphere.normalize(rng.normal(size=(200, 3)))
    moved = m.advance(dirs)
    expected = sphere.rodrigues(omega, dirs)
    err = np.rad2deg(sphere.geodesic_angle(moved, expected))
    # Only grid bilinear interpolation separates the two evaluations.
    assert err.max() < 1e-5
    # Re-expressing the same motion in per-direction tangent bases must not
    # change the transported target direction.
    view = tangent_transport(m, dirs)
    err2 = np.rad2deg(sphere.geodesic_angle(view["target"], moved))
    # tangent_components is undefined for near-zero motion; compare only the
    # directions whose geodesic arc is numerically meaningful.
    meaningful = sphere.geodesic_angle(dirs, moved) > 1e-7
    assert err2[meaningful].max() < 1e-5
    assert np.allclose(view["target"], moved, atol=1e-12)


def test_seam_transport_has_small_geodesic_arc_not_360_degrees():
    # Source just below +pi and target just above -pi: the same physical step
    # reads as a huge raw longitude subtraction but a small geodesic arc.
    d0 = sphere.lonlat_to_direction(np.pi - np.deg2rad(3.0), 0.3)
    d1 = sphere.lonlat_to_direction(-np.pi + np.deg2rad(3.0), 0.3)
    omega = sphere.rotation_vector_between(d0[None, :], d1[None, :])[0]
    m = _constant_motion(omega)
    moved = m.advance(d0[None, :])[0]
    assert sphere.geodesic_angle(moved[None], d1[None])[0] < np.deg2rad(0.01)
    lon0, _ = sphere.direction_to_lonlat(d0[None, :])
    lon1, _ = sphere.direction_to_lonlat(moved[None, :])
    wrapped = topology.longitude_step(lon0[0], lon1[0])
    assert abs(float(wrapped)) < np.deg2rad(7.0)
    assert abs(float(lon1[0] - lon0[0]) - float(wrapped)) > np.pi / 2
    true_arc = np.rad2deg(sphere.geodesic_angle(d0[None], d1[None]))[0]
    assert m.arc_deg(d0[None, :])[0] == pytest.approx(float(true_arc), abs=1e-4)
    assert true_arc < 7.0


def test_pole_crossing_trajectory_flags_boundary_with_small_arc():
    # Hand-built chain that crosses the north pole (lon flip ~ pi).
    chain = []
    p = np.array([1.0, 0.0, 0.0])
    yaxis = np.array([0.0, 1.0, 0.0])
    for t in range(5):
        a = np.deg2rad(80.0 + 5.0 * t)
        chain.append(sphere.normalize(p * np.cos(a) + yaxis * np.sin(a)))
    chain = np.stack(chain)
    traj = SphereMotionTrajectory(chain[0], motions=[], chain=chain)
    vels = traj.boundary_velocities()
    pole_steps = [v for v in vels if v.crosses_pole]
    assert pole_steps
    v = max(pole_steps, key=lambda x: x.naive_lon_jump_deg)
    assert v.naive_lon_jump_deg > 120.0
    assert v.arc_deg < 12.0  # geodesic step stays small over the pole
    assert not v.crosses_seam


def test_seam_crossing_trajectory_flags_seam_not_pole():
    lons = np.deg2rad([170.0, 176.0, -178.0, -172.0])
    lats = np.deg2rad([10.0, 10.5, 11.0, 10.7])
    chain = sphere.lonlat_to_direction(lons, lats)
    traj = SphereMotionTrajectory(chain[0], motions=[], chain=chain)
    vels = traj.boundary_velocities()
    seam_steps = [v for v in vels if v.crosses_seam]
    assert seam_steps
    assert all(not v.crosses_pole for v in seam_steps)
    # Wrapped longitude step is small while the naive jump is ~360-ish.
    v = seam_steps[0]
    assert abs(v.lon_step_deg) < 12.0
    assert v.naive_lon_jump_deg > 120.0


def test_angular_equivalence_reports_real_error_and_valid_fraction():
    rng = np.random.default_rng(2)
    a = sphere.normalize(rng.normal(size=(50, 3)))
    b = sphere.rodrigues(np.array([1e-4, 0.0, 0.0]), a)
    valid = np.ones(50, bool)
    valid[:10] = False
    ok, observed, fraction = angular_equivalence(
        a, b, valid_mask=valid, max_err_deg=0.01)
    assert ok and fraction == pytest.approx(0.8)
    assert observed > 0.0  # a measured number, not a hard-coded pass
    bad = sphere.rodrigues(np.array([0.05, 0.0, 0.0]), a)
    ok2, observed2, _ = angular_equivalence(a, bad, valid, max_err_deg=0.01)
    assert not ok2 and observed2 > 1.0


def test_trajectory_identity_is_single_object_through_boundaries():
    lons = np.deg2rad([176.0, -179.0, -174.0])
    lats = np.deg2rad([86.0, 87.5, 86.5])
    chain = sphere.lonlat_to_direction(lons, lats)
    traj = SphereMotionTrajectory(chain[0], motions=[], chain=chain)
    # One continuous anchor chain regardless of projection singularities.
    assert len(traj) == 3
    assert np.allclose(np.linalg.norm(traj.chain, axis=1), 1.0)
    doc = traj.to_dict()
    assert len(doc["boundary"]) == 2
