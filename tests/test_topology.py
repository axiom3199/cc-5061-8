"""Topology boundary tests: longitude wraparound and pole basis changes.

These rules were split out of the image sampler; the sampler now consumes
:mod:`panomotion.topology` instead of implementing them.
"""

import numpy as np
import pytest

from panomotion import sphere, topology


def test_longitude_wrap_is_seam_adjacency():
    lon = np.array([np.pi - 0.01, -np.pi + 0.01])
    step = topology.longitude_step(lon[0], lon[1])
    # Across the seam the shortest step is a few hundredths of a radian,
    # never the naive ~2*pi gap.
    assert abs(step) < 0.05
    assert topology.crosses_seam_lonlat(lon[0], lon[1])


def test_wrap_pixel_u_matches_modular_sampler_index():
    u = np.array([-1.0, -0.5, 8.0, 9.7])
    np.testing.assert_allclose(topology.wrap_pixel_u(u, 8),
                               np.mod(u, 8))


def test_pole_reflection_shifts_opposite_longitude():
    u2, v2 = topology.reflect_polar_pixel(
        np.array([4.0]), np.array([-2.0]), width=16, height=8)
    assert v2[0] == pytest.approx(1.0)
    assert u2[0] == pytest.approx(12.0)  # half-width longitude shift
    u3, v3 = topology.reflect_polar_pixel(
        np.array([3.0]), np.array([9.0]), width=16, height=8)
    assert v3[0] == pytest.approx(6.0)
    assert u3[0] == pytest.approx(11.0)


def test_polar_basis_mask_and_switch_keep_orthonormal_frame():
    d = np.array([[0.0, 1.0, 0.0], [0.0, -1.0, 0.0],
                  [1.0, 0.0, 0.0], [0.2, 0.96, 0.1]])
    mask = topology.polar_basis_mask(d, polar_threshold=0.85)
    assert mask[0] and mask[1] and not mask[2]
    b1, b2, polar = topology.tangent_basis(d)
    np.testing.assert_array_equal(polar, mask)
    for k in range(len(d)):
        assert np.linalg.norm(b1[k]) == pytest.approx(1.0, abs=1e-10)
        assert np.linalg.norm(b2[k]) == pytest.approx(1.0, abs=1e-10)
        assert b1[k] @ b2[k] == pytest.approx(0.0, abs=1e-10)
        assert b1[k] @ d[k] == pytest.approx(0.0, abs=1e-10)


def test_crosses_pole_flag_uses_latitude_not_seam():
    # A near-180 lon flip at high latitude is a pole crossing; an actual
    # +/-pi seam crossing near the equator is flagged as a seam adjacency.
    lon_pole = np.array([0.2, 0.2 + np.pi - 0.05])
    lat_pole = np.array([1.52, 1.50])
    lat_seam = np.array([0.1, 0.12])
    pole_flags = topology.crosses_pole_lonlat(lon_pole, lat_pole)
    assert pole_flags[1]
    assert not topology.crosses_pole_lonlat(lon_pole, lat_seam)[1]
    # A genuine seam step: from just below +pi to just above -pi.
    lon_seam = np.array([np.pi - 0.03, -np.pi + 0.03])
    assert topology.crosses_seam_lonlat(lon_seam[0], lon_seam[1])
    assert not topology.crosses_pole_lonlat(lon_seam, lat_seam)[1]


def test_seam_neighbour_pairs_are_physically_adjacent():
    cols, rows = 24, 6
    pairs = topology.seam_neighbour_pairs(cols, rows)
    g = sphere.make_grid(240, 120, cols, rows)
    d = g["dir"].reshape(-1, 3)
    for right, left in pairs:
        gap = sphere.geodesic_angle(d[right], d[left])
        assert gap < np.deg2rad(16.0)  # one grid spacing, not 360 degrees
