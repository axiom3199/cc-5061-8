import numpy as np
import pytest

from panomotion import sphere


def test_pixel_direction_roundtrip():
    W, H = 320, 160
    rng = np.random.default_rng(0)
    u = rng.uniform(0, W - 1, 2000)
    v = rng.uniform(0, H - 1, 2000)
    d = sphere.pixel_to_direction(u, v, W, H)
    assert np.allclose(np.linalg.norm(d, axis=1), 1.0, atol=1e-10)
    u2, v2 = sphere.direction_to_pixel(d, W, H)
    # Longitude is modulo the seam.
    du = (u2 - u + W / 2) % W - W / 2
    assert np.max(np.abs(du)) < 1e-8
    assert np.max(np.abs(v2 - v)) < 1e-8


def test_cardinal_directions_and_poles():
    W, H = 960, 480
    # Image centre -> lon 0, lat 0 -> +x.
    d = sphere.pixel_to_direction((W - 1) / 2, (H - 1) / 2, W, H)
    assert np.allclose(d, [1, 0, 0], atol=1e-10)
    # Top centre is north pole (+y), bottom centre south pole.
    north = sphere.pixel_to_direction((W - 1) / 2, -0.5, W, H)
    south = sphere.pixel_to_direction((W - 1) / 2, H - 0.5, W, H)
    assert north[1] == pytest.approx(1.0, abs=1e-12)
    assert south[1] == pytest.approx(-1.0, abs=1e-12)
    # Left/right borders map to -pi/+pi seam and are the same direction.
    dl = sphere.pixel_to_direction(-0.5, H / 2 - 0.5, W, H)
    dr = sphere.pixel_to_direction(W - 0.5, H / 2 - 0.5, W, H)
    assert sphere.geodesic_angle(dl, dr) == pytest.approx(0.0, abs=1e-12)


def test_safe_basis_orthonormal_including_poles():
    dirs = np.array([
        [0, 1, 0], [0, -1, 0], [0.3, 0.95, 0.08],
        [1, 0, 0], [0, 0, 1],
    ], float)
    b1, b2, polar = sphere.safe_tangent_basis(dirs)
    assert polar[0] and polar[1]
    assert b1.shape == dirs.shape
    for i in range(len(dirs)):
        assert np.linalg.norm(b1[i]) == pytest.approx(1, abs=1e-10)
        assert np.linalg.norm(b2[i]) == pytest.approx(1, abs=1e-10)
        assert b1[i] @ b2[i] == pytest.approx(0, abs=1e-10)
        assert b1[i] @ dirs[i] == pytest.approx(0, abs=1e-10)
        assert b2[i] @ dirs[i] == pytest.approx(0, abs=1e-10)


def test_exp_log_roundtrip():
    rng = np.random.default_rng(1)
    lon = rng.uniform(-np.pi, np.pi, 50)
    lat = rng.uniform(-1.45, 1.45, 50)
    d0 = sphere.lonlat_to_direction(lon, lat)
    b1, b2, _ = sphere.safe_tangent_basis(d0)
    tx = rng.uniform(-0.08, 0.08, 50)
    ty = rng.uniform(-0.08, 0.08, 50)
    d1 = sphere.exp_map(d0, tx, ty, b1, b2)
    rx, ry, theta = sphere.tangent_components(d0, d1, b1, b2)
    assert np.allclose(rx, tx, atol=1e-8)
    assert np.allclose(ry, ty, atol=1e-8)
    arc = sphere.geodesic_angle(d0, d1)
    assert np.allclose(arc, np.hypot(tx, ty), atol=1e-8)
    assert np.allclose(theta, np.hypot(tx, ty), atol=1e-8)


def test_rotation_vector_maps_d0_to_d1():
    rng = np.random.default_rng(2)
    d0 = sphere.normalize(rng.normal(size=(100, 3)))
    axis = sphere.normalize(rng.normal(size=(100, 3)))
    # Rotation axes need not be perpendicular; Rodrigues handles that.
    theta = rng.uniform(0.05, 0.6, 100)
    omega = axis * theta[:, None]
    d1 = sphere.rodrigues(omega, d0)
    omega_01 = sphere.rotation_vector_between(d0, d1)
    back = sphere.rodrigues(omega_01, d0)
    assert np.max(sphere.geodesic_angle(back, d1)) < 1e-7
    # The minimal rotation vector magnitude equals the geodesic arc.
    assert np.allclose(np.linalg.norm(omega_01, axis=1),
                       sphere.geodesic_angle(d0, d1), atol=1e-9)


def test_rotation_yxz_and_rotate_dirs():
    R = sphere.rotation_yxz(0.4, -0.2, 0.1)
    assert np.allclose(R.T @ R, np.eye(3), atol=1e-12)
    d = sphere.normalize(np.array([[0.3, 0.5, 0.8]]))
    out = sphere.rotate_dirs(R, d)
    assert np.allclose(out, (R @ d[0])[None, :], atol=1e-12)


def test_grid_and_seam_pairs_are_adjacent():
    cols, rows = 32, 8
    g = sphere.make_grid(320, 160, cols, rows)
    u, v = g["u"].ravel(), g["v"].ravel()
    assert u.shape == (rows * cols,)
    pairs = sphere.seam_pairs(cols, rows)
    assert pairs.shape == (rows, 2)
    d = sphere.pixel_to_direction(u, v, 320, 160)
    for right, left in pairs[:3]:
        # Right-edge node and left-edge node at the same row are neighbours
        # across the seam: their geodesic gap is one grid spacing, not 360 deg.
        gap = sphere.geodesic_angle(d[right], d[left])
        assert gap < np.deg2rad(13)
