import numpy as np

from panomotion import sphere, warp


def test_seam_wrap_sampling_averages_both_borders():
    img = np.zeros((4, 8, 3), np.uint8)
    img[:, 0] = [255, 0, 0]
    img[:, 7] = [0, 0, 255]
    # u = -0.5 sits exactly halfway between col 7 (blue) and col 0 (red).
    out = warp.sample_equirect(img, np.array([-0.5, 7.5]), np.array([1.5, 1.5]))
    assert np.allclose(out[0], [127.5, 0, 127.5], atol=1.0)
    assert np.allclose(out[1], [127.5, 0, 127.5], atol=1.0)


def test_polar_reflection_opposite_longitude():
    u = np.array([4.0])
    v = np.array([-2.0])  # 1.5 px beyond the north edge (v < -0.5)
    u2, v2 = warp.reflect_polar_coords(u, v, width=16, height=8)
    assert v2[0] == -(-2.0) - 1.0
    assert u2[0] == 4.0 + 8.0  # shifted by half the width


def test_constant_rotation_reproject_matches_direct_resample():
    rng = np.random.default_rng(3)
    frame_a = rng.integers(0, 256, size=(80, 160, 3), dtype=np.uint8)
    R = sphere.rotation_yxz(np.deg2rad(3.0), np.deg2rad(-2.0), 0.0)
    omega = sphere.rotation_matrix_to_vector(R)
    grid = np.broadcast_to(omega, (6, 12, 3)).copy()

    warped = warp.reproject_frame(frame_a, grid)

    # Independent construction of frame B: source dir = R^{-1} d_out.
    uu, vv = np.meshgrid(np.arange(160), np.arange(80))
    d_out = sphere.pixel_to_direction(uu, vv, 160, 80)
    src = sphere.rotate_dirs(R.T, d_out.reshape(-1, 3)).reshape(80, 160, 3)
    expected = warp.sample_directions(warp.to_float(frame_a), src)

    err = np.abs(warp.to_float(warped) - expected).mean()
    assert err < 0.01  # 0-1 units; fixed-point inversion is exact for const field


def test_fill_missing_omega_keeps_supported_and_finite():
    rows, cols = 6, 12
    omega = np.zeros((rows, cols, 3))
    support = np.zeros((rows, cols), bool)
    omega[0, 0] = [0.05, 0.0, 0.0]
    support[0, 0] = True
    out = warp.fill_missing_omega(omega, support,
                                  np.array([0.01, 0.02, 0.03]), passes=10)
    assert np.all(np.isfinite(out))
    assert np.allclose(out[0, 0], omega[0, 0])
    assert np.allclose(out[3, 6], [0.01, 0.02, 0.03], atol=0.02)
