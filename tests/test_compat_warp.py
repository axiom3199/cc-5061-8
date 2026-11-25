"""Backward-compatibility boundary for the pre-split ``warp`` entry points.

The projection/sampler/field split must not break historical callers: every
old ``panomotion.warp`` symbol keeps its original semantics and now delegates
to the owning module.
"""

import numpy as np

from panomotion import sphere, warp
from panomotion import field, sampling, topology


def test_warp_reexports_sampler_and_field_symbols():
    assert warp.sample_equirect is sampling.sample_equirect
    assert warp.sample_directions is sampling.sample_directions
    assert warp.to_float is sampling.to_float
    assert warp.to_uint8 is sampling.to_uint8
    assert warp.fill_missing_omega is field.fill_missing_omega
    assert warp.interpolate_omega_grid is field.interpolate_omega_grid
    assert warp.invert_omega_field is field.invert_omega_field
    assert warp.reproject_frame is field.reproject_frame
    assert warp.photometric_error is field.photometric_error


def test_legacy_reflect_polar_coords_matches_topology():
    u = np.array([4.0, 3.0])
    v = np.array([-2.0, 9.0])
    u_leg, v_leg = warp.reflect_polar_coords(u, v, 16, 8)
    u_top, v_top = topology.reflect_polar_pixel(u, v, 16, 8)
    np.testing.assert_allclose(u_leg, u_top)
    np.testing.assert_allclose(v_leg, v_top)


def test_legacy_reproject_pipeline_still_runs():
    rng = np.random.default_rng(9)
    frame = rng.integers(0, 256, size=(64, 128, 3), dtype=np.uint8)
    R = sphere.rotation_yxz(np.deg2rad(2.5), np.deg2rad(-1.5), 0.0)
    omega = sphere.rotation_matrix_to_vector(R)
    grid = np.broadcast_to(omega, (4, 8, 3)).copy()
    warped = warp.reproject_frame(frame, grid)
    assert warped.shape == frame.shape and warped.dtype == np.uint8
    metrics, diff = warp.photometric_error(frame, warped)
    assert set(metrics) == {"mae", "rmse", "psnr"}
    assert diff.shape == frame.shape
