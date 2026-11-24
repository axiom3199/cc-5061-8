import numpy as np
import pytest

from panomotion import sphere, warp
from panomotion.synth import generate_synthetic


@pytest.fixture(scope="module")
def seq():
    return generate_synthetic(width=480, height=240, n_frames=6,
                              seed=11, verbose=False)


def test_camera_matrices_are_rotations(seq):
    for R in seq.Rs:
        assert np.allclose(R.T @ R, np.eye(3), atol=1e-10)


def test_object_matrix_maps_center_to_next_frame(seq):
    for obj in seq.objects:
        for t in range(seq.n_frames - 1):
            cam0 = seq.Rs[t].T @ obj.centers[t]
            cam1 = seq.Rs[t + 1].T @ obj.centers[t + 1]
            M = seq.object_matrix(t, obj)
            moved = sphere.rotate_dirs(M, cam0)
            assert np.rad2deg(sphere.geodesic_angle(moved, cam1)) < 1e-8


def test_polar_object_crosses_equirect_pole(seq):
    obj = seq.objects[0]
    cam = np.stack([seq.Rs[t].T @ obj.centers[t]
                    for t in range(seq.n_frames)])
    lon, lat = sphere.direction_to_lonlat(cam)
    assert np.max(np.abs(np.rad2deg(lat))) > 85.0
    jumps = np.abs(np.diff(lon))
    j = int(np.argmax(jumps))
    assert np.rad2deg(jumps[j]) > 120.0
    steps = sphere.geodesic_angle(cam[:-1], cam[1:])
    assert np.rad2deg(steps[j]) < 15.0


def test_seam_object_has_360_style_jump_but_small_arc(seq):
    obj = seq.objects[1]
    cam = np.stack([seq.Rs[t].T @ obj.centers[t]
                    for t in range(seq.n_frames)])
    lon, _ = sphere.direction_to_lonlat(cam)
    jumps = np.abs(np.diff(lon))
    j = int(np.argmax(jumps))
    assert np.rad2deg(jumps[j]) > 120.0
    steps = sphere.geodesic_angle(cam[:-1], cam[1:])
    assert np.rad2deg(steps[j]) < 12.0


def test_gt_background_warp_is_photometrically_consistent(seq):
    t = 1
    H, W = seq.height, seq.width
    uu, vv = np.meshgrid(np.arange(W), np.arange(H))
    d = sphere.pixel_to_direction(uu, vv, W, H)
    labels_a = seq.labels[t]
    gt = seq.gt_targets(d.reshape(-1, 3), t,
                        labels=labels_a.ravel()).reshape(H, W, 3)
    labels_b = seq.labels_at(gt.reshape(-1, 3), t + 1).reshape(H, W)

    mask = (labels_a == 0) & (labels_b == 0)
    assert mask.mean() > 0.4
    b = seq.frames[t + 1].astype(np.float32) / 255.0
    sampled = warp.sample_directions(b, gt)
    a = seq.frames[t].astype(np.float32) / 255.0
    mae = np.abs(a - sampled).mean(axis=2)[mask].mean()
    assert mae < 0.05
