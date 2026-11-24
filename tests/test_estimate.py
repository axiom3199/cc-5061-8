import numpy as np

from panomotion import sphere, warp
from panomotion.estimate import estimate_pair


def _rotated_pair(seed=7, W=960, H=480):
    rng = np.random.default_rng(seed)
    a = rng.integers(0, 256, size=(H, W)).astype(np.float32)
    # Smooth slightly so bilinear resampling keeps strong gradients.
    try:
        from scipy.ndimage import gaussian_filter  # optional
        a = gaussian_filter(a, 0.8)
    except Exception:
        pass
    frame_a = np.clip(a, 0, 255).astype(np.uint8)
    R = sphere.rotation_yxz(np.deg2rad(3.2), np.deg2rad(-1.8), np.deg2rad(0.4))
    uu, vv = np.meshgrid(np.arange(W), np.arange(H))
    d_out = sphere.pixel_to_direction(uu, vv, W, H)
    src = sphere.rotate_dirs(R.T, d_out.reshape(-1, 3)).reshape(H, W, 3)
    b = warp.sample_directions(frame_a.astype(np.float32) / 255.0, src)
    frame_b = np.clip(b * 255, 0, 255).astype(np.uint8)
    return frame_a, frame_b, R


def test_estimate_recovers_known_camera_rotation():
    a, b, R_true = _rotated_pair()
    res = estimate_pair(a, b, cols=32, rows=16)
    R_est = res["global_R"]
    # Relative rotation angle between estimate and truth.
    dR = R_est.T @ R_true
    err = np.arccos(np.clip((np.trace(dR) - 1) / 2, -1, 1))
    assert np.rad2deg(err) < 0.25

    good = res["ncc"] > 0.5
    assert good.mean() > 0.15
    pred = sphere.rotate_dirs(R_true, res["src"][good])
    e = np.rad2deg(sphere.geodesic_angle(pred, res["target"][good]))
    assert np.median(e) < 0.2
