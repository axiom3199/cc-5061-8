"""World-frame rotation-vector field operations and spherical reprojection.

Split out of the old sampler module: the field lives on the equirect grid but
all interpolation is seam-wrapped and the inversion works on unit directions,
so neither the longitude seam nor the poles introduce spikes.  Image access
goes through :mod:`panomotion.sampling`; topology rules through
:mod:`panomotion.topology`.
"""

from __future__ import annotations

import numpy as np

from . import sphere
from .sampling import sample_equirect, to_float, to_uint8


def fill_missing_omega(omega: np.ndarray, support: np.ndarray,
                       global_omega: np.ndarray, passes: int = 24) -> np.ndarray:
    """Replace unsupported nodes with the global rotation, then diffuse holes.

    Supported estimates are kept fixed; unsupported nodes start from the robust
    global camera rotation and relax through a seam-wrapped Jacobi diffusion so
    bilinear interpolation cannot create spikes.
    """
    out = omega.copy()
    sup = support.astype(bool).copy()
    out[~sup] = global_omega

    rows, cols = out.shape[:2]
    for _ in range(passes):
        acc = np.zeros_like(out)
        cnt = np.zeros((rows, cols, 1))
        acc += np.roll(out, 1, axis=1)
        acc += np.roll(out, -1, axis=1)
        cnt += 2.0
        up = np.empty_like(out)
        dn = np.empty_like(out)
        up[1:] = out[:-1]
        up[0] = out[0]
        dn[:-1] = out[1:]
        dn[-1] = out[-1]
        acc += up + dn
        cnt += 2.0
        blended = acc / cnt
        out = np.where(sup[..., None], out, 0.5 * out + 0.5 * blended)
    return out


def interpolate_omega_grid(omega_grid: np.ndarray, width: int, height: int):
    """Bilinearly sample a rows x cols x 3 omega grid at every output pixel.

    Horizontal indices wrap at the longitude seam.  Vertical coordinates clamp
    to the node rows; polar rows carry world rotation vectors, which do not
    blow up at the pole.
    """
    rows, cols = omega_grid.shape[:2]
    uu = np.arange(width)
    vv = np.arange(height)
    UU, VV = np.meshgrid(uu, vv)
    fi = (UU + 0.5) * cols / width - 0.5
    fj = (VV + 0.5) * rows / height - 0.5

    fi0 = np.floor(fi).astype(np.int64)
    tx = fi - fi0
    i0 = fi0 % cols
    i1 = (fi0 + 1) % cols

    fj_c = np.clip(fj, 0.0, rows - 1.000001)
    fj0 = np.floor(fj_c).astype(np.int64)
    ty = fj_c - fj0
    j0 = np.clip(fj0, 0, rows - 1)
    j1 = np.clip(fj0 + 1, 0, rows - 1)

    w00 = (1 - tx) * (1 - ty)
    w10 = tx * (1 - ty)
    w01 = (1 - tx) * ty
    w11 = tx * ty
    return (
        omega_grid[j0, i0] * w00[..., None]
        + omega_grid[j0, i1] * w10[..., None]
        + omega_grid[j1, i0] * w01[..., None]
        + omega_grid[j1, i1] * w11[..., None]
    )


def invert_omega_field(omega_pixels: np.ndarray, width: int, height: int,
                       iterations: int = 8) -> np.ndarray:
    """Invert a forward map ``d' = Rodrigues(omega(d)) d`` per output pixel.

    Fixed-point iteration ``d <- Exp(-omega(d)) d_out``; per-frame motion is
    small so it converges in a few steps.  Returns source unit directions.
    """
    UU, VV = np.meshgrid(np.arange(width), np.arange(height))
    d_out = sphere.pixel_to_direction(UU, VV, width, height)
    d = d_out.copy()
    for _ in range(iterations):
        u, v = sphere.direction_to_pixel(d, width, height)
        w_at_d = sample_equirect(omega_pixels, u, v)
        d = sphere.rodrigues(-w_at_d, d_out)
    return d


def reproject_frame(frame_a: np.ndarray, omega_grid: np.ndarray) -> np.ndarray:
    """Warp ``frame_a`` into the next frame using a forward omega grid."""
    h, w = frame_a.shape[:2]
    omega_pixels = interpolate_omega_grid(omega_grid, w, h)
    src_dirs = invert_omega_field(omega_pixels, w, h)
    u, v = sphere.direction_to_pixel(src_dirs, w, h)
    warped = sample_equirect(to_float(frame_a), u, v)
    return to_uint8(np.clip(warped, 0.0, 1.0))


def photometric_error(frame_b: np.ndarray, warped_b_from_a: np.ndarray):
    a = to_float(frame_b)
    b = to_float(warped_b_from_a)
    diff = np.abs(a - b)
    mae = float(diff.mean())
    rmse = float(np.sqrt(((a - b) ** 2).mean()))
    psnr = float("inf") if rmse < 1e-12 else 20.0 * np.log10(1.0 / rmse)
    return {"mae": mae, "rmse": rmse, "psnr": psnr}, diff
