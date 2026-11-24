"""Equirectangular image sampling and spherical reprojection warps."""

from __future__ import annotations

import numpy as np

from . import sphere


def to_float(img: np.ndarray) -> np.ndarray:
    if img.dtype == np.uint8:
        return img.astype(np.float64) / 255.0
    return img.astype(np.float64)


def to_uint8(img: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(img * 255.0), 0, 255).astype(np.uint8)


def reflect_polar_coords(u, v, width: int, height: int):
    """Reflect sample coordinates that cross a pole.

    Walking past the north/south edge of an equirect plate is not clamping:
    the ray continues on the opposite longitude (``u += width/2``) and the
    row coordinate is mirrored.  Longitude wrapping is handled separately by
    modular x indexing.
    """
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)

    north = v < -0.5
    if np.any(north):
        v = np.where(north, -v - 1.0, v)
        u = np.where(north, u + width / 2.0, u)

    south = v > height - 0.5
    if np.any(south):
        v = np.where(south, 2.0 * height - v - 1.0, v)
        u = np.where(south, u + width / 2.0, u)

    # After one reflection coordinates are inside; guard extreme footprints.
    v = np.clip(v, 0.0, height - 1.0)
    return u, v


def sample_equirect(img: np.ndarray, u, v) -> np.ndarray:
    """Bilinear sample ``img`` at pixel coords ``u, v`` with seam/pole rules.

    ``img`` may be ``H x W`` or ``H x W x C``.  Output shape is
    ``u.shape + (C,)`` (or ``u.shape`` for 2-D images).
    """
    img = np.asarray(img)
    h, w = img.shape[:2]
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    u, v = reflect_polar_coords(u, v, w, h)

    x0 = np.floor(u).astype(np.int64)
    y0 = np.floor(v).astype(np.int64)
    fx = u - x0
    fy = v - y0

    x0_mod = x0 % w
    x1_mod = (x0 + 1) % w  # longitude seam adjacency
    y0c = np.clip(y0, 0, h - 1)
    y1c = np.clip(y0 + 1, 0, h - 1)

    if img.ndim == 3:
        fx = fx[..., None]
        fy = fy[..., None]
        Ia = img[y0c, x0_mod]
        Ib = img[y0c, x1_mod]
        Ic = img[y1c, x0_mod]
        Id = img[y1c, x1_mod]
        top = Ia * (1.0 - fx) + Ib * fx
        bot = Ic * (1.0 - fx) + Id * fx
        return top * (1.0 - fy) + bot * fy

    Ia = img[y0c, x0_mod]
    Ib = img[y0c, x1_mod]
    Ic = img[y1c, x0_mod]
    Id = img[y1c, x1_mod]
    top = Ia * (1.0 - fx) + Ib * fx
    bot = Ic * (1.0 - fx) + Id * fx
    return top * (1.0 - fy) + bot * fy


def sample_directions(img: np.ndarray, dirs: np.ndarray) -> np.ndarray:
    """Sample an equirect image for an array of unit directions."""
    h, w = img.shape[:2]
    u, v = sphere.direction_to_pixel(dirs, w, h)
    return sample_equirect(img, u, v)


# ---------------------------------------------------------------------------
# Rotation-vector field interpolation on the equirect grid
# ---------------------------------------------------------------------------

def fill_missing_omega(omega: np.ndarray, support: np.ndarray,
                       global_omega: np.ndarray, passes: int = 24) -> np.ndarray:
    """Replace unsupported nodes with the global rotation and diffuse holes.

    ``omega`` has shape ``rows x cols x 3``.  Unsupported nodes start from the
    robust global camera rotation; a seam-wrapped Jacobi diffusion then blends
    values near supported neighbours so bilinear interpolation cannot create
    spikes.  Supported estimates are kept fixed.
    """
    out = omega.copy()
    sup = support.astype(bool).copy()
    out[~sup] = global_omega

    rows, cols = out.shape[:2]
    for _ in range(passes):
        acc = np.zeros_like(out)
        cnt = np.zeros((rows, cols, 1))
        # 4-neighbour average with horizontal seam wrap and vertical clamp.
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
        # Only relax currently-unsupported cells; supported stay as measured.
        out = np.where(sup[..., None], out, 0.5 * out + 0.5 * blended)
    return out


def interpolate_omega_grid(omega_grid: np.ndarray, width: int, height: int):
    """Bilinearly sample a rows x cols x 3 world-frame omega grid at all pixels.

    Horizontal indices wrap at the longitude seam.  Vertical coordinates are
    clamped to the node rows (polar rows carry the nearest measured rotation
    vectors, which are world vectors and therefore do not blow up).
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
    out = (
        omega_grid[j0, i0] * w00[..., None]
        + omega_grid[j0, i1] * w10[..., None]
        + omega_grid[j1, i0] * w01[..., None]
        + omega_grid[j1, i1] * w11[..., None]
    )
    return out


def invert_omega_field(omega_pixels: np.ndarray, width: int, height: int,
                       iterations: int = 8) -> np.ndarray:
    """Invert a forward map ``d' = Rodrigues(omega(d)) d`` per output pixel.

    For every output direction ``d_out`` we seek the source direction ``d``
    satisfying ``d_out = Exp(omega(d)) d`` with fixed-point iteration
    ``d <- Exp(-omega(d)) d_out``.  Motion per frame is small so this converges
    in a few iterations.  Returns source unit directions.
    """
    UU, VV = np.meshgrid(np.arange(width), np.arange(height))
    d_out = sphere.pixel_to_direction(UU, VV, width, height)
    d = d_out.copy()
    for _ in range(iterations):
        # omega_pixels is already a per-pixel field (H x W x 3); sample it at
        # the current source-direction guess using seam/pole-aware indexing.
        u, v = sphere.direction_to_pixel(d, width, height)
        w_at_d = sample_equirect(omega_pixels, u, v)
        d = sphere.rodrigues(-w_at_d, d_out)
    return d


def reproject_frame(frame_a: np.ndarray, omega_grid: np.ndarray) -> np.ndarray:
    """Warp ``frame_a`` into the next frame using a forward omega grid.

    The forward motion field is inverted per pixel and the source is sampled
    with seam/pole-correct bilinear interpolation.
    """
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
