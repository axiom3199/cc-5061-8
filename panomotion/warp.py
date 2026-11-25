"""Backward-compatible adapter for the pre-split warp entry points.

The projection/sampler/field boundary split moved the implementations to
:mod:`panomotion.sampling` and :mod:`panomotion.field`, and the seam/pole
rules to :mod:`panomotion.topology`.  This module keeps every historical
import path working (``from panomotion.warp import sample_equirect, ...``) so
existing spherical-motion callers and samples run unchanged; new code should
import from the owning modules directly.
"""

from __future__ import annotations

from .sampling import sample_directions, sample_equirect, to_float, to_uint8
from .field import (fill_missing_omega, interpolate_omega_grid,
                    invert_omega_field, photometric_error, reproject_frame)
from .topology import reflect_polar_pixel as reflect_polar_coords

__all__ = [
    "to_float", "to_uint8", "reflect_polar_coords", "sample_equirect",
    "sample_directions", "fill_missing_omega", "interpolate_omega_grid",
    "invert_omega_field", "reproject_frame", "photometric_error",
]
