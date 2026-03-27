# SPDX-License-Identifier: GPL-2.0-only
# Copyright (C) 2026 Cooper Stuntz
# See LICENSE for full license terms.

"""Pure line-profile extraction helpers used by measurement controllers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import numpy as np  # type: ignore[reportMissingImports]

import unit_utils


@dataclass(frozen=True)
class RectMapping:
    """Affine mapping from view coordinates to pixel coordinates.

    Attributes:
        x_origin: View-space X origin corresponding to pixel index 0.
        y_origin: View-space Y origin corresponding to pixel index 0.
        scale_x_px: Pixel-per-view-unit factor on X.
        scale_y_px: Pixel-per-view-unit factor on Y.
    """

    x_origin: float
    y_origin: float
    scale_x_px: float
    scale_y_px: float


@dataclass(frozen=True)
class AxisCalibration:
    """Axis scale/offset values used for coordinate conversion.

    Attributes:
        scale_x: X axis scale in world-units per pixel.
        scale_y: Y axis scale in world-units per pixel.
        offset_x: X axis world offset.
        offset_y: Y axis world offset.
    """

    scale_x: float
    scale_y: float
    offset_x: float
    offset_y: float


def rect_mapping_from_rect(
    *,
    image_width: int,
    image_height: int,
    rect_left: float,
    rect_top: float,
    rect_width: float,
    rect_height: float,
) -> Optional[RectMapping]:
    """Build a rect-based view-to-pixel mapping when rect dimensions are valid.

    Args:
        image_width: Image width in pixels.
        image_height: Image height in pixels.
        rect_left: View-space left coordinate of the image rect.
        rect_top: View-space top coordinate of the image rect.
        rect_width: View-space rect width.
        rect_height: View-space rect height.

    Returns:
        A mapping object when dimensions are non-zero, otherwise ``None``.
    """
    if rect_width == 0.0 or rect_height == 0.0:
        return None

    return RectMapping(
        x_origin=float(rect_left),
        y_origin=float(rect_top),
        scale_x_px=float(image_width - 1) / float(rect_width),
        scale_y_px=float(image_height - 1) / float(rect_height),
    )


def map_view_points_to_pixel(
    *,
    p1: Tuple[float, float],
    p2: Tuple[float, float],
    rect_mapping: Optional[RectMapping],
    axis_calibration: Optional[AxisCalibration],
) -> Optional[tuple[float, float, float, float]]:
    """Map two view-space points into pixel coordinates.

    Args:
        p1: First profile point in view coordinates.
        p2: Second profile point in view coordinates.
        rect_mapping: Optional view-rect mapping.
        axis_calibration: Optional axis calibration values.

    Returns:
        Tuple ``(x0, y0, x1, y1)`` in pixel coordinates, or ``None`` when axis
        calibration is invalid (for example zero scale).
    """
    if rect_mapping is not None:
        x0 = (float(p1[0]) - rect_mapping.x_origin) * rect_mapping.scale_x_px
        y0 = (float(p1[1]) - rect_mapping.y_origin) * rect_mapping.scale_y_px
        x1 = (float(p2[0]) - rect_mapping.x_origin) * rect_mapping.scale_x_px
        y1 = (float(p2[1]) - rect_mapping.y_origin) * rect_mapping.scale_y_px
        return x0, y0, x1, y1

    if axis_calibration is not None:
        if axis_calibration.scale_x == 0.0 or axis_calibration.scale_y == 0.0:
            return None
        x0 = (float(p1[0]) - axis_calibration.offset_x) / axis_calibration.scale_x
        y0 = (float(p1[1]) - axis_calibration.offset_y) / axis_calibration.scale_y
        x1 = (float(p2[0]) - axis_calibration.offset_x) / axis_calibration.scale_x
        y1 = (float(p2[1]) - axis_calibration.offset_y) / axis_calibration.scale_y
        return x0, y0, x1, y1

    return float(p1[0]), float(p1[1]), float(p2[0]), float(p2[1])


def endpoints_are_finite(x0: float, y0: float, x1: float, y1: float) -> bool:
    """Return whether all endpoint coordinates are finite numbers.

    Args:
        x0: First endpoint X coordinate.
        y0: First endpoint Y coordinate.
        x1: Second endpoint X coordinate.
        y1: Second endpoint Y coordinate.

    Returns:
        ``True`` when all coordinates are finite.
    """
    return bool(np.isfinite((x0, y0, x1, y1)).all())


def clamp_profile_endpoints(
    *,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    width: int,
    height: int,
) -> tuple[float, float, float, float]:
    """Clamp profile endpoints to image bounds.

    Args:
        x0: First endpoint X coordinate.
        y0: First endpoint Y coordinate.
        x1: Second endpoint X coordinate.
        y1: Second endpoint Y coordinate.
        width: Image width in pixels.
        height: Image height in pixels.

    Returns:
        Clamped endpoint tuple ``(x0, y0, x1, y1)``.
    """
    return (
        float(np.clip(x0, 0.0, float(width - 1))),
        float(np.clip(y0, 0.0, float(height - 1))),
        float(np.clip(x1, 0.0, float(width - 1))),
        float(np.clip(y1, 0.0, float(height - 1))),
    )


def compute_sample_count(
    *,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    width: int,
    height: int,
) -> int:
    """Compute bounded sample count policy for a line segment.

    Args:
        x0: First endpoint X coordinate.
        y0: First endpoint Y coordinate.
        x1: Second endpoint X coordinate.
        y1: Second endpoint Y coordinate.
        width: Image width in pixels.
        height: Image height in pixels.

    Returns:
        Number of sample points, clamped to a safe maximum proportional to
        image size.
    """
    dx = x1 - x0
    dy = y1 - y0
    sample_count = int(max(abs(dx), abs(dy))) + 1
    max_samples = max(width, height) * 4
    return max(2, min(sample_count, max_samples))


def sample_line_coordinates(
    *,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    sample_count: int,
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate clipped line sample coordinates.

    Args:
        x0: First endpoint X coordinate.
        y0: First endpoint Y coordinate.
        x1: Second endpoint X coordinate.
        y1: Second endpoint Y coordinate.
        sample_count: Number of requested sample points.
        width: Image width in pixels.
        height: Image height in pixels.

    Returns:
        Tuple of sampled and clipped ``(xs, ys)`` arrays.
    """
    xs = np.linspace(x0, x1, sample_count)
    ys = np.linspace(y0, y1, sample_count)
    xs_clipped = np.clip(xs, 0.0, float(width - 1))
    ys_clipped = np.clip(ys, 0.0, float(height - 1))
    return xs_clipped, ys_clipped


def pixel_distance_axis(xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """Compute cumulative distance axis in pixel units.

    Args:
        xs: Sampled X coordinates in pixel space.
        ys: Sampled Y coordinates in pixel space.

    Returns:
        Distance from the first sample point for each sample.
    """
    return np.sqrt((xs - xs[0]) ** 2 + (ys - ys[0]) ** 2)


def resolve_profile_axis_unit(
    *,
    view_mode: str,
    axis_units: Optional[str],
    freq_axis_base_unit: Optional[str],
) -> str:
    """Return the effective base unit for profile x-axis formatting.

    Args:
        view_mode: Current viewer mode (for example ``image`` or ``fft``).
        axis_units: Axis units from the current view when available.
        freq_axis_base_unit: FFT base unit used for reciprocal-space displays.

    Returns:
        Normalized unit string used for downstream distance labeling.
    """
    if view_mode == "fft":
        axis_unit, _ = unit_utils.scale_bar_unit_and_mode(
            freq_axis_base_unit, reciprocal_hint=True
        )
        return axis_unit

    return unit_utils.normalize_axis_unit(axis_units, default="")


def world_distance_axis(
    *,
    xs: np.ndarray,
    ys: np.ndarray,
    rect_mapping: Optional[RectMapping],
    axis_calibration: Optional[AxisCalibration],
) -> np.ndarray:
    """Compute distance axis in world units from sampled pixel coordinates.

    Args:
        xs: Sampled X coordinates in pixel space.
        ys: Sampled Y coordinates in pixel space.
        rect_mapping: Optional view-to-pixel mapping.
        axis_calibration: Optional axis calibration values.

    Returns:
        World-space distance from the first sample point for each sample.
    """
    if rect_mapping is not None:
        dx_units = (xs - xs[0]) / rect_mapping.scale_x_px
        dy_units = (ys - ys[0]) / rect_mapping.scale_y_px
    elif axis_calibration is not None:
        dx_units = (xs - xs[0]) * axis_calibration.scale_x
        dy_units = (ys - ys[0]) * axis_calibration.scale_y
    else:
        dx_units = xs - xs[0]
        dy_units = ys - ys[0]

    return np.sqrt(dx_units**2 + dy_units**2)


def scaled_distance_axis(
    *,
    distances_world: np.ndarray,
    axis_unit: str,
    view_mode: str,
    is_reciprocal_space: bool,
    format_reciprocal_scale: Callable[[float, str], tuple[float, str]],
    format_si_scale: Callable[[float, str], tuple[float, str]],
) -> tuple[np.ndarray, str, Optional[str], Optional[float], Optional[float]]:
    """Scale world distances and return display-axis metadata.

    Args:
        distances_world: Distance samples in world units.
        axis_unit: Base unit for formatting.
        view_mode: Current viewer mode.
        is_reciprocal_space: Whether the active view is reciprocal-space.
        format_reciprocal_scale: Formatter for reciprocal units.
        format_si_scale: Formatter for SI units.

    Returns:
        Tuple of ``(distances, x_axis_label, display_unit, reference_distance, scaled_reference)``.
    """
    reference_distance = float(distances_world[-1]) if distances_world.size else 0.0
    if reference_distance <= 0 or not np.isfinite(reference_distance):
        return (
            distances_world,
            f"Distance ({axis_unit})",
            None,
            reference_distance,
            None,
        )

    if view_mode == "fft" or is_reciprocal_space:
        scaled_ref, display_unit = format_reciprocal_scale(
            reference_distance, axis_unit
        )
    else:
        scaled_ref, display_unit = format_si_scale(reference_distance, axis_unit)

    scale_factor = (
        float(scaled_ref) / reference_distance if reference_distance != 0 else 1.0
    )
    return (
        distances_world * scale_factor,
        f"Distance ({display_unit})",
        display_unit,
        reference_distance,
        float(scaled_ref),
    )
