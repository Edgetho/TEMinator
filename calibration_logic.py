# SPDX-License-Identifier: GPL-2.0-only
# Copyright (C) 2026 Cooper Stuntz
# See LICENSE for full license terms.

"""Pure parsing and validation helpers for calibration workflows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import unit_utils


@dataclass(frozen=True)
class ReferencePPUResult:
    """Result of parsing reference calibration inputs.

    Attributes:
        ppu: Computed pixels-per-unit value, or ``None`` when parsing failed or
            inputs were incomplete.
        target_units: Normalized target unit selected for conversion.
        error: Human-readable validation message, or ``None`` when valid.
    """

    ppu: Optional[float]
    target_units: str
    error: Optional[str]


@dataclass(frozen=True)
class ManualCalibrationResult:
    """Result of parsing manual calibration dialog fields.

    Attributes:
        ppu_x: Parsed X pixels-per-unit value when valid.
        ppu_y: Parsed Y pixels-per-unit value when valid.
        units: Normalized axis unit when parsing succeeded.
        error: Human-readable validation message, or ``None`` when valid.
    """

    ppu_x: Optional[float]
    ppu_y: Optional[float]
    units: Optional[str]
    error: Optional[str]


def default_pixels_per_unit(axis_scale: float) -> float:
    """Return default pixels-per-unit derived from an axis scale.

    Args:
        axis_scale: Axis scale in world-units per pixel.

    Returns:
        ``1/axis_scale`` when positive, otherwise a safe default of ``1.0``.
    """
    return 1.0 / axis_scale if axis_scale > 0 else 1.0


def parse_reference_ppu(
    *,
    reference_pixels_text: str,
    reference_distance_text: str,
    target_units_text: str,
) -> ReferencePPUResult:
    """Parse reference distance fields and compute pixels-per-unit.

    Args:
        reference_pixels_text: Pixel distance text from the dialog.
        reference_distance_text: Physical/reference distance text from the dialog.
        target_units_text: Current unit text in the dialog.

    Returns:
        A result with computed ``ppu`` and normalized ``target_units`` when valid,
        otherwise a human-readable error.
    """
    text_pixels = reference_pixels_text.strip()
    text_distance = reference_distance_text.strip()
    target_units = unit_utils.normalize_axis_unit(target_units_text, default="nm")

    try:
        reference_pixels = float(text_pixels)
    except ValueError:
        if text_pixels:
            return ReferencePPUResult(
                ppu=None,
                target_units=target_units,
                error="Reference pixels must be a valid number.",
            )
        return ReferencePPUResult(ppu=None, target_units=target_units, error=None)

    raw_distance = unit_utils.split_value_and_unit(text_distance)
    if raw_distance is None:
        if text_distance:
            return ReferencePPUResult(
                ppu=None,
                target_units=target_units,
                error="Reference distance must be like '10', '10 nm', or '2 nm-1' (not '1/nm').",
            )
        return ReferencePPUResult(ppu=None, target_units=target_units, error=None)

    _value_raw, explicit_unit = raw_distance
    if explicit_unit and unit_utils.unit_kind(explicit_unit) != unit_utils.unit_kind(
        target_units
    ):
        target_units = unit_utils.normalize_axis_unit(explicit_unit, default="nm")

    parsed = unit_utils.parse_distance_to_target_units(text_distance, target_units)
    if parsed is None:
        if text_distance:
            return ReferencePPUResult(
                ppu=None,
                target_units=target_units,
                error="Could not parse/convert reference distance. Use reciprocal units as '<unit>-1' (e.g., 'nm-1').",
            )
        return ReferencePPUResult(ppu=None, target_units=target_units, error=None)

    reference_units, _parsed_unit = parsed
    if reference_pixels <= 0 or reference_units <= 0:
        return ReferencePPUResult(
            ppu=None,
            target_units=target_units,
            error="Reference pixel and distance values must be greater than zero.",
        )

    return ReferencePPUResult(
        ppu=reference_pixels / reference_units,
        target_units=target_units,
        error=None,
    )


def parse_manual_calibration(
    *,
    ppu_x_text: str,
    ppu_y_text: str,
    units_text: str,
    lock_xy: bool,
) -> ManualCalibrationResult:
    """Parse and validate manual calibration dialog values.

    Args:
        ppu_x_text: Raw X pixels-per-unit text.
        ppu_y_text: Raw Y pixels-per-unit text.
        units_text: Raw axis-unit text.
        lock_xy: Whether Y should be forced to match X.

    Returns:
        Parsed values and normalized units when valid, or an error describing
        the first validation failure encountered.
    """
    try:
        ppu_x = float(ppu_x_text.strip())
        ppu_y = float(ppu_y_text.strip())
        units = unit_utils.normalize_axis_unit(units_text, default="nm")
    except ValueError:
        return ManualCalibrationResult(
            ppu_x=None,
            ppu_y=None,
            units=None,
            error="Pixels-per-unit values must be valid numbers.",
        )

    if lock_xy:
        ppu_y = ppu_x

    if ppu_x <= 0 or ppu_y <= 0:
        return ManualCalibrationResult(
            ppu_x=None,
            ppu_y=None,
            units=None,
            error="Pixels-per-unit values must be greater than zero.",
        )

    return ManualCalibrationResult(
        ppu_x=ppu_x,
        ppu_y=ppu_y,
        units=units,
        error=None,
    )


def should_preserve_metadata_status(
    *,
    metadata_reloaded_in_dialog: bool,
    proposed_scale_x: float,
    proposed_scale_y: float,
    current_scale_x: float,
    current_scale_y: float,
    new_units: str,
    current_units: str,
    tolerance: float = 1e-15,
) -> bool:
    """Return whether accepted values match reloaded metadata calibration.

    Args:
        metadata_reloaded_in_dialog: Whether metadata reload was triggered in
            the current dialog session.
        proposed_scale_x: Proposed X scale from dialog values.
        proposed_scale_y: Proposed Y scale from dialog values.
        current_scale_x: Current X scale on the axes.
        current_scale_y: Current Y scale on the axes.
        new_units: Proposed normalized unit string.
        current_units: Current normalized unit string.
        tolerance: Absolute tolerance for scale comparison.

    Returns:
        ``True`` when metadata was reloaded and accepted values are unchanged,
        otherwise ``False``.
    """
    if not metadata_reloaded_in_dialog:
        return False

    same_x = abs(proposed_scale_x - current_scale_x) <= tolerance
    same_y = abs(proposed_scale_y - current_scale_y) <= tolerance
    same_units = new_units == current_units
    return bool(same_x and same_y and same_units)
