# SPDX-License-Identifier: GPL-2.0-only
# Copyright (C) 2026 Cooper Stuntz
# See LICENSE for full license terms.

"""Shared unit parsing, conversion, and scale-bar helpers."""

from __future__ import annotations

import re
from typing import Optional, Tuple

_UNDEFINED_UNIT_TOKENS = {"", "<undefined>", "undefined", "none", "null"}


def normalize_axis_unit(unit_text: str | None, default: str = "nm") -> str:
    """Normalize user/axis unit text and apply a sane default when unset.

    Args:
        unit_text: Input value for unit text.
        default: Input value for default.

    Returns:
        Detailed parameter description.

    """

    if unit_text is None:
        raw = ""
    else:
        try:
            raw = str(unit_text).strip()
        except Exception:
            raw = ""
    if raw.lower() in _UNDEFINED_UNIT_TOKENS:
        return default
    return re.sub(r"\s+", "", raw)


def reciprocal_denominator(unit_text: str | None) -> Optional[str]:
    """Return denominator unit for reciprocal forms like '1/nm' or 'nm^-1'.

    Args:
        unit_text: Input value for unit text.

    Returns:
        Detailed parameter description.

    """

    unit = normalize_axis_unit(unit_text, default="")
    if not unit:
        return None

    if unit.startswith("1/"):
        return unit[2:] or None
    if unit.endswith("^-1"):
        return unit[:-3] or None
    if unit.endswith("⁻¹"):
        return unit[:-2] or None
    if unit.endswith("-1") and len(unit) > 2 and unit[-2].isdigit() is False:
        return unit[:-2] or None
    return None


def is_reciprocal_unit(unit_text: str | None) -> bool:
    """Return True when a unit string is explicitly reciprocal.

    Args:
        unit_text: Input value for unit text.

    Returns:
        Detailed parameter description.

    """

    return reciprocal_denominator(unit_text) is not None


def unit_kind(unit_text: str | None) -> Optional[str]:
    """Return 'linear', 'reciprocal', or None for missing unit text.

    Args:
        unit_text: Input value for unit text.

    Returns:
        Detailed parameter description.

    """

    unit = normalize_axis_unit(unit_text, default="")
    if not unit:
        return None
    return "reciprocal" if is_reciprocal_unit(unit) else "linear"


def split_value_and_unit(text: str) -> Optional[Tuple[float, Optional[str]]]:
    """Parse strings like '10', '10nm', '10 nm', or '2 nm-1'.

    Reciprocal units in user input must use suffix notation, e.g. ``nm-1``.
    Slash notation such as ``1/nm`` is intentionally rejected for input.

    Args:
        text: User-facing text value for this operation.

    Returns:
        Detailed parameter description.

    """

    raw = (text or "").strip()
    if not raw:
        return None

    match = re.match(
        r"^([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)(?:\s*(\S+))?$",
        raw,
    )
    if match is None:
        return None

    value_text = match.group(1)
    unit_text = match.group(2)

    try:
        value = float(value_text)
    except ValueError:
        return None

    if unit_text is None:
        return value, None

    normalized_unit = normalize_axis_unit(unit_text, default="")
    if "/" in normalized_unit:
        return None

    return value, normalized_unit or None


def _linear_unit_to_meter_factor(unit_text: str | None) -> Optional[float]:
    """Return multiplicative factor to convert a linear unit to meters.

    Args:
        unit_text: Input value for unit text.

    Returns:
        Detailed parameter description.

    """

    unit = (
        normalize_axis_unit(unit_text, default="").replace("μ", "u").replace("Å", "A")
    )
    if not unit:
        return None

    unit_lower = unit.lower()
    if unit_lower in {"a", "ang", "angstrom", "angstroem"}:
        return 1e-10

    if unit_lower == "m":
        return 1.0

    prefixes = {
        "k": 1e3,
        "": 1.0,
        "m": 1e-3,
        "u": 1e-6,
        "n": 1e-9,
        "p": 1e-12,
        "f": 1e-15,
    }

    if len(unit_lower) == 2 and unit_lower.endswith("m"):
        return prefixes.get(unit_lower[0])

    return None


def convert_distance_value(
    value: float, source_unit: str, target_unit: str
) -> Optional[float]:
    """Convert a distance value between compatible linear or reciprocal units.

    Args:
        value: Input value for value.
        source_unit: Input value for source unit.
        target_unit: Input value for target unit.

    Returns:
        Detailed parameter description.

    """

    source = normalize_axis_unit(source_unit, default="")
    target = normalize_axis_unit(target_unit, default="")
    if not source or not target:
        return None

    source_denom = reciprocal_denominator(source)
    target_denom = reciprocal_denominator(target)

    source_kind = "reciprocal" if source_denom is not None else "linear"
    target_kind = "reciprocal" if target_denom is not None else "linear"
    if source_kind != target_kind:
        return None

    source_linear = source_denom if source_denom is not None else source
    target_linear = target_denom if target_denom is not None else target

    source_factor = _linear_unit_to_meter_factor(source_linear)
    target_factor = _linear_unit_to_meter_factor(target_linear)
    if source_factor is None or target_factor is None:
        return None

    if source_kind == "linear":
        value_m = value * source_factor
        return float(value_m / target_factor)

    value_inv_m = value / source_factor
    return float(value_inv_m * target_factor)


def parse_distance_to_target_units(
    distance_text: str,
    target_unit: str,
) -> Optional[Tuple[float, Optional[str]]]:
    """Parse and convert a typed distance expression to target units.

    Args:
        distance_text: Input value for distance text.
        target_unit: Input value for target unit.

    Returns:
        Detailed parameter description.

    """

    parsed = split_value_and_unit(distance_text)
    if parsed is None:
        return None

    value, explicit_unit = parsed
    clean_target = normalize_axis_unit(target_unit, default="nm")
    source_unit = explicit_unit or clean_target

    converted = convert_distance_value(value, source_unit, clean_target)
    if converted is None:
        return None

    return converted, explicit_unit


def scale_bar_unit_and_mode(
    axis_unit: str | None, reciprocal_hint: bool
) -> Tuple[str, bool]:
    """Return (display_base_unit, reciprocal_mode) for DynamicScaleBar.

    Args:
        axis_unit: Input value for axis unit.
        reciprocal_hint: Input value for reciprocal hint.

    Returns:
        Detailed parameter description.

    """

    normalized = normalize_axis_unit(axis_unit, default="nm")
    denom = reciprocal_denominator(normalized)
    if denom:
        return normalize_axis_unit(denom, default="nm"), True
    return normalized, bool(reciprocal_hint)
