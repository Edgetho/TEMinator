# SPDX-License-Identifier: GPL-2.0-only
# Copyright (C) 2026 Cooper Stuntz
# See LICENSE for full license terms.

"""Pure EDS metadata parsing helpers for validation and tests."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from eds_models import EDSMetadataContext


def coerce_float(value: Any) -> Optional[float]:
    """Attempt numeric conversion, returning None for invalid values."""
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def first_present_number(candidates: Sequence[Any]) -> Optional[float]:
    """Return first candidate that can be converted to float."""
    for value in candidates:
        converted = coerce_float(value)
        if converted is not None:
            return converted
    return None


def _extract_custom_property_numeric(meta: Dict[str, Any], key_suffix: str) -> Optional[float]:
    custom_props = meta.get("CustomProperties", {})
    if not isinstance(custom_props, dict):
        return None
    for key, entry in custom_props.items():
        if not isinstance(key, str):
            continue
        if not key.endswith(key_suffix):
            continue
        if isinstance(entry, dict) and "value" in entry:
            value = coerce_float(entry.get("value"))
        else:
            value = coerce_float(entry)
        if value is not None:
            return value
    return None


def build_eds_metadata_context(
    original_meta: Dict[str, Any],
    mapped_meta: Optional[Dict[str, Any]] = None,
) -> EDSMetadataContext:
    """Resolve EDS calibration and timing metadata via fallback hierarchy."""
    mapped_meta = mapped_meta or {}
    mapped_sample = mapped_meta.get("Sample", {}) if isinstance(mapped_meta, dict) else {}
    mapped_tem = (
        mapped_meta.get("Acquisition_instrument", {}).get("TEM", {})
        if isinstance(mapped_meta, dict)
        else {}
    )

    original_tem = original_meta.get("Acquisition_instrument", {}).get("TEM", {})
    beam_energy_ev = first_present_number(
        [
            mapped_tem.get("beam_energy") if isinstance(mapped_tem, dict) else None,
            original_tem.get("beam_energy") if isinstance(original_tem, dict) else None,
            original_meta.get("Acquisition", {}).get("BeamEnergy"),
        ]
    )

    dispersion_ev_per_channel = None
    offset_ev = None
    begin_energy_ev = None
    live_time_s = None
    real_time_s = None

    detectors = original_meta.get("Detectors", {})
    if isinstance(detectors, dict):
        for det_info in detectors.values():
            if not isinstance(det_info, dict):
                continue
            if det_info.get("DetectorType") != "AnalyticalDetector":
                continue
            if dispersion_ev_per_channel is None:
                dispersion_ev_per_channel = coerce_float(det_info.get("Dispersion"))
            if offset_ev is None:
                offset_ev = coerce_float(det_info.get("OffsetEnergy"))
            if begin_energy_ev is None:
                begin_energy_ev = coerce_float(det_info.get("BeginEnergy"))
            if live_time_s is None:
                live_time_s = coerce_float(det_info.get("LiveTime"))
            if real_time_s is None:
                real_time_s = coerce_float(det_info.get("RealTime"))

    if dispersion_ev_per_channel is None:
        dispersion_ev_per_channel = _extract_custom_property_numeric(original_meta, ".Dispersion")
    if begin_energy_ev is None:
        begin_energy_ev = _extract_custom_property_numeric(original_meta, ".SpectrumBeginEnergy")
    if live_time_s is None:
        live_time_s = _extract_custom_property_numeric(original_meta, ".LiveTime")
    if real_time_s is None:
        real_time_s = _extract_custom_property_numeric(original_meta, ".RealTime")

    if offset_ev is None and begin_energy_ev is not None:
        offset_ev = begin_energy_ev

    xray_lines: List[str] = []
    lines_from_mapped = mapped_sample.get("xray_lines", []) if isinstance(mapped_sample, dict) else []
    lines_from_original = original_meta.get("Sample", {}).get("xray_lines", [])
    for candidate in (lines_from_mapped, lines_from_original):
        if isinstance(candidate, list) and candidate:
            xray_lines = [str(line) for line in candidate]
            break

    return EDSMetadataContext(
        beam_energy_ev=beam_energy_ev,
        dispersion_ev_per_channel=dispersion_ev_per_channel,
        offset_ev=offset_ev,
        live_time_s=live_time_s,
        real_time_s=real_time_s,
        xray_lines=xray_lines,
    )
