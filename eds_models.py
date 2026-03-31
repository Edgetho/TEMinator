# SPDX-License-Identifier: GPL-2.0-only
# Copyright (C) 2026 Cooper Stuntz
# See LICENSE for full license terms.

"""Shared EDS data contracts used by viewer and quantification workflows."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class EDSSpectrumSample:
    """Canonical spectrum sample contract for EDS operations."""

    channel_index: int
    energy_ev: float
    counts: float
    source_name: str
    source_metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EDSROIRegion:
    """Canonical region contract for ROI-driven integration."""

    region_id: int
    geometry_type: str
    coordinates: Sequence[Tuple[float, float]]
    calibration_units: str
    timestamp_iso: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EDSIntegrationSettings:
    """Canonical integration settings contract."""

    integration_windows_ev: Sequence[Tuple[float, float]] = field(default_factory=list)
    background_mode: str = "none"
    included_lines: Sequence[str] = field(default_factory=list)


@dataclass(frozen=True)
class EDSQuantResultRow:
    """Canonical quantification result row contract."""

    region_id: int
    element: str
    counts: float
    weight_percent: Optional[float]
    atomic_percent: Optional[float]
    method: str
    warnings: Sequence[str] = field(default_factory=list)


@dataclass(frozen=True)
class EDSMetadataContext:
    """Resolved metadata needed for EDS calibration and quantification."""

    beam_energy_ev: Optional[float] = None
    dispersion_ev_per_channel: Optional[float] = None
    offset_ev: Optional[float] = None
    live_time_s: Optional[float] = None
    real_time_s: Optional[float] = None
    xray_lines: Sequence[str] = field(default_factory=list)


@dataclass(frozen=True)
class EDSCapabilityState:
    """Runtime feature availability snapshot used for menus and UI controls."""

    has_edx_data: bool
    has_elemental_maps: bool
    has_spectra: bool
    has_integration_regions: bool
    has_energy_calibration: bool
    has_timing_metadata: bool
    has_xray_lines: bool

    def as_dict(self) -> Dict[str, bool]:
        """Return a dict representation for diagnostics/logging."""
        return {
            "has_edx_data": self.has_edx_data,
            "has_elemental_maps": self.has_elemental_maps,
            "has_spectra": self.has_spectra,
            "has_integration_regions": self.has_integration_regions,
            "has_energy_calibration": self.has_energy_calibration,
            "has_timing_metadata": self.has_timing_metadata,
            "has_xray_lines": self.has_xray_lines,
        }
