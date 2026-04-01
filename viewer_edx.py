# SPDX-License-Identifier: GPL-2.0-only
# Copyright (C) 2026 Cooper Stuntz
# See LICENSE for full license terms.

"""EDX spectrum and elemental map viewer for image-viewer windows."""

from __future__ import annotations

import csv
import json
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from time import monotonic
from typing import Any, Dict, List, Optional, Protocol, Sequence, Tuple

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

from eds_models import EDSCapabilityState, EDSMetadataContext, EDSROIRegion
from eds_quantification import EDSQuantificationService, QuantificationRequest
from types_common import LoggerLike


class _SpectrumAnalysisManagerOwner(Protocol):
    """Protocol for objects that own a SpectrumAnalysisManager."""

    signal: Any
    data: np.ndarray
    ax_x: Any
    ax_y: Any
    p1: Any
    glw: Any
    img_orig: Any
    file_path: str
    view_mode: str
    is_reciprocal_space: bool

    def _update_image_display(self) -> None:
        """Refresh the main image display."""
        ...

    def _update_edx_legend_overlay(self) -> None:
        """Refresh the EDX legend overlay."""
        ...

    def edx_manager_start_region_selection(self) -> None:
        """Enter EDX integration region selection mode."""
        ...

    def refresh_edx_menu_state(self) -> None:
        """Refresh dynamic EDS menu enablement from capability state."""
        ...

    def _get_original_metadata_dict_from_signal(self, signal: Any) -> Dict[str, Any]:
        """Get original metadata from signal."""
        ...

    def _apply_axis_calibration_values(
        self,
        dx: float,
        dy: float,
        units: str,
        ox: Optional[float] = None,
        oy: Optional[float] = None,
        source: str = "",
    ) -> None:
        """Apply calibration values to axes."""
        ...


class SpectrumAnalysisManager:
    """Owns EDX spectrum and elemental map data and visualization state."""

    def __init__(self, viewer: _SpectrumAnalysisManagerOwner, logger: LoggerLike):
        """Initialize the EDX spectrum analysis manager.

        Args:
            viewer: The image viewer window that owns this manager.
            logger: Logger for debug output.
        """
        self.viewer = viewer
        self.logger = logger

        # Spectrum data state
        self.spectra: Dict[str, np.ndarray] = {}  # spectrum_name -> 1D array
        self.spectrum_metadata: Dict[str, Dict[str, Any]] = {}  # spectrum_name -> metadata dict
        self.active_spectra: set[str] = set()  # which spectra are currently displayed

        # Elemental map data state
        self.elemental_maps: Dict[str, np.ndarray] = {}  # element_name -> 2D array
        self.map_metadata: Dict[str, Dict[str, Any]] = {}  # element_name -> metadata dict
        self.active_elements: set[str] = set()  # which elements are currently displayed
        
        # Color assignment state
        self.element_colors: Dict[str, Tuple[int, int, int]] = {}  # element -> (R, G, B)
        self.spectrum_colors: Dict[str, Tuple[int, int, int]] = {}  # spectrum -> (R, G, B)
        
        # Integration region state
        self.integration_regions: List[EDSROIRegion] = []
        self.region_count: int = 0
        self.active_region_selector: bool = False
        self.region_rois: Dict[int, pg.RectROI] = {}
        self._region_draw_start: Optional[Tuple[float, float]] = None
        self._region_draw_preview: Optional[pg.RectROI] = None
        self._region_draw_prev_handlers: Optional[Tuple[Any, Any, Any, Any]] = None
        self._region_hover_hint_shown: bool = False
        
        # Energy calibration state
        self.beam_energy_ev: float = 200.0  # default 200 eV
        self.spectrum_dispersion: float = 5.0  # eV per channel
        self.spectrum_offset: float = 0.0  # eV offset
        self.live_time_s: Optional[float] = None
        self.real_time_s: Optional[float] = None
        self.xray_lines: List[str] = []
        self._calibration_source: str = "uncalibrated"
        
        # Cached composite map for display
        self._cached_composite_map: Optional[np.ndarray] = None
        self._cached_composite_needs_update: bool = True
        self._last_hover_update_s: float = 0.0
        self._hover_interval_s: float = 0.05
        
        # UI References (set by image viewer)
        self.edx_panel: Optional[QtWidgets.QWidget] = None
        self.edx_tabs: Optional[QtWidgets.QTabWidget] = None
        self.spectrum_plot: Optional[pg.PlotItem] = None
        self.maps_list: Optional[QtWidgets.QListWidget] = None
        self.results_table: Optional[QtWidgets.QTableWidget] = None
        self.hover_status_label: Optional[QtWidgets.QLabel] = None
        self.quant_method_combo: Optional[QtWidgets.QComboBox] = None
        self.quant_units_combo: Optional[QtWidgets.QComboBox] = None
        self.quant_factors_edit: Optional[QtWidgets.QLineEdit] = None
        self.quant_absorption_checkbox: Optional[QtWidgets.QCheckBox] = None
        self.element_checkboxes: Dict[str, QtWidgets.QCheckBox] = {}
        self.element_name_labels: Dict[str, QtWidgets.QLabel] = {}
        self.element_color_buttons: Dict[str, QtWidgets.QPushButton] = {}
        
        self.quantification_service = EDSQuantificationService()
        self._has_edx_data = False
        self.hover_updates_enabled: bool = True
        self.output_artifacts: List[Dict[str, Any]] = []

    def _notify_capability_state_changed(self) -> None:
        """Notify owner that EDS capabilities may have changed."""
        refresh_hook = getattr(self.viewer, "refresh_edx_menu_state", None)
        if callable(refresh_hook):
            try:
                refresh_hook()
            except Exception:
                pass

    def _trace_event(self, event: str, **fields: Any) -> None:
        """Forward structured export trace logs to the owning viewer when enabled."""
        trace_hook = getattr(self.viewer, "_trace_event", None)
        if not callable(trace_hook):
            return
        trace_hook(f"edx_{event}", **fields)

    @staticmethod
    def _trace_array_stats(array: Any, label: str) -> str:
        """Return compact numeric stats for EDX map/composite diagnostics."""
        if array is None:
            return f"{label}:none"
        try:
            arr = np.asarray(array)
        except Exception as exc:
            return f"{label}:unavailable({type(exc).__name__})"
        if arr.size == 0:
            return f"{label}:shape={arr.shape} dtype={arr.dtype} size=0"

        finite_mask = np.isfinite(arr)
        finite_count = int(np.count_nonzero(finite_mask))
        if finite_count == 0:
            return f"{label}:shape={arr.shape} dtype={arr.dtype} finite=0/{arr.size}"

        finite_values = arr[finite_mask]
        min_val = float(np.min(finite_values))
        max_val = float(np.max(finite_values))
        mean_val = float(np.mean(finite_values))
        std_val = float(np.std(finite_values))
        channels = arr.shape[2] if arr.ndim == 3 else 1
        return (
            f"{label}:shape={arr.shape} dtype={arr.dtype} channels={channels} "
            f"finite={finite_count}/{arr.size} min={min_val:.6g} max={max_val:.6g} "
            f"mean={mean_val:.6g} std={std_val:.6g}"
        )

    def detect_and_load_edx_data(
        self,
        elemental_map_signals: Optional[List[Tuple[str, Any]]] = None,
    ) -> bool:
        """Detect and auto-load EDX data from signal metadata or provided map signals.

        Args:
            elemental_map_signals: Optional list of (element_name, signal) tuples for EDX maps.

        Returns:
            True if EDX data was detected and loaded, False otherwise.
        """
        if self.viewer.signal is None:
            return False

        try:
            # If elemental map signals were provided, load them directly
            if elemental_map_signals:
                self._load_elemental_maps_from_signals(elemental_map_signals)
                self._load_energy_calibration_from_metadata(
                    self.viewer._get_original_metadata_dict_from_signal(self.viewer.signal) or {}
                )
                self._load_element_colors_from_metadata(
                    self.viewer._get_original_metadata_dict_from_signal(self.viewer.signal) or {}
                )
                self._has_edx_data = len(self.elemental_maps) > 0
                if self._has_edx_data:
                    self.logger.info(
                        f"EDX data loaded from signals: {len(self.elemental_maps)} elemental map(s)"
                    )
                return self._has_edx_data

            # Otherwise, try to detect from metadata
            meta = self.viewer._get_original_metadata_dict_from_signal(self.viewer.signal)
            if not meta:
                self.logger.debug("No original metadata found in signal")
                return False

            # Check if Features/SIFeature indicates EDS analysis
            if "Features" not in meta or "SIFeature" not in meta["Features"]:
                self.logger.debug("No SIFeature in metadata (no EDS data)")
                return False

            self._load_spectra_from_metadata(meta)
            self._load_elemental_maps_from_metadata(meta)
            self._load_energy_calibration_from_metadata(meta)
            self._load_element_colors_from_metadata(meta)

            has_spectra = len(self.spectra) > 0
            has_maps = len(self.elemental_maps) > 0

            self._has_edx_data = has_spectra or has_maps
            if self._has_edx_data:
                self.logger.info(
                    f"EDX data loaded: {len(self.spectra)} spectra, "
                    f"{len(self.elemental_maps)} elemental maps"
                )
            return self._has_edx_data

        except Exception as e:
            self.logger.warning(f"Failed to auto-load EDX data: {e}", exc_info=True)
            return False

    def _load_spectra_from_metadata(self, meta: Dict[str, Any]) -> None:
        """Load spectrum data from metadata.

        Args:
            meta: The original metadata dictionary.
        """
        try:
            si_feature = meta["Features"]["SIFeature"].get(list(meta["Features"]["SIFeature"].keys())[0])
            if "spectrums" in si_feature:
                spectrum_refs = si_feature["spectrums"]
                if isinstance(spectrum_refs, list):
                    for spec_ref in spectrum_refs:
                        if isinstance(spec_ref, dict):
                            for name, path in spec_ref.items():
                                # Extract spectrum data from signal if available
                                # For now, store metadata reference
                                self.spectra[name] = None  # Will be loaded on demand
                                self.spectrum_metadata[name] = {"path": path, "detector": name}
                                self.logger.debug(f"Found spectrum: {name} at {path}")
        except (KeyError, IndexError, TypeError) as e:
            self.logger.debug(f"Could not parse spectra from metadata: {e}")

    def _load_elemental_maps_from_metadata(self, meta: Dict[str, Any]) -> None:
        """Load elemental map references from metadata.

        Args:
            meta: The original metadata dictionary.
        """
        try:
            # Try to get element list from Sample metadata
            elements = []
            if "Sample" in meta and "elements" in meta["Sample"]:
                elements = meta["Sample"]["elements"]
            
            # Try to get from mapped metadata
            if not elements and hasattr(self.viewer.signal, "metadata"):
                if "Sample" in self.viewer.signal.metadata:
                    if "elements" in self.viewer.signal.metadata["Sample"]:
                        elements = self.viewer.signal.metadata["Sample"]["elements"]
            
            if elements:
                for element in elements:
                    # Create placeholder for elemental map
                    self.elemental_maps[element] = None  # Will be loaded on demand
                    self.map_metadata[element] = {
                        "element": element,
                        "loaded": False,
                    }
                    self.logger.debug(f"Found element: {element}")
        except (KeyError, TypeError) as e:
            self.logger.debug(f"Could not parse elemental maps from metadata: {e}")

    def _load_elemental_maps_from_signals(
        self, elemental_map_signals: List[Tuple[str, Any]]
    ) -> None:
        """Load elemental maps directly from provided signals.

        Args:
            elemental_map_signals: List of (element_name, signal) tuples.
        """
        try:
            for element_name, signal in elemental_map_signals:
                # Store the signal data directly
                try:
                    map_data = signal.data
                    self.elemental_maps[element_name] = map_data
                    self.map_metadata[element_name] = {
                        "element": element_name,
                        "loaded": True,
                        "signal": signal,
                    }
                    self.logger.debug(
                        f"Loaded elemental map for {element_name}: shape={map_data.shape}"
                    )
                except Exception as e:
                    self.logger.warning(
                        f"Failed to load map for {element_name}: {e}"
                    )
        except Exception as e:
            self.logger.debug(f"Error loading elemental maps from signals: {e}")

    def _load_energy_calibration_from_metadata(self, meta: Dict[str, Any]) -> None:
        """Load energy calibration from detector metadata.

        Args:
            meta: The original metadata dictionary.
        """
        try:
            ctx = self._build_metadata_context(meta)

            if ctx.beam_energy_ev is not None:
                self.beam_energy_ev = float(ctx.beam_energy_ev)
            if ctx.dispersion_ev_per_channel is not None:
                self.spectrum_dispersion = float(ctx.dispersion_ev_per_channel)
            if ctx.offset_ev is not None:
                self.spectrum_offset = float(ctx.offset_ev)

            self.live_time_s = ctx.live_time_s
            self.real_time_s = ctx.real_time_s
            self.xray_lines = list(ctx.xray_lines)

            has_any_calibration = (
                ctx.beam_energy_ev is not None
                or ctx.dispersion_ev_per_channel is not None
                or ctx.offset_ev is not None
            )
            self._calibration_source = "metadata" if has_any_calibration else "uncalibrated"
            self.logger.debug(
                "Resolved EDS metadata context: beam=%s eV dispersion=%s eV/ch "
                "offset=%s eV live_time=%s s real_time=%s s lines=%s",
                ctx.beam_energy_ev,
                ctx.dispersion_ev_per_channel,
                ctx.offset_ev,
                ctx.live_time_s,
                ctx.real_time_s,
                list(ctx.xray_lines),
            )
        except Exception as e:
            self.logger.debug(f"Could not parse energy calibration from metadata: {e}")
            self._calibration_source = "uncalibrated"

    @staticmethod
    def _coerce_float(value: Any) -> Optional[float]:
        """Attempt numeric conversion, returning None for invalid values."""
        try:
            if value is None:
                return None
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _first_present_number(candidates: Sequence[Any]) -> Optional[float]:
        """Return the first valid numeric value found in candidates."""
        for value in candidates:
            converted = SpectrumAnalysisManager._coerce_float(value)
            if converted is not None:
                return converted
        return None

    def _build_metadata_context(self, meta: Dict[str, Any]) -> EDSMetadataContext:
        """Resolve EDS calibration/timing metadata using a fallback hierarchy."""
        def _extract_custom_property_numeric(key_suffix: str) -> Optional[float]:
            custom_props = meta.get("CustomProperties", {})
            if not isinstance(custom_props, dict):
                return None
            for key, entry in custom_props.items():
                if not isinstance(key, str):
                    continue
                if not key.endswith(key_suffix):
                    continue
                if isinstance(entry, dict) and "value" in entry:
                    value = self._coerce_float(entry.get("value"))
                else:
                    value = self._coerce_float(entry)
                if value is not None:
                    return value
            return None

        mapped_meta = getattr(self.viewer.signal, "metadata", None)
        mapped_sample: Dict[str, Any] = {}
        mapped_tem: Dict[str, Any] = {}
        if mapped_meta is not None:
            try:
                if "Sample" in mapped_meta:
                    mapped_sample = dict(mapped_meta["Sample"])
            except Exception:
                mapped_sample = {}
            try:
                if "Acquisition_instrument" in mapped_meta and "TEM" in mapped_meta["Acquisition_instrument"]:
                    mapped_tem = dict(mapped_meta["Acquisition_instrument"]["TEM"])
            except Exception:
                mapped_tem = {}

        original_tem = meta.get("Acquisition_instrument", {}).get("TEM", {})
        beam_energy_ev = self._first_present_number(
            [
                mapped_tem.get("beam_energy"),
                original_tem.get("beam_energy"),
                meta.get("Acquisition", {}).get("BeamEnergy"),
            ]
        )

        dispersion_ev_per_channel = None
        offset_ev = None
        begin_energy_ev = None
        live_time_s = None
        real_time_s = None
        detectors = meta.get("Detectors", {})
        if isinstance(detectors, dict):
            for det_info in detectors.values():
                if not isinstance(det_info, dict):
                    continue
                if det_info.get("DetectorType") != "AnalyticalDetector":
                    continue
                if dispersion_ev_per_channel is None:
                    dispersion_ev_per_channel = self._coerce_float(det_info.get("Dispersion"))
                if offset_ev is None:
                    offset_ev = self._coerce_float(det_info.get("OffsetEnergy"))
                if begin_energy_ev is None:
                    begin_energy_ev = self._coerce_float(det_info.get("BeginEnergy"))
                if live_time_s is None:
                    live_time_s = self._coerce_float(det_info.get("LiveTime"))
                if real_time_s is None:
                    real_time_s = self._coerce_float(det_info.get("RealTime"))

        # Fallbacks from vendor custom properties when detector records are incomplete.
        if dispersion_ev_per_channel is None:
            dispersion_ev_per_channel = _extract_custom_property_numeric(".Dispersion")
        if begin_energy_ev is None:
            begin_energy_ev = _extract_custom_property_numeric(".SpectrumBeginEnergy")
        if live_time_s is None:
            live_time_s = _extract_custom_property_numeric(".LiveTime")
        if real_time_s is None:
            real_time_s = _extract_custom_property_numeric(".RealTime")

        # Prefer explicit offset; otherwise use detector begin energy as channel-0 anchor.
        if offset_ev is None and begin_energy_ev is not None:
            offset_ev = begin_energy_ev

        xray_lines: List[str] = []
        lines_from_mapped = mapped_sample.get("xray_lines", [])
        lines_from_original = meta.get("Sample", {}).get("xray_lines", [])
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

    def _load_element_colors_from_metadata(self, meta: Dict[str, Any]) -> None:
        """Load color assignments for elements from metadata.

        Args:
            meta: The original metadata dictionary.
        """
        try:
            # Try to find colormixSelection in operations
            if "Operations" in meta:
                ops = meta["Operations"]
                if "ImageQuantificationOperation" in ops:
                    for op_id, op_data in ops["ImageQuantificationOperation"].items():
                        if "colormixSelection" in op_data:
                            colormix = op_data["colormixSelection"]
                            if isinstance(colormix, list):
                                for entry in colormix:
                                    if isinstance(entry, dict):
                                        name = entry.get("name")
                                        color = entry.get("color", "#ffffff")
                                        # Convert hex color to RGB tuple
                                        rgb = self._hex_to_rgb(color)
                                        self.element_colors[name] = rgb
                                        self.logger.debug(f"Loaded color for {name}: {rgb}")
        except (KeyError, TypeError) as e:
            self.logger.debug(f"Could not parse element colors from metadata: {e}")

    @staticmethod
    def _hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
        """Convert hex color string to RGB tuple.

        Args:
            hex_color: Color string like "#ff0000".

        Returns:
            Tuple of (R, G, B) integers 0-255.
        """
        hex_color = hex_color.lstrip("#")
        if len(hex_color) == 6:
            return tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))
        return (255, 255, 255)  # default to white

    @staticmethod
    def _rgb_to_hex(rgb: Tuple[int, int, int]) -> str:
        """Convert RGB tuple to hex color string.

        Args:
            rgb: Tuple of (R, G, B) integers 0-255.

        Returns:
            Color string like "#ff0000".
        """
        return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"

    def get_energy_axis(self) -> np.ndarray:
        """Get energy axis in eV for spectra display.

        Returns:
            1D array of energy values in eV.
        """
        # Assuming spectrum has fixed number of channels
        # This should be determined from actual spectrum shape
        num_channels = 1024  # placeholder
        energy = self.spectrum_offset + np.arange(num_channels) * self.spectrum_dispersion
        return energy

    def render_composite_map(
        self,
        active_elements: Optional[List[str]] = None,
        normalize_intensity: bool = True,
    ) -> Optional[np.ndarray]:
        """Render composite colored elemental map from selected elements.

        Args:
            active_elements: List of element names to composite. If None, uses self.active_elements.
            normalize_intensity: Whether to normalize each map to [0, 1] before coloring.

        Returns:
            RGB composite image as uint8 array, or None if no maps available.
        """
        if active_elements is None:
            active_elements = list(self.active_elements)

        self._trace_event(
            "render_composite_enter",
            requested_active=active_elements,
            normalize=normalize_intensity,
            total_maps=len(self.elemental_maps),
        )

        if not active_elements:
            self._trace_event("render_composite_empty", reason="no_active_elements")
            return None

        # Load maps if not already loaded
        for elem in active_elements:
            if elem in self.elemental_maps and self.elemental_maps[elem] is None:
                self._load_elemental_map(elem)

        # Get first map to determine output shape
        first_map = None
        for elem in active_elements:
            if elem in self.elemental_maps and self.elemental_maps[elem] is not None:
                first_map = self.elemental_maps[elem]
                break

        if first_map is None:
            self._trace_event("render_composite_empty", reason="no_loaded_maps")
            return None

        # Initialize composite RGB
        composite = np.zeros((*first_map.shape, 3), dtype=np.float32)

        # Composite each map with its assigned color
        for elem in active_elements:
            if elem not in self.elemental_maps or self.elemental_maps[elem] is None:
                self._trace_event(
                    "render_composite_skip_element",
                    element=elem,
                    reason="missing_map_data",
                )
                continue

            map_data = self.elemental_maps[elem].astype(np.float32)
            self._trace_event(
                "render_composite_element_source",
                element=elem,
                source_stats=self._trace_array_stats(map_data, "source"),
            )

            # Normalize to [0, 1]
            if normalize_intensity:
                map_min = np.min(map_data)
                map_max = np.max(map_data)
                if map_max > map_min:
                    map_data = (map_data - map_min) / (map_max - map_min)
                else:
                    map_data = np.zeros_like(map_data)
                self._trace_event(
                    "render_composite_element_normalized",
                    element=elem,
                    map_min=float(map_min),
                    map_max=float(map_max),
                    normalized_stats=self._trace_array_stats(map_data, "normalized"),
                )

            # Get color for this element
            rgb = self.element_colors.get(elem, (255, 255, 255))
            rgb_norm = np.array(rgb, dtype=np.float32) / 255.0

            # Expand map to RGB and multiply by color
            colored_map = np.stack([map_data] * 3, axis=-1)
            colored_map = colored_map * rgb_norm[np.newaxis, np.newaxis, :]

            # Additive blending
            composite += colored_map
            self._trace_event(
                "render_composite_element_blended",
                element=elem,
                color=rgb,
                blended_stats=self._trace_array_stats(composite, "composite_accum"),
            )

        # Clamp to [0, 1] and convert to uint8
        composite = np.clip(composite, 0, 1) * 255
        out = composite.astype(np.uint8)
        self._trace_event(
            "render_composite_exit",
            output_stats=self._trace_array_stats(out, "output_u8"),
        )
        return out

    def _load_elemental_map(self, element: str) -> bool:
        """Load elemental map data from signal or lazy-load from metadata.

        Args:
            element: Element name (e.g., "C", "Ni").

        Returns:
            True if loaded successfully, False otherwise.
        """
        try:
            # Check if already loaded
            if element in self.elemental_maps and self.elemental_maps[element] is not None:
                return True
            
            # Check if we have a signal reference
            if element in self.map_metadata:
                meta = self.map_metadata[element]
                if "signal" in meta and meta["signal"] is not None:
                    map_data = meta["signal"].data
                    self.elemental_maps[element] = map_data
                    self.logger.debug(f"Loaded elemental map for {element}: shape={map_data.shape}")
                    return True
            
            self.logger.debug(f"No map data available for {element}")
            return False
        except Exception as e:
            self.logger.warning(f"Failed to load elemental map for {element}: {e}")
            return False

    def get_has_edx_data(self) -> bool:
        """Check if any EDX data was detected.

        Returns:
            True if spectra or maps were detected.
        """
        return self._has_edx_data

    def get_capability_state(self) -> EDSCapabilityState:
        """Return runtime EDS capability flags used by menus and controls."""
        has_energy_calibration = (
            self._calibration_source != "uncalibrated"
            or self.spectrum_dispersion is not None
            or self.spectrum_offset is not None
        )
        has_timing_metadata = self.live_time_s is not None or self.real_time_s is not None
        return EDSCapabilityState(
            has_edx_data=self._has_edx_data,
            has_elemental_maps=bool(self.elemental_maps),
            has_spectra=bool(self.spectra),
            has_integration_regions=bool(self.integration_regions),
            has_energy_calibration=has_energy_calibration,
            has_timing_metadata=has_timing_metadata,
            has_xray_lines=bool(self.xray_lines),
        )

    def add_integration_region(self, region_data: Dict[str, Any]) -> int:
        """Add an integration region and return its ID.

        Args:
            region_data: Dictionary with region definition (coordinates, elements, etc.).

        Returns:
            Region ID.
        """
        region_id = self.region_count
        geometry_type = str(region_data.get("type", "unknown"))
        calibration_units = str(region_data.get("calibration_units", "px"))
        timestamp_iso = str(
            region_data.get("timestamp")
            or datetime.now(timezone.utc).isoformat()
        )

        coordinates_raw = region_data.get("coordinates", [])
        coordinates: List[Tuple[float, float]] = []
        if isinstance(coordinates_raw, list):
            for coord in coordinates_raw:
                if not isinstance(coord, (list, tuple)) or len(coord) != 2:
                    continue
                x = self._coerce_float(coord[0])
                y = self._coerce_float(coord[1])
                if x is None or y is None:
                    continue
                coordinates.append((x, y))

        reserved_keys = {"id", "type", "calibration_units", "timestamp", "coordinates"}
        metadata = {k: v for k, v in region_data.items() if k not in reserved_keys}

        region = EDSROIRegion(
            region_id=region_id,
            geometry_type=geometry_type,
            coordinates=coordinates,
            calibration_units=calibration_units,
            timestamp_iso=timestamp_iso,
            metadata=metadata,
        )
        self.integration_regions.append(region)
        self.region_count += 1
        self._notify_capability_state_changed()
        self.logger.debug(f"Added integration region {region_id}")
        return region_id

    def remove_integration_region(self, region_id: int) -> bool:
        """Remove an integration region by ID.

        Args:
            region_id: ID of region to remove.

        Returns:
            True if removed, False if not found.
        """
        for i, region in enumerate(self.integration_regions):
            if region.region_id == region_id:
                self.integration_regions.pop(i)
                roi_item = self.region_rois.pop(region_id, None)
                if roi_item is not None and hasattr(self.viewer, "p1") and self.viewer.p1 is not None:
                    try:
                        self.viewer.p1.removeItem(roi_item)
                    except Exception:
                        pass
                self._notify_capability_state_changed()
                self.logger.debug(f"Removed integration region {region_id}")
                return True
        return False

    def clear_integration_regions(self) -> None:
        """Clear all integration regions."""
        if hasattr(self.viewer, "p1") and self.viewer.p1 is not None:
            for roi_item in self.region_rois.values():
                try:
                    self.viewer.p1.removeItem(roi_item)
                except Exception:
                    pass
        self.region_rois.clear()
        self.integration_regions.clear()
        self.region_count = 0
        self.logger.debug("Cleared all integration regions")

    def build_edx_panel(self) -> Optional[QtWidgets.QWidget]:
        """Build the EDX control panel with tabs for spectra, maps, and integration results.

        Returns:
            QWidget containing tabbed interface, or None if no EDX data available.
        """
        if not self._has_edx_data:
            return None

        panel = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(panel)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(5)

        # Create tab widget
        tabs = QtWidgets.QTabWidget()
        
        # Spectrum tab
        spectrum_tab = self._build_spectrum_tab()
        if spectrum_tab:
            tabs.addTab(spectrum_tab, "Spectra")

        # Maps tab
        maps_tab = self._build_maps_tab()
        if maps_tab:
            tabs.addTab(maps_tab, "Maps")

        # Integration tab
        integration_tab = self._build_integration_tab()
        if integration_tab:
            tabs.addTab(integration_tab, "Integration")

        layout.addWidget(tabs)
        self.edx_tabs = tabs
        self.edx_panel = panel
        return panel

    def _build_spectrum_tab(self) -> Optional[QtWidgets.QWidget]:
        """Build the spectrum display and control tab."""
        if not self.spectra and not self.elemental_maps:
            return None

        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)

        # Spectrum plot
        self.spectrum_plot = pg.PlotItem(title="EDS Spectrum")
        self.spectrum_plot.setLabel("bottom", "Energy", units="eV")
        self.spectrum_plot.setLabel("left", "Counts")
        plot_widget = pg.GraphicsLayoutWidget()
        plot_widget.addItem(self.spectrum_plot, row=0, col=0)
        layout.addWidget(plot_widget, 1)

        self.hover_status_label = QtWidgets.QLabel(
            "Hover over the image to inspect spectra."
        )
        self.hover_status_label.setWordWrap(True)
        layout.addWidget(self.hover_status_label)

        # Spectrum selector with checkboxes
        spectrum_list_group = QtWidgets.QGroupBox("Spectrum Sources")
        spectrum_list_layout = QtWidgets.QVBoxLayout()

        for spectrum_name in self.spectra.keys():
            checkbox = QtWidgets.QCheckBox(spectrum_name)
            checkbox.setToolTip(f"Toggle {spectrum_name} spectrum display")
            checkbox.stateChanged.connect(
                lambda state, name=spectrum_name: self._on_spectrum_checkbox_changed(
                    name, state
                )
            )
            spectrum_list_layout.addWidget(checkbox)

        spectrum_list_layout.addStretch()
        spectrum_list_group.setLayout(spectrum_list_layout)
        layout.addWidget(spectrum_list_group, 0)

        # Energy calibration info
        calib_info = QtWidgets.QLabel(
            f"Calibration: {self._calibration_source}\n"
            f"Dispersion: {self.spectrum_dispersion:.1f} eV/ch\n"
            f"Offset: {self.spectrum_offset:.1f} eV\n"
            f"Live/Real time: {self.live_time_s if self.live_time_s is not None else '-'} / "
            f"{self.real_time_s if self.real_time_s is not None else '-'} s"
        )
        calib_info.setWordWrap(True)
        layout.addWidget(calib_info)

        return widget

    def _build_maps_tab(self) -> Optional[QtWidgets.QWidget]:
        """Build the elemental maps display and control tab."""
        if not self.elemental_maps:
            return None

        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)

        # Element/map selector with color pickers
        maps_list_group = QtWidgets.QGroupBox("Elemental Maps")
        maps_list_layout = QtWidgets.QVBoxLayout()
        self.element_checkboxes.clear()
        self.element_name_labels.clear()
        self.element_color_buttons.clear()

        for element_name in self.elemental_maps.keys():
            row_layout = QtWidgets.QHBoxLayout()

            # Checkbox for visibility
            checkbox = QtWidgets.QCheckBox()
            checkbox.setToolTip(f"Toggle {element_name} map display")
            checkbox.stateChanged.connect(
                lambda state, elem=element_name: self._on_element_checkbox_changed(
                    elem, state
                )
            )
            self.element_checkboxes[element_name] = checkbox
            row_layout.addWidget(checkbox)

            element_label = QtWidgets.QLabel(element_name)
            self.element_name_labels[element_name] = element_label
            self._set_label_color(
                element_label,
                self.element_colors.get(element_name, (255, 255, 255)),
            )
            row_layout.addWidget(element_label)

            # Color picker button
            color = self.element_colors.get(element_name, (255, 255, 255))
            color_btn = QtWidgets.QPushButton("")
            color_btn.setMaximumWidth(50)
            self._set_button_color(color_btn, color)
            self.element_color_buttons[element_name] = color_btn
            color_btn.clicked.connect(
                lambda checked=False, elem=element_name: self._on_color_picker_clicked(
                    elem
                )
            )
            row_layout.addWidget(color_btn)

            row_layout.addStretch()
            maps_list_layout.addLayout(row_layout)

        maps_list_layout.addStretch()
        maps_list_group.setLayout(maps_list_layout)
        layout.addWidget(maps_list_group, 1)

        # Normalization checkbox
        normalize_cb = QtWidgets.QCheckBox("Normalize Intensity")
        normalize_cb.setChecked(True)
        normalize_cb.stateChanged.connect(self._on_normalize_changed)
        layout.addWidget(normalize_cb)

        return widget

    def _build_integration_tab(self) -> QtWidgets.QWidget:
        """Build the integration results display tab."""
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)

        quant_group = QtWidgets.QGroupBox("Quantification Options")
        quant_layout = QtWidgets.QFormLayout(quant_group)

        self.quant_method_combo = QtWidgets.QComboBox()
        self.quant_method_combo.addItems(["CL", "Custom"])
        self.quant_method_combo.currentTextChanged.connect(
            lambda _text: self._refresh_results_table()
        )
        quant_layout.addRow("Method:", self.quant_method_combo)

        self.quant_units_combo = QtWidgets.QComboBox()
        self.quant_units_combo.addItems(["atomic", "weight"])
        self.quant_units_combo.setToolTip(
            "Primary display preference for downstream exports; table always shows both wt% and at%."
        )
        quant_layout.addRow("Primary Units:", self.quant_units_combo)

        self.quant_factors_edit = QtWidgets.QLineEdit()
        self.quant_factors_edit.setPlaceholderText("Fe=1.45, Pt=5.08")
        self.quant_factors_edit.textChanged.connect(lambda _text: self._refresh_results_table())
        quant_layout.addRow("Factors:", self.quant_factors_edit)

        self.quant_absorption_checkbox = QtWidgets.QCheckBox("Absorption correction")
        capability = self.get_capability_state()
        self.quant_absorption_checkbox.setEnabled(capability.has_timing_metadata)
        if not capability.has_timing_metadata:
            self.quant_absorption_checkbox.setToolTip(
                "Disabled until required timing metadata is available."
            )
        quant_layout.addRow("Advanced:", self.quant_absorption_checkbox)

        layout.addWidget(quant_group)

        # Results table
        self.results_table = QtWidgets.QTableWidget()
        self.results_table.setColumnCount(6)
        self.results_table.setHorizontalHeaderLabels(
            ["Region ID", "Element", "Counts", "wt%", "at%", "Method"]
        )
        self.results_table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.results_table, 1)

        # Action buttons
        button_layout = QtWidgets.QHBoxLayout()

        select_region_btn = QtWidgets.QPushButton("Select Region")
        select_region_btn.clicked.connect(self._on_select_region_clicked)
        button_layout.addWidget(select_region_btn)

        export_btn = QtWidgets.QPushButton("Export Results")
        export_btn.clicked.connect(self._on_export_results_clicked)
        button_layout.addWidget(export_btn)

        clear_btn = QtWidgets.QPushButton("Clear Results")
        clear_btn.clicked.connect(self._on_clear_results_clicked)
        button_layout.addWidget(clear_btn)

        remove_btn = QtWidgets.QPushButton("Remove Selected")
        remove_btn.clicked.connect(self._on_remove_selected_region_clicked)
        button_layout.addWidget(remove_btn)

        layout.addLayout(button_layout)

        return widget

    @staticmethod
    def _set_button_color(button: QtWidgets.QPushButton, rgb: Tuple[int, int, int]) -> None:
        """Set button background color based on RGB tuple.

        Args:
            button: Button to update.
            rgb: (R, G, B) tuple with values 0-255.
        """
        hex_color = f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"
        button.setStyleSheet(f"background-color: {hex_color};")

    @staticmethod
    def _set_label_color(label: QtWidgets.QLabel, rgb: Tuple[int, int, int]) -> None:
        """Set label text color based on RGB tuple.

        Args:
            label: Label to update.
            rgb: (R, G, B) tuple with values 0-255.
        """
        hex_color = f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"
        label.setStyleSheet(f"color: {hex_color}; font-weight: 600;")

    def _on_spectrum_checkbox_changed(self, spectrum_name: str, state: int) -> None:
        """Handle spectrum visibility toggle.

        Args:
            spectrum_name: Name of spectrum.
            state: Qt checkbox state.
        """
        if state == QtCore.Qt.Checked:
            self.active_spectra.add(spectrum_name)
        else:
            self.active_spectra.discard(spectrum_name)
        self.logger.debug(f"Active spectra: {self.active_spectra}")

    def _on_element_checkbox_changed(self, element: str, state: int) -> None:
        """Handle elemental map visibility toggle.

        Args:
            element: Element name.
            state: Qt checkbox state.
        """
        if state == QtCore.Qt.Checked:
            self.active_elements.add(element)
        else:
            self.active_elements.discard(element)
        self._trace_event(
            "element_checkbox_changed",
            element=element,
            state=int(state),
            active_elements=sorted(list(self.active_elements)),
        )
        self.logger.debug(f"Active elements: {self.active_elements}")
        self._cached_composite_needs_update = True
        if hasattr(self.viewer, "_update_edx_legend_overlay"):
            self.viewer._update_edx_legend_overlay()
        # Trigger image update in parent viewer
        if hasattr(self.viewer, "_update_image_display"):
            self.viewer._update_image_display()

    def _on_color_picker_clicked(self, element: str) -> None:
        """Handle color picker button click.

        Args:
            element: Element name.
        """
        current_color = self.element_colors.get(element, (255, 255, 255))
        current_qcolor = QtGui.QColor(*current_color)

        color = QtWidgets.QColorDialog.getColor(
            initial=current_qcolor,
            parent=self.edx_panel,
            title=f"Pick color for {element}",
        )
        if color.isValid():
            rgb = (color.red(), color.green(), color.blue())
            self.element_colors[element] = rgb
            button = self.element_color_buttons.get(element)
            if button is not None:
                self._set_button_color(button, rgb)
            label = self.element_name_labels.get(element)
            if label is not None:
                self._set_label_color(label, rgb)
            self.logger.debug(f"Color changed for {element}: {rgb}")
            self._cached_composite_needs_update = True
            if hasattr(self.viewer, "_update_edx_legend_overlay"):
                self.viewer._update_edx_legend_overlay()
            # Trigger image update and update button color
            if hasattr(self.viewer, "_update_image_display"):
                self.viewer._update_image_display()

    def _on_normalize_changed(self, state: int) -> None:
        """Handle intensity normalization toggle.

        Args:
            state: Qt checkbox state.
        """
        normalize = state == QtCore.Qt.Checked
        self.logger.debug(f"Normalization toggled: {normalize}")
        self._cached_composite_needs_update = True
        # Trigger image update in parent viewer
        if hasattr(self.viewer, "_update_image_display"):
            self.viewer._update_image_display()

    def _on_select_region_clicked(self) -> None:
        """Handle region selection button click."""
        self.active_region_selector = True
        self.logger.debug("Region selector activated")
        self.show_tab("Integration")
        # Tell parent viewer to enter region selection mode
        if hasattr(self.viewer, "edx_manager_start_region_selection"):
            self.viewer.edx_manager_start_region_selection()

    def _on_export_results_clicked(self) -> None:
        """Handle export results button click."""
        if not self.integration_regions:
            QtWidgets.QMessageBox.information(
                self.edx_panel, "Export", "No integration regions to export."
            )
            return
        self.logger.debug(f"Exporting {len(self.integration_regions)} regions")
        self.prompt_save_all_results()

    def _on_clear_results_clicked(self) -> None:
        """Handle clear results button click."""
        self._safe_remove_preview_roi()
        self._restore_region_draw_handlers()
        self.active_region_selector = False
        self.clear_integration_regions()
        if self.results_table:
            self.results_table.setRowCount(0)
        self._notify_capability_state_changed()
        self.logger.debug("Integration results cleared")

    def show_tab(self, tab_name: str) -> None:
        """Switch EDS tabs by visible tab title."""
        if self.edx_tabs is None:
            return
        for idx in range(self.edx_tabs.count()):
            if self.edx_tabs.tabText(idx).strip().lower() == tab_name.strip().lower():
                self.edx_tabs.setCurrentIndex(idx)
                return

    def set_quant_method(self, method: str) -> None:
        """Set quantification method and refresh rows."""
        if self.quant_method_combo is None:
            return
        target = method.strip().lower()
        for idx in range(self.quant_method_combo.count()):
            if self.quant_method_combo.itemText(idx).strip().lower() == target:
                self.quant_method_combo.setCurrentIndex(idx)
                self._refresh_results_table()
                return

    def toggle_absorption_correction(self) -> None:
        """Toggle absorption correction option when available."""
        if self.quant_absorption_checkbox is None:
            return
        if not self.quant_absorption_checkbox.isEnabled():
            return
        self.quant_absorption_checkbox.setChecked(
            not self.quant_absorption_checkbox.isChecked()
        )

    def toggle_hover_updates(self) -> None:
        """Toggle hover spectra updates on/off."""
        self.hover_updates_enabled = not self.hover_updates_enabled
        if self.hover_status_label is not None:
            status = "enabled" if self.hover_updates_enabled else "disabled"
            self.hover_status_label.setText(f"Hover spectra updates {status}.")

    def _on_remove_selected_region_clicked(self) -> None:
        """Remove the region tied to the selected results-table row."""
        if self.results_table is None:
            return
        row = self.results_table.currentRow()
        if row < 0:
            return
        item = self.results_table.item(row, 0)
        if item is None:
            return
        try:
            region_id = int(item.text())
        except Exception:
            return
        self.remove_integration_region(region_id)
        self._refresh_results_table()

    def begin_rectangle_region_selection(self) -> None:
        """Enter click-drag rectangle ROI mode for EDS integration."""
        plot = getattr(self.viewer, "p1", None)
        if plot is None or not hasattr(plot, "vb"):
            return

        vb = plot.vb
        if self._region_draw_prev_handlers is None:
            self._region_draw_prev_handlers = (
                vb.mousePressEvent,
                vb.mouseMoveEvent,
                vb.mouseReleaseEvent,
                getattr(vb, "mouseDragEvent", None),
            )

        vb.mousePressEvent = self._on_region_mouse_press
        vb.mouseMoveEvent = self._on_region_mouse_move
        vb.mouseReleaseEvent = self._on_region_mouse_release
        if getattr(vb, "mouseDragEvent", None) is not None:
            vb.mouseDragEvent = self._on_region_mouse_drag

        self.active_region_selector = True
        self._notify_capability_state_changed()
        if not self._region_hover_hint_shown:
            self._region_hover_hint_shown = True
            QtWidgets.QMessageBox.information(
                self.edx_panel,
                "EDS Region Selection",
                "Click and drag on the image to define an integration rectangle.",
            )

    def _restore_region_draw_handlers(self) -> None:
        """Restore ViewBox mouse handlers after region selection."""
        plot = getattr(self.viewer, "p1", None)
        if plot is None or not hasattr(plot, "vb"):
            return
        if self._region_draw_prev_handlers is None:
            return
        press, move, release, drag = self._region_draw_prev_handlers
        vb = plot.vb
        vb.mousePressEvent = press
        vb.mouseMoveEvent = move
        vb.mouseReleaseEvent = release
        if drag is not None:
            vb.mouseDragEvent = drag
        self._region_draw_prev_handlers = None

    def _safe_remove_preview_roi(self) -> None:
        """Remove temporary preview ROI if present."""
        if self._region_draw_preview is None:
            return
        plot = getattr(self.viewer, "p1", None)
        try:
            if plot is not None:
                plot.removeItem(self._region_draw_preview)
        except Exception:
            pass
        self._region_draw_preview = None

    def _on_region_mouse_press(self, event: Any) -> None:
        """Handle region-draw mouse press."""
        plot = getattr(self.viewer, "p1", None)
        if plot is None:
            return
        scene_pos = event.scenePos()
        if not plot.sceneBoundingRect().contains(scene_pos):
            return
        view_pos = plot.vb.mapSceneToView(scene_pos)
        self._region_draw_start = (view_pos.x(), view_pos.y())
        self._safe_remove_preview_roi()
        preview = pg.RectROI(
            [view_pos.x(), view_pos.y()],
            [1e-9, 1e-9],
            pen=pg.mkPen(255, 165, 0, width=2, style=QtCore.Qt.DashLine),
            movable=False,
            rotatable=False,
            resizable=False,
        )
        self._region_draw_preview = preview
        plot.addItem(preview)
        event.accept()

    def _on_region_mouse_move(self, event: Any) -> None:
        """Handle region-draw mouse move preview."""
        if self._region_draw_start is None or self._region_draw_preview is None:
            return
        plot = getattr(self.viewer, "p1", None)
        if plot is None:
            return
        scene_pos = event.scenePos()
        if not plot.sceneBoundingRect().contains(scene_pos):
            return
        view_pos = plot.vb.mapSceneToView(scene_pos)
        x0, y0 = self._region_draw_start
        x1 = view_pos.x()
        y1 = view_pos.y()
        left = min(x0, x1)
        top = min(y0, y1)
        width = max(abs(x1 - x0), 1e-9)
        height = max(abs(y1 - y0), 1e-9)
        self._region_draw_preview.setPos((left, top), update=False)
        self._region_draw_preview.setSize((width, height), update=False)
        event.accept()

    def _finalize_region_from_end(self, end_x: float, end_y: float) -> None:
        """Finalize ROI draw, persist region, and refresh integration results."""
        if self._region_draw_start is None:
            return
        x0, y0 = self._region_draw_start
        self._region_draw_start = None
        left = min(x0, end_x)
        top = min(y0, end_y)
        width = abs(end_x - x0)
        height = abs(end_y - y0)
        self._safe_remove_preview_roi()
        if width <= 0.0 or height <= 0.0:
            self._restore_region_draw_handlers()
            self.active_region_selector = False
            return

        region_id = self.add_integration_region(
            {
                "type": "rectangle",
                "coordinates": [
                    (left, top),
                    (left + width, top),
                    (left + width, top + height),
                    (left, top + height),
                ],
                "calibration_units": str(getattr(self.viewer.ax_x, "units", "px") or "px"),
            }
        )

        plot = getattr(self.viewer, "p1", None)
        if plot is not None:
            roi_item = pg.RectROI(
                [left, top],
                [width, height],
                pen=pg.mkPen(0, 255, 255, width=2),
                movable=True,
                rotatable=False,
            )
            roi_item.addScaleHandle((1, 1), (0, 0))
            roi_item.addScaleHandle((0, 0), (1, 1))
            roi_item.sigRegionChanged.connect(
                lambda _roi=roi_item, rid=region_id: self._on_roi_item_changed(rid, _roi)
            )
            plot.addItem(roi_item)
            self.region_rois[region_id] = roi_item

        self._refresh_results_table()
        self.prompt_save_region_artifacts(region_id)
        self._restore_region_draw_handlers()
        self.active_region_selector = False
        self._notify_capability_state_changed()

    def _on_region_mouse_release(self, event: Any) -> None:
        """Handle region-draw mouse release completion."""
        plot = getattr(self.viewer, "p1", None)
        if plot is None:
            return
        scene_pos = event.scenePos()
        if not plot.sceneBoundingRect().contains(scene_pos):
            self._safe_remove_preview_roi()
            self._restore_region_draw_handlers()
            self.active_region_selector = False
            return
        view_pos = plot.vb.mapSceneToView(scene_pos)
        self._finalize_region_from_end(view_pos.x(), view_pos.y())
        event.accept()

    def _on_region_mouse_drag(self, event: Any) -> None:
        """Support drag events for robust region creation."""
        plot = getattr(self.viewer, "p1", None)
        if plot is None:
            return
        scene_pos = event.scenePos()
        if not plot.sceneBoundingRect().contains(scene_pos):
            if event.isFinish():
                self._safe_remove_preview_roi()
                self._restore_region_draw_handlers()
                self.active_region_selector = False
            event.accept()
            return
        view_pos = plot.vb.mapSceneToView(scene_pos)
        if event.isStart() or self._region_draw_start is None:
            class _Evt:
                def __init__(self, pos: Any):
                    self._pos = pos
                def scenePos(self) -> Any:
                    return self._pos
                def accept(self) -> None:
                    return
            self._on_region_mouse_press(_Evt(scene_pos))
        elif event.isFinish():
            self._finalize_region_from_end(view_pos.x(), view_pos.y())
        else:
            class _Evt:
                def __init__(self, pos: Any):
                    self._pos = pos
                def scenePos(self) -> Any:
                    return self._pos
                def accept(self) -> None:
                    return
            self._on_region_mouse_move(_Evt(scene_pos))
        event.accept()

    def _on_roi_item_changed(self, region_id: int, roi_item: pg.RectROI) -> None:
        """Update stored region contract when ROI is moved/resized."""
        region = next((r for r in self.integration_regions if r.region_id == region_id), None)
        if region is None:
            return
        x = float(roi_item.pos().x())
        y = float(roi_item.pos().y())
        w = float(roi_item.size().x())
        h = float(roi_item.size().y())
        updated = replace(
            region,
            coordinates=[(x, y), (x + w, y), (x + w, y + h), (x, y + h)],
        )
        for idx, existing in enumerate(self.integration_regions):
            if existing.region_id == region_id:
                self.integration_regions[idx] = updated
                break
        self._refresh_results_table()

    def _view_xy_to_pixel_rc(self, x: float, y: float) -> Optional[Tuple[int, int]]:
        """Convert calibrated view coordinates to pixel row/column indices."""
        data = getattr(self.viewer, "data", None)
        ax_x = getattr(self.viewer, "ax_x", None)
        ax_y = getattr(self.viewer, "ax_y", None)
        if data is None or ax_x is None or ax_y is None:
            return None
        try:
            col = int(round((x - float(ax_x.offset)) / float(ax_x.scale)))
            row = int(round((y - float(ax_y.offset)) / float(ax_y.scale)))
        except Exception:
            return None
        if row < 0 or col < 0:
            return None
        if row >= int(data.shape[0]) or col >= int(data.shape[1]):
            return None
        return row, col

    def update_hover_spectrum(self, x: float, y: float) -> None:
        """Update spectrum plot from current cursor location."""
        if not self.hover_updates_enabled:
            return
        if self.spectrum_plot is None:
            return
        now = monotonic()
        if now - self._last_hover_update_s < self._hover_interval_s:
            return
        self._last_hover_update_s = now

        pixel = self._view_xy_to_pixel_rc(x, y)
        if pixel is None:
            return
        row, col = pixel

        energy_axis, counts, source_label = self._resolve_hover_spectrum(row, col)
        if energy_axis is None or counts is None:
            if self.hover_status_label is not None:
                self.hover_status_label.setText(
                    f"Hover ({col}, {row}) - no spectrum data available"
                )
            return

        self.spectrum_plot.clear()
        self.spectrum_plot.plot(energy_axis, counts, pen=pg.mkPen(0, 255, 255, width=2))
        if source_label == "elemental maps":
            self.spectrum_plot.setLabel("bottom", "Element index")
        else:
            self.spectrum_plot.setLabel("bottom", "Energy", units="eV")
        # Keep both axes responsive to the hovered signal range.
        self.spectrum_plot.enableAutoRange(x=True, y=True)
        view_box = self.spectrum_plot.getViewBox()
        if view_box is not None:
            view_box.autoRange()
        if self.hover_status_label is not None:
            source_text = source_label
            if source_label == "elemental maps":
                source_text = "elemental maps (pseudo-spectrum; limited resolution)"
            self.hover_status_label.setText(
                f"Hover ({col}, {row}) from {source_text}"
            )

    def _resolve_hover_spectrum(
        self, row: int, col: int
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], str]:
        """Resolve the best available spectrum for a pixel location."""
        selected_names = list(self.active_spectra) if self.active_spectra else list(self.spectra.keys())
        for name in selected_names:
            data = self.spectra.get(name)
            if data is None:
                continue
            arr = np.asarray(data)
            if arr.ndim == 1:
                energy = self.spectrum_offset + np.arange(arr.size) * self.spectrum_dispersion
                return energy.astype(float), arr.astype(float), name
            if arr.ndim == 3 and row < arr.shape[0] and col < arr.shape[1]:
                counts = arr[row, col, :]
                energy = self.spectrum_offset + np.arange(counts.size) * self.spectrum_dispersion
                return energy.astype(float), counts.astype(float), name
            if arr.ndim == 2 and row < arr.shape[0]:
                counts = arr[row, :]
                energy = self.spectrum_offset + np.arange(counts.size) * self.spectrum_dispersion
                return energy.astype(float), counts.astype(float), name

        # Fallback: pseudo spectrum from elemental map intensities at this pixel.
        elements: List[str] = []
        values: List[float] = []
        for element, map_data in sorted(self.elemental_maps.items()):
            if map_data is None:
                continue
            arr = np.asarray(map_data)
            if arr.ndim != 2 or row >= arr.shape[0] or col >= arr.shape[1]:
                continue
            elements.append(element)
            values.append(float(arr[row, col]))

        if values:
            counts = np.asarray(values, dtype=float)
            energy = np.arange(len(values), dtype=float)
            return energy, counts, "elemental maps"
        return None, None, "none"

    def _region_bounds_in_pixels(self, region: EDSROIRegion) -> Optional[Tuple[int, int, int, int]]:
        """Return clamped pixel bounds [r0:r1, c0:c1] for a rectangular region."""
        if len(region.coordinates) < 2:
            return None
        xs = [p[0] for p in region.coordinates]
        ys = [p[1] for p in region.coordinates]
        top_left = self._view_xy_to_pixel_rc(min(xs), min(ys))
        bottom_right = self._view_xy_to_pixel_rc(max(xs), max(ys))
        if top_left is None or bottom_right is None:
            return None
        r0, c0 = top_left
        r1, c1 = bottom_right
        if r1 < r0:
            r0, r1 = r1, r0
        if c1 < c0:
            c0, c1 = c1, c0
        return r0, r1 + 1, c0, c1 + 1

    def _compute_region_element_counts(self, region: EDSROIRegion) -> List[Tuple[str, float]]:
        """Compute simple integrated counts from elemental maps within region."""
        bounds = self._region_bounds_in_pixels(region)
        if bounds is None:
            return []
        r0, r1, c0, c1 = bounds
        rows: List[Tuple[str, float]] = []
        for element, map_data in sorted(self.elemental_maps.items()):
            if map_data is None:
                continue
            arr = np.asarray(map_data)
            if arr.ndim != 2:
                continue
            r1c = min(r1, arr.shape[0])
            c1c = min(c1, arr.shape[1])
            if r0 >= r1c or c0 >= c1c:
                continue
            counts = float(np.nansum(arr[r0:r1c, c0:c1c]))
            rows.append((element, counts))
        return rows

    def _current_quant_method(self) -> str:
        """Return active quantification method from UI."""
        if self.quant_method_combo is None:
            return "CL"
        return str(self.quant_method_combo.currentText() or "CL")

    def _current_quant_factor_text(self) -> str:
        """Return quantification factors from UI input."""
        if self.quant_factors_edit is None:
            return ""
        return str(self.quant_factors_edit.text() or "")

    def _quant_rows_for_region(self, region: EDSROIRegion):
        """Compute quantification rows for one region."""
        count_pairs = self._compute_region_element_counts(region)
        if not count_pairs:
            return []
        request = QuantificationRequest(
            region_id=region.region_id,
            element_counts={k: v for k, v in count_pairs},
            method=self._current_quant_method(),
            factor_text=self._current_quant_factor_text(),
        )
        return self.quantification_service.quantify(request)

    def _refresh_results_table(self) -> None:
        """Rebuild integration table rows from current region contracts."""
        if self.results_table is None:
            return
        self.results_table.setRowCount(0)
        for region in self.integration_regions:
            rows = list(self._quant_rows_for_region(region))
            if not rows:
                row_index = self.results_table.rowCount()
                self.results_table.insertRow(row_index)
                self.results_table.setItem(row_index, 0, QtWidgets.QTableWidgetItem(str(region.region_id)))
                self.results_table.setItem(row_index, 1, QtWidgets.QTableWidgetItem("(no data)"))
                self.results_table.setItem(row_index, 2, QtWidgets.QTableWidgetItem("0"))
                self.results_table.setItem(row_index, 3, QtWidgets.QTableWidgetItem("-"))
                self.results_table.setItem(row_index, 4, QtWidgets.QTableWidgetItem("-"))
                self.results_table.setItem(row_index, 5, QtWidgets.QTableWidgetItem(self._current_quant_method()))
                continue
            for qrow in rows:
                row_index = self.results_table.rowCount()
                self.results_table.insertRow(row_index)
                id_item = QtWidgets.QTableWidgetItem(str(qrow.region_id))
                element_item = QtWidgets.QTableWidgetItem(qrow.element)
                counts_item = QtWidgets.QTableWidgetItem(f"{qrow.counts:.6g}")
                wt_text = "-" if qrow.weight_percent is None else f"{qrow.weight_percent:.3f}"
                at_text = "-" if qrow.atomic_percent is None else f"{qrow.atomic_percent:.3f}"
                wt_item = QtWidgets.QTableWidgetItem(wt_text)
                at_item = QtWidgets.QTableWidgetItem(at_text)
                method_item = QtWidgets.QTableWidgetItem(qrow.method)

                warnings = list(qrow.warnings)
                if warnings:
                    tooltip = "\n".join(warnings)
                    element_item.setToolTip(tooltip)
                    method_item.setToolTip(tooltip)

                self.results_table.setItem(row_index, 0, id_item)
                self.results_table.setItem(row_index, 1, element_item)
                self.results_table.setItem(row_index, 2, counts_item)
                self.results_table.setItem(row_index, 3, wt_item)
                self.results_table.setItem(row_index, 4, at_item)
                self.results_table.setItem(row_index, 5, method_item)
            self._notify_capability_state_changed()

    def _register_output_artifact(
        self,
        *,
        kind: str,
        path: Path,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record generated output artifact for phase-4 persistence tracking."""
        self.output_artifacts.append(
            {
                "kind": kind,
                "path": str(path),
                "created_at": datetime.now(timezone.utc).isoformat(),
                "metadata": metadata or {},
            }
        )

    def _default_output_dir(self) -> Path:
        """Resolve preferred output directory near the source file."""
        try:
            return Path(self.viewer.file_path).parent
        except Exception:
            return Path.cwd()

    def _quant_rows_all_regions(self) -> List[Any]:
        """Return flattened quantification rows for all regions."""
        rows: List[Any] = []
        for region in self.integration_regions:
            rows.extend(self._quant_rows_for_region(region))
        return rows

    def _quant_rows_for_region_id(self, region_id: int) -> List[Any]:
        """Return quantification rows for one region id."""
        for region in self.integration_regions:
            if region.region_id == region_id:
                return list(self._quant_rows_for_region(region))
        return []

    def _write_quant_rows_csv(self, path: Path, rows: Sequence[Any]) -> bool:
        """Write quantification rows to CSV."""
        try:
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(
                    [
                        "region_id",
                        "element",
                        "counts",
                        "weight_percent",
                        "atomic_percent",
                        "method",
                        "warnings",
                    ]
                )
                for row in rows:
                    writer.writerow(
                        [
                            row.region_id,
                            row.element,
                            f"{row.counts:.12g}",
                            "" if row.weight_percent is None else f"{row.weight_percent:.12g}",
                            "" if row.atomic_percent is None else f"{row.atomic_percent:.12g}",
                            row.method,
                            " | ".join(row.warnings),
                        ]
                    )
            return True
        except Exception as exc:
            self.logger.warning("Failed to write EDS CSV %s: %s", path, exc)
            return False

    def _write_region_metadata_json(
        self,
        path: Path,
        *,
        scope: str,
        region_id: Optional[int] = None,
    ) -> bool:
        """Write sidecar JSON with region/calibration metadata."""
        try:
            if region_id is None:
                regions = [
                    {
                        "region_id": r.region_id,
                        "geometry_type": r.geometry_type,
                        "coordinates": list(r.coordinates),
                        "calibration_units": r.calibration_units,
                        "timestamp_iso": r.timestamp_iso,
                        "metadata": dict(r.metadata),
                    }
                    for r in self.integration_regions
                ]
            else:
                regions = [
                    {
                        "region_id": r.region_id,
                        "geometry_type": r.geometry_type,
                        "coordinates": list(r.coordinates),
                        "calibration_units": r.calibration_units,
                        "timestamp_iso": r.timestamp_iso,
                        "metadata": dict(r.metadata),
                    }
                    for r in self.integration_regions
                    if r.region_id == region_id
                ]

            payload = {
                "scope": scope,
                "calibration_source": self._calibration_source,
                "beam_energy_ev": self.beam_energy_ev,
                "dispersion_ev_per_channel": self.spectrum_dispersion,
                "offset_ev": self.spectrum_offset,
                "live_time_s": self.live_time_s,
                "real_time_s": self.real_time_s,
                "quant_method": self._current_quant_method(),
                "quant_factors": self._current_quant_factor_text(),
                "regions": regions,
            }
            path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            return True
        except Exception as exc:
            self.logger.warning("Failed to write EDS JSON %s: %s", path, exc)
            return False

    def _save_snapshot_png(self, path: Path) -> bool:
        """Save a snapshot image of the current EDS view."""
        try:
            glw = getattr(self.viewer, "glw", None)
            if glw is None:
                return False
            pixmap = glw.grab()
            return bool(pixmap.save(str(path), "PNG"))
        except Exception:
            return False

    def prompt_save_region_artifacts(self, region_id: int) -> None:
        """Prompt-save outputs for a newly created region-derived result."""
        rows = self._quant_rows_for_region_id(region_id)
        if not rows:
            return
        answer = QtWidgets.QMessageBox.question(
            self.edx_panel,
            "Save EDS Region Output",
            f"Save derived outputs for EDS region {region_id}?",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.Yes,
        )
        if answer != QtWidgets.QMessageBox.Yes:
            return

        default_dir = self._default_output_dir()
        default_name = f"{Path(self.viewer.file_path).stem}_eds_region{region_id}.csv"
        selected_path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self.edx_panel,
            "Save EDS Region Results",
            str(default_dir / default_name),
            "CSV (*.csv)",
        )
        if not selected_path:
            return
        csv_path = Path(selected_path)
        if csv_path.suffix.lower() != ".csv":
            csv_path = csv_path.with_suffix(".csv")

        json_path = csv_path.with_suffix(".json")
        png_path = csv_path.with_suffix(".png")

        ok_csv = self._write_quant_rows_csv(csv_path, rows)
        ok_json = self._write_region_metadata_json(json_path, scope="region", region_id=region_id)
        ok_png = self._save_snapshot_png(png_path)

        if ok_csv:
            self._register_output_artifact(
                kind="eds-region-csv",
                path=csv_path,
                metadata={"region_id": region_id},
            )
        if ok_json:
            self._register_output_artifact(
                kind="eds-region-json",
                path=json_path,
                metadata={"region_id": region_id},
            )
        if ok_png:
            self._register_output_artifact(
                kind="eds-region-snapshot",
                path=png_path,
                metadata={"region_id": region_id},
            )

    def prompt_save_all_results(self) -> bool:
        """Prompt-save all integrated EDS results and return True when CSV saved."""
        rows = self._quant_rows_all_regions()
        if not rows:
            QtWidgets.QMessageBox.information(
                self.edx_panel,
                "Save EDS Results",
                "No quantified region rows are available to save.",
            )
            return False

        default_dir = self._default_output_dir()
        default_name = f"{Path(self.viewer.file_path).stem}_eds_results.csv"
        selected_path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self.edx_panel,
            "Save EDS Results",
            str(default_dir / default_name),
            "CSV (*.csv)",
        )
        if not selected_path:
            return False

        csv_path = Path(selected_path)
        if csv_path.suffix.lower() != ".csv":
            csv_path = csv_path.with_suffix(".csv")
        json_path = csv_path.with_suffix(".json")

        ok_csv = self._write_quant_rows_csv(csv_path, rows)
        ok_json = self._write_region_metadata_json(json_path, scope="all")

        if ok_csv:
            self._register_output_artifact(
                kind="eds-all-csv",
                path=csv_path,
                metadata={"row_count": len(rows)},
            )
        if ok_json:
            self._register_output_artifact(
                kind="eds-all-json",
                path=json_path,
                metadata={"row_count": len(rows)},
            )
        return ok_csv
