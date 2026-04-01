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

from eds_metadata import build_eds_metadata_context, coerce_float, first_present_number
from eds_models import (
    EDSCapabilityState,
    EDSIntegrationSettings,
    EDSMetadataContext,
    EDSROIRegion,
)
from eds_quantification import (
    EDS_CSV_HEADER,
    EDSQuantificationService,
    QuantificationRequest,
    quant_rows_to_csv_records,
)
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
        self.spectrum_energy_axes: Dict[str, np.ndarray] = {}
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
        self.model_results_table: Optional[QtWidgets.QTableWidget] = None
        self.hover_status_label: Optional[QtWidgets.QLabel] = None
        self.model_status_label: Optional[QtWidgets.QLabel] = None
        self.model_fit_btn: Optional[QtWidgets.QPushButton] = None
        self.model_fit_bg_btn: Optional[QtWidgets.QPushButton] = None
        self.quant_method_combo: Optional[QtWidgets.QComboBox] = None
        self.quant_units_combo: Optional[QtWidgets.QComboBox] = None
        self.quant_factors_edit: Optional[QtWidgets.QLineEdit] = None
        self.quant_absorption_checkbox: Optional[QtWidgets.QCheckBox] = None
        self.background_mode_combo: Optional[QtWidgets.QComboBox] = None
        self.integration_width_edit: Optional[QtWidgets.QLineEdit] = None
        self.element_checkboxes: Dict[str, QtWidgets.QCheckBox] = {}
        self.element_name_labels: Dict[str, QtWidgets.QLabel] = {}
        self.element_color_buttons: Dict[str, QtWidgets.QPushButton] = {}
        self._line_family_checkboxes: Dict[str, QtWidgets.QCheckBox] = {}
        self._enabled_line_families: set[str] = {"Ka", "Kb", "La", "Lb", "Ma"}
        self._line_overlay_items: List[Any] = []
        self._window_overlay_items: List[Any] = []
        self.integration_settings = EDSIntegrationSettings(
            integration_windows_ev=tuple(),
            background_mode="none",
            included_lines=tuple(),
        )
        
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
        spectrum_signals: Optional[List[Tuple[str, Any]]] = None,
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
            # If EDX signals were provided, load them directly.
            if elemental_map_signals or spectrum_signals:
                self._load_elemental_maps_from_signals(elemental_map_signals)
                self._load_spectra_from_signals(spectrum_signals)
                self._load_energy_calibration_from_metadata(
                    self.viewer._get_original_metadata_dict_from_signal(self.viewer.signal) or {}
                )
                self._load_element_colors_from_metadata(
                    self.viewer._get_original_metadata_dict_from_signal(self.viewer.signal) or {}
                )
                self._has_edx_data = len(self.elemental_maps) > 0 or len(self.spectra) > 0
                if self._has_edx_data:
                    self.logger.info(
                        "EDX data loaded from signals: %d spectra, %d elemental map(s)",
                        len(self.spectra),
                        len(self.elemental_maps),
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

    def _energy_axis_from_signal(self, signal: Any, channels: int) -> np.ndarray:
        """Build calibrated energy axis for a spectrum signal when possible."""
        try:
            axis = signal.axes_manager.signal_axes[0]
            scale = float(getattr(axis, "scale", self.spectrum_dispersion))
            offset = float(getattr(axis, "offset", self.spectrum_offset))
            units = str(getattr(axis, "units", "") or "").strip().lower()
            if units == "kev":
                scale *= 1000.0
                offset *= 1000.0
            return offset + np.arange(channels, dtype=float) * scale
        except Exception:
            return self.spectrum_offset + np.arange(channels, dtype=float) * self.spectrum_dispersion

    def _load_spectra_from_signals(
        self,
        spectrum_signals: Optional[List[Tuple[str, Any]]],
    ) -> None:
        """Load EDS spectra directly from provided HyperSpy signal objects."""
        if not spectrum_signals:
            return

        for spectrum_name, signal in spectrum_signals:
            try:
                data = np.asarray(signal.data, dtype=float)
            except Exception as exc:
                self.logger.warning(
                    "Failed to access spectrum data for %s: %s",
                    spectrum_name,
                    exc,
                )
                continue

            if data.ndim < 1:
                continue

            channels = int(data.shape[-1])
            self.spectra[spectrum_name] = data
            self.spectrum_energy_axes[spectrum_name] = self._energy_axis_from_signal(
                signal, channels
            )
            self.spectrum_metadata[spectrum_name] = {
                "signal": signal,
                "shape": tuple(int(v) for v in data.shape),
                "source": "hyperspy-signal",
            }

        if self.spectra and not self.active_spectra:
            self.active_spectra = set(self.spectra.keys())

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
        self, elemental_map_signals: Optional[List[Tuple[str, Any]]]
    ) -> None:
        """Load elemental maps directly from provided signals.

        Args:
            elemental_map_signals: List of (element_name, signal) tuples.
        """
        if not elemental_map_signals:
            return

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
        return coerce_float(value)

    @staticmethod
    def _first_present_number(candidates: Sequence[Any]) -> Optional[float]:
        """Return the first valid numeric value found in candidates."""
        return first_present_number(candidates)

    def _build_metadata_context(self, meta: Dict[str, Any]) -> EDSMetadataContext:
        """Resolve EDS calibration/timing metadata using a fallback hierarchy."""
        mapped_meta = getattr(self.viewer.signal, "metadata", None)
        mapped_dict: Dict[str, Any] = {}
        if mapped_meta is not None:
            try:
                mapped_dict = dict(mapped_meta.as_dictionary()) if hasattr(mapped_meta, "as_dictionary") else dict(mapped_meta)
            except Exception:
                mapped_dict = {}
        return build_eds_metadata_context(
            original_meta=meta,
            mapped_meta=mapped_dict,
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
        if self.active_spectra:
            candidate = next(iter(self.active_spectra))
            axis = self.spectrum_energy_axes.get(candidate)
            if axis is not None:
                return axis

        for axis in self.spectrum_energy_axes.values():
            return axis

        num_channels = 1024
        return self.spectrum_offset + np.arange(num_channels) * self.spectrum_dispersion

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
        has_model_fit_results = bool(self._serialize_model_fit_result())
        return EDSCapabilityState(
            has_edx_data=self._has_edx_data,
            has_elemental_maps=bool(self.elemental_maps),
            has_spectra=bool(self.spectra),
            has_integration_regions=bool(self.integration_regions),
            has_energy_calibration=has_energy_calibration,
            has_timing_metadata=has_timing_metadata,
            has_xray_lines=bool(self.xray_lines),
            has_model_fit_results=has_model_fit_results,
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
            checkbox.setChecked(True)
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

        line_group = QtWidgets.QGroupBox("X-ray Line Markers")
        line_layout = QtWidgets.QHBoxLayout(line_group)
        self._line_family_checkboxes.clear()
        for family in ("Ka", "Kb", "La", "Lb", "Ma"):
            checkbox = QtWidgets.QCheckBox(family)
            checkbox.setChecked(True)
            checkbox.stateChanged.connect(
                lambda state, fam=family: self._on_line_family_checkbox_changed(
                    fam, state
                )
            )
            self._line_family_checkboxes[family] = checkbox
            line_layout.addWidget(checkbox)
        line_layout.addStretch()
        layout.addWidget(line_group)

        windows_group = QtWidgets.QGroupBox("Integration/Background Windows")
        windows_layout = QtWidgets.QFormLayout(windows_group)
        self.background_mode_combo = QtWidgets.QComboBox()
        self.background_mode_combo.addItems(["none", "auto"])
        self.background_mode_combo.currentTextChanged.connect(
            lambda _text: self._refresh_integration_windows_from_controls()
        )
        windows_layout.addRow("Background:", self.background_mode_combo)

        self.integration_width_edit = QtWidgets.QLineEdit("120")
        self.integration_width_edit.setToolTip("Half-width around each selected line (eV)")
        self.integration_width_edit.textChanged.connect(
            lambda _text: self._refresh_integration_windows_from_controls()
        )
        windows_layout.addRow("Half-width (eV):", self.integration_width_edit)
        layout.addWidget(windows_group)

        model_group = QtWidgets.QGroupBox("Model Fitting")
        model_layout = QtWidgets.QHBoxLayout(model_group)
        self.model_fit_bg_btn = QtWidgets.QPushButton("Fit Background")
        self.model_fit_bg_btn.clicked.connect(self._on_model_fit_background_clicked)
        model_layout.addWidget(self.model_fit_bg_btn)
        self.model_fit_btn = QtWidgets.QPushButton("Run Model Fit")
        self.model_fit_btn.clicked.connect(self._on_model_fit_clicked)
        model_layout.addWidget(self.model_fit_btn)
        layout.addWidget(model_group)

        self.model_status_label = QtWidgets.QLabel("Model fit idle.")
        self.model_status_label.setWordWrap(True)
        layout.addWidget(self.model_status_label)

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

        self._refresh_integration_windows_from_controls()

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
        self.quant_method_combo.addItems(["CL", "Custom", "Zeta", "Cross-Section"])
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

        model_results_group = QtWidgets.QGroupBox("Model Fit Intensities")
        model_results_layout = QtWidgets.QVBoxLayout(model_results_group)
        self.model_results_table = QtWidgets.QTableWidget()
        self.model_results_table.setColumnCount(3)
        self.model_results_table.setHorizontalHeaderLabels(
            ["Spectrum", "Line", "Intensity"]
        )
        self.model_results_table.horizontalHeader().setStretchLastSection(True)
        model_results_layout.addWidget(self.model_results_table)
        layout.addWidget(model_results_group, 1)

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

    def _on_line_family_checkbox_changed(self, family: str, state: int) -> None:
        """Handle X-ray line-family marker visibility toggles."""
        if state == QtCore.Qt.Checked:
            self._enabled_line_families.add(family)
        else:
            self._enabled_line_families.discard(family)
        self._draw_line_markers()
        self._refresh_integration_windows_from_controls()

    @staticmethod
    def _safe_get_line_energy_ev(element: str, family: str) -> Optional[float]:
        """Resolve a line energy from HyperSpy's elemental database in eV."""
        try:
            import hyperspy.api as hs

            node = getattr(hs.material.elements, element)
            xray = getattr(node.Atomic_properties.Xray_lines, family)
            energy_kev = float(getattr(xray, "energy_keV", getattr(xray, "energy")))
            return energy_kev * 1000.0
        except Exception:
            return None

    def _line_marker_candidates(self) -> List[Tuple[float, str]]:
        """Build marker candidates from selected elements and line families."""
        elements: set[str] = set()
        for line in self.xray_lines:
            if "_" in line:
                elements.add(line.split("_", 1)[0])
        for element in self.elemental_maps.keys():
            token = str(element).split()[0].strip()
            if token and token[0].isalpha():
                elements.add(token)

        markers: List[Tuple[float, str]] = []
        for element in sorted(elements):
            for family in sorted(self._enabled_line_families):
                energy_ev = self._safe_get_line_energy_ev(element, family)
                if energy_ev is None:
                    continue
                if self.beam_energy_ev and energy_ev > self.beam_energy_ev:
                    continue
                markers.append((energy_ev, f"{element}_{family}"))
        return markers

    def _draw_line_markers(self) -> None:
        """Draw X-ray line overlays on the active spectrum plot."""
        if self.spectrum_plot is None:
            return
        for item in self._line_overlay_items:
            try:
                self.spectrum_plot.removeItem(item)
            except Exception:
                pass
        self._line_overlay_items = []

        for energy_ev, label in self._line_marker_candidates():
            marker = pg.InfiniteLine(
                pos=float(energy_ev),
                angle=90,
                movable=False,
                pen=pg.mkPen(255, 215, 0, width=1),
                label=label,
                labelOpts={"position": 0.95, "color": (255, 215, 0)},
            )
            self.spectrum_plot.addItem(marker)
            self._line_overlay_items.append(marker)

    def _refresh_integration_windows_from_controls(self) -> None:
        """Recompute integration windows from UI controls and redraw overlays."""
        half_width_ev = 120.0
        if self.integration_width_edit is not None:
            try:
                parsed = float(self.integration_width_edit.text().strip())
                if parsed > 0:
                    half_width_ev = parsed
            except Exception:
                pass

        background_mode = "none"
        if self.background_mode_combo is not None:
            background_mode = str(self.background_mode_combo.currentText() or "none").strip().lower()

        markers = self._line_marker_candidates()
        line_windows: List[Tuple[float, float]] = []
        included_lines: List[str] = []
        for energy_ev, label in markers:
            line_windows.append((energy_ev - half_width_ev, energy_ev + half_width_ev))
            included_lines.append(label)

        self.integration_settings = EDSIntegrationSettings(
            integration_windows_ev=tuple(line_windows),
            background_mode=background_mode,
            included_lines=tuple(included_lines),
        )
        self._draw_integration_windows()

    def _draw_integration_windows(self) -> None:
        """Draw integration/background window overlays on the spectrum plot."""
        if self.spectrum_plot is None:
            return

        for item in self._window_overlay_items:
            try:
                self.spectrum_plot.removeItem(item)
            except Exception:
                pass
        self._window_overlay_items = []

        for low_ev, high_ev in self.integration_settings.integration_windows_ev:
            region = pg.LinearRegionItem(
                values=(float(low_ev), float(high_ev)),
                movable=False,
                brush=pg.mkBrush(0, 255, 255, 25),
                pen=pg.mkPen(0, 255, 255, 80),
            )
            self.spectrum_plot.addItem(region)
            self._window_overlay_items.append(region)

            if self.integration_settings.background_mode == "auto":
                width = max((high_ev - low_ev), 1.0)
                pad = 0.25 * width
                left = pg.LinearRegionItem(
                    values=(float(low_ev - width - pad), float(low_ev - pad)),
                    movable=False,
                    brush=pg.mkBrush(255, 180, 0, 20),
                    pen=pg.mkPen(255, 180, 0, 60),
                )
                right = pg.LinearRegionItem(
                    values=(float(high_ev + pad), float(high_ev + width + pad)),
                    movable=False,
                    brush=pg.mkBrush(255, 180, 0, 20),
                    pen=pg.mkPen(255, 180, 0, 60),
                )
                self.spectrum_plot.addItem(left)
                self.spectrum_plot.addItem(right)
                self._window_overlay_items.extend([left, right])

    def _get_model_input_spectrum(self) -> Tuple[Optional[str], Optional[np.ndarray], Optional[np.ndarray]]:
        """Return (name, counts_1d, energy_axis_eV) for model fitting."""
        candidate_names = list(self.active_spectra) if self.active_spectra else list(self.spectra.keys())
        for name in candidate_names:
            arr = self.spectra.get(name)
            if arr is None:
                continue
            data = np.asarray(arr, dtype=float)
            if data.ndim == 1:
                counts = data
            else:
                try:
                    counts = np.nanmean(data.reshape((-1, data.shape[-1])), axis=0)
                except Exception:
                    continue
            if counts.size == 0:
                continue
            energy = self.spectrum_energy_axes.get(name)
            if energy is None or energy.shape[0] != counts.shape[0]:
                energy = self.spectrum_offset + np.arange(counts.shape[0], dtype=float) * self.spectrum_dispersion
            return name, counts.astype(float), energy.astype(float)
        return None, None, None

    def _selected_elements_for_model(self) -> List[str]:
        """Resolve unique element symbols used for model setup."""
        elements: set[str] = set()
        for label in self.integration_settings.included_lines:
            if "_" in label:
                token = label.split("_", 1)[0].strip()
                if token:
                    elements.add(token)
        for element in self.elemental_maps.keys():
            token = str(element).split()[0].strip()
            if token:
                elements.add(token)
        return sorted(elements)

    def _build_hyperspy_eds_signal(self, counts: np.ndarray, energy_ev: np.ndarray) -> Any:
        """Create a HyperSpy Signal1D configured for EDS model fitting."""
        import hyperspy.api as hs

        signal = hs.signals.Signal1D(counts)
        axis = signal.axes_manager[-1]
        axis.name = "E"
        axis.units = "eV"
        if energy_ev.size >= 2:
            axis.scale = float(energy_ev[1] - energy_ev[0])
            axis.offset = float(energy_ev[0])
        signal.set_signal_type("EDS_TEM")

        elements = self._selected_elements_for_model()
        if elements:
            try:
                signal.set_elements(elements)
                signal.add_lines()
            except Exception:
                pass
        return signal

    def _on_model_fit_background_clicked(self) -> None:
        """Run HyperSpy EDS background fit on the selected spectrum."""
        name, counts, energy_ev = self._get_model_input_spectrum()
        if counts is None or energy_ev is None:
            if self.model_status_label is not None:
                self.model_status_label.setText("Model fit unavailable: no spectrum selected.")
            return
        try:
            signal = self._build_hyperspy_eds_signal(counts, energy_ev)
            model = signal.create_model()
            model.fit_background()
            meta = self.spectrum_metadata.setdefault(name or "spectrum", {})
            meta["last_model"] = model
            meta["last_background_fit_timestamp"] = datetime.now(timezone.utc).isoformat()
            self._refresh_model_results_table()
            if self.model_status_label is not None:
                self.model_status_label.setText(f"Background fit complete for {name}.")
        except Exception as exc:
            if self.model_status_label is not None:
                self.model_status_label.setText(f"Background fit failed: {type(exc).__name__}: {exc}")

    def run_model_fit_background(self) -> None:
        """Public entrypoint for background fit action."""
        self._on_model_fit_background_clicked()

    def _on_model_fit_clicked(self) -> None:
        """Run HyperSpy EDS model fit and store line intensity summary."""
        name, counts, energy_ev = self._get_model_input_spectrum()
        if counts is None or energy_ev is None:
            if self.model_status_label is not None:
                self.model_status_label.setText("Model fit unavailable: no spectrum selected.")
            return
        try:
            signal = self._build_hyperspy_eds_signal(counts, energy_ev)
            model = signal.create_model()
            model.fit_background()
            model.fit()
            result = model.get_lines_intensity()
            meta = self.spectrum_metadata.setdefault(name or "spectrum", {})
            meta["last_fit_result"] = result
            meta["last_fit_timestamp"] = datetime.now(timezone.utc).isoformat()
            self._refresh_model_results_table()
            if self.model_status_label is not None:
                self.model_status_label.setText(f"Model fit complete for {name}.")
        except Exception as exc:
            if self.model_status_label is not None:
                self.model_status_label.setText(f"Model fit failed: {type(exc).__name__}: {exc}")

    def run_model_fit_full(self) -> None:
        """Public entrypoint for full fit action."""
        self._on_model_fit_clicked()

    def _serialize_model_fit_result(self) -> List[Dict[str, Any]]:
        """Serialize latest model-fit line intensity records for export."""
        rows: List[Dict[str, Any]] = []
        for spectrum_name, meta in sorted(self.spectrum_metadata.items()):
            if not isinstance(meta, dict):
                continue
            result = meta.get("last_fit_result")
            if result is None:
                continue

            if isinstance(result, list):
                iterable = result
            else:
                iterable = [result]

            for item in iterable:
                title = "line"
                try:
                    meta_obj = getattr(item, "metadata", None)
                    if meta_obj is not None and hasattr(meta_obj, "General"):
                        t = getattr(meta_obj.General, "title", None)
                        if t:
                            title = str(t)
                    elif isinstance(meta_obj, dict):
                        t = meta_obj.get("General", {}).get("title")
                        if t:
                            title = str(t)
                except Exception:
                    title = "line"

                data_arr = np.asarray(getattr(item, "data", np.asarray([])), dtype=float)
                intensity = float(np.nansum(data_arr)) if data_arr.size else 0.0
                rows.append(
                    {
                        "spectrum": spectrum_name,
                        "line": title,
                        "intensity": intensity,
                    }
                )
        return rows

    def _write_model_fit_rows_csv(self, path: Path, rows: Sequence[Dict[str, Any]]) -> bool:
        """Write model-fit intensity rows to CSV."""
        try:
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(["spectrum", "line", "intensity"])
                for row in rows:
                    writer.writerow([
                        str(row.get("spectrum", "")),
                        str(row.get("line", "")),
                        f"{float(row.get('intensity', 0.0)):.12g}",
                    ])
            return True
        except Exception as exc:
            self.logger.warning("Failed to write model-fit CSV %s: %s", path, exc)
            return False

    def _model_fit_summary_by_spectrum(self) -> List[Dict[str, Any]]:
        """Build compact per-spectrum model-fit summaries for export sidecars."""
        rows = self._serialize_model_fit_result()
        aggregate: Dict[str, Dict[str, Any]] = {}
        for row in rows:
            spectrum = str(row.get("spectrum", ""))
            if not spectrum:
                continue
            entry = aggregate.setdefault(
                spectrum,
                {"spectrum": spectrum, "line_count": 0, "total_intensity": 0.0},
            )
            entry["line_count"] = int(entry["line_count"]) + 1
            entry["total_intensity"] = float(entry["total_intensity"]) + float(
                row.get("intensity", 0.0)
            )

        for spectrum, entry in aggregate.items():
            meta = self.spectrum_metadata.get(spectrum, {}) if isinstance(self.spectrum_metadata, dict) else {}
            if isinstance(meta, dict) and meta.get("last_fit_timestamp"):
                entry["last_fit_timestamp"] = str(meta["last_fit_timestamp"])

        return [aggregate[key] for key in sorted(aggregate.keys())]

    def prompt_save_model_fit_results(self) -> bool:
        """Prompt-save model-fit line intensities to CSV."""
        rows = self._serialize_model_fit_result()
        if not rows:
            QtWidgets.QMessageBox.information(
                self.edx_panel,
                "Save Model Fit Results",
                "No model-fit line intensities are available to save.",
            )
            return False

        default_dir = self._default_output_dir()
        default_name = f"{Path(self.viewer.file_path).stem}_eds_model_fit.csv"
        selected_path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self.edx_panel,
            "Save Model Fit Results",
            str(default_dir / default_name),
            "CSV (*.csv)",
        )
        if not selected_path:
            return False

        csv_path = Path(selected_path)
        if csv_path.suffix.lower() != ".csv":
            csv_path = csv_path.with_suffix(".csv")

        ok = self._write_model_fit_rows_csv(csv_path, rows)
        if ok:
            self._register_output_artifact(
                kind="eds-model-fit-csv",
                path=csv_path,
                metadata={"row_count": len(rows)},
            )
        return ok

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
        self._draw_line_markers()
        self._draw_integration_windows()
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
                energy = self.spectrum_energy_axes.get(name)
                if energy is None or energy.shape[0] != arr.size:
                    energy = self.spectrum_offset + np.arange(arr.size) * self.spectrum_dispersion
                return energy.astype(float), arr.astype(float), name
            if arr.ndim == 3 and row < arr.shape[0] and col < arr.shape[1]:
                counts = arr[row, col, :]
                energy = self.spectrum_energy_axes.get(name)
                if energy is None or energy.shape[0] != counts.size:
                    energy = self.spectrum_offset + np.arange(counts.size) * self.spectrum_dispersion
                return energy.astype(float), counts.astype(float), name
            if arr.ndim == 2 and row < arr.shape[0]:
                counts = arr[row, :]
                energy = self.spectrum_energy_axes.get(name)
                if energy is None or energy.shape[0] != counts.size:
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
        """Compute integrated counts from spectra when available, else elemental maps."""
        spectral_rows = self._compute_region_line_counts_from_spectra(region)
        if spectral_rows:
            return spectral_rows

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

    def _compute_region_line_counts_from_spectra(
        self,
        region: EDSROIRegion,
    ) -> List[Tuple[str, float]]:
        """Integrate per-element counts from spectral cubes using selected windows."""
        bounds = self._region_bounds_in_pixels(region)
        if bounds is None:
            return []

        r0, r1, c0, c1 = bounds
        candidate_names = list(self.active_spectra) if self.active_spectra else list(self.spectra.keys())
        cube_name: Optional[str] = None
        cube_data: Optional[np.ndarray] = None
        for name in candidate_names:
            arr = self.spectra.get(name)
            if arr is None:
                continue
            data = np.asarray(arr)
            if data.ndim != 3:
                continue
            if r0 >= data.shape[0] or c0 >= data.shape[1]:
                continue
            cube_name = name
            cube_data = data
            break

        if cube_data is None or cube_name is None:
            return []

        r1c = min(r1, cube_data.shape[0])
        c1c = min(c1, cube_data.shape[1])
        if r0 >= r1c or c0 >= c1c:
            return []

        region_cube = np.asarray(cube_data[r0:r1c, c0:c1c, :], dtype=float)
        spectrum = np.nansum(region_cube, axis=(0, 1))
        energy = self.spectrum_energy_axes.get(cube_name)
        if energy is None or energy.shape[0] != spectrum.shape[0]:
            energy = self.spectrum_offset + np.arange(spectrum.shape[0], dtype=float) * self.spectrum_dispersion

        windows = list(self.integration_settings.integration_windows_ev)
        lines = list(self.integration_settings.included_lines)
        if not windows or not lines or len(windows) != len(lines):
            return []

        return self._integrate_spectrum_windows(
            energy=energy,
            spectrum=spectrum,
            windows=windows,
            lines=lines,
            background_mode=self.integration_settings.background_mode,
        )

    @staticmethod
    def _integrate_spectrum_windows(
        *,
        energy: np.ndarray,
        spectrum: np.ndarray,
        windows: Sequence[Tuple[float, float]],
        lines: Sequence[str],
        background_mode: str,
    ) -> List[Tuple[str, float]]:
        """Integrate per-element counts from explicit line windows on one spectrum."""
        by_element: Dict[str, float] = {}
        for (low_ev, high_ev), line_label in zip(windows, lines):
            mask = (energy >= low_ev) & (energy <= high_ev)
            if not np.any(mask):
                continue
            signal_counts = float(np.nansum(spectrum[mask]))
            if background_mode == "auto":
                width = max(float(high_ev - low_ev), 1.0)
                pad = 0.25 * width
                left_mask = (energy >= (low_ev - width - pad)) & (energy <= (low_ev - pad))
                right_mask = (energy >= (high_ev + pad)) & (energy <= (high_ev + width + pad))
                background_samples: List[float] = []
                if np.any(left_mask):
                    background_samples.append(float(np.nanmean(spectrum[left_mask])))
                if np.any(right_mask):
                    background_samples.append(float(np.nanmean(spectrum[right_mask])))
                if background_samples:
                    bg_level = float(np.nanmean(np.asarray(background_samples)))
                    signal_counts = max(0.0, signal_counts - bg_level * int(np.count_nonzero(mask)))

            element = str(line_label).split("_", 1)[0]
            by_element[element] = by_element.get(element, 0.0) + signal_counts

        return sorted(by_element.items(), key=lambda kv: kv[0])

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

    def _extract_beam_current_na(self) -> Optional[float]:
        """Resolve beam current (nA) from mapped or original metadata when available."""
        try:
            mapped = getattr(self.viewer.signal, "metadata", None)
            if mapped is not None and hasattr(mapped, "as_dictionary"):
                mapped_dict = mapped.as_dictionary()
                if isinstance(mapped_dict, dict):
                    tem = mapped_dict.get("Acquisition_instrument", {}).get("TEM", {})
                    if isinstance(tem, dict):
                        beam_current = self._coerce_float(tem.get("beam_current"))
                        if beam_current is not None:
                            return beam_current
        except Exception:
            pass

        meta = self.viewer._get_original_metadata_dict_from_signal(self.viewer.signal) or {}
        acquisition = meta.get("Acquisition", {}) if isinstance(meta, dict) else {}
        if isinstance(acquisition, dict):
            return self._coerce_float(acquisition.get("BeamCurrent"))
        return None

    def _quant_rows_for_region(self, region: EDSROIRegion):
        """Compute quantification rows for one region."""
        count_pairs = self._compute_region_element_counts(region)
        if not count_pairs:
            return []
        method_ui = self._current_quant_method().strip().lower()
        method_map = {
            "cl": "CL",
            "custom": "CUSTOM",
            "zeta": "ZETA",
            "cross-section": "CROSS_SECTION",
            "cross_section": "CROSS_SECTION",
        }
        method = method_map.get(method_ui, "CL")
        request = QuantificationRequest(
            region_id=region.region_id,
            element_counts={k: v for k, v in count_pairs},
            method=method,
            factor_text=self._current_quant_factor_text(),
            absorption_correction=bool(
                self.quant_absorption_checkbox is not None
                and self.quant_absorption_checkbox.isChecked()
            ),
            thickness_nm=self._coerce_float(region.metadata.get("thickness_nm")),
            beam_current_na=self._extract_beam_current_na(),
            real_time_s=self.real_time_s,
            detector_count=len(
                [
                    v
                    for v in (
                        self.viewer._get_original_metadata_dict_from_signal(self.viewer.signal) or {}
                    ).get("Detectors", {}).values()
                    if isinstance(v, dict) and v.get("DetectorType") == "AnalyticalDetector"
                ]
            ) or None,
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
        self._refresh_model_results_table()

    def _refresh_model_results_table(self) -> None:
        """Populate model-fit line-intensity rows in the integration tab."""
        if self.model_results_table is None:
            return
        self.model_results_table.setRowCount(0)
        rows = self._serialize_model_fit_result()
        for row in rows:
            idx = self.model_results_table.rowCount()
            self.model_results_table.insertRow(idx)
            self.model_results_table.setItem(
                idx, 0, QtWidgets.QTableWidgetItem(str(row.get("spectrum", "")))
            )
            self.model_results_table.setItem(
                idx, 1, QtWidgets.QTableWidgetItem(str(row.get("line", "")))
            )
            self.model_results_table.setItem(
                idx,
                2,
                QtWidgets.QTableWidgetItem(
                    f"{float(row.get('intensity', 0.0)):.6g}"
                ),
            )

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
                writer.writerow(list(EDS_CSV_HEADER))
                for record in quant_rows_to_csv_records(rows):
                    writer.writerow([record[column] for column in EDS_CSV_HEADER])
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
                "integration_settings": {
                    "background_mode": self.integration_settings.background_mode,
                    "integration_windows_ev": [
                        [float(lo), float(hi)]
                        for lo, hi in self.integration_settings.integration_windows_ev
                    ],
                    "included_lines": list(self.integration_settings.included_lines),
                },
                "model_fit": {
                    "status_text": (
                        self.model_status_label.text()
                        if self.model_status_label is not None
                        else ""
                    ),
                    "available_spectra": sorted(list(self.spectra.keys())),
                    "line_intensities": self._serialize_model_fit_result(),
                    "spectrum_summary": self._model_fit_summary_by_spectrum(),
                },
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
        model_csv_path = csv_path.with_name(csv_path.stem + "_model_fit.csv")

        ok_csv = self._write_quant_rows_csv(csv_path, rows)
        ok_json = self._write_region_metadata_json(json_path, scope="region", region_id=region_id)
        ok_png = self._save_snapshot_png(png_path)
        model_rows = self._serialize_model_fit_result()
        ok_model_csv = bool(model_rows) and self._write_model_fit_rows_csv(model_csv_path, model_rows)

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
        if ok_model_csv:
            self._register_output_artifact(
                kind="eds-model-fit-csv",
                path=model_csv_path,
                metadata={"region_id": region_id, "row_count": len(model_rows)},
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
        model_csv_path = csv_path.with_name(csv_path.stem + "_model_fit.csv")

        ok_csv = self._write_quant_rows_csv(csv_path, rows)
        ok_json = self._write_region_metadata_json(json_path, scope="all")
        model_rows = self._serialize_model_fit_result()
        ok_model_csv = bool(model_rows) and self._write_model_fit_rows_csv(model_csv_path, model_rows)

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
        if ok_model_csv:
            self._register_output_artifact(
                kind="eds-model-fit-csv",
                path=model_csv_path,
                metadata={"row_count": len(model_rows)},
            )
        return ok_csv
