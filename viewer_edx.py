# SPDX-License-Identifier: GPL-2.0-only
# Copyright (C) 2026 Cooper Stuntz
# See LICENSE for full license terms.

"""EDX spectrum and elemental map viewer for image-viewer windows."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Protocol, Tuple

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

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
        self.integration_regions: List[Dict[str, Any]] = []  # list of region definitions
        self.region_count: int = 0
        self.active_region_selector: bool = False
        
        # Energy calibration state
        self.beam_energy_ev: float = 200.0  # default 200 eV
        self.spectrum_dispersion: float = 5.0  # eV per channel
        self.spectrum_offset: float = 0.0  # eV offset
        self._calibration_source: str = "uncalibrated"
        
        # Cached composite map for display
        self._cached_composite_map: Optional[np.ndarray] = None
        self._cached_composite_needs_update: bool = True
        
        # UI References (set by image viewer)
        self.edx_panel: Optional[QtWidgets.QWidget] = None
        self.spectrum_plot: Optional[pg.PlotItem] = None
        self.maps_list: Optional[QtWidgets.QListWidget] = None
        self.results_table: Optional[QtWidgets.QTableWidget] = None
        self.element_checkboxes: Dict[str, QtWidgets.QCheckBox] = {}
        self.element_name_labels: Dict[str, QtWidgets.QLabel] = {}
        self.element_color_buttons: Dict[str, QtWidgets.QPushButton] = {}
        
        self._has_edx_data = False

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
            # Try to get beam energy
            if "Acquisition_instrument" in meta and "TEM" in meta["Acquisition_instrument"]:
                tem_info = meta["Acquisition_instrument"]["TEM"]
                if "beam_energy" in tem_info:
                    self.beam_energy_ev = float(tem_info["beam_energy"])
                    self.logger.debug(f"Loaded beam energy: {self.beam_energy_ev} eV")

            # Try to get detector spectral parameters
            if "Detectors" in meta:
                detectors = meta["Detectors"]
                # Look for analytical detectors with EDS info
                for det_name, det_info in detectors.items():
                    if isinstance(det_info, dict):
                        if det_info.get("DetectorType") == "AnalyticalDetector":
                            if "Dispersion" in det_info:
                                self.spectrum_dispersion = float(det_info["Dispersion"])
                            if "OffsetEnergy" in det_info:
                                self.spectrum_offset = float(det_info["OffsetEnergy"])
                            self.logger.debug(
                                f"Loaded EDS calibration: dispersion={self.spectrum_dispersion} eV/ch, "
                                f"offset={self.spectrum_offset} eV"
                            )
                            break
            
            self._calibration_source = "metadata"
        except (KeyError, ValueError, TypeError) as e:
            self.logger.debug(f"Could not parse energy calibration from metadata: {e}")
            self._calibration_source = "uncalibrated"

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

    def add_integration_region(self, region_data: Dict[str, Any]) -> int:
        """Add an integration region and return its ID.

        Args:
            region_data: Dictionary with region definition (coordinates, elements, etc.).

        Returns:
            Region ID.
        """
        region_id = self.region_count
        region_data["id"] = region_id
        self.integration_regions.append(region_data)
        self.region_count += 1
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
            if region.get("id") == region_id:
                self.integration_regions.pop(i)
                self.logger.debug(f"Removed integration region {region_id}")
                return True
        return False

    def clear_integration_regions(self) -> None:
        """Clear all integration regions."""
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
        self.edx_panel = panel
        return panel

    def _build_spectrum_tab(self) -> Optional[QtWidgets.QWidget]:
        """Build the spectrum display and control tab."""
        if not self.spectra:
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
            f"Offset: {self.spectrum_offset:.1f} eV"
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

        # Results table
        self.results_table = QtWidgets.QTableWidget()
        self.results_table.setColumnCount(5)
        self.results_table.setHorizontalHeaderLabels(
            ["Region ID", "Element", "Counts", "wt%", "at%"]
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

    def _on_clear_results_clicked(self) -> None:
        """Handle clear results button click."""
        self.clear_integration_regions()
        if self.results_table:
            self.results_table.setRowCount(0)
        self.logger.debug("Integration results cleared")
