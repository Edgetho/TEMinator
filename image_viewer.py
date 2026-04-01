# SPDX-License-Identifier: GPL-2.0-only
# Copyright (C) 2026 Cooper Stuntz
# See LICENSE for full license terms.

"""Image viewer window and image-opening helper."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import hyperspy.api as hs
import numpy as np
import pyqtgraph as pg
import pyqtgraph.exporters as pg_exporters
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

import calibration_logic
import unit_utils
import utils
from command_utils import CommandModeController
from dialogs import (
    LineProfileWindow,
    MeasurementHistoryWindow,
    MetadataWindow,
    RenderSettingsDialog,
    ToneCurveDialog,
)
from file_navigation import IMAGE_FILE_FILTER
from image_loader import open_image_file
from measurement_tools import (
    DRAWN_LINE_PEN,
    LABEL_BRUSH_COLOR,
    FFTBoxROI,
    LineDrawingTool,
    MeasurementLabel,
)
from menu_manager import MenuBuilder, build_menu_config_for_role
from scale_bars import DynamicLegendBox, DynamicScaleBar
from utils import (
    HelpDialogActions,
    open_file_dialog,
    open_parameters_dialog,
)
from viewer_commands import (
    ViewerCommandRouter,
)
from viewer_edx import SpectrumAnalysisManager
from viewer_fft import FFTWindowManager
from viewer_measurements import MeasurementController
from viewer_settings import (
    RESAMPLING_BALANCED,
    RESAMPLING_FAST,
    RESAMPLING_HIGH,
    get_effective_render_settings,
    global_render_config_options,
    hardware_acceleration_available,
    load_render_settings,
    save_render_settings,
)

FFT_COLORS = ["r", "g", "b", "y", "c", "m"]
DEFAULT_IMAGE_WINDOW_SIZE = (1000, 900)
DEFAULT_FFT_WINDOW_SIZE = (700, 700)
RECIPROCAL_SIGNAL_TYPE_TOKENS = {"diffraction", "electron_diffraction", "fft"}
REAL_SIGNAL_TYPE_TOKENS = {"image", "tem", "stem"}
DIFFRACTION_MODE_TOKENS = {
    "diffraction",
    "electron diffraction",
    "selected area diffraction",
    "saed",
    "nanodiffraction",
    "cbed",
    "kikuchi",
}
RECIPROCAL_SIGNAL_TYPE_SUBSTRINGS = {"diffraction", "fft", "kikuchi", "cbed"}
logger = logging.getLogger(__name__)


class ImageViewerWindow(QtWidgets.QMainWindow):
    """Window for viewing and analyzing a single image with FFT boxes."""

    def __init__(
        self,
        file_path: str,
        signal=None,
        window_suffix: Optional[str] = None,
        view_mode: str = "image",
        source_region: Optional[np.ndarray] = None,
        fft_region: Optional[np.ndarray] = None,
        fft_name: Optional[str] = None,
        parent_image_window: Optional["ImageViewerWindow"] = None,
        elemental_map_signals: Optional[List[Tuple[str, Any]]] = None,
        eds_spectrum_signals: Optional[List[Tuple[str, Any]]] = None,
    ):
        """Initialize an image viewer window for displaying TEM images and measurements.

        Args:
            file_path: Path to the source image file.
            signal: HyperSpy Signal object to display. If None, loaded from file_path.
            window_suffix: Optional suffix to append to window title (e.g., for multi-image files).
            view_mode: Display mode - "image" for direct view or "fft" for Fourier transform.
            source_region: Optional array specifying source region for FFT windows.
            fft_region: Optional array specifying FFT region for inverse FFT windows.
            fft_name: Optional name/label for FFT windows.
            parent_image_window: Reference to parent viewer for FFT/inverse FFT relationships.
            elemental_map_signals: Optional list of (element_name, signal) tuples for EDX elemental maps.
            eds_spectrum_signals: Optional list of (spectrum_name, signal) tuples for EDX spectra.
        """
        super().__init__()
        self.setAcceptDrops(True)
        self.file_path = file_path
        self.view_mode = view_mode
        self.signal = None
        self.data = None
        self.ax_x = None
        self.ax_y = None
        self.p1 = None
        self.img_orig = None
        self.glw = None
        self.fft_boxes: List[pg.RectROI] = []
        self.fft_count = 0
        self.is_reciprocal_space = False
        self.line_tool: Optional[LineDrawingTool] = None
        self.fft_windows: List["ImageViewerWindow"] = []
        self.fft_to_fft_window: Dict[pg.RectROI, "ImageViewerWindow"] = {}
        self.fft_box_meta: Dict[pg.RectROI, Dict[str, object]] = {}
        self.selected_fft_box: Optional[pg.RectROI] = None
        self.inverse_fft_windows: List["ImageViewerWindow"] = []
        self.inverse_fft_to_window: Dict[pg.RectROI, "ImageViewerWindow"] = {}
        self.inverse_fft_box_meta: Dict[pg.RectROI, Dict[str, object]] = {}
        self.inverse_fft_boxes: List[pg.RectROI] = []
        self.inverse_fft_count = 0
        self.selected_inverse_fft_box: Optional[pg.RectROI] = None
        self.scale_bar: Optional[DynamicScaleBar] = None
        self.edx_legend: Optional[DynamicLegendBox] = None
        self.measurement_history_window: Optional[MeasurementHistoryWindow] = None
        self.metadata_window: Optional[MetadataWindow] = None
        self.measurement_count = 0
        self.profile_measurement_count = 0
        self.btn_measure: Optional[QtWidgets.QPushButton] = None
        self.measurement_items: List[Tuple[int, pg.PlotDataItem, pg.TextItem]] = []
        self.profile_measurement_items: Dict[int, Dict[str, Any]] = {}
        self.selected_measurement_index: Optional[int] = None

        # Colormap state for the main image view
        self._available_colormaps: List[str] = [
            "gray",
            "viridis",
            "plasma",
            "magma",
            "inferno",
            "cividis",
        ]
        self._current_colormap_index: int = 0
        self.btn_colormap: Optional[QtWidgets.QAbstractButton] = None

        # Vim-style command line (":" commands)
        self.command_edit: Optional[QtWidgets.QLineEdit] = None

        self.display_min: Optional[float] = None
        self.display_max: Optional[float] = None
        self.display_gamma: float = 1.0
        self._tone_dialog: Optional[ToneCurveDialog] = None

        self.fft_name = fft_name
        self.parent_image_window = parent_image_window
        self.elemental_map_signals: Optional[List[Tuple[str, Any]]] = elemental_map_signals
        self.eds_spectrum_signals: Optional[List[Tuple[str, Any]]] = eds_spectrum_signals
        self._magnitude_spectrum = None
        self._fft_complex = None
        self._inverse_fft_cache = None
        self._inverse_fft_image = None
        self._nyq_x = None
        self._nyq_y = None
        self._last_region_id = None
        self._source_region = source_region if source_region is not None else fft_region
        self._fft_region = self._source_region
        self.freq_axis_base_unit: str = "m"
        self._calibration_status: str = "metadata"
        self._manual_calibrated: bool = False
        self._line_draw_mode: str = "measurement"
        self._calibration_dialog_state: Optional[Dict[str, Any]] = None
        self._on_calibration_pixels_selected: Optional[Callable[[float], None]] = None
        self._base_window_title: str = ""
        self._render_settings: Dict[str, Any] = get_effective_render_settings()
        self._mipmap_levels: List[np.ndarray] = []
        self._current_mipmap_level: int = -1
        self._display_image_full_res: Optional[np.ndarray] = None
        self._export_trace_enabled: bool = str(
            os.environ.get("TEMINATOR_EXPORT_TRACE", "")
        ).strip().lower() in {"1", "true", "yes", "on"}
        self._edx_hover_connected: bool = False
        self.measurements = MeasurementController(self, logger)
        self.fft_manager = FFTWindowManager(self, logger, FFT_COLORS)
        self.edx_manager = SpectrumAnalysisManager(self, logger)
        self.commands = ViewerCommandRouter(self, logger)
        self._command_mode = CommandModeController(
            command_edit_getter=lambda: self.command_edit,
            run_command=self.commands.run_vim_command,
            on_unknown_command=lambda cmd: QtWidgets.QMessageBox.information(
                self,
                "Command",
                f"Unknown command: {cmd}",
            ),
            focus_target_getter=lambda: (
                self.glw if hasattr(self.glw, "setFocus") else None
            ),
        )
        self.help_actions = HelpDialogActions(
            parent_widget=self,
            menu_config_provider=lambda: build_menu_config_for_role(
                role="viewer", callbacks_map={}
            ),
            extra_shortcuts_provider=lambda: {
                "Enter command mode": ":",
                "Exit command mode": "Esc",
                "Delete selected ROI": "Delete",
            },
            additional_colormaps_provider=lambda: self._available_colormaps,
            logger_instance=logger,
        )

        if self.view_mode in {"fft", "inverse_fft"}:
            self._setup_transform_view()
        elif signal is not None:
            self._setup_from_signal(signal, window_suffix)
        else:
            self._load_and_setup()

        # Install a global key event filter so we can capture ':'
        # even when the plot widget has focus.
        app = QtWidgets.QApplication.instance()
        if app is not None:
            app.installEventFilter(self)

    # Drag and drop -----------------------------------------------------

    def dragEnterEvent(self, event: QtGui.QDragEnterEvent):  # type: ignore[override]
        """Accept drag events for image files.

        Args:
            event: Qt event object carrying user interaction details.

        """
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QtGui.QDropEvent):  # type: ignore[override]
        """Handle dropped local files and open valid image paths.

        Args:
            event: Qt event object carrying user interaction details.

        """
        from pathlib import Path as _Path

        for url in event.mimeData().urls():
            file_path = url.toLocalFile()
            if not file_path:
                continue
            try:
                if not _Path(file_path).is_file():
                    continue
            except Exception:
                continue

            open_image_file(file_path)

    # Loading and setup -------------------------------------------------

    def _load_and_setup(self):
        """Load the image signal from disk and initialize viewer state."""
        try:
            logger.debug("Loading image via HyperSpy: %s", self.file_path)
            s = hs.load(self.file_path)
            if s.axes_manager.navigation_dimension != 0:
                logger.debug(
                    "Loaded signal has navigation_dimension=%s; using first navigation element",
                    s.axes_manager.navigation_dimension,
                )
                s = s.inav[0, 0]
            self._setup_from_signal(s)
            logger.debug("Image setup complete for: %s", self.file_path)
        except Exception as e:
            logger.exception("Failed to load file: %s", self.file_path)
            QtWidgets.QMessageBox.critical(self, "Error Loading File", str(e))

    def _setup_from_signal(self, signal, window_suffix: Optional[str] = None):
        """Initialize viewer data, calibration, and display mode from a signal.

        Args:
            signal: HyperSpy signal containing image data, axes, and metadata.
            window_suffix: Optional suffix appended to the window title for derived views.
        """
        self.signal = signal
        self.data = signal.data
        self.ax_x = signal.axes_manager[0]
        self.ax_y = signal.axes_manager[1]

        calibrated = self._apply_calibration_from_original_metadata()
        if not calibrated:
            self._calibration_status = "uncalibrated"
            logger.debug(
                "Startup metadata calibration failed; entering uncalibrated mode"
            )
            QtWidgets.QMessageBox.warning(
                self,
                "Calibration",
                "This image could not be calibrated successfully from metadata; "
                "using default pixel scaling instead.",
            )
        else:
            self._calibration_status = "metadata"
            logger.debug("Startup metadata calibration applied successfully")

        # Decide whether this view should be treated as reciprocal space.
        # Prefer explicit metadata / axis units when available so that
        # multiple images from the same file are classified consistently,
        # falling back to an image‑content heuristic only as a last resort.
        self.is_reciprocal_space = self._detect_reciprocal_space(signal)
        logger.debug(
            "Signal setup: shape=%s units=(%s,%s) reciprocal=%s",
            getattr(self.data, "shape", None),
            getattr(self.ax_x, "units", None),
            getattr(self.ax_y, "units", None),
            self.is_reciprocal_space,
        )

        self._init_display_window()

        file_name = Path(self.file_path).stem
        if window_suffix:
            title = f"Image Viewer - {file_name} {window_suffix}"
        else:
            title = f"Image Viewer - {file_name}"
        self._set_base_window_title(title)

        # Load EDX data BEFORE setup_ui so menu items can be properly configured
        self.edx_manager.detect_and_load_edx_data(
            elemental_map_signals=self.elemental_map_signals,
            spectrum_signals=self.eds_spectrum_signals,
        )
        self.setup_ui()
        self.resize(*DEFAULT_IMAGE_WINDOW_SIZE)

    def _setup_transform_view(self):
        """Initialize this window as an FFT or inverse-FFT transform view."""
        if self._source_region is None:
            QtWidgets.QMessageBox.critical(
                self, "FFT", "Could not create transform view: missing ROI data."
            )
            self.close()
            return

        logger.debug(
            "Setting up transform view: mode=%s source_shape=%s source_dtype=%s",
            self.view_mode,
            getattr(self._source_region, "shape", None),
            getattr(self._source_region, "dtype", None),
        )

        parent = self.parent_image_window
        if parent is None or parent.ax_x is None or parent.ax_y is None:
            QtWidgets.QMessageBox.critical(
                self,
                "FFT",
                "Could not create transform view: missing parent calibration.",
            )
            self.close()
            return

        is_fft_view = self.view_mode == "fft"
        self.is_reciprocal_space = is_fft_view
        self.ax_x = parent.ax_x
        self.ax_y = parent.ax_y
        self.freq_axis_base_unit = self.ax_x.units or "m"

        parent_title = parent.windowTitle().replace("Image Viewer - ", "")
        default_name = "FFT" if is_fft_view else "iFFT"
        display_name = self.fft_name or default_name
        self._set_base_window_title(f"Image Viewer - {parent_title} - {display_name}")

        self._refresh_transform_data()
        if self.view_mode == "fft":
            self.data = (
                self._magnitude_spectrum
            )  # set the image date to the magnitude spectrum for the FFT view
        elif self.view_mode == "inverse_fft":
            self.data = (
                self._inverse_fft_image
            )  # set the image data to the inverse FFT result for the iFFT view
        else:
            QtWidgets.QMessageBox.critical(
                self, "FFT", f"Unknown transform view mode: {self.view_mode}"
            )
            self.close()
            return
        self._init_display_window()
        self.setup_ui()
        self.resize(*DEFAULT_FFT_WINDOW_SIZE)

    def _get_original_metadata_dict(self) -> Optional[dict]:
        """Return original metadata dictionary for the current signal.

        Returns:
            Computed result produced by this operation.
        """
        return self._get_original_metadata_dict_from_signal(self.signal)

    def _get_original_metadata_dict_from_signal(self, signal) -> Optional[dict]:
        """Return original metadata dictionary from a provided signal object.

        Args:
            signal: HyperSpy signal containing image data, axes, and metadata.

        Returns:
            Computed result produced by this operation.
        """
        if signal is None:
            return None

        original_meta = None

        if hasattr(signal, "original_metadata"):
            original_meta = signal.original_metadata
        elif hasattr(signal.metadata, "original_metadata"):
            original_meta = signal.metadata.original_metadata

        if original_meta is None:
            return None

        if hasattr(original_meta, "as_dictionary"):
            return original_meta.as_dictionary()

        try:
            return dict(original_meta)
        except Exception:
            return None

    def _extract_ser_calibration(
        self, meta: dict
    ) -> Optional[Tuple[float, float, Optional[float], Optional[float]]]:
        """Extract calibration deltas/offsets from SER reader metadata.

        Args:
            meta: Metadata dictionary extracted from image headers or HyperSpy objects.

        Returns:
            Tuple of (dx, dy, ox, oy) calibration values, or None when metadata is incomplete.
        """
        ser_params = None
        if "ser_header_parameters" in meta:
            ser_params = meta["ser_header_parameters"]
        else:
            for key, value in meta.items():
                if isinstance(key, str) and key.lower() == "ser_header_parameters":
                    ser_params = value
                    break

        if not isinstance(ser_params, dict):
            logger.debug("Calibration metadata missing ser_header_parameters")
            return None

        dx = ser_params.get("CalibrationDeltaX")
        dy = ser_params.get("CalibrationDeltaY")
        if not (isinstance(dx, (int, float)) and isinstance(dy, (int, float))):
            logger.debug("Calibration metadata has invalid deltas: dx=%r dy=%r", dx, dy)
            return None
        if dx <= 0 or dy <= 0:
            logger.debug(
                "Calibration metadata deltas must be positive: dx=%r dy=%r", dx, dy
            )
            return None

        ox = ser_params.get("CalibrationOffsetX")
        oy = ser_params.get("CalibrationOffsetY")
        out_ox = float(ox) if isinstance(ox, (int, float)) else None
        out_oy = float(oy) if isinstance(oy, (int, float)) else None

        return float(dx), float(dy), out_ox, out_oy

    def _metadata_mode_values(self, root: Any) -> List[str]:
        """Collect potential acquisition/display mode strings from metadata."""
        values: List[str] = []

        def _visit(node: Any) -> None:
            if isinstance(node, dict):
                for key, value in node.items():
                    key_text = str(key).strip().lower().replace(" ", "")
                    if key_text in {
                        "mode",
                        "displaymode",
                        "display_mode",
                        "imagingmode",
                        "acquisitionmode",
                    } and isinstance(value, str):
                        mode_text = value.strip()
                        if mode_text:
                            values.append(mode_text)

                    if isinstance(value, (dict, list, tuple, set)):
                        _visit(value)
            elif isinstance(node, (list, tuple, set)):
                for item in node:
                    if isinstance(item, (dict, list, tuple, set)):
                        _visit(item)

        _visit(root)
        return values

    def _metadata_mode_indicates_reciprocal(self, meta: Optional[dict]) -> bool:
        """Return True when metadata mode text indicates reciprocal-space data."""
        if not meta:
            return False

        for mode_value in self._metadata_mode_values(meta):
            normalized = " ".join(mode_value.lower().split())
            if any(token in normalized for token in DIFFRACTION_MODE_TOKENS):
                logger.debug(
                    "Metadata mode indicates reciprocal-space data: mode=%s",
                    mode_value,
                )
                return True
        return False

    def _reciprocal_unit_from_axis_unit(self, axis_unit: Optional[str]) -> str:
        """Build a reciprocal unit string from the indicated axis unit."""
        normalized = unit_utils.normalize_axis_unit(axis_unit, default="m")
        if unit_utils.is_reciprocal_unit(normalized):
            return normalized
        return f"1/{normalized}"

    def _apply_axis_calibration_values(
        self,
        dx: float,
        dy: float,
        units: str,
        ox: Optional[float] = None,
        oy: Optional[float] = None,
        source: str = "unknown",
    ) -> bool:
        """Apply axis scale/unit/offset calibration values to both plot axes.

        Args:
            dx: Calibrated pixel spacing along the x-axis in physical units.
            dy: Calibrated pixel spacing along the y-axis in physical units.
            units: Physical unit label applied to both image axes.
            ox: Optional x-axis origin offset in calibrated units.
            oy: Optional y-axis origin offset in calibrated units.
            source: Human-readable source label used in diagnostics and logs.

        Returns:
            True when axis scales/offsets are applied successfully; otherwise False.
        """
        if self.ax_x is None or self.ax_y is None:
            logger.debug(
                "Skipping axis calibration apply (%s): axes unavailable", source
            )
            return False

        try:
            self.ax_x.scale = float(dx)
            self.ax_y.scale = float(dy)
            self.ax_x.units = units
            self.ax_y.units = units

            if ox is not None:
                self.ax_x.offset = float(ox)
            if oy is not None:
                self.ax_y.offset = float(oy)
        except Exception:
            logger.exception("Failed applying axis calibration (%s)", source)
            return False

        logger.debug(
            "Applied axis calibration (%s): scale_x=%s scale_y=%s units=%s offset_x=%s offset_y=%s",
            source,
            dx,
            dy,
            units,
            ox,
            oy,
        )
        return True

    def _apply_calibration_from_original_metadata(
        self,
        meta_override: Optional[dict] = None,
        source: str = "signal metadata",
    ) -> bool:
        """Apply SER calibration values extracted from original metadata.

        Args:
            meta_override: Optional metadata dictionary used instead of signal metadata.
            source: Human-readable source label used in diagnostics and logs.

        Returns:
            True when usable calibration metadata is found and applied.
        """
        meta = (
            meta_override
            if meta_override is not None
            else self._get_original_metadata_dict()
        )
        if not meta:
            logger.debug("No original metadata available for calibration (%s)", source)
            return False

        extracted = self._extract_ser_calibration(meta)
        if extracted is None:
            logger.debug(
                "Could not extract calibration values from metadata (%s)", source
            )
            return False

        dx, dy, ox, oy = extracted
        units = "m"
        if self._metadata_mode_indicates_reciprocal(meta):
            units = self._reciprocal_unit_from_axis_unit(
                self.ax_x.units if self.ax_x is not None else "m"
            )
            logger.debug(
                "Applying reciprocal calibration units from metadata mode (%s)",
                source,
            )

            if unit_utils.is_reciprocal_unit(units):
                converted_dx = unit_utils.convert_distance_value(dx, "1/m", units)
                converted_dy = unit_utils.convert_distance_value(dy, "1/m", units)
                converted_ox = (
                    unit_utils.convert_distance_value(ox, "1/m", units)
                    if ox is not None
                    else None
                )
                converted_oy = (
                    unit_utils.convert_distance_value(oy, "1/m", units)
                    if oy is not None
                    else None
                )

                if (
                    converted_dx is not None
                    and converted_dy is not None
                    and np.isfinite(converted_dx)
                    and np.isfinite(converted_dy)
                    and converted_dx > 0
                    and converted_dy > 0
                ):
                    logger.debug(
                        "Converted reciprocal metadata calibration to %s: "
                        "dx=%s->%s dy=%s->%s ox=%s->%s oy=%s->%s",
                        units,
                        dx,
                        converted_dx,
                        dy,
                        converted_dy,
                        ox,
                        converted_ox,
                        oy,
                        converted_oy,
                    )
                    dx = float(converted_dx)
                    dy = float(converted_dy)
                    if converted_ox is not None and np.isfinite(converted_ox):
                        ox = float(converted_ox)
                    if converted_oy is not None and np.isfinite(converted_oy):
                        oy = float(converted_oy)
                else:
                    logger.debug(
                        "Reciprocal metadata calibration conversion failed for units=%s; "
                        "using raw metadata values",
                        units,
                    )
        return self._apply_axis_calibration_values(
            dx, dy, units, ox=ox, oy=oy, source=source
        )

    def _reload_calibration_from_file_metadata(self) -> bool:
        """Reload pixel calibration from the source image file metadata.

        Returns:
            True if calibration was successfully loaded, False otherwise.
        """
        if not self.file_path:
            logger.debug("Cannot reload metadata calibration: file_path is empty")
            return False

        logger.debug("Reloading calibration metadata from file: %s", self.file_path)
        try:
            signal = hs.load(self.file_path)
            if signal.axes_manager.navigation_dimension != 0:
                signal = signal.inav[0, 0]
        except Exception:
            logger.exception(
                "Failed to load file for metadata calibration reload: %s",
                self.file_path,
            )
            return False

        meta = self._get_original_metadata_dict_from_signal(signal)
        ok = self._apply_calibration_from_original_metadata(
            meta_override=meta,
            source="file metadata reload",
        )
        logger.debug("Metadata calibration reload result: success=%s", ok)
        return ok

    def _open_calibration_dialog(
        self, initial_state: Optional[Dict[str, Any]] = None
    ) -> None:
        """Open the interactive calibration dialog for setting pixel size manually.

        Args:
            initial_state: Optional dict with initial calibration values.
        """
        if self.view_mode != "image" or self.ax_x is None or self.ax_y is None:
            QtWidgets.QMessageBox.information(
                self,
                "Parameters",
                "Calibration parameters are only editable for image views.",
            )
            return

        logger.debug(
            "Opening calibration dialog: initial_state_keys=%s",
            sorted((initial_state or {}).keys()),
        )

        state = dict(initial_state or {})
        current_scale_x = float(self.ax_x.scale)
        current_scale_y = float(self.ax_y.scale)
        default_ppu_x = calibration_logic.default_pixels_per_unit(current_scale_x)
        default_ppu_y = calibration_logic.default_pixels_per_unit(current_scale_y)

        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("Parameters")
        layout = QtWidgets.QFormLayout(dialog)

        unit_default = unit_utils.normalize_axis_unit(self.ax_x.units, default="nm")

        edit_ppu_x = QtWidgets.QLineEdit(
            f"{float(state.get('ppu_x', default_ppu_x)):.12g}"
        )
        edit_ppu_y = QtWidgets.QLineEdit(
            f"{float(state.get('ppu_y', default_ppu_y)):.12g}"
        )
        edit_units = QtWidgets.QLineEdit(str(state.get("units", unit_default)))
        edit_ref_pixels = QtWidgets.QLineEdit(
            f"{float(state.get('reference_pixels', 0.0)):.12g}"
        )
        initial_ref_distance = state.get("reference_distance")
        if initial_ref_distance is None:
            initial_ref_distance = f"{float(state.get('reference_units', 1.0)):.12g}"
        edit_ref_distance = QtWidgets.QLineEdit(str(initial_ref_distance))
        chk_lock_ratio = QtWidgets.QCheckBox("Lock X/Y pixel ratio")
        chk_lock_ratio.setChecked(bool(state.get("lock_xy", True)))
        lbl_ref_error = QtWidgets.QLabel("")
        lbl_ref_error.setStyleSheet("color: #d32f2f;")
        lbl_ref_error.setVisible(False)

        def set_ref_error(message: Optional[str]) -> None:
            text = (message or "").strip()
            lbl_ref_error.setText(text)
            lbl_ref_error.setVisible(bool(text))
            if text:
                logger.debug("Calibration reference validation message: %s", text)

        def sync_locked_ppu() -> None:
            if chk_lock_ratio.isChecked():
                edit_ppu_y.setText(edit_ppu_x.text().strip())
                edit_ppu_y.setEnabled(False)
            else:
                edit_ppu_y.setEnabled(True)

        def update_ppu_from_reference() -> None:
            result = calibration_logic.parse_reference_ppu(
                reference_pixels_text=edit_ref_pixels.text(),
                reference_distance_text=edit_ref_distance.text(),
                target_units_text=edit_units.text(),
            )

            if result.target_units != unit_utils.normalize_axis_unit(
                edit_units.text(), default="nm"
            ):
                edit_units.blockSignals(True)
                edit_units.setText(result.target_units)
                edit_units.blockSignals(False)

            if result.error:
                set_ref_error(result.error)
                return

            if result.ppu is None:
                set_ref_error(None)
                return

            ppu = result.ppu
            edit_ppu_x.setText(f"{ppu:.12g}")
            if chk_lock_ratio.isChecked():
                edit_ppu_y.setText(f"{ppu:.12g}")
            set_ref_error(None)
            logger.debug(
                "Calibration reference parsed: units=%s computed_ppu=%s",
                result.target_units,
                ppu,
            )

        chk_lock_ratio.toggled.connect(sync_locked_ppu)
        edit_ppu_x.textChanged.connect(sync_locked_ppu)
        edit_ref_pixels.textChanged.connect(update_ppu_from_reference)
        edit_ref_distance.textChanged.connect(update_ppu_from_reference)
        edit_units.textChanged.connect(update_ppu_from_reference)
        sync_locked_ppu()

        layout.addRow(chk_lock_ratio)
        layout.addRow("Pixels per unit X:", edit_ppu_x)
        layout.addRow("Pixels per unit Y:", edit_ppu_y)
        layout.addRow("Units:", edit_units)
        layout.addRow("Reference distance (pixels):", edit_ref_pixels)
        layout.addRow("Reference distance:", edit_ref_distance)
        layout.addRow("", lbl_ref_error)

        draw_button = QtWidgets.QPushButton("Draw distance on canvas")
        layout.addRow("Reference picker:", draw_button)

        draw_result_code = 2
        metadata_reloaded_in_dialog = False

        def _collect_state() -> Dict[str, Any]:
            return {
                "ppu_x": edit_ppu_x.text().strip() or f"{default_ppu_x:.12g}",
                "ppu_y": edit_ppu_y.text().strip() or f"{default_ppu_y:.12g}",
                "units": edit_units.text().strip() or "nm",
                "reference_pixels": edit_ref_pixels.text().strip() or "0",
                "reference_distance": edit_ref_distance.text().strip() or "1",
                "lock_xy": chk_lock_ratio.isChecked(),
            }

        def _request_draw() -> None:
            logger.debug("Calibration dialog draw requested")
            self._calibration_dialog_state = _collect_state()
            dialog.done(draw_result_code)

        draw_button.clicked.connect(_request_draw)

        def _reload_from_metadata() -> None:
            nonlocal metadata_reloaded_in_dialog
            logger.debug("Calibration dialog metadata reload requested")
            if not self._reload_calibration_from_file_metadata():
                set_ref_error("Could not reload calibration metadata from file.")
                QtWidgets.QMessageBox.warning(
                    dialog,
                    "Calibration",
                    "Could not reload calibration metadata from this file.",
                )
                return

            self._manual_calibrated = False
            self._calibration_status = "metadata"
            self.is_reciprocal_space = self._detect_reciprocal_space(self.signal)
            self._refresh_view_after_calibration_change()
            metadata_reloaded_in_dialog = True

            scale_x = float(self.ax_x.scale)
            scale_y = float(self.ax_y.scale)
            ppu_x = 1.0 / scale_x if scale_x > 0 else 1.0
            ppu_y = 1.0 / scale_y if scale_y > 0 else 1.0
            edit_ppu_x.setText(f"{ppu_x:.12g}")
            edit_ppu_y.setText(f"{ppu_y:.12g}")
            edit_units.setText(
                unit_utils.normalize_axis_unit(self.ax_x.units, default="nm")
            )
            set_ref_error(None)
            logger.debug(
                "Calibration dialog populated from reloaded metadata: ppu_x=%s ppu_y=%s units=%s",
                ppu_x,
                ppu_y,
                unit_utils.normalize_axis_unit(self.ax_x.units, default="nm"),
            )

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel,
            parent=dialog,
        )
        btn_reload_meta = buttons.addButton(
            "Reload Metadata Calibration",
            QtWidgets.QDialogButtonBox.ActionRole,
        )
        btn_reload_meta.clicked.connect(_reload_from_metadata)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addRow(buttons)

        result = dialog.exec_()
        if result == draw_result_code:
            logger.debug("Calibration dialog closed for distance drawing")
            self._start_calibration_distance_pick()
            return

        if result != QtWidgets.QDialog.Accepted:
            logger.debug("Calibration dialog cancelled")
            self._calibration_dialog_state = None
            return

        parsed_manual = calibration_logic.parse_manual_calibration(
            ppu_x_text=edit_ppu_x.text(),
            ppu_y_text=edit_ppu_y.text(),
            units_text=edit_units.text(),
            lock_xy=chk_lock_ratio.isChecked(),
        )
        if parsed_manual.error:
            QtWidgets.QMessageBox.warning(
                self,
                "Parameters",
                parsed_manual.error,
            )
            return

        new_ppu_x = float(parsed_manual.ppu_x)
        new_ppu_y = float(parsed_manual.ppu_y)
        new_units = str(parsed_manual.units)

        logger.debug(
            "Applying manual calibration from dialog: ppu_x=%s ppu_y=%s units=%s lock_xy=%s",
            new_ppu_x,
            new_ppu_y,
            new_units,
            chk_lock_ratio.isChecked(),
        )

        proposed_scale_x = 1.0 / new_ppu_x
        proposed_scale_y = 1.0 / new_ppu_y
        current_units = unit_utils.normalize_axis_unit(self.ax_x.units, default="nm")
        current_scale_x = float(self.ax_x.scale)
        current_scale_y = float(self.ax_y.scale)

        if calibration_logic.should_preserve_metadata_status(
            metadata_reloaded_in_dialog=metadata_reloaded_in_dialog,
            proposed_scale_x=proposed_scale_x,
            proposed_scale_y=proposed_scale_y,
            current_scale_x=current_scale_x,
            current_scale_y=current_scale_y,
            new_units=new_units,
            current_units=current_units,
        ):
            logger.debug(
                "Calibration dialog accepted after metadata reload with unchanged values; preserving metadata status"
            )
            self._manual_calibrated = False
            self._calibration_status = "metadata"
            self.is_reciprocal_space = self._detect_reciprocal_space(self.signal)
            self._refresh_view_after_calibration_change()
            self._calibration_dialog_state = None
            return

        self.ax_x.scale = proposed_scale_x
        self.ax_y.scale = proposed_scale_y
        self.ax_x.units = new_units
        self.ax_y.units = new_units
        self._manual_calibrated = True
        self._calibration_status = "manual"
        self._calibration_dialog_state = None
        metadata_reciprocal = self._detect_reciprocal_space(self.signal)
        self.is_reciprocal_space = bool(
            metadata_reciprocal or unit_utils.is_reciprocal_unit(new_units)
        )
        self._refresh_view_after_calibration_change()
        logger.debug("Manual calibration applied successfully")

    def _prepare_for_measurement_input(self) -> None:
        """Clear transform-ROI selection state before entering any measurement flow."""
        had_fft = self.selected_fft_box is not None
        had_ifft = self.selected_inverse_fft_box is not None

        self.selected_fft_box = None
        self.selected_inverse_fft_box = None

        if had_fft or had_ifft:
            logger.debug(
                "Cleared ROI selection before measurement input: fft_selected=%s inverse_fft_selected=%s",
                had_fft,
                had_ifft,
            )

    def _start_calibration_distance_pick(self) -> None:
        """Start interactive distance picking used by manual calibration."""
        if self.line_tool is None:
            logger.debug(
                "Calibration distance pick requested but line tool is unavailable"
            )
            QtWidgets.QMessageBox.warning(
                self,
                "Calibration",
                "Line drawing tool is not available.",
            )
            return

        logger.debug("Starting calibration distance pick")
        self._prepare_for_measurement_input()

        if self.btn_measure is not None:
            self.btn_measure.blockSignals(True)
            self.btn_measure.setChecked(False)
            self.btn_measure.blockSignals(False)
            self.btn_measure.setStyleSheet("")

        self.line_tool.disable()
        self._on_measurement_drawing_state_changed(False)

        state = dict(self._calibration_dialog_state or {})
        self._line_draw_mode = "calibration"

        def _on_pixels_selected(pixel_distance: float) -> None:
            logger.debug("Calibration distance picked: %.6g px", pixel_distance)
            next_state = dict(state)
            next_state["reference_pixels"] = f"{pixel_distance:.12g}"
            self._calibration_dialog_state = None
            QtCore.QTimer.singleShot(
                0, lambda: self._open_calibration_dialog(next_state)
            )

        self._on_calibration_pixels_selected = _on_pixels_selected
        self.line_tool.enable()
        self._on_measurement_drawing_state_changed(False)

    def _set_base_window_title(self, title: str) -> None:
        """Set and store the base window title used by stateful title updates.

        Args:
            title: Title text displayed in the associated UI element.

        """
        self._base_window_title = title
        self._on_measurement_drawing_state_changed(False)

    def _on_measurement_drawing_state_changed(self, is_drawing: bool) -> None:
        """Refresh window-title hints when measurement drawing mode changes.

        Args:
            is_drawing: Boolean flag indicating whether drawing.

        """
        title = self._base_window_title or self.windowTitle()
        for prefix in ("Measurement mode - ", "<esc> to exit measurement mode - "):
            if title.startswith(prefix):
                title = title[len(prefix) :]

        measurement_ready = bool(
            self.line_tool is not None and getattr(self.line_tool, "is_enabled", False)
        )
        if measurement_ready:
            title = f"<esc> to exit measurement mode - {title}"

        if self.windowTitle() != title:
            self.setWindowTitle(title)
            logger.debug(
                "Window title updated for measurement mode: drawing=%s ready=%s title=%s",
                is_drawing,
                measurement_ready,
                title,
            )

    def _sync_scale_bar_units_from_axes(self) -> None:
        """Sync scale-bar unit text/mode from current axis units."""
        if self.scale_bar is None or self.ax_x is None:
            return

        display_unit, reciprocal_mode = unit_utils.scale_bar_unit_and_mode(
            self.ax_x.units,
            reciprocal_hint=self.is_reciprocal_space,
        )
        self.scale_bar.units = display_unit
        self.scale_bar.reciprocal = reciprocal_mode
        logger.debug(
            "Scale bar sync from axes: axis_units=%s display_unit=%s reciprocal=%s",
            self.ax_x.units,
            display_unit,
            reciprocal_mode,
        )

    def _refresh_view_after_calibration_change(self) -> None:
        """Apply updated calibration to image rect, ranges, and scale bar."""
        x_offset, y_offset, w, h = self.image_bounds
        logger.debug(
            "Refreshing view after calibration change: x_offset=%s y_offset=%s w=%s h=%s",
            x_offset,
            y_offset,
            w,
            h,
        )

        if self.img_orig is not None:
            self.img_orig.setRect(QtCore.QRectF(x_offset, y_offset, w, h))

        if self.p1 is not None:
            self.p1.setXRange(x_offset, x_offset + w, padding=0)
            self.p1.setYRange(y_offset, y_offset + h, padding=0)

        if self.scale_bar is not None:
            self._sync_scale_bar_units_from_axes()
            if hasattr(self.scale_bar, "_update_geometry"):
                self.scale_bar._update_geometry()

        self._refresh_scale_bar_calibration_tag()

    def _refresh_scale_bar_calibration_tag(self) -> None:
        """Update the scale-bar tag indicating current calibration status."""
        if self.scale_bar is None or not hasattr(self.scale_bar, "set_status_tag"):
            return

        if self._calibration_status == "uncalibrated":
            self.scale_bar.set_status_tag("UNCALIBRATED")
        elif self._calibration_status == "manual":
            self.scale_bar.set_status_tag("manually calibrated")
        else:
            self.scale_bar.set_status_tag(None)
        logger.debug(
            "Scale bar calibration tag refreshed: status=%s", self._calibration_status
        )

    def _ensure_edx_legend_overlay(self) -> None:
        """Create the bottom-right EDX legend overlay when EDX maps exist."""
        if self.view_mode != "image" or self.p1 is None:
            return

        if not (
            hasattr(self, "edx_manager") and self.edx_manager.get_has_edx_data()
        ):
            if self.edx_legend is not None:
                self.edx_legend.set_entries([])
            return

        if self.edx_legend is None:
            self.edx_legend = DynamicLegendBox(self.p1.vb)

    def _update_edx_legend_overlay(self) -> None:
        """Refresh EDX legend entries from selected maps and their colors."""
        if self.view_mode != "image":
            return

        self._ensure_edx_legend_overlay()
        if self.edx_legend is None:
            return

        active_names: List[str] = []
        if hasattr(self, "edx_manager") and self.edx_manager.get_has_edx_data():
            active_names = [
                name
                for name in self.edx_manager.elemental_maps.keys()
                if name in self.edx_manager.active_elements
            ]

        entries = [
            (name, self.edx_manager.element_colors.get(name, (255, 255, 255)))
            for name in active_names
        ]
        self.edx_legend.set_entries(entries)

    def _blank_edx_display_image(self) -> np.ndarray:
        """Return a blank frame matching the current EDX map/image shape."""
        if isinstance(self.data, np.ndarray) and self.data.ndim >= 2:
            return np.zeros(self.data.shape[:2], dtype=np.float32)

        if hasattr(self, "edx_manager"):
            for map_data in self.edx_manager.elemental_maps.values():
                if isinstance(map_data, np.ndarray) and map_data.ndim >= 2:
                    return np.zeros(map_data.shape[:2], dtype=np.float32)

        return np.zeros((1, 1), dtype=np.float32)

    def _detect_reciprocal_space(self, signal) -> bool:
        """Determine if the signal should be treated as reciprocal space.

        Args:
            signal: HyperSpy signal containing image data, axes, and metadata.

        Returns:
            Computed result produced by this operation.
        """

        # 1) Prefer explicit HyperSpy metadata signal_type when available
        signal_type_hint: Optional[bool] = None
        try:
            meta = getattr(signal, "metadata", None)
            sig_meta = getattr(meta, "Signal", None)
            sig_type = getattr(sig_meta, "signal_type", None)
            if isinstance(sig_type, str):
                st = " ".join(sig_type.strip().lower().replace("_", " ").split())
                if (
                    st in RECIPROCAL_SIGNAL_TYPE_TOKENS
                    or any(token in st for token in RECIPROCAL_SIGNAL_TYPE_SUBSTRINGS)
                ):
                    return True
                if st in REAL_SIGNAL_TYPE_TOKENS:
                    signal_type_hint = False
        except Exception:
            pass

        # 2) Original metadata Mode is authoritative when it indicates diffraction
        try:
            original_meta = self._get_original_metadata_dict_from_signal(signal)
            if self._metadata_mode_indicates_reciprocal(original_meta):
                return True
        except Exception:
            pass

        if signal_type_hint is False:
            return False

        # 3) Inspect axis unit strings for common reciprocal-space patterns
        units_parts: list[str] = []
        try:
            ax0 = signal.axes_manager[0]
            ax1 = signal.axes_manager[1]
            units_parts.append(str(getattr(ax0, "units", "") or ""))
            units_parts.append(str(getattr(ax1, "units", "") or ""))
        except Exception:
            pass

        if any(unit_utils.is_reciprocal_unit(part) for part in units_parts):
            return True

        # 4) Default: treat as real-space image. We no longer rely on
        # intensity-based heuristics for diffraction detection because
        # they can classify visually similar images from the same file
        # differently, leading to inconsistent scale-bar behaviour.
        return False

    @property
    def image_bounds(self) -> Tuple[float, float, float, float]:
        """Get the pixel bounds of the displayed image.

        Returns:
            (x_min, x_max, y_min, y_max) in image pixel coordinates.
        """
        x_offset = self.ax_x.offset if self.ax_x else 0
        y_offset = self.ax_y.offset if self.ax_y else 0
        w = self.ax_x.size * self.ax_x.scale if self.ax_x else 1
        h = self.ax_y.size * self.ax_y.scale if self.ax_y else 1
        return x_offset, y_offset, w, h

    def _init_display_window(self):
        """Initialize display min/max/gamma from current source data."""
        source_data = self.data
        if self.view_mode == "fft":
            source_data = self._magnitude_spectrum
        elif self.view_mode == "inverse_fft":
            source_data = self._inverse_fft_image

        if source_data is None:
            self.display_min = 0.0
            self.display_max = 1.0
            self.display_gamma = 1.0
            return

        arr = np.asarray(source_data, dtype=float)
        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            self.display_min = 0.0
            self.display_max = 1.0
        else:
            self.display_min = float(finite.min())
            self.display_max = float(finite.max())
        self.display_gamma = 1.0
        logger.debug(
            "Initialized display window: min=%s max=%s gamma=%s",
            self.display_min,
            self.display_max,
            self.display_gamma,
        )

    def setup_ui(self):
        """Set up the main UI layout with plot, controls, and toolbars."""
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        main_layout = QtWidgets.QVBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        self._setup_menu_bar()

        self.glw = pg.GraphicsLayoutWidget()
        logger.debug(
            "setup_ui: mode=%s useOpenGL=%s QT_QPA_PLATFORM=%r QT_OPENGL=%r",
            self.view_mode,
            pg.getConfigOption("useOpenGL"),
            os.environ.get("QT_QPA_PLATFORM"),
            os.environ.get("QT_OPENGL"),
        )
        self.glw.ci.setContentsMargins(0, 0, 0, 0)
        if hasattr(self.glw.ci, "layout"):
            self.glw.ci.layout.setContentsMargins(0, 0, 0, 0)
            self.glw.ci.layout.setSpacing(0)

        # Create layout with EDX panel if available
        if (
            hasattr(self, "edx_manager")
            and self.edx_manager.get_has_edx_data()
            and self.view_mode == "image"
        ):
            edx_splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
            edx_splitter.addWidget(self.glw)

            edx_panel = self.edx_manager.build_edx_panel()
            if edx_panel:
                edx_splitter.addWidget(edx_panel)
                edx_splitter.setStretchFactor(0, 3)
                edx_splitter.setStretchFactor(1, 1)

            main_layout.addWidget(edx_splitter)
        else:
            main_layout.addWidget(self.glw)

        # Hidden command line at the bottom, shown when ':' is pressed
        self.command_edit = QtWidgets.QLineEdit()
        self.command_edit.setPlaceholderText("Vim command (:F, :d, :e <file>, :E, :a)")
        self.command_edit.returnPressed.connect(self._execute_command_from_line)
        self.command_edit.hide()
        main_layout.addWidget(self.command_edit)

        # Shortcut for ':' so command mode is easy to enter while this
        # image window is active.
        colon_shortcut = QtGui.QShortcut(QtGui.QKeySequence(":"), self)
        colon_shortcut.activated.connect(self._enter_command_mode)

        self.btn_measure = QtWidgets.QPushButton("Measure Distance")
        self.btn_measure.setCheckable(True)
        self.btn_measure.clicked.connect(self.measurements.toggle_line_measurement)
        self.btn_measure.hide()

        self.p1 = self.glw.addPlot()
        self.p1.hideAxis("bottom")
        self.p1.hideAxis("left")
        self.p1.showAxis("top", False)
        self.p1.showAxis("right", False)
        self.p1.hideButtons()
        self.p1.setMenuEnabled(False)

        self.p1.vb.setAspectLocked(True, ratio=1.0)
        if hasattr(self.p1.vb, "setBorder"):
            self.p1.vb.setBorder(None)

        self.img_orig = pg.ImageItem(axisOrder="row-major")
        self.p1.addItem(self.img_orig)
        self._apply_render_preferences_to_view()
        self.p1.invertY(True)
        if self.view_mode == "fft":
            cmap = pg.colormap.get("magma")
            self.img_orig.setLookupTable(cmap.getLookupTable())
            self._update_image_display()
            if self._nyq_x is not None and self._nyq_y is not None:
                self.img_orig.setRect(
                    QtCore.QRectF(
                        -self._nyq_x, -self._nyq_y, 2 * self._nyq_x, 2 * self._nyq_y
                    )
                )
            if hasattr(self.p1.vb, "setPadding"):
                self.p1.vb.setPadding(0.0)
            fft_unit, fft_reciprocal = unit_utils.scale_bar_unit_and_mode(
                self.freq_axis_base_unit,
                reciprocal_hint=True,
            )
            self.scale_bar = DynamicScaleBar(self.p1.vb, units=fft_unit)
            self.scale_bar.reciprocal = fft_reciprocal
        elif self.view_mode == "inverse_fft":
            self.img_orig.setLookupTable(None)
            self._update_image_display()
            x_offset, y_offset, w, h = self.image_bounds
            self.img_orig.setRect(QtCore.QRectF(x_offset, y_offset, w, h))

            if hasattr(self.p1.vb, "setPadding"):
                self.p1.vb.setPadding(0.0)
            self.p1.setXRange(x_offset, x_offset + w, padding=0)
            self.p1.setYRange(y_offset, y_offset + h, padding=0)

            axis_units = self.ax_x.units if self.ax_x is not None else "nm"
            display_unit, reciprocal_mode = unit_utils.scale_bar_unit_and_mode(
                axis_units,
                reciprocal_hint=False,
            )
            self.scale_bar = DynamicScaleBar(self.p1.vb, units=display_unit)
            self.scale_bar.reciprocal = reciprocal_mode
        else:
            self._apply_colormap()
            self._update_image_display()

            x_offset, y_offset, w, h = self.image_bounds
            self.img_orig.setRect(QtCore.QRectF(x_offset, y_offset, w, h))

            if hasattr(self.p1.vb, "setPadding"):
                self.p1.vb.setPadding(0.0)
            self.p1.setXRange(x_offset, x_offset + w, padding=0)
            self.p1.setYRange(y_offset, y_offset + h, padding=0)

            axis_units = self.ax_x.units if self.ax_x is not None else "nm"
            display_unit, reciprocal_mode = unit_utils.scale_bar_unit_and_mode(
                axis_units,
                reciprocal_hint=self.is_reciprocal_space,
            )
            self.scale_bar = DynamicScaleBar(self.p1.vb, units=display_unit)
            self.scale_bar.reciprocal = reciprocal_mode
            if not reciprocal_mode:
                try:
                    overlay_label = self._build_export_overlay_label()
                    if overlay_label and hasattr(self.scale_bar, "set_extra_label"):
                        self.scale_bar.set_extra_label(overlay_label)
                except Exception:
                    pass

            self._refresh_scale_bar_calibration_tag()
            self._ensure_edx_legend_overlay()
            self._update_edx_legend_overlay()

        self.line_tool = LineDrawingTool(
            self.p1,
            self.measurements.on_line_drawn,
            self._on_measurement_drawing_state_changed,
        )

        # Route scene hover to EDS manager for mouse-over spectra updates.
        if (
            not self._edx_hover_connected
            and self.view_mode == "image"
            and hasattr(self, "edx_manager")
            and self.edx_manager.get_has_edx_data()
        ):
            try:
                scene = self.glw.scene() if self.glw is not None else None
                if scene is not None and hasattr(scene, "sigMouseMoved"):
                    scene.sigMouseMoved.connect(self._on_scene_mouse_moved_for_edx)
                    self._edx_hover_connected = True
            except Exception:
                logger.debug("Could not connect EDS hover handler", exc_info=True)

        if hasattr(self.p1, "vb") and hasattr(self.p1.vb, "sigRangeChanged"):
            self.p1.vb.sigRangeChanged.connect(self._on_view_range_changed)

        self.setup_keyboard_shortcuts()

    def _render_quality_mode(self) -> str:
        """Return configured image-resampling quality mode.

        Returns:
            Normalized render quality mode key used by the viewer pipeline.
        """
        return (
            str(self._render_settings.get("image_resampling_quality", RESAMPLING_HIGH))
            .strip()
            .lower()
        )

    def _apply_render_preferences_to_view(self) -> None:
        """Apply rendering preference flags to the image view backend."""
        if self.img_orig is None or self.glw is None:
            return

        quality = self._render_quality_mode()
        auto_downsample = quality in {RESAMPLING_BALANCED, RESAMPLING_HIGH}
        smooth = quality in {RESAMPLING_BALANCED, RESAMPLING_HIGH}

        if hasattr(self.img_orig, "setAutoDownsample"):
            self.img_orig.setAutoDownsample(auto_downsample)

        if hasattr(self.glw, "setRenderHint"):
            self.glw.setRenderHint(QtGui.QPainter.SmoothPixmapTransform, smooth)

        logger.debug(
            "Applied render preferences: mode=%s quality=%s auto_downsample=%s smooth_transform=%s useOpenGL=%s",
            self.view_mode,
            quality,
            auto_downsample,
            smooth,
            pg.getConfigOption("useOpenGL"),
        )

    @staticmethod
    def _downsample_by_2_mean(image: np.ndarray) -> Optional[np.ndarray]:
        """Downsample a 2D image by 2x2 mean pooling.

        Args:
            image: 2D image array used for display, FFT, or tone processing.

        Returns:
            Half-resolution image created by 2x2 mean pooling, or None for unsupported inputs.
        """
        if image.ndim != 2:
            return None

        src = np.asarray(image, dtype=np.float32)
        h, w = src.shape
        h2 = (h // 2) * 2
        w2 = (w // 2) * 2
        if h2 < 2 or w2 < 2:
            return None

        view = src[:h2, :w2]
        return view.reshape(h2 // 2, 2, w2 // 2, 2).mean(axis=(1, 3), dtype=np.float32)

    def _build_mipmap_levels(self, image: np.ndarray) -> None:
        """Build a pyramid of downsampled levels for high-quality zooming.

        Args:
            image: Input image array used by the computation.

        """
        self._mipmap_levels = []
        base = np.asarray(image, dtype=np.float32)
        self._mipmap_levels.append(base)

        while len(self._mipmap_levels) < 9:
            nxt = self._downsample_by_2_mean(self._mipmap_levels[-1])
            if nxt is None:
                break
            self._mipmap_levels.append(nxt)

        self._current_mipmap_level = -1

    def _compute_target_mipmap_level(self) -> int:
        """Select mipmap level that best matches current visible resolution.

        Returns:
            Best mipmap pyramid level for the current zoom and viewport.
        """
        if not self._mipmap_levels or self.p1 is None or self.img_orig is None:
            return 0

        try:
            rect = self.img_orig.rect()
            rect_w = float(rect.width())
            rect_h = float(rect.height())
            if rect_w <= 0 or rect_h <= 0:
                return 0

            vr = self.p1.vb.viewRange()
            x0, x1 = float(vr[0][0]), float(vr[0][1])
            y0, y1 = float(vr[1][0]), float(vr[1][1])

            visible_cols = abs((x1 - x0) / rect_w) * float(
                self._mipmap_levels[0].shape[1]
            )
            visible_rows = abs((y1 - y0) / rect_h) * float(
                self._mipmap_levels[0].shape[0]
            )

            view_px_w = max(float(self.p1.vb.width()), 1.0)
            view_px_h = max(float(self.p1.vb.height()), 1.0)

            src_per_screen = max(
                visible_cols / view_px_w, visible_rows / view_px_h, 1.0
            )
            level = int(np.floor(np.log2(src_per_screen)))
        except Exception:
            return 0

        return max(0, min(level, len(self._mipmap_levels) - 1))

    def _trace_event(self, event: str, **fields: Any) -> None:
        """Emit a structured debug trace line when export tracing is enabled."""
        if not self._export_trace_enabled:
            return
        payload_parts: List[str] = []
        for key, value in fields.items():
            payload_parts.append(f"{key}={value}")
        payload = " ".join(payload_parts)
        logger.debug("EXPORT_TRACE event=%s %s", event, payload)

    @staticmethod
    def _trace_array_stats(array: Any, label: str) -> str:
        """Return compact numeric stats for tracing array state transitions."""
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
            return (
                f"{label}:shape={arr.shape} dtype={arr.dtype} finite=0/{arr.size} "
                "min=nan max=nan"
            )

        finite_values = arr[finite_mask]
        min_val = float(np.min(finite_values))
        max_val = float(np.max(finite_values))
        mean_val = float(np.mean(finite_values))
        std_val = float(np.std(finite_values))
        p01, p50, p99 = np.percentile(finite_values, [1, 50, 99])
        channels = arr.shape[2] if arr.ndim == 3 else 1
        return (
            f"{label}:shape={arr.shape} dtype={arr.dtype} channels={channels} "
            f"finite={finite_count}/{arr.size} min={min_val:.6g} max={max_val:.6g} "
            f"mean={mean_val:.6g} std={std_val:.6g} p01={float(p01):.6g} "
            f"p50={float(p50):.6g} p99={float(p99):.6g}"
        )

    def _set_display_image(self, image: np.ndarray) -> None:
        """Push image data into the display item with current quality mode.

        Args:
            image: Input image array used by the computation.

        """
        if self.img_orig is None:
            return

        self._trace_event(
            "set_display_image_enter",
            view_mode=self.view_mode,
            quality=self._render_quality_mode(),
            source_stats=self._trace_array_stats(image, "input"),
        )

        self._display_image_full_res = np.asarray(image, dtype=np.float32)
        quality = self._render_quality_mode()

        self._trace_event(
            "set_display_image_after_cast",
            full_res_stats=self._trace_array_stats(self._display_image_full_res, "full_res"),
        )

        if quality == RESAMPLING_HIGH:
            self._build_mipmap_levels(self._display_image_full_res)
            self._trace_event(
                "set_display_image_mipmap_built",
                levels=len(self._mipmap_levels),
                level0_stats=self._trace_array_stats(
                    self._mipmap_levels[0] if self._mipmap_levels else None,
                    "mipmap0",
                ),
            )
            self._on_view_range_changed(force=True)
            return

        self._mipmap_levels = []
        self._current_mipmap_level = -1
        self.img_orig.setImage(
            self._display_image_full_res, autoLevels=False, levels=(0.0, 1.0)
        )
        self._trace_event(
            "set_display_image_after_setImage",
            levels="(0.0,1.0)",
            image_item_stats=self._trace_array_stats(getattr(self.img_orig, "image", None), "img_item"),
        )

    def _on_view_range_changed(self, *_args, force: bool = False) -> None:
        """Update displayed mipmap image after pan/zoom range changes.

        Args:
            *_args: Additional callback arguments from Qt signal emissions.
            force: When True, refreshes display state even if cached values match.
        """
        if self.img_orig is None:
            return
        if self._render_quality_mode() != RESAMPLING_HIGH:
            return
        if not self._mipmap_levels:
            return

        level = self._compute_target_mipmap_level()
        self._trace_event(
            "view_range_changed",
            force=force,
            chosen_level=level,
            current_level=self._current_mipmap_level,
            mipmap_levels=len(self._mipmap_levels),
        )
        if not force and level == self._current_mipmap_level:
            return

        self._current_mipmap_level = level
        self.img_orig.setImage(
            self._mipmap_levels[level], autoLevels=False, levels=(0.0, 1.0)
        )
        self._trace_event(
            "view_range_after_setImage",
            level=level,
            level_stats=self._trace_array_stats(self._mipmap_levels[level], "mipmap"),
            image_item_stats=self._trace_array_stats(getattr(self.img_orig, "image", None), "img_item"),
        )

    def _update_image_display(self):
        """Render source data into the image item using tone and colormap settings."""
        self._trace_event(
            "update_display_enter",
            view_mode=self.view_mode,
            has_data=self.data is not None,
            has_img_item=self.img_orig is not None,
            display_min=self.display_min,
            display_max=self.display_max,
            display_gamma=self.display_gamma,
        )
        if self.data is None or self.img_orig is None:
            if self.view_mode not in {"fft", "inverse_fft"}:
                return

        if self.display_min is None or self.display_max is None:
            self._init_display_window()

        # EDX checkbox-driven map rendering
        if (
            self.view_mode == "image"
            and hasattr(self, "edx_manager")
            and self.edx_manager.get_has_edx_data()
        ):
            self._trace_event(
                "update_display_edx_branch",
                active_elements=sorted(list(self.edx_manager.active_elements)),
                map_count=len(self.edx_manager.elemental_maps),
            )
            self._update_edx_legend_overlay()
            if self.edx_manager.active_elements:
                composite_map = self.edx_manager.render_composite_map()
                if composite_map is not None:
                    display_data = composite_map.astype(np.float32) / 255.0
                    self._trace_event(
                        "update_display_edx_composite",
                        composite_stats=self._trace_array_stats(composite_map, "composite_u8"),
                        display_stats=self._trace_array_stats(display_data, "display_f32"),
                    )
                    self._set_display_image(display_data)
                    return
            else:
                blank = self._blank_edx_display_image()
                self._trace_event(
                    "update_display_edx_blank",
                    reason="no_active_elements",
                    blank_stats=self._trace_array_stats(blank, "blank"),
                )
                self._set_display_image(blank)
                return

        if self.view_mode == "fft":
            if self._magnitude_spectrum is None:
                return

            display_data = self._magnitude_spectrum

            adjusted = utils.apply_intensity_transform(
                display_data,
                self.display_min,
                self.display_max,
                self.display_gamma,
            )
            if adjusted is None:
                return

            self._trace_event(
                "update_display_fft",
                source_stats=self._trace_array_stats(display_data, "fft_source"),
                adjusted_stats=self._trace_array_stats(adjusted, "fft_adjusted"),
            )

            self._set_display_image(adjusted)
            if self._nyq_x is not None and self._nyq_y is not None:
                self.img_orig.setRect(
                    QtCore.QRectF(
                        -self._nyq_x, -self._nyq_y, 2 * self._nyq_x, 2 * self._nyq_y
                    )
                )
            return

        if self.view_mode == "inverse_fft":
            if self._inverse_fft_image is None:
                return

            adjusted = utils.apply_intensity_transform(
                self._inverse_fft_image,
                self.display_min,
                self.display_max,
                self.display_gamma,
            )
            if adjusted is None:
                return

            self._trace_event(
                "update_display_inverse_fft",
                source_stats=self._trace_array_stats(self._inverse_fft_image, "ifft_source"),
                adjusted_stats=self._trace_array_stats(adjusted, "ifft_adjusted"),
            )

            self._set_display_image(adjusted)
            return

        display_data = self.data

        adjusted = utils.apply_intensity_transform(
            display_data,
            self.display_min,
            self.display_max,
            self.display_gamma,
        )
        if adjusted is None:
            return

        self._trace_event(
            "update_display_image",
            source_stats=self._trace_array_stats(display_data, "image_source"),
            adjusted_stats=self._trace_array_stats(adjusted, "image_adjusted"),
        )

        self._set_display_image(adjusted)

    def _apply_colormap(self) -> None:
        """Apply the currently selected colormap to the main image."""

        if self.img_orig is None:
            return

        if not self._available_colormaps:
            return

        name = self._available_colormaps[
            self._current_colormap_index % len(self._available_colormaps)
        ]
        lut_obj = getattr(self.img_orig, "lut", None)
        lut_len = len(lut_obj) if lut_obj is not None else 0
        self._trace_event(
            "apply_colormap_enter",
            selected=name,
            display_stats=self._trace_array_stats(self._display_image_full_res, "display"),
            lut_len_before=lut_len,
        )

        # "gray" means the default grayscale appearance (no custom LUT)
        if name == "gray":
            try:
                self.img_orig.setLookupTable(None)
            except Exception:
                pass
            self._trace_event("apply_colormap_gray", action="setLookupTable(None)")
            return

        try:
            cmap = pg.colormap.get(name)
            lut = cmap.getLookupTable()
            self.img_orig.setLookupTable(lut)
            self._trace_event(
                "apply_colormap_named",
                action="setLookupTable(lut)",
                lut_len=len(lut),
            )
        except Exception:
            # Fall back to default grayscale if something goes wrong
            try:
                self.img_orig.setLookupTable(None)
            except Exception:
                pass
            self._trace_event(
                "apply_colormap_error",
                action="fallback_setLookupTable(None)",
            )

    def setup_keyboard_shortcuts(self):
        """Configure keyboard shortcuts for common image operations.

        Sets up Delete key for removing FFT regions and other navigation shortcuts.
        """
        delete_shortcut = QtGui.QShortcut(QtGui.QKeySequence.Delete, self)
        delete_shortcut.activated.connect(self.fft_manager.delete_selected_roi)

        backspace_shortcut = QtGui.QShortcut(
            QtGui.QKeySequence(QtCore.Qt.Key_Backspace), self
        )
        backspace_shortcut.activated.connect(self.fft_manager.delete_selected_roi)

        escape_shortcut = QtGui.QShortcut(
            QtGui.QKeySequence(QtCore.Qt.Key_Escape), self
        )
        escape_shortcut.activated.connect(self.measurements.exit_measure_mode)

    def _set_colormap_by_name(self, name: str) -> None:
        """Set the active colormap by name and update the button label.

        Args:
            name: Name identifier used to locate or label an entity.

        """

        if not self._available_colormaps:
            return

        try:
            index = self._available_colormaps.index(name)
        except ValueError:
            return

        self._current_colormap_index = index

        if self.btn_colormap is not None:
            self.btn_colormap.setText(f"Colormap: {name}")

        self._apply_colormap()

    def _cycle_colormap_forward(self) -> None:
        """Cycle to the next colormap in the list.

        Triggered by the + key. Keyboard shortcut for quick colormap navigation.
        """
        if not self._available_colormaps:
            return

        self._current_colormap_index = (self._current_colormap_index + 1) % len(
            self._available_colormaps
        )
        current_cmap = self._available_colormaps[self._current_colormap_index]

        if self.btn_colormap is not None:
            self.btn_colormap.setText(f"Colormap: {current_cmap}")

        self._apply_colormap()
        logger.debug(f"Colormap cycled forward to: {current_cmap}")

    def _cycle_colormap_backward(self) -> None:
        """Cycle to the previous colormap in the list.

        Triggered by the - key. Keyboard shortcut for quick colormap navigation.
        """
        if not self._available_colormaps:
            return

        self._current_colormap_index = (self._current_colormap_index - 1) % len(
            self._available_colormaps
        )
        current_cmap = self._available_colormaps[self._current_colormap_index]

        if self.btn_colormap is not None:
            self.btn_colormap.setText(f"Colormap: {current_cmap}")

        self._apply_colormap()
        logger.debug(f"Colormap cycled backward to: {current_cmap}")

    def _refresh_transform_data(self):
        """Refresh cached FFT/inverse-FFT data from the current source region."""
        logger.debug(
            "Refreshing transform data: mode=%s source_shape=%s last_region_id=%s",
            self.view_mode,
            getattr(self._source_region, "shape", None),
            self._last_region_id,
        )
        if self.view_mode == "fft":
            self._compute_fft()
        elif self.view_mode == "inverse_fft":
            self._compute_inverse_fft_from_region()

    def _compute_fft(self):
        """Compute and cache FFT magnitude/complex arrays for the source ROI."""
        if self._source_region is None or self.ax_x is None or self.ax_y is None:
            logger.debug("Skipping FFT compute: missing source region or axes")
            return

        current_region_id = id(self._source_region)
        if (
            self._last_region_id == current_region_id
            and self._magnitude_spectrum is not None
        ):
            logger.debug(
                "Skipping FFT compute: region id unchanged (%s)", current_region_id
            )
            return

        self._last_region_id = current_region_id

        self._magnitude_spectrum, self._nyq_x, self._nyq_y = utils.compute_fft(
            self._source_region,
            self.ax_x.scale,
            self.ax_y.scale,
        )
        logger.debug(
            "Computed FFT: shape=%s nyq_x=%s nyq_y=%s min=%s max=%s",
            getattr(self._magnitude_spectrum, "shape", None),
            self._nyq_x,
            self._nyq_y,
            float(np.nanmin(self._magnitude_spectrum))
            if self._magnitude_spectrum is not None
            else None,
            float(np.nanmax(self._magnitude_spectrum))
            if self._magnitude_spectrum is not None
            else None,
        )

        window = (
            np.hanning(self._source_region.shape[0])[:, None]
            * np.hanning(self._source_region.shape[1])[None, :]
        )
        windowed = self._source_region * window
        self._fft_complex = np.fft.fftshift(np.fft.fft2(windowed))
        self._inverse_fft_cache = None

    def _compute_inverse_fft_from_region(self):
        """Compute inverse FFT image from the current source-region data."""
        if self._source_region is None or self.ax_x is None or self.ax_y is None:
            logger.debug("Skipping inverse FFT compute: missing source region or axes")
            return

        current_region_id = id(self._source_region)
        if (
            self._last_region_id == current_region_id
            and self._inverse_fft_image is not None
        ):
            logger.debug(
                "Skipping inverse FFT compute: region id unchanged (%s)",
                current_region_id,
            )
            return

        self._last_region_id = current_region_id

        source_region = np.asarray(self._source_region, dtype=np.float32)
        if source_region.ndim != 2:
            logger.debug(
                "Skipping inverse FFT compute: source region is not 2D (ndim=%s)",
                source_region.ndim,
            )
            return

        self._inverse_fft_image = np.abs(
            np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(source_region)))
        )
        logger.debug(
            "Computed inverse FFT: input_shape=%s output_shape=%s output_min=%s output_max=%s",
            source_region.shape,
            getattr(self._inverse_fft_image, "shape", None),
            float(np.nanmin(self._inverse_fft_image))
            if self._inverse_fft_image is not None
            else None,
            float(np.nanmax(self._inverse_fft_image))
            if self._inverse_fft_image is not None
            else None,
        )

    # Vim-style command handling -------------------------------------

    def eventFilter(self, obj, event):  # type: ignore[override]
        """Capture ':' globally when this window is active.

        Args:
            obj: QObject currently being filtered by the event filter.
            event: Qt drag/drop or input event provided by the GUI framework.
        """
        if self._command_mode.handle_key_event(self.isActiveWindow(), event):
            return True

        return super().eventFilter(obj, event)

    def _enter_command_mode(self) -> None:
        """Move focus into command-line mode for vim-style commands."""
        self._command_mode.enter_mode()

    def _exit_command_mode(self) -> None:
        """Exit command mode and restore focus to the graphics widget."""
        self._command_mode.exit_mode()

    def _execute_command_from_line(self) -> None:
        """Parse and execute the current command-line text input."""
        self._command_mode.execute_from_line()

    def _open_adjust_dialog(self):
        """Open or focus the tone-curve adjustment dialog for current view data."""
        if self.view_mode == "fft" and self._magnitude_spectrum is None:
            return

        if self.view_mode == "inverse_fft" and self._inverse_fft_image is None:
            return

        if self.view_mode == "image" and self.data is None:
            return

        if self._tone_dialog is not None and self._tone_dialog.isVisible():
            self._tone_dialog.raise_()
            self._tone_dialog.activateWindow()
            return

        initial_min = self.display_min
        initial_max = self.display_max
        initial_gamma = self.display_gamma

        def on_params_changed(min_val: float, max_val: float, gamma: float):
            self.display_min = float(min_val)
            self.display_max = float(max_val)
            self.display_gamma = float(gamma)
            self._update_image_display()

        if self.view_mode == "fft":
            source_data = self._magnitude_spectrum
        elif self.view_mode == "inverse_fft":
            source_data = self._inverse_fft_image
        else:
            source_data = self.data

        dialog = ToneCurveDialog(
            source_data,
            initial_min=initial_min,
            initial_max=initial_max,
            initial_gamma=initial_gamma,
            parent=self,
            on_params_changed=on_params_changed,
        )
        self._tone_dialog = dialog

        orig_min, orig_max, orig_gamma = (
            self.display_min,
            self.display_max,
            self.display_gamma,
        )

        def handle_rejected():
            self.display_min, self.display_max, self.display_gamma = (
                orig_min,
                orig_max,
                orig_gamma,
            )
            self._update_image_display()

        def handle_finished(_result):
            self._tone_dialog = None

        dialog.rejected.connect(handle_rejected)
        dialog.finished.connect(handle_finished)

        dialog.setModal(False)
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()

    def _setup_menu_bar(self) -> None:
        """Create and wire the menu bar for image-viewer actions."""
        callbacks_map = {
            "Open": self._open_file_dialog,
            "Save View": self._save_view_and_ffts,
            "Build Figure": lambda: self._show_not_implemented("Build Figure"),
            "Calibrate": self._open_calibration_dialog,
            "Parameters": self._open_parameters_dialog,
            "FFT": lambda _checked=False: self.fft_manager.add_new_fft(),
            "Inverse FFT": lambda _checked=False: (
                self.fft_manager.add_new_inverse_fft()
            ),
            "Adjust": self._open_adjust_dialog,
            "Distance": self._menu_start_distance_measurement,
            "History": self.measurements.show_measurement_history,
            "Intensity": lambda: self._show_not_implemented("Intensity"),
            "Profile": self._menu_start_profile_measurement,
            "Select Peaks": self._menu_start_peak_selection,
            "Export Peaks CSV": self._menu_export_peaks_csv,
            "Metadata": self._show_metadata_window,
            "Render Diagnostics": self._show_render_diagnostics,
            "Cycle Colormap Forward": self._cycle_colormap_forward,
            "Cycle Colormap Backward": self._cycle_colormap_backward,
            "Toggle Spectra Panel": self._edx_toggle_panel,
            "Select Integration Region": self._edx_start_region_selection,
            "Export EDS Results": self._edx_export_results,
            "Show Spectra Tab": self._edx_show_spectra_tab,
            "Show Maps Tab": self._edx_show_maps_tab,
            "Show Integration Tab": self._edx_show_integration_tab,
            "Toggle Hover Spectra": self._edx_toggle_hover_spectra,
            "Clear Integration Regions": self._edx_clear_regions,
            "Quant Method: CL": self._edx_set_quant_method_cl,
            "Quant Method: Custom": self._edx_set_quant_method_custom,
            "Quant Method: Zeta": self._edx_set_quant_method_zeta,
            "Quant Method: Cross-Section": self._edx_set_quant_method_cross_section,
            "Toggle Absorption Correction": self._edx_toggle_absorption_correction,
            "Keyboard Shortcuts": self.help_actions.show_keyboard_shortcuts,
            "About": self.help_actions.show_about,
            "README": self.help_actions.show_readme,
        }

        config = build_menu_config_for_role(
            role="viewer",
            callbacks_map=callbacks_map,
        )

        # Determine if an image is available for this viewer
        if self.view_mode == "image":
            image_available = self.data is not None
        elif self.view_mode == "fft":
            image_available = self._magnitude_spectrum is not None
        elif self.view_mode == "inverse_fft":
            image_available = self._inverse_fft_image is not None
        else:
            image_available = False

        # Determine if EDX data is available
        edx_available = (
            hasattr(self, "edx_manager") 
            and self.edx_manager.get_capability_state().has_edx_data
        )
        edx_capabilities = (
            self.edx_manager.get_capability_state().as_dict()
            if hasattr(self, "edx_manager")
            else {}
        )

        # Build menus using the builder
        self.menu_builder = MenuBuilder(self, logger)
        self.menu_builder.build_from_config(
            config, 
            image_available=image_available,
            edx_available=edx_available,
            edx_capabilities=edx_capabilities,
        )

        # Add Colormap submenu with individual colormap options to the Display menu
        if "Display" in self.menu_builder.menus:
            display_menu = self.menu_builder.menus["Display"]
            colormap_submenu = display_menu.addMenu("Colormap")
            for cmap_name in self._available_colormaps:
                action = colormap_submenu.addAction(cmap_name.capitalize())
                action.triggered.connect(
                    lambda checked=False, name=cmap_name: self._set_colormap_by_name(
                        name
                    )
                )
                logger.debug(f"Added colormap menu item: {cmap_name}")

        logger.debug(
            "Image viewer menu bar setup complete with keyboard shortcuts and colormap menu"
        )

    def _show_not_implemented(self, feature_name: str) -> None:
        """Show a generic placeholder message for unfinished features.

        Args:
            feature_name: Feature label shown in the not-implemented dialog.
        """
        QtWidgets.QMessageBox.information(
            self,
            feature_name,
            f"{feature_name} is planned but not implemented yet.",
        )

    def _show_render_diagnostics(self) -> None:
        """Display current rendering backend and quality diagnostics."""
        gl_available = hardware_acceleration_available(force_refresh=True)
        effective = global_render_config_options(
            self._render_settings, hardware_available=gl_available
        )

        diagnostics: list[str] = [
            f"Window mode: {self.view_mode}",
            f"File: {self.file_path}",
            f"Requested hardware acceleration: {bool(self._render_settings.get('use_hardware_acceleration', True))}",
            f"Detected OpenGL available: {gl_available}",
            f"Effective pyqtgraph useOpenGL (settings): {effective.get('useOpenGL')}",
            f"Current pyqtgraph useOpenGL (global): {pg.getConfigOption('useOpenGL')}",
            f"Image resampling quality: {self._render_settings.get('image_resampling_quality')}",
            f"QT_QPA_PLATFORM: {os.environ.get('QT_QPA_PLATFORM')}",
            f"QT_OPENGL: {os.environ.get('QT_OPENGL')}",
            f"QT_XCB_GL_INTEGRATION: {os.environ.get('QT_XCB_GL_INTEGRATION')}",
            f"XDG_SESSION_TYPE: {os.environ.get('XDG_SESSION_TYPE')}",
            f"WAYLAND_DISPLAY: {os.environ.get('WAYLAND_DISPLAY')}",
            f"DISPLAY: {os.environ.get('DISPLAY')}",
            f"Graphics widget class: {type(self.glw).__name__ if self.glw is not None else 'None'}",
            f"Image item class: {type(self.img_orig).__name__ if self.img_orig is not None else 'None'}",
        ]

        if self._display_image_full_res is not None:
            diagnostics.append(
                f"Display image shape: {self._display_image_full_res.shape}"
            )
            diagnostics.append(
                f"Display image dtype: {self._display_image_full_res.dtype}"
            )

        if self._source_region is not None:
            diagnostics.append(f"Source region shape: {self._source_region.shape}")
            diagnostics.append(f"Source region dtype: {self._source_region.dtype}")

        if self._magnitude_spectrum is not None:
            diagnostics.append(f"FFT magnitude shape: {self._magnitude_spectrum.shape}")

        if self._inverse_fft_image is not None:
            diagnostics.append(f"iFFT image shape: {self._inverse_fft_image.shape}")

        details = "\n".join(diagnostics)
        logger.debug("Render diagnostics requested:\n%s", details)

        # Create a non-modal, resizable dialog
        dialog = QtWidgets.QDialog(self, QtCore.Qt.Window)
        dialog.setWindowTitle("Render Diagnostics")
        dialog.setWindowModality(QtCore.Qt.NonModal)
        dialog.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        dialog.resize(600, 400)

        layout = QtWidgets.QVBoxLayout()

        lab_title = QtWidgets.QLabel("Render diagnostics for current window")
        layout.addWidget(lab_title)

        text_edit = QtWidgets.QTextEdit()
        text_edit.setReadOnly(True)
        text_edit.setPlainText(details)
        text_edit.setFontFamily("Courier")
        layout.addWidget(text_edit)

        btn_close = QtWidgets.QPushButton("Close")
        btn_close.clicked.connect(dialog.close)
        layout.addWidget(btn_close)

        dialog.setLayout(layout)
        dialog.show()

    def _open_parameters_dialog(self) -> None:
        """Open render-parameter dialog and apply accepted settings."""
        current = load_render_settings()
        updated = open_parameters_dialog(self, current)
        if updated is None:
            return

        save_render_settings(updated)
        self._render_settings = updated
        gl_available = hardware_acceleration_available()
        pg.setConfigOptions(
            **global_render_config_options(updated, hardware_available=gl_available)
        )
        self._apply_render_preferences_to_view()
        self._update_image_display()

    def _open_file_dialog(self) -> None:
        """Open file picker and load selected image into a new viewer window."""
        start_dir = (
            str(Path(self.file_path).parent) if self.file_path else str(Path.cwd())
        )
        selected_file = open_file_dialog(self, start_dir)
        if selected_file:
            open_image_file(selected_file)

    def _menu_start_distance_measurement(self) -> None:
        """Handle menu action that starts distance measurement mode."""
        logger.debug("Menu action: start distance measurement")
        self.measurements.start_distance_measurement()

    def _menu_start_profile_measurement(self) -> None:
        """Handle menu action that starts profile measurement mode."""
        logger.debug("Menu action: start profile measurement")
        self.measurements.start_profile_measurement()

    # EDS menu callbacks
    def _edx_toggle_panel(self) -> None:
        """Toggle EDX panel visibility."""
        if not hasattr(self, "edx_manager") or not self.edx_manager.get_has_edx_data():
            return
        logger.debug("EDX menu action: toggle panel")
        if self.edx_manager.edx_panel:
            visible = self.edx_manager.edx_panel.isVisible()
            self.edx_manager.edx_panel.setVisible(not visible)

    def _edx_start_region_selection(self) -> None:
        """Start EDX integration region selection."""
        if not hasattr(self, "edx_manager") or not self.edx_manager.get_has_edx_data():
            return
        logger.debug("EDX menu action: start region selection")
        self.edx_manager._on_select_region_clicked()

    def edx_manager_start_region_selection(self) -> None:
        """Enter EDS rectangle region selection mode from the manager."""
        if self.view_mode != "image" or self.p1 is None:
            QtWidgets.QMessageBox.information(
                self,
                "EDS",
                "Region selection is available only on image views.",
            )
            return

        # Ensure measurement tools do not compete for mouse events.
        try:
            self.measurements.exit_measure_mode()
        except Exception:
            logger.debug("Could not exit measurement mode before EDS region draw", exc_info=True)

        self._prepare_for_measurement_input()
        if self.btn_measure is not None:
            self.btn_measure.blockSignals(True)
            self.btn_measure.setChecked(False)
            self.btn_measure.blockSignals(False)
            self.btn_measure.setStyleSheet("")

        self.edx_manager.begin_rectangle_region_selection()

    def _on_scene_mouse_moved_for_edx(self, scene_pos: object) -> None:
        """Forward scene mouse movement to EDS hover-spectrum inspector."""
        if self.view_mode != "image":
            return
        if not hasattr(self, "edx_manager") or not self.edx_manager.get_has_edx_data():
            return
        if self.p1 is None:
            return
        try:
            if not self.p1.sceneBoundingRect().contains(scene_pos):
                return
            view_pos = self.p1.vb.mapSceneToView(scene_pos)
            self.edx_manager.update_hover_spectrum(view_pos.x(), view_pos.y())
        except Exception:
            # Hover updates are best-effort and should never break interaction.
            return

    def _edx_export_results(self) -> None:
        """Export EDX integration results."""
        if not hasattr(self, "edx_manager") or not self.edx_manager.get_has_edx_data():
            return
        logger.debug("EDX menu action: export results")
        self.edx_manager._on_export_results_clicked()

    def _edx_show_spectra_tab(self) -> None:
        """Switch EDS panel to the Spectra tab."""
        if not hasattr(self, "edx_manager"):
            return
        self.edx_manager.show_tab("Spectra")

    def _edx_show_maps_tab(self) -> None:
        """Switch EDS panel to the Maps tab."""
        if not hasattr(self, "edx_manager"):
            return
        self.edx_manager.show_tab("Maps")

    def _edx_show_integration_tab(self) -> None:
        """Switch EDS panel to the Integration tab."""
        if not hasattr(self, "edx_manager"):
            return
        self.edx_manager.show_tab("Integration")

    def _edx_toggle_hover_spectra(self) -> None:
        """Enable or disable hover-driven spectra updates."""
        if not hasattr(self, "edx_manager"):
            return
        self.edx_manager.toggle_hover_updates()

    def _edx_clear_regions(self) -> None:
        """Clear all EDS integration regions."""
        if not hasattr(self, "edx_manager"):
            return
        self.edx_manager._on_clear_results_clicked()

    def _edx_set_quant_method_cl(self) -> None:
        """Switch quantification method to CL."""
        if not hasattr(self, "edx_manager"):
            return
        self.edx_manager.set_quant_method("CL")

    def _edx_set_quant_method_custom(self) -> None:
        """Switch quantification method to Custom."""
        if not hasattr(self, "edx_manager"):
            return
        self.edx_manager.set_quant_method("Custom")

    def _edx_set_quant_method_zeta(self) -> None:
        """Switch quantification method to Zeta."""
        if not hasattr(self, "edx_manager"):
            return
        self.edx_manager.set_quant_method("Zeta")

    def _edx_set_quant_method_cross_section(self) -> None:
        """Switch quantification method to Cross-Section."""
        if not hasattr(self, "edx_manager"):
            return
        self.edx_manager.set_quant_method("Cross-Section")

    def _edx_toggle_absorption_correction(self) -> None:
        """Toggle absorption correction option in the integration panel."""
        if not hasattr(self, "edx_manager"):
            return
        self.edx_manager.toggle_absorption_correction()

    def refresh_edx_menu_state(self) -> None:
        """Refresh enablement of capability-gated EDS menu actions."""
        if not hasattr(self, "edx_manager") or not hasattr(self, "menu_builder"):
            return
        capabilities = self.edx_manager.get_capability_state().as_dict()
        title_to_state = {
            "Show Spectra Tab": capabilities.get("has_edx_data", False),
            "Show Maps Tab": capabilities.get("has_elemental_maps", False),
            "Show Integration Tab": capabilities.get("has_edx_data", False),
            "Toggle Hover Spectra": capabilities.get("has_edx_data", False),
            "Clear Integration Regions": capabilities.get("has_integration_regions", False),
            "Quant Method: CL": capabilities.get("has_integration_regions", False),
            "Quant Method: Custom": capabilities.get("has_integration_regions", False),
            "Quant Method: Zeta": capabilities.get("has_integration_regions", False),
            "Quant Method: Cross-Section": capabilities.get("has_integration_regions", False),
            "Toggle Absorption Correction": capabilities.get("has_timing_metadata", False),
        }
        for title, enabled in title_to_state.items():
            self.menu_builder.set_action_enabled("EDS", title, bool(enabled))

    def _menu_start_peak_selection(self) -> None:
        """Handle menu action that starts peak selection mode."""
        logger.debug("Menu action: start peak selection")
        self.measurements.start_peak_selection()

    def _menu_export_peaks_csv(self) -> None:
        """Handle menu action that exports selected peaks to CSV."""
        logger.debug("Menu action: export peaks csv")
        self.measurements.export_peaks_to_csv()

    def _iter_transform_windows_recursive(self):
        """Yield all descendant FFT/iFFT windows for this image file."""
        visited: set[int] = set()
        stack: list["ImageViewerWindow"] = []
        stack.extend(self.fft_windows or [])
        stack.extend(self.inverse_fft_windows or [])

        while stack:
            win = stack.pop()
            if win is None:
                continue

            win_id = id(win)
            if win_id in visited:
                continue
            visited.add(win_id)

            try:
                mode = getattr(win, "view_mode", None)
                same_file = getattr(win, "file_path", None) == self.file_path
                if same_file and mode in {"fft", "inverse_fft"}:
                    yield win

                stack.extend(getattr(win, "fft_windows", []) or [])
                stack.extend(getattr(win, "inverse_fft_windows", []) or [])
            except RuntimeError:
                # Qt object may already be deleted
                continue

    def _child_transform_segment(
        self, parent: "ImageViewerWindow", child: "ImageViewerWindow"
    ) -> str:
        """Return child segment label relative to parent, e.g. FFT0 or iFFT0.

        Args:
            parent: Optional parent widget that owns the dialog window.
            child: Child transform window being evaluated in the export tree.

        Returns:
            Human-readable segment describing how a child transform derives from its parent.
        """
        try:
            for i, w in enumerate(parent.fft_windows or []):
                if w is child:
                    return f"FFT{i}"
            for i, w in enumerate(parent.inverse_fft_windows or []):
                if w is child:
                    return f"iFFT{i}"
        except RuntimeError:
            pass

        # Fallback if linkage lists are stale
        mode = getattr(child, "view_mode", "")
        return "iFFT0" if mode == "inverse_fft" else "FFT0"

    def _transform_chain_label(self, window: "ImageViewerWindow") -> str:
        """Build chain label from this root window to target transform window.

        Args:
            window: Transform window for which a chain label is being generated.

        Returns:
            Full transform lineage label used in exported overlays and diagnostics.
        """
        parts: list[str] = []
        cur = window
        visited: set[int] = set()

        while cur is not None and cur is not self:
            cur_id = id(cur)
            if cur_id in visited:
                break
            visited.add(cur_id)

            parent = getattr(cur, "parent_image_window", None)
            if parent is None:
                break

            parts.append(self._child_transform_segment(parent, cur))
            cur = parent

        parts.reverse()
        return "-".join(parts) if parts else "oops"

    @staticmethod
    def _capture_plot_pixmap(
        plot_item: Any,
        fallback_widget: QtWidgets.QWidget,
        trace_event: Optional[Callable[..., None]] = None,
        trace_target: str = "unknown",
    ) -> QtGui.QPixmap:
        """Capture a plot item via ImageExporter, falling back to widget grab."""
        try:
            exporter = pg_exporters.ImageExporter(plot_item)
            exported = exporter.export(toBytes=True)
            if isinstance(exported, QtGui.QImage) and not exported.isNull():
                pixmap = QtGui.QPixmap.fromImage(exported)
                if not pixmap.isNull():
                    if callable(trace_event):
                        trace_event(
                            "capture_path",
                            target=trace_target,
                            mode="image_exporter",
                            width=pixmap.width(),
                            height=pixmap.height(),
                            depth=pixmap.depth(),
                        )
                    return pixmap
        except Exception:
            pass

        pixmap = fallback_widget.grab()
        if callable(trace_event):
            trace_event(
                "capture_path",
                target=trace_target,
                mode="widget_grab_fallback",
                width=pixmap.width(),
                height=pixmap.height(),
                depth=pixmap.depth(),
            )
        return pixmap

    def _save_view_and_ffts(self) -> None:
        """Save the current view (with annotations) and any FFT windows and their children, recursively.

        The save location, base filename, and output format are selected through
        a single system file picker. Main, FFT, and profile exports all use
        the same selected image format.
        """

        if self.data is None or self.glw is None:
            QtWidgets.QMessageBox.information(
                self,
                "Save Images",
                "No image is currently loaded to save.",
            )
            return
        logger.debug("Saving view/FFTs for file: %s", self.file_path)

        # Resolve default location/name for the save picker
        try:
            default_dir = str(Path(self.file_path).parent)
        except Exception:
            default_dir = str(Path.cwd())

        suggested_base = Path(self.file_path).stem or "image"
        initial_path = str(Path(default_dir) / f"{suggested_base}_view.png")

        file_filters = "PNG (*.png);;TIFF (*.tif *.tiff);;JPEG (*.jpg *.jpeg)"
        selected_path, selected_filter = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save Figure Set",
            initial_path,
            file_filters,
        )
        if not selected_path:
            return

        main_path = Path(selected_path)

        filter_upper = (selected_filter or "").upper()
        if "TIFF" in filter_upper:
            fmt_name = "TIFF"
            default_ext = ".tif"
        elif "JPEG" in filter_upper:
            fmt_name = "JPEG"
            default_ext = ".jpg"
        else:
            fmt_name = "PNG"
            default_ext = ".png"

        suffix = main_path.suffix.lower()
        if suffix in {".png"}:
            fmt_name, ext = "PNG", ".png"
        elif suffix in {".tif", ".tiff"}:
            fmt_name, ext = "TIFF", suffix
        elif suffix in {".jpg", ".jpeg"}:
            fmt_name, ext = "JPEG", suffix
        else:
            ext = default_ext
            main_path = main_path.with_suffix(ext)

        self._trace_event(
            "save_config",
            selected_path=selected_path,
            selected_filter=selected_filter,
            resolved_main_path=str(main_path),
            fmt_name=fmt_name,
            ext=ext,
            display_stats=self._trace_array_stats(self._display_image_full_res, "display_full_res"),
            image_item_stats=self._trace_array_stats(
                getattr(self.img_orig, "image", None) if self.img_orig is not None else None,
                "img_item",
            ),
        )

        directory_path = main_path.parent
        typed_name = main_path.stem
        if typed_name.endswith("_view") and len(typed_name) > len("_view"):
            base_name = typed_name[: -len("_view")]
        else:
            base_name = typed_name

        # Compose extra label for export using metadata (file, instrument,
        # acquisition mode, and AcquireDate from original metadata).
        overlay_label = self._build_export_overlay_label()

        # Temporarily overlay metadata label above the scale bar during export
        extra_label_applied = False
        if self.scale_bar is not None and hasattr(self.scale_bar, "set_extra_label"):
            try:
                self.scale_bar.set_extra_label(overlay_label)
                extra_label_applied = True
            except Exception:
                extra_label_applied = False

        # Capture main view (graphics widget) including annotations and label
        try:
            if self.p1 is not None:
                pixmap = self._capture_plot_pixmap(
                    self.p1,
                    self.glw,
                    trace_event=self._trace_event,
                    trace_target="main",
                )
            else:
                pixmap = self.glw.grab()
                self._trace_event(
                    "capture_path",
                    target="main",
                    mode="widget_grab_direct",
                    width=pixmap.width(),
                    height=pixmap.height(),
                    depth=pixmap.depth(),
                )
        except Exception as e:
            # Clean up overlay on failure
            if extra_label_applied:
                try:
                    self.scale_bar.set_extra_label(None)  # type: ignore[union-attr]
                except Exception:
                    pass
            QtWidgets.QMessageBox.critical(
                self,
                "Save Images",
                f"Failed to capture main view: {e}",
            )
            return

        qimg = pixmap.toImage()
        qimg_bytes = (
            int(qimg.sizeInBytes()) if hasattr(qimg, "sizeInBytes") else int(qimg.byteCount())
        )
        self._trace_event(
            "save_main_grabbed",
            pixmap_w=pixmap.width(),
            pixmap_h=pixmap.height(),
            pixmap_depth=pixmap.depth(),
            qimg_format=int(qimg.format()),
            qimg_has_alpha=qimg.hasAlphaChannel(),
            qimg_bytes=qimg_bytes,
        )

        save_ok = pixmap.save(str(main_path), fmt_name)
        self._trace_event(
            "save_main_result",
            ok=save_ok,
            path=str(main_path),
            fmt=fmt_name,
        )
        if not save_ok:
            QtWidgets.QMessageBox.warning(
                self,
                "Save Images",
                f"Failed to save main view to {main_path}.",
            )
        else:
            logger.debug("Saved main view: %s (%s)", main_path, fmt_name)

        # Remove temporary overlay label after saving
        if extra_label_applied:
            try:
                self.scale_bar.set_extra_label(None)  # type: ignore[union-attr]
            except Exception:
                pass

        # Capture all open FFT windows using the selected main-view format
        fft_saved = 0
        transform_windows = list(self._iter_transform_windows_recursive())
        for idx, fft_window in enumerate(transform_windows, start=1):
            # Build label: metadata summary and ROI number (from fft_name)
            parent_window = getattr(fft_window, "parent_image_window", None)
            base_label = None
            if parent_window is self:
                base_label = overlay_label
            elif parent_window is not None and hasattr(
                parent_window, "_build_export_overlay_label"
            ):
                try:
                    base_label = parent_window._build_export_overlay_label()
                except Exception:
                    base_label = None

            if not base_label:
                # Fallback to file name if we cannot get metadata
                try:
                    parent_file = getattr(parent_window, "file_path", None)
                    base_label = Path(parent_file).name if parent_file else "image"
                except Exception:
                    base_label = "image"

            roi_label = getattr(fft_window, "fft_name", f"FFT {idx}")
            extra_label = f"{base_label} | {roi_label}"

            extra_label_applied_fft = False
            scale_bar = getattr(fft_window, "scale_bar", None)
            if scale_bar is not None and hasattr(scale_bar, "set_extra_label"):
                try:
                    scale_bar.set_extra_label(extra_label)
                    extra_label_applied_fft = True
                except Exception:
                    extra_label_applied_fft = False

            # Grab only the FFT view (graphics widget) so that
            # window chrome and buttons are not included, mirroring
            # how the main image view is exported.
            try:
                fft_view_widget = getattr(fft_window, "glw", fft_window)
                fft_plot_item = getattr(fft_window, "p1", None)
                if fft_plot_item is not None:
                    fft_pixmap = self._capture_plot_pixmap(
                        fft_plot_item,
                        fft_view_widget,
                        trace_event=self._trace_event,
                        trace_target=f"fft:{getattr(fft_window, 'fft_name', 'unknown')}",
                    )
                else:
                    fft_pixmap = fft_view_widget.grab()
                    self._trace_event(
                        "capture_path",
                        target=f"fft:{getattr(fft_window, 'fft_name', 'unknown')}",
                        mode="widget_grab_direct",
                        width=fft_pixmap.width(),
                        height=fft_pixmap.height(),
                        depth=fft_pixmap.depth(),
                    )
            except Exception:
                if extra_label_applied_fft:
                    try:
                        scale_bar.set_extra_label(None)  # type: ignore[union-attr]
                    except Exception:
                        pass
                continue

            fft_qimg = fft_pixmap.toImage()
            fft_bytes = (
                int(fft_qimg.sizeInBytes())
                if hasattr(fft_qimg, "sizeInBytes")
                else int(fft_qimg.byteCount())
            )

            chain = self._transform_chain_label(fft_window)
            fft_path = directory_path / f"{base_name}_{chain}{ext}"
            fft_ok = fft_pixmap.save(str(fft_path), fmt_name)
            self._trace_event(
                "save_fft_result",
                chain=chain,
                ok=fft_ok,
                path=str(fft_path),
                fmt=fmt_name,
                pixmap_w=fft_pixmap.width(),
                pixmap_h=fft_pixmap.height(),
                qimg_format=int(fft_qimg.format()),
                qimg_has_alpha=fft_qimg.hasAlphaChannel(),
                qimg_bytes=fft_bytes,
            )
            if fft_ok:
                fft_saved += 1
                logger.debug("Saved FFT view: %s", fft_path)

            # Remove temporary overlay label after saving
            if extra_label_applied_fft:
                try:
                    scale_bar.set_extra_label(None)  # type: ignore[union-attr]
                except Exception:
                    pass

        # Capture open profile views using the selected main-view format
        profile_saved = 0
        profile_records = getattr(self, "profile_measurement_items", {}) or {}
        for profile_id, record in sorted(profile_records.items(), key=lambda item: item[0]):
            if not isinstance(record, dict):
                continue

            profile_window = record.get("window")
            if profile_window is None:
                continue

            try:
                plot_widget = getattr(profile_window, "plot_widget", None)
                capture_widget = plot_widget if plot_widget is not None else profile_window
                profile_plot_item = None
                if plot_widget is not None and hasattr(plot_widget, "getPlotItem"):
                    profile_plot_item = plot_widget.getPlotItem()

                if profile_plot_item is not None:
                    profile_pixmap = self._capture_plot_pixmap(
                        profile_plot_item,
                        capture_widget,
                        trace_event=self._trace_event,
                        trace_target=f"profile:{profile_id}",
                    )
                else:
                    profile_pixmap = capture_widget.grab()
                    self._trace_event(
                        "capture_path",
                        target=f"profile:{profile_id}",
                        mode="widget_grab_direct",
                        width=profile_pixmap.width(),
                        height=profile_pixmap.height(),
                        depth=profile_pixmap.depth(),
                    )
            except Exception:
                continue

            profile_qimg = profile_pixmap.toImage()
            profile_bytes = (
                int(profile_qimg.sizeInBytes())
                if hasattr(profile_qimg, "sizeInBytes")
                else int(profile_qimg.byteCount())
            )

            profile_path = directory_path / f"{base_name}_profile{profile_id}{ext}"
            profile_ok = profile_pixmap.save(str(profile_path), fmt_name)
            self._trace_event(
                "save_profile_result",
                profile_id=profile_id,
                ok=profile_ok,
                path=str(profile_path),
                fmt=fmt_name,
                pixmap_w=profile_pixmap.width(),
                pixmap_h=profile_pixmap.height(),
                qimg_format=int(profile_qimg.format()),
                qimg_has_alpha=profile_qimg.hasAlphaChannel(),
                qimg_bytes=profile_bytes,
            )
            if profile_ok:
                profile_saved += 1
                logger.debug("Saved profile view: %s", profile_path)

        message_lines = [f"Saved main view to:\n{main_path}"]
        if fft_saved:
            message_lines.append(f"Saved {fft_saved} FFT view {fmt_name} file(s).")
        else:
            message_lines.append(
                "No FFT windows were open."
            )

        if profile_saved:
            message_lines.append(
                f"Saved {profile_saved} profile view {fmt_name} file(s)."
            )
        else:
            message_lines.append("No profile windows were open; no profile views were saved.")

        # Optionally persist EDS result artifacts when available.
        if (
            hasattr(self, "edx_manager")
            and self.edx_manager.get_capability_state().has_integration_regions
        ):
            save_eds_reply = QtWidgets.QMessageBox.question(
                self,
                "Save EDS Results",
                "Save EDS quantification data artifacts as well?",
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                QtWidgets.QMessageBox.Yes,
            )
            if save_eds_reply == QtWidgets.QMessageBox.Yes:
                if self.edx_manager.prompt_save_all_results():
                    message_lines.append("Saved EDS quantification data artifacts.")
                else:
                    message_lines.append("EDS quantification artifacts were not saved.")

        QtWidgets.QMessageBox.information(
            self,
            "Save Images",
            "\n".join(message_lines),
        )

    # ------------------------------------------------------------------
    # Metadata helpers for export overlays
    # ------------------------------------------------------------------

    def _build_export_overlay_label(self) -> str:
        """Return a one-line label for export overlays.

        Returns:
            Overlay label string containing file and transform context for export.
        """

        # File name
        try:
            file_name = Path(self.file_path).name if self.file_path else "image"
        except Exception:
            file_name = "image"

        microscope_tag = ""
        mode_tag = ""
        acq_date = ""

        raw_meta = self._get_original_metadata_dict()
        if isinstance(raw_meta, dict):
            obj_info = (
                raw_meta.get("ObjectInfo")
                or raw_meta.get("objectInfo")
                or raw_meta.get("object_info")
            )

            if isinstance(obj_info, dict):
                # AcquireDate: full string
                acq_val = (
                    obj_info.get("AcquireDate")
                    or obj_info.get("acquiredate")
                    or obj_info.get("Acquiredate")
                )
                if acq_val:
                    acq_date = str(acq_val)

                # ExperimentalDescription contains Microscope + Mode
                exp_desc = (
                    obj_info.get("ExperimentalDescription")
                    or obj_info.get("experimentaldescription")
                    or obj_info.get("experimental_description")
                )

                if isinstance(exp_desc, dict):
                    # Microscope: take second word
                    microscope_val = exp_desc.get("Microscope") or exp_desc.get(
                        "microscope"
                    )
                    if isinstance(microscope_val, str):
                        words = microscope_val.split()
                        if len(words) >= 2:
                            microscope_tag = words[1]

                    # Mode: take first word
                    mode_val = exp_desc.get("Mode") or exp_desc.get("mode")
                    if isinstance(mode_val, str):
                        parts = mode_val.strip().split()
                        if parts:
                            mode_tag = parts[0]

        parts = [file_name]
        if microscope_tag:
            parts.append(microscope_tag)
        if mode_tag:
            parts.append(mode_tag)
        if acq_date:
            parts.append(acq_date)

        return " | ".join(str(p) for p in parts if p)

    def _show_metadata_window(self):
        """Open or refresh the metadata window for the current signal."""
        if self.signal is None:
            QtWidgets.QMessageBox.information(
                self, "Metadata", "No metadata available for this image."
            )
            return

        raw_metadata = self._get_original_metadata_dict()
        cleaned_metadata = None
        if self.signal is not None and hasattr(self.signal, "metadata"):
            try:
                cleaned_metadata = self.signal.metadata.as_dictionary()
            except Exception:
                cleaned_metadata = None

        file_name = Path(self.file_path).name
        title = f"Metadata - {file_name}"

        if self.metadata_window is None:
            self.metadata_window = MetadataWindow(
                self,
                title=title,
                raw_metadata=raw_metadata,
                cleaned_metadata=cleaned_metadata,
            )
        else:
            self.metadata_window.setWindowTitle(title)
            self.metadata_window.update_metadata(
                raw_metadata=raw_metadata,
                cleaned_metadata=cleaned_metadata,
            )

        self.metadata_window.show()
        self.metadata_window.raise_()
        self.metadata_window.activateWindow()
