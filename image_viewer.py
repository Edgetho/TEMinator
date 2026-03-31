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
from scale_bars import DynamicScaleBar
from utils import (
    HelpDialogActions,
    open_file_dialog,
    open_parameters_dialog,
)
from viewer_commands import (
    ViewerCommandRouter,
)
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
        self.measurements = MeasurementController(self, logger)
        self.fft_manager = FFTWindowManager(self, logger, FFT_COLORS)
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
        return self._apply_axis_calibration_values(
            dx, dy, "m", ox=ox, oy=oy, source=source
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

    def _detect_reciprocal_space(self, signal) -> bool:
        """Determine if the signal should be treated as reciprocal space.

        Args:
            signal: HyperSpy signal containing image data, axes, and metadata.

        Returns:
            Computed result produced by this operation.
        """

        # 1) Prefer explicit HyperSpy metadata when available
        try:
            meta = getattr(signal, "metadata", None)
            sig_meta = getattr(meta, "Signal", None)
            sig_type = getattr(sig_meta, "signal_type", None)
            if isinstance(sig_type, str):
                st = sig_type.strip().lower()
                if st in {"diffraction", "electron_diffraction", "fft"}:
                    return True
                if st in {"image", "tem", "stem"}:
                    return False
        except Exception:
            pass

        # 2) Inspect axis unit strings for common reciprocal-space patterns
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

        # 3) Default: treat as real-space image. We no longer rely on
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

        self.line_tool = LineDrawingTool(
            self.p1,
            self.measurements.on_line_drawn,
            self._on_measurement_drawing_state_changed,
        )

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

    def _set_display_image(self, image: np.ndarray) -> None:
        """Push image data into the display item with current quality mode.

        Args:
            image: Input image array used by the computation.

        """
        if self.img_orig is None:
            return

        self._display_image_full_res = np.asarray(image, dtype=np.float32)
        quality = self._render_quality_mode()

        if quality == RESAMPLING_HIGH:
            self._build_mipmap_levels(self._display_image_full_res)
            self._on_view_range_changed(force=True)
            return

        self._mipmap_levels = []
        self._current_mipmap_level = -1
        self.img_orig.setImage(
            self._display_image_full_res, autoLevels=False, levels=(0.0, 1.0)
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
        if not force and level == self._current_mipmap_level:
            return

        self._current_mipmap_level = level
        self.img_orig.setImage(
            self._mipmap_levels[level], autoLevels=False, levels=(0.0, 1.0)
        )

    def _update_image_display(self):
        """Render source data into the image item using tone and colormap settings."""
        if self.data is None or self.img_orig is None:
            if self.view_mode not in {"fft", "inverse_fft"}:
                return

        if self.display_min is None or self.display_max is None:
            self._init_display_window()

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

        # "gray" means the default grayscale appearance (no custom LUT)
        if name == "gray":
            try:
                self.img_orig.setLookupTable(None)
            except Exception:
                pass
            return

        try:
            cmap = pg.colormap.get(name)
            self.img_orig.setLookupTable(cmap.getLookupTable())
        except Exception:
            # Fall back to default grayscale if something goes wrong
            try:
                self.img_orig.setLookupTable(None)
            except Exception:
                pass

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
        config = build_menu_config_for_role(
            role="viewer",
            callbacks_map={
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
                "Keyboard Shortcuts": self.help_actions.show_keyboard_shortcuts,
                "About": self.help_actions.show_about,
                "README": self.help_actions.show_readme,
            },
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

        # Build menus using the builder
        self.menu_builder = MenuBuilder(self, logger)
        self.menu_builder.build_from_config(config, image_available=image_available)

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

    def _save_view_and_ffts(self) -> None:
        """Save the current view (with annotations) and any FFT windows and their children, recursively.

        The main image view is saved in a user-selected format (PNG, TIFF,
        or JPEG). All FFT views are saved as PNG files.
        """

        if self.data is None or self.glw is None:
            QtWidgets.QMessageBox.information(
                self,
                "Save Images",
                "No image is currently loaded to save.",
            )
            return
        logger.debug("Saving view/FFTs for file: %s", self.file_path)

        # Choose output directory
        try:
            default_dir = str(Path(self.file_path).parent)
        except Exception:
            default_dir = str(Path.cwd())

        directory = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            "Select Output Directory",
            default_dir,
        )
        if not directory:
            return

        directory_path = Path(directory)

        # Choose base name
        suggested_base = Path(self.file_path).stem or "image"
        base_name, ok = QtWidgets.QInputDialog.getText(
            self,
            "Base File Name",
            "Base name for saved images:",
            text=suggested_base,
        )
        if not ok:
            return
        base_name = base_name.strip()
        if not base_name:
            QtWidgets.QMessageBox.information(
                self,
                "Save Images",
                "Base name cannot be empty.",
            )
            return

        # Choose format for main view
        format_labels = ["PNG (.png)", "TIFF (.tif)", "JPEG (.jpg)"]
        format_map = {
            0: ("PNG", ".png"),
            1: ("TIFF", ".tif"),
            2: ("JPEG", ".jpg"),
        }

        fmt_label, ok = QtWidgets.QInputDialog.getItem(
            self,
            "Image Format",
            "Format for main view:",
            format_labels,
            0,
            False,
        )
        if not ok:
            return

        try:
            fmt_index = format_labels.index(fmt_label)
        except ValueError:
            QtWidgets.QMessageBox.warning(
                self,
                "Save Images",
                "Unknown image format selected.",
            )
            return

        fmt_name, ext = format_map.get(fmt_index, ("PNG", ".png"))

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
            pixmap = self.glw.grab()
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

        main_path = directory_path / f"{base_name}_view{ext}"
        if not pixmap.save(str(main_path), fmt_name):
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

        # Capture all open FFT windows as PNGs, with metadata + ROI label
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
                fft_pixmap = fft_view_widget.grab()
            except Exception:
                if extra_label_applied_fft:
                    try:
                        scale_bar.set_extra_label(None)  # type: ignore[union-attr]
                    except Exception:
                        pass
                continue

            chain = self._transform_chain_label(fft_window)
            fft_path = directory_path / f"{base_name}_{chain}.png"
            if fft_pixmap.save(str(fft_path), "PNG"):
                fft_saved += 1
                logger.debug("Saved FFT view: %s", fft_path)

            # Remove temporary overlay label after saving
            if extra_label_applied_fft:
                try:
                    scale_bar.set_extra_label(None)  # type: ignore[union-attr]
                except Exception:
                    pass

        message_lines = [f"Saved main view to:\n{main_path}"]
        if fft_saved:
            message_lines.append(f"Saved {fft_saved} FFT view PNG file(s).")
        else:
            message_lines.append(
                "No FFT windows were open; only the main view was saved."
            )

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
