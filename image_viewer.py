# SPDX-License-Identifier: GPL-2.0-only
# Copyright (C) 2026 Cooper Stuntz
# See LICENSE for full license terms.

"""Image viewer window and image-opening helper."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Callable, Any

import numpy as np
import hyperspy.api as hs
import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets, QtCore, QtGui

import utils
import unit_utils
from command_utils import enter_command_mode, exit_command_mode, parse_command_input
from dialogs import (
    MeasurementHistoryWindow,
    LineProfileWindow,
    MetadataWindow,
    ToneCurveDialog,
)
from file_navigation import IMAGE_FILE_FILTER
from image_loader import open_image_file
from measurement_tools import (
    LineDrawingTool,
    FFTBoxROI,
    MeasurementLabel,
    LABEL_BRUSH_COLOR,
    DRAWN_LINE_PEN,
)
from scale_bars import DynamicScaleBar
from viewer_fft import FFTWindowManager
from viewer_measurements import MeasurementController
from viewer_commands import (
    ViewerCommandRouter,
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
        fft_region: Optional[np.ndarray] = None,
        fft_name: Optional[str] = None,
        parent_image_window: Optional["ImageViewerWindow"] = None,
    ):
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
        self._nyq_x = None
        self._nyq_y = None
        self._last_region_id = None
        self._fft_region = fft_region
        self.chk_inverse: Optional[QtWidgets.QCheckBox] = None
        self._inverse_action: Optional[QtWidgets.QAction] = None
        self.freq_axis_base_unit: str = "m"
        self._calibration_status: str = "metadata"
        self._manual_calibrated: bool = False
        self._image_inverse_cache: Optional[np.ndarray] = None
        self._line_draw_mode: str = "measurement"
        self._calibration_dialog_state: Optional[Dict[str, Any]] = None
        self._on_calibration_pixels_selected: Optional[Callable[[float], None]] = None
        self._base_window_title: str = ""
        self.measurements = MeasurementController(self, logger)
        self.fft_manager = FFTWindowManager(self, logger, FFT_COLORS)
        self.commands = ViewerCommandRouter(self, logger)

        if self.view_mode == "fft":
            self._setup_fft_view()
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
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QtGui.QDropEvent):  # type: ignore[override]
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
        self.signal = signal
        self.data = signal.data
        self.ax_x = signal.axes_manager[0]
        self.ax_y = signal.axes_manager[1]

        calibrated = self._apply_calibration_from_original_metadata()
        if not calibrated:
            self._calibration_status = "uncalibrated"
            logger.debug("Startup metadata calibration failed; entering uncalibrated mode")
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


    def _setup_fft_view(self):
        if self._fft_region is None:
            QtWidgets.QMessageBox.critical(self, "FFT", "Could not create FFT view: missing ROI data.")
            self.close()
            return

        parent = self.parent_image_window
        if parent is None or parent.ax_x is None or parent.ax_y is None:
            QtWidgets.QMessageBox.critical(self, "FFT", "Could not create FFT view: missing parent calibration.")
            self.close()
            return

        self.is_reciprocal_space = True
        self.ax_x = parent.ax_x
        self.ax_y = parent.ax_y
        self.freq_axis_base_unit = self.ax_x.units or "m"

        parent_title = parent.windowTitle().replace("Image Viewer - ", "")
        display_name = self.fft_name or "FFT"
        self._set_base_window_title(f"Image Viewer - {parent_title} - {display_name}")

        self._compute_fft()
        self._init_display_window()
        self.setup_ui()
        self.resize(*DEFAULT_FFT_WINDOW_SIZE)

    def _get_original_metadata_dict(self) -> Optional[dict]:
        return self._get_original_metadata_dict_from_signal(self.signal)

    def _get_original_metadata_dict_from_signal(self, signal) -> Optional[dict]:
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

    def _extract_ser_calibration(self, meta: dict) -> Optional[Tuple[float, float, Optional[float], Optional[float]]]:
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
            logger.debug("Calibration metadata deltas must be positive: dx=%r dy=%r", dx, dy)
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
        if self.ax_x is None or self.ax_y is None:
            logger.debug("Skipping axis calibration apply (%s): axes unavailable", source)
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
        meta = meta_override if meta_override is not None else self._get_original_metadata_dict()
        if not meta:
            logger.debug("No original metadata available for calibration (%s)", source)
            return False

        extracted = self._extract_ser_calibration(meta)
        if extracted is None:
            logger.debug("Could not extract calibration values from metadata (%s)", source)
            return False

        dx, dy, ox, oy = extracted
        return self._apply_axis_calibration_values(dx, dy, "m", ox=ox, oy=oy, source=source)

    def _reload_calibration_from_file_metadata(self) -> bool:
        if not self.file_path:
            logger.debug("Cannot reload metadata calibration: file_path is empty")
            return False

        logger.debug("Reloading calibration metadata from file: %s", self.file_path)
        try:
            signal = hs.load(self.file_path)
            if signal.axes_manager.navigation_dimension != 0:
                signal = signal.inav[0, 0]
        except Exception:
            logger.exception("Failed to load file for metadata calibration reload: %s", self.file_path)
            return False

        meta = self._get_original_metadata_dict_from_signal(signal)
        ok = self._apply_calibration_from_original_metadata(
            meta_override=meta,
            source="file metadata reload",
        )
        logger.debug("Metadata calibration reload result: success=%s", ok)
        return ok


    def _open_calibration_dialog(self, initial_state: Optional[Dict[str, Any]] = None) -> None:
        if self.view_mode != "image" or self.ax_x is None or self.ax_y is None:
            QtWidgets.QMessageBox.information(
                self,
                "Parameters",
                "Calibration parameters are only editable for image views.",
            )
            return

        logger.debug("Opening calibration dialog: initial_state_keys=%s", sorted((initial_state or {}).keys()))

        state = dict(initial_state or {})
        current_scale_x = float(self.ax_x.scale)
        current_scale_y = float(self.ax_y.scale)
        default_ppu_x = 1.0 / current_scale_x if current_scale_x > 0 else 1.0
        default_ppu_y = 1.0 / current_scale_y if current_scale_y > 0 else 1.0

        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("Parameters")
        layout = QtWidgets.QFormLayout(dialog)

        unit_default = unit_utils.normalize_axis_unit(self.ax_x.units, default="nm")

        edit_ppu_x = QtWidgets.QLineEdit(f"{float(state.get('ppu_x', default_ppu_x)):.12g}")
        edit_ppu_y = QtWidgets.QLineEdit(f"{float(state.get('ppu_y', default_ppu_y)):.12g}")
        edit_units = QtWidgets.QLineEdit(str(state.get("units", unit_default)))
        edit_ref_pixels = QtWidgets.QLineEdit(f"{float(state.get('reference_pixels', 0.0)):.12g}")
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
            try:
                ref_pixels = float(edit_ref_pixels.text().strip())
            except ValueError:
                if edit_ref_pixels.text().strip():
                    set_ref_error("Reference pixels must be a valid number.")
                else:
                    set_ref_error(None)
                return

            target_units = unit_utils.normalize_axis_unit(edit_units.text(), default="nm")
            distance_text = edit_ref_distance.text().strip()
            raw_distance = unit_utils.split_value_and_unit(distance_text)
            if raw_distance is None:
                if distance_text:
                    set_ref_error(
                        "Reference distance must be like '10', '10 nm', or '2 nm-1' (not '1/nm')."
                    )
                else:
                    set_ref_error(None)
                return

            _value_raw, explicit_unit = raw_distance
            if explicit_unit and unit_utils.unit_kind(explicit_unit) != unit_utils.unit_kind(target_units):
                target_units = unit_utils.normalize_axis_unit(explicit_unit, default="nm")
                edit_units.blockSignals(True)
                edit_units.setText(target_units)
                edit_units.blockSignals(False)

            parsed_units = unit_utils.parse_distance_to_target_units(distance_text, target_units)
            if parsed_units is None:
                if distance_text:
                    set_ref_error(
                        "Could not parse/convert reference distance. Use reciprocal units as '<unit>-1' (e.g., 'nm-1')."
                    )
                else:
                    set_ref_error(None)
                return

            ref_units, _parsed_unit = parsed_units

            if ref_pixels <= 0 or ref_units <= 0:
                set_ref_error("Reference pixel and distance values must be greater than zero.")
                return

            ppu = ref_pixels / ref_units
            edit_ppu_x.setText(f"{ppu:.12g}")
            if chk_lock_ratio.isChecked():
                edit_ppu_y.setText(f"{ppu:.12g}")
            set_ref_error(None)
            logger.debug(
                "Calibration reference parsed: pixels=%s distance=%s units=%s computed_ppu=%s",
                ref_pixels,
                ref_units,
                target_units,
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
            edit_units.setText(unit_utils.normalize_axis_unit(self.ax_x.units, default="nm"))
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

        try:
            new_ppu_x = float(edit_ppu_x.text().strip())
            new_ppu_y = float(edit_ppu_y.text().strip())
            new_units = unit_utils.normalize_axis_unit(edit_units.text(), default="nm")
        except ValueError:
            QtWidgets.QMessageBox.warning(
                self,
                "Parameters",
                "Pixels-per-unit values must be valid numbers.",
            )
            return

        if chk_lock_ratio.isChecked():
            new_ppu_y = new_ppu_x

        if new_ppu_x <= 0 or new_ppu_y <= 0:
            QtWidgets.QMessageBox.warning(
                self,
                "Parameters",
                "Pixels-per-unit values must be greater than zero.",
            )
            return

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

        if (
            metadata_reloaded_in_dialog
            and abs(proposed_scale_x - current_scale_x) <= 1e-15
            and abs(proposed_scale_y - current_scale_y) <= 1e-15
            and new_units == current_units
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
        self.is_reciprocal_space = bool(metadata_reciprocal or unit_utils.is_reciprocal_unit(new_units))
        self._refresh_view_after_calibration_change()
        logger.debug("Manual calibration applied successfully")

    def _start_calibration_distance_pick(self) -> None:
        if self.line_tool is None:
            logger.debug("Calibration distance pick requested but line tool is unavailable")
            QtWidgets.QMessageBox.warning(
                self,
                "Calibration",
                "Line drawing tool is not available.",
            )
            return

        logger.debug("Starting calibration distance pick")

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
            QtCore.QTimer.singleShot(0, lambda: self._open_calibration_dialog(next_state))

        self._on_calibration_pixels_selected = _on_pixels_selected
        self.line_tool.enable()
        self._on_measurement_drawing_state_changed(False)

    def _set_base_window_title(self, title: str) -> None:
        self._base_window_title = title
        self._on_measurement_drawing_state_changed(False)

    def _on_measurement_drawing_state_changed(self, is_drawing: bool) -> None:
        title = self._base_window_title or self.windowTitle()
        for prefix in ("Measurement mode - ", "<esc> to exit measurement mode - "):
            if title.startswith(prefix):
                title = title[len(prefix):]

        measurement_ready = bool(self.line_tool is not None and getattr(self.line_tool, "is_enabled", False))
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
        if self.scale_bar is None or not hasattr(self.scale_bar, "set_status_tag"):
            return

        if self._calibration_status == "uncalibrated":
            self.scale_bar.set_status_tag("UNCALIBRATED")
        elif self._calibration_status == "manual":
            self.scale_bar.set_status_tag("manually calibrated")
        else:
            self.scale_bar.set_status_tag(None)
        logger.debug("Scale bar calibration tag refreshed: status=%s", self._calibration_status)

    def _detect_reciprocal_space(self, signal) -> bool:
        """Determine if the signal should be treated as reciprocal space.

        The logic prefers explicit metadata and axis units so that images
        originating from the same file (e.g. multi-image EMI/DM stacks)
        receive consistent classification and therefore consistent
        scale-bar behaviour.
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
        x_offset = self.ax_x.offset if self.ax_x else 0
        y_offset = self.ax_y.offset if self.ax_y else 0
        w = self.ax_x.size * self.ax_x.scale if self.ax_x else 1
        h = self.ax_y.size * self.ax_y.scale if self.ax_y else 1
        return x_offset, y_offset, w, h

    def _init_display_window(self):
        source_data = self.data
        if self.view_mode == "fft":
            source_data = self._magnitude_spectrum

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
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        main_layout = QtWidgets.QVBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        self._setup_menu_bar()

        self.glw = pg.GraphicsLayoutWidget()
        self.glw.ci.setContentsMargins(0, 0, 0, 0)
        if hasattr(self.glw.ci, "layout"):
            self.glw.ci.layout.setContentsMargins(0, 0, 0, 0)
            self.glw.ci.layout.setSpacing(0)
        main_layout.addWidget(self.glw)

        # Hidden command line at the bottom, shown when ':' is pressed
        self.command_edit = QtWidgets.QLineEdit()
        self.command_edit.setPlaceholderText(
            "Vim command (:F, :d, :e <file>, :E, :a)"
        )
        self.command_edit.returnPressed.connect(self._execute_command_from_line)
        self.command_edit.hide()
        main_layout.addWidget(self.command_edit)

        # Shortcut for ':' so command mode is easy to enter while this
        # image window is active.
        colon_shortcut = QtGui.QShortcut(QtGui.QKeySequence(":"), self)
        colon_shortcut.activated.connect(self._enter_command_mode)

        self.btn_measure = QtWidgets.QPushButton("Measure Distance")
        self.btn_measure.setCheckable(True)
        self.btn_measure.clicked.connect(self._toggle_line_measurement)
        self.btn_measure.hide()

        self.chk_inverse = QtWidgets.QCheckBox("Show Inverse FFT")
        self.chk_inverse.hide()
        self.chk_inverse.stateChanged.connect(lambda _state: self._update_image_display())

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
        self.p1.invertY(True)
        if self.view_mode == "fft":
            cmap = pg.colormap.get("magma")
            self.img_orig.setLookupTable(cmap.getLookupTable())
            self._update_image_display()
            if self._nyq_x is not None and self._nyq_y is not None:
                self.img_orig.setRect(
                    QtCore.QRectF(-self._nyq_x, -self._nyq_y, 2 * self._nyq_x, 2 * self._nyq_y)
                )
            if hasattr(self.p1.vb, "setPadding"):
                self.p1.vb.setPadding(0.0)
            fft_unit, fft_reciprocal = unit_utils.scale_bar_unit_and_mode(
                self.freq_axis_base_unit,
                reciprocal_hint=True,
            )
            self.scale_bar = DynamicScaleBar(self.p1.vb, units=fft_unit)
            self.scale_bar.reciprocal = fft_reciprocal
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
            self._on_line_drawn,
            self._on_measurement_drawing_state_changed,
        )

        self.setup_keyboard_shortcuts()

    def _update_image_display(self):
        if self.data is None or self.img_orig is None:
            if self.view_mode != "fft":
                return

        if self.display_min is None or self.display_max is None:
            self._init_display_window()

        if self.view_mode == "fft":
            if self._magnitude_spectrum is None:
                return

            if self.chk_inverse is not None and self.chk_inverse.isChecked():
                if self._inverse_fft_cache is None:
                    self._inverse_fft_cache = utils.compute_inverse_fft(self._fft_complex)
                display_data = self._inverse_fft_cache
            else:
                display_data = self._magnitude_spectrum

            adjusted = utils.apply_intensity_transform(
                display_data,
                self.display_min,
                self.display_max,
                self.display_gamma,
            )
            if adjusted is None:
                return

            self.img_orig.setImage(adjusted, autoLevels=False, levels=(0.0, 1.0))
            if self._nyq_x is not None and self._nyq_y is not None:
                self.img_orig.setRect(
                    QtCore.QRectF(-self._nyq_x, -self._nyq_y, 2 * self._nyq_x, 2 * self._nyq_y)
                )
            return

        display_data = self.data
        if self.chk_inverse is not None and self.chk_inverse.isChecked() and self.data is not None:
            if self._image_inverse_cache is None:
                self._image_inverse_cache = np.abs(np.fft.ifft2(np.fft.ifftshift(self.data)))
            display_data = self._image_inverse_cache

        adjusted = utils.apply_intensity_transform(
            display_data,
            self.display_min,
            self.display_max,
            self.display_gamma,
        )
        if adjusted is None:
            return

        self.img_orig.setImage(adjusted, autoLevels=False, levels=(0.0, 1.0))

    def _apply_colormap(self) -> None:
        """Apply the currently selected colormap to the main image."""

        if self.img_orig is None:
            return

        if not self._available_colormaps:
            return

        name = self._available_colormaps[self._current_colormap_index % len(self._available_colormaps)]

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
        delete_shortcut = QtGui.QShortcut(QtGui.QKeySequence.Delete, self)
        delete_shortcut.activated.connect(self._delete_selected_roi)

        backspace_shortcut = QtGui.QShortcut(
            QtGui.QKeySequence(QtCore.Qt.Key_Backspace), self
        )
        backspace_shortcut.activated.connect(self._delete_selected_roi)

        escape_shortcut = QtGui.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key_Escape), self)
        escape_shortcut.activated.connect(self._exit_measure_mode)

    def _set_colormap_by_name(self, name: str) -> None:
        """Set the active colormap by name and update the button label."""

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

    def _compute_fft(self):
        if self._fft_region is None or self.ax_x is None or self.ax_y is None:
            return

        current_region_id = id(self._fft_region)
        if self._last_region_id == current_region_id and self._magnitude_spectrum is not None:
            return

        self._last_region_id = current_region_id

        self._magnitude_spectrum, self._nyq_x, self._nyq_y = utils.compute_fft(
            self._fft_region,
            self.ax_x.scale,
            self.ax_y.scale,
        )

        window = np.hanning(self._fft_region.shape[0])[:, None] * np.hanning(
            self._fft_region.shape[1]
        )[None, :]
        windowed = self._fft_region * window
        self._fft_complex = np.fft.fftshift(np.fft.fft2(windowed))
        self._inverse_fft_cache = None

    # Vim-style command handling -------------------------------------

    def eventFilter(self, obj, event):  # type: ignore[override]
        """Capture ':' globally when this window is active."""
        if (
            self.isActiveWindow()
            and event.type() == QtCore.QEvent.KeyPress
        ):
            key_event = event  # QKeyEvent
            # Enter command mode on ':' with no modifiers
            if getattr(key_event, "text", lambda: "")() == ":" and not key_event.modifiers():
                self._enter_command_mode()
                return True

            # Allow Esc to cancel command mode
            if (
                self.command_edit is not None
                and self.command_edit.isVisible()
                and getattr(key_event, "key", lambda: None)() == QtCore.Qt.Key_Escape
            ):
                self._exit_command_mode()
                return True

        return super().eventFilter(obj, event)

    def _enter_command_mode(self) -> None:
        enter_command_mode(self.command_edit)

    def _exit_command_mode(self) -> None:
        focus_target = self.glw if hasattr(self.glw, "setFocus") else None
        exit_command_mode(self.command_edit, focus_target=focus_target)

    def _execute_command_from_line(self) -> None:
        if self.command_edit is None:
            return

        parsed = parse_command_input(self.command_edit.text())
        if parsed is None:
            self._exit_command_mode()
            return

        cmd, arg = parsed

        handled = self._run_vim_command(cmd, arg)
        if not handled:
            QtWidgets.QMessageBox.information(
                self,
                "Command",
                f"Unknown command: {cmd}",
            )

        self._exit_command_mode()

    def _run_vim_command(self, cmd: str, arg: str) -> bool:
        return self.commands.run_vim_command(cmd, arg)

    def _open_file_by_name(self, filename: str) -> None:
        self.commands.open_file_by_name(filename)

    def _open_directory_fuzzy_view(self) -> None:
        self.commands.open_directory_fuzzy_view()

    def _open_adjust_dialog(self):
        if self.view_mode == "fft" and self._magnitude_spectrum is None:
            return

        if self.view_mode != "fft" and self.data is None:
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
            if self.chk_inverse is not None and self.chk_inverse.isChecked():
                if self._inverse_fft_cache is None and self._fft_complex is not None:
                    self._inverse_fft_cache = utils.compute_inverse_fft(self._fft_complex)
                source_data = self._inverse_fft_cache if self._inverse_fft_cache is not None else self._magnitude_spectrum
            else:
                source_data = self._magnitude_spectrum
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

    def _delete_selected_roi(self):
        self.fft_manager.delete_selected_roi()

    def _setup_menu_bar(self) -> None:
        menu_bar = self.menuBar()
        menu_bar.clear()

        file_menu = menu_bar.addMenu("File")
        act_open = file_menu.addAction("Open")
        act_open.triggered.connect(self._open_file_dialog)

        act_save_view = file_menu.addAction("Save View")
        act_save_view.triggered.connect(self._save_view_and_ffts)

        act_build_figure = file_menu.addAction("Build Figure")
        act_build_figure.triggered.connect(lambda: self._show_not_implemented("Build Figure"))

        act_calibrate_image = file_menu.addAction("Calibrate Image")
        act_calibrate_image.triggered.connect(self._open_calibration_dialog)

        act_parameters = file_menu.addAction("Parameters")
        act_parameters.triggered.connect(lambda: self._show_not_implemented("Parameters"))

        manipulate_menu = menu_bar.addMenu("Manipulate")
        act_fft = manipulate_menu.addAction("FFT")
        act_fft.triggered.connect(self._add_new_fft)
        act_fft.setEnabled(self.view_mode == "image")

        act_inverse_fft = manipulate_menu.addAction("Inverse FFT")
        if self.view_mode in {"image", "fft"}:
            act_inverse_fft.setCheckable(True)
            act_inverse_fft.toggled.connect(self._on_inverse_fft_toggled)
            self._inverse_action = act_inverse_fft

        measure_menu = menu_bar.addMenu("Measure")
        act_distance = measure_menu.addAction("Distance")
        act_distance.triggered.connect(self._menu_start_distance_measurement)

        act_history = measure_menu.addAction("History")
        act_history.triggered.connect(self._show_measurement_history)

        act_intensity = measure_menu.addAction("Intensity")
        act_intensity.triggered.connect(lambda: self._show_not_implemented("Intensity"))

        act_profile = measure_menu.addAction("Profile")
        act_profile.triggered.connect(self._menu_start_profile_measurement)

        display_menu = menu_bar.addMenu("Display")
        act_adjust = display_menu.addAction("Adjust")
        act_adjust.triggered.connect(self._open_adjust_dialog)

        act_metadata = display_menu.addAction("Metadata")
        act_metadata.triggered.connect(self._show_metadata_window)
        act_metadata.setEnabled(self.view_mode == "image")

    def _show_not_implemented(self, feature_name: str) -> None:
        QtWidgets.QMessageBox.information(
            self,
            feature_name,
            f"{feature_name} is planned but not implemented yet.",
        )

    def _open_file_dialog(self) -> None:
        start_dir = str(Path(self.file_path).parent) if self.file_path else str(Path.cwd())
        selected_file, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Open Image",
            start_dir,
            IMAGE_FILE_FILTER,
        )
        if selected_file:
            open_image_file(selected_file)

    def _menu_start_distance_measurement(self) -> None:
        logger.debug("Menu action: start distance measurement")
        if self.btn_measure is not None and not self.btn_measure.isChecked():
            self.btn_measure.setChecked(True)
        self._toggle_line_measurement()

    def _menu_start_profile_measurement(self) -> None:
        logger.debug("Menu action: start profile measurement")
        self.measurements.start_profile_measurement()

    def _on_inverse_fft_toggled(self, checked: bool) -> None:
        if self.chk_inverse is not None:
            self.chk_inverse.blockSignals(True)
            self.chk_inverse.setChecked(checked)
            self.chk_inverse.blockSignals(False)
        self.display_min = None
        self.display_max = None
        self._update_image_display()

    def _save_view_and_ffts(self) -> None:
        """Save the current view (with annotations) and any FFT windows.

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
        for idx, fft_window in enumerate(self.fft_windows, start=1):
            # Build label: metadata summary and ROI number (from fft_name)
            parent_window = getattr(fft_window, "parent_image_window", None)
            base_label = None
            if parent_window is self:
                base_label = overlay_label
            elif parent_window is not None and hasattr(parent_window, "_build_export_overlay_label"):
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

            fft_path = directory_path / f"{base_name}_fft{idx}.png"
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

    def _add_new_fft(self, x_offset=None, y_offset=None, w=None, h=None):
        self.fft_manager.add_new_fft(x_offset, y_offset, w, h)

    def _on_fft_finished(self, fft_box: pg.RectROI, fft_id: int, text_item: pg.TextItem):
        self.fft_manager.on_fft_finished(fft_box, fft_id, text_item)

    def _open_or_update_fft_window(
        self, fft_box: pg.RectROI, fft_id: int, text_item: pg.TextItem, region: np.ndarray
    ):
        self.fft_manager.open_or_update_fft_window(fft_box, fft_id, text_item, region)

    def _on_fft_box_clicked(self, fft_box: pg.RectROI):
        self.fft_manager.on_fft_box_clicked(fft_box)

    def _on_fft_box_double_clicked(
        self, fft_box: pg.RectROI, fft_id: int, text_item: pg.TextItem
    ):
        self.fft_manager.on_fft_box_double_clicked(fft_box, fft_id, text_item)

    def _exit_measure_mode(self):
        self.measurements.exit_measure_mode()

    def _toggle_line_measurement(self):
        self.measurements.toggle_line_measurement()

    def _on_line_drawn(self, p1: Tuple[float, float], p2: Tuple[float, float]):
        self.measurements.on_line_drawn(p1, p2)

    def _on_measurement_label_clicked(self, label: pg.TextItem):
        self.measurements.on_measurement_label_clicked(label)

    def _clear_measurements(self):
        self.measurements.clear_measurements()

    def clear_measurements_from_history(self):
        self.measurements.clear_measurements_from_history()

    def _delete_selected_measurement(self):
        self.measurements.delete_selected_measurement()

    def delete_measurement_by_history_id(self, entry_id: int, entry_type: str):
        self.measurements.delete_measurement_by_history_id(entry_id, entry_type)

    def open_measurement_by_history_id(self, entry_id: int, entry_type: str):
        self.measurements.open_measurement_by_history_id(entry_id, entry_type)

    def rename_measurement_by_history_id(self, entry_id: int, entry_type: str, new_text: str):
        self.measurements.rename_measurement_by_history_id(entry_id, entry_type, new_text)

    def _set_label_fill(self, text_item: pg.TextItem, brush: pg.QtGui.QBrush):
        self.measurements.set_label_fill(text_item, brush)

    def delete_measurement_by_label(self, label_text: str):
        self.measurements.delete_measurement_by_label(label_text)

    def _format_measurement_label(
        self, result: dict, measurement_id: Optional[int] = None
    ) -> str:
        return self.measurements.format_measurement_label(result, measurement_id)

    def _show_measurement_history(self):
        self.measurements.show_measurement_history()

    def _add_to_measurement_history(self, measurement_text: str):
        self.measurements.add_to_measurement_history(measurement_text)

    # ------------------------------------------------------------------
    # Metadata helpers for export overlays
    # ------------------------------------------------------------------

    def _build_export_overlay_label(self) -> str:
        """Return a one-line label for export overlays.

        Uses *original* metadata to add:
        - Microscope: second word of ``ObjectInfo.ExperimentalDescription["Microscope"]``
        - Mode: first word of ``ObjectInfo.ExperimentalDescription["Mode"]``
        - AcquireDate: full string from ``ObjectInfo["AcquireDate"]``

        Final format:
            ``<filename> | <Microscope-second-word> | <Mode-first-word> | <AcquireDate>``
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
                    microscope_val = (
                        exp_desc.get("Microscope")
                        or exp_desc.get("microscope")
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
