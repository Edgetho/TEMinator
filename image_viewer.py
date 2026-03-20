# SPDX-License-Identifier: GPL-2.0-only
# Copyright (C) 2026 Cooper Stuntz
# See LICENSE for full license terms.

"""Image viewer window and image-opening helper."""
from __future__ import annotations

from pathlib import Path
from typing import Optional, List, Tuple, Dict

import numpy as np
import hyperspy.api as hs
import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets, QtCore, QtGui

import utils
from dialogs import MeasurementHistoryWindow, MetadataWindow, ToneCurveDialog
from measurement_tools import (
    LineDrawingTool,
    FFTBoxROI,
    MeasurementLabel,
    LABEL_BRUSH_COLOR,
    DRAWN_LINE_PEN,
)
from scale_bars import DynamicScaleBar


FFT_COLORS = ["r", "g", "b", "y", "c", "m"]
DEFAULT_IMAGE_WINDOW_SIZE = (1000, 900)
DEFAULT_FFT_WINDOW_SIZE = (700, 700)


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
        self.btn_measure: Optional[QtWidgets.QPushButton] = None
        self.measurement_items: List[Tuple[pg.PlotDataItem, pg.TextItem]] = []
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
            s = hs.load(self.file_path)
            if s.axes_manager.navigation_dimension != 0:
                s = s.inav[0, 0]
            self._setup_from_signal(s)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error Loading File", str(e))

    def _setup_from_signal(self, signal, window_suffix: Optional[str] = None):
        self.signal = signal
        self.data = signal.data
        self.ax_x = signal.axes_manager[0]
        self.ax_y = signal.axes_manager[1]

        calibrated = self._apply_calibration_from_original_metadata()
        if not calibrated:
            QtWidgets.QMessageBox.warning(
                self,
                "Calibration",
                "This image could not be calibrated successfully from metadata; "
                "using default pixel scaling instead.",
            )

        # Decide whether this view should be treated as reciprocal space.
        # Prefer explicit metadata / axis units when available so that
        # multiple images from the same file are classified consistently,
        # falling back to an image‑content heuristic only as a last resort.
        self.is_reciprocal_space = self._detect_reciprocal_space(signal)

        self._init_display_window()

        file_name = Path(self.file_path).stem
        if window_suffix:
            title = f"Image Viewer - {file_name} {window_suffix}"
        else:
            title = f"Image Viewer - {file_name}"
        self.setWindowTitle(title)

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
        self.setWindowTitle(f"Image Viewer - {parent_title} - {display_name}")

        self._compute_fft()
        self._init_display_window()
        self.setup_ui()
        self.resize(*DEFAULT_FFT_WINDOW_SIZE)

    def _get_original_metadata_dict(self) -> Optional[dict]:
        if self.signal is None:
            return None

        original_meta = None

        if hasattr(self.signal, "original_metadata"):
            original_meta = self.signal.original_metadata
        elif hasattr(self.signal.metadata, "original_metadata"):
            original_meta = self.signal.metadata.original_metadata

        if original_meta is None:
            return None

        if hasattr(original_meta, "as_dictionary"):
            return original_meta.as_dictionary()

        try:
            return dict(original_meta)
        except Exception:
            return None

    def _apply_calibration_from_original_metadata(self) -> bool:
        meta = self._get_original_metadata_dict()
        if not meta:
            return False

        ser_params = None
        if "ser_header_parameters" in meta:
            ser_params = meta["ser_header_parameters"]
        else:
            for key, value in meta.items():
                if isinstance(key, str) and key.lower() == "ser_header_parameters":
                    ser_params = value
                    break

        if not isinstance(ser_params, dict):
            return False

        dx = ser_params.get("CalibrationDeltaX")
        dy = ser_params.get("CalibrationDeltaY")

        if not (isinstance(dx, (int, float)) and isinstance(dy, (int, float))):
            return False
        if dx <= 0 or dy <= 0:
            return False

        try:
            self.ax_x.scale = float(dx)
            self.ax_y.scale = float(dy)
            self.ax_x.units = "m"
            self.ax_y.units = "m"

            ox = ser_params.get("CalibrationOffsetX")
            oy = ser_params.get("CalibrationOffsetY")
            if isinstance(ox, (int, float)):
                self.ax_x.offset = float(ox)
            if isinstance(oy, (int, float)):
                self.ax_y.offset = float(oy)
        except Exception:
            return False

        return True

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

        units_str = " ".join(units_parts).lower()
        reciprocal_markers = [
            "1/",
            "/m",
            "1/m",
            "1/nm",
            "/nm",
            "^-1",
            "-1",
        ]
        if any(marker in units_str for marker in reciprocal_markers):
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
            self.scale_bar = DynamicScaleBar(self.p1.vb, units=self.freq_axis_base_unit)
            self.scale_bar.reciprocal = True
        else:
            self._apply_colormap()
            self._update_image_display()

            x_offset, y_offset, w, h = self.image_bounds
            self.img_orig.setRect(QtCore.QRectF(x_offset, y_offset, w, h))

            if hasattr(self.p1.vb, "setPadding"):
                self.p1.vb.setPadding(0.0)
            self.p1.setXRange(x_offset, x_offset + w, padding=0)
            self.p1.setYRange(y_offset, y_offset + h, padding=0)

            units = self.ax_x.units if self.ax_x is not None and self.ax_x.units else "m"
            if self.is_reciprocal_space:
                self.scale_bar = DynamicScaleBar(self.p1.vb, units=units)
                self.scale_bar.reciprocal = True
            else:
                self.scale_bar = DynamicScaleBar(self.p1.vb, units=units)
                try:
                    overlay_label = self._build_export_overlay_label()
                    if overlay_label and hasattr(self.scale_bar, "set_extra_label"):
                        self.scale_bar.set_extra_label(overlay_label)
                except Exception:
                    pass

        self.line_tool = LineDrawingTool(self.p1, self._on_line_drawn)

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

        adjusted = utils.apply_intensity_transform(
            self.data,
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
        if self.command_edit is None:
            return
        self.command_edit.show()
        self.command_edit.clear()
        self.command_edit.setText(":")
        self.command_edit.setFocus()
        self.command_edit.setCursorPosition(len(self.command_edit.text()))

    def _exit_command_mode(self) -> None:
        if self.command_edit is None:
            return
        self.command_edit.clear()
        self.command_edit.hide()
        # Return focus to the graphics view so normal keys work again
        if hasattr(self.glw, "setFocus"):
            self.glw.setFocus()

    def _execute_command_from_line(self) -> None:
        if self.command_edit is None:
            return

        text = self.command_edit.text().strip()
        if text.startswith(":"):
            text = text[1:]
        text = text.strip()
        if not text:
            self._exit_command_mode()
            return

        parts = text.split(maxsplit=1)
        cmd = parts[0]
        arg = parts[1] if len(parts) > 1 else ""

        handled = self._run_vim_command(cmd, arg)
        if not handled:
            QtWidgets.QMessageBox.information(
                self,
                "Command",
                f"Unknown command: {cmd}",
            )

        self._exit_command_mode()

    def _run_vim_command(self, cmd: str, arg: str) -> bool:
        """Dispatch vim-like commands for this image window.

        Supported commands:
          :F              – add a new FFT ROI using current view
          :d              – start distance measurement mode
          :e <filename>   – open a file in this directory
          :E              – fuzzy-open a file in this image's directory
          :a              – open the adjustment window
        """

        cmd_str = cmd.strip()
        if not cmd_str:
            return False

        upper_cmd = cmd_str.upper()
        lower_cmd = cmd_str.lower()

        # :F – add new FFT ROI
        if upper_cmd == "F":
            self._add_new_fft()
            return True

        # :d – enable distance measurement mode
        if upper_cmd == "D":
            if self.btn_measure is not None:
                if not self.btn_measure.isChecked():
                    self.btn_measure.setChecked(True)
                    self._toggle_line_measurement()
            return True

        # :a – open adjustment window
        if upper_cmd == "A":
            self._open_adjust_dialog()
            return True

        # :E – directory fuzzy finder for current image directory
        if upper_cmd == "E" and not arg:
            self._open_directory_fuzzy_view()
            return True

        # :e <filename> – open file (relative to current image directory by default)
        if lower_cmd == "e":
            if not arg:
                QtWidgets.QMessageBox.information(
                    self,
                    "Command",
                    "Usage: :e <filename>",
                )
                return True
            self._open_file_by_name(arg)
            return True

        return False

    def _open_file_by_name(self, filename: str) -> None:
        from pathlib import Path as _Path

        name = filename.strip().strip("\"").strip("'")
        if not name:
            return

        path = _Path(name)
        if not path.is_absolute():
            try:
                base = _Path(self.file_path).parent
            except Exception:
                base = _Path.cwd()
            path = base / name

        if not path.is_file():
            QtWidgets.QMessageBox.warning(
                self,
                "Open File",
                f"File not found: {path}",
            )
            return

        open_image_file(str(path))

    def _open_directory_fuzzy_view(self) -> None:
        from pathlib import Path as _Path

        try:
            directory = _Path(self.file_path).parent
        except Exception:
            directory = _Path.cwd()

        if not directory.is_dir():
            QtWidgets.QMessageBox.warning(
                self,
                "Directory",
                f"Directory not found: {directory}",
            )
            return

        dialog = DirectoryFuzzyOpenDialog(self, directory)
        dialog.exec_()

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
        if self.selected_measurement_index is not None:
            self._delete_selected_measurement()
            return

        fft_box = self.selected_fft_box
        if fft_box is None or fft_box not in self.fft_boxes:
            QtWidgets.QMessageBox.information(
                self, "Delete", "No measurement or ROI selected."
            )
            return

        index = self.fft_boxes.index(fft_box)
        self.p1.removeItem(fft_box)

        meta = self.fft_box_meta.pop(fft_box, None)
        if meta is not None:
            text_item = meta.get("text_item")
            if text_item is not None:
                self.p1.removeItem(text_item)

        fft_window = self.fft_to_fft_window.pop(fft_box, None)
        if fft_window is not None:
            fft_window.close()
            if fft_window in self.fft_windows:
                self.fft_windows.remove(fft_window)

        self.fft_boxes.pop(index)
        self.selected_fft_box = None

        QtWidgets.QMessageBox.information(self, "Deleted", f"FFT Box {index} deleted.")

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

        act_parameters = file_menu.addAction("Parameters")
        act_parameters.triggered.connect(lambda: self._show_not_implemented("Parameters"))

        manipulate_menu = menu_bar.addMenu("Manipulate")
        act_fft = manipulate_menu.addAction("FFT")
        act_fft.triggered.connect(self._add_new_fft)
        act_fft.setEnabled(self.view_mode == "image")

        act_inverse_fft = manipulate_menu.addAction("Inverse FFT")
        if self.view_mode == "fft":
            act_inverse_fft.setCheckable(True)
            act_inverse_fft.toggled.connect(self._on_inverse_fft_toggled)
            self._inverse_action = act_inverse_fft
        else:
            act_inverse_fft.setEnabled(False)

        measure_menu = menu_bar.addMenu("Measure")
        act_distance = measure_menu.addAction("Distance")
        act_distance.triggered.connect(self._menu_start_distance_measurement)

        act_history = measure_menu.addAction("History")
        act_history.triggered.connect(self._show_measurement_history)

        act_intensity = measure_menu.addAction("Intensity")
        act_intensity.triggered.connect(lambda: self._show_not_implemented("Intensity"))

        act_profile = measure_menu.addAction("Profile")
        act_profile.triggered.connect(lambda: self._show_not_implemented("Profile"))

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
            "Image files (*.dm3 *.dm4 *.tif *.tiff *.mrc *.ser *.png *.jpg *.jpeg);;All files (*)",
        )
        if selected_file:
            open_image_file(selected_file)

    def _menu_start_distance_measurement(self) -> None:
        if self.btn_measure is not None and not self.btn_measure.isChecked():
            self.btn_measure.setChecked(True)
        self._toggle_line_measurement()

    def _on_inverse_fft_toggled(self, checked: bool) -> None:
        if self.chk_inverse is not None:
            self.chk_inverse.blockSignals(True)
            self.chk_inverse.setChecked(checked)
            self.chk_inverse.blockSignals(False)
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
        if self.view_mode != "image":
            return

        if x_offset is None or y_offset is None or w is None or h is None:
            try:
                (x_range, y_range) = self.p1.vb.viewRange()
                x0, x1 = x_range
                y0, y1 = y_range
                x_offset = float(x0)
                y_offset = float(y0)
                w = float(x1 - x0)
                h = float(y1 - y0)
            except Exception:
                x_offset, y_offset, w, h = self.image_bounds

        if w is None or h is None or w <= 0 or h <= 0:
            QtWidgets.QMessageBox.warning(self, "Error", "Invalid image bounds")
            return

        color = FFT_COLORS[self.fft_count % len(FFT_COLORS)]
        roi_x = float(x_offset) + float(w) / 4.0
        roi_y = float(y_offset) + float(h) / 4.0
        roi_w = float(w) / 2.0
        roi_h = float(h) / 2.0
        fft_box = FFTBoxROI([roi_x, roi_y], [roi_w, roi_h], pen=pg.mkPen(color, width=2))
        fft_box.addScaleHandle([1, 1], [0, 0])
        fft_box.addScaleHandle([0, 0], [1, 1])
        self.p1.addItem(fft_box)

        fft_id = self.fft_count
        self.fft_count += 1

        text_item = pg.TextItem(f"FFT {fft_id}", anchor=(0, 0), fill=pg.mkBrush(color))
        text_item.setPos(fft_box.pos()[0], fft_box.pos()[1])
        self.p1.addItem(text_item)

        fft_box.sigRegionChangeFinished.connect(
            lambda: self._on_fft_finished(fft_box, fft_id, text_item)
        )
        fft_box.sigBoxClicked.connect(lambda roi=fft_box: self._on_fft_box_clicked(roi))
        fft_box.sigBoxDoubleClicked.connect(
            lambda roi=fft_box: self._on_fft_box_double_clicked(roi, fft_id, text_item)
        )
        self.fft_boxes.append(fft_box)
        self.fft_box_meta[fft_box] = {"id": fft_id, "text_item": text_item}

    def _on_fft_finished(self, fft_box: pg.RectROI, fft_id: int, text_item: pg.TextItem):
        region = fft_box.getArrayRegion(self.data, self.img_orig)

        if region is None or region.shape[0] < 2 or region.shape[1] < 2:
            return

        text_item.setPos(fft_box.pos()[0], fft_box.pos()[1])

        self._open_or_update_fft_window(fft_box, fft_id, text_item, region)

    def _open_or_update_fft_window(
        self, fft_box: pg.RectROI, fft_id: int, text_item: pg.TextItem, region: np.ndarray
    ):
        if fft_box in self.fft_to_fft_window:
            fft_window = self.fft_to_fft_window[fft_box]
            fft_window._fft_region = region
            fft_window._compute_fft()
            fft_window._init_display_window()
            fft_window._update_image_display()
            fft_window.show()
            fft_window.raise_()
            fft_window.activateWindow()
        else:
            fft_name = f"FFT {fft_id}"
            fft_window = ImageViewerWindow(
                self.file_path,
                view_mode="fft",
                fft_region=region,
                fft_name=fft_name,
                parent_image_window=self,
            )
            fft_window.show()
            self.fft_windows.append(fft_window)
            self.fft_to_fft_window[fft_box] = fft_window

    def _on_fft_box_clicked(self, fft_box: pg.RectROI):
        if fft_box in self.fft_boxes:
            self.selected_fft_box = fft_box

    def _on_fft_box_double_clicked(
        self, fft_box: pg.RectROI, fft_id: int, text_item: pg.TextItem
    ):
        region = fft_box.getArrayRegion(self.data, self.img_orig)
        if region is None or region.shape[0] < 2 or region.shape[1] < 2:
            return

        text_item.setPos(fft_box.pos()[0], fft_box.pos()[1])
        self._open_or_update_fft_window(fft_box, fft_id, text_item, region)

    def _exit_measure_mode(self):
        if self.btn_measure is not None and self.btn_measure.isChecked():
            self.btn_measure.setChecked(False)

        if self.line_tool is not None:
            self.line_tool.disable()
        if self.btn_measure is not None:
            self.btn_measure.setStyleSheet("")

    def _toggle_line_measurement(self):
        if self.line_tool is None:
            return

        if self.btn_measure is not None and self.btn_measure.isChecked():
            self.line_tool.enable()
            self.btn_measure.setStyleSheet(
                "background-color: #4caf50; color: white;"
            )
        else:
            self.line_tool.disable()
            if self.btn_measure is not None:
                self.btn_measure.setStyleSheet("")

    def _on_line_drawn(self, p1: Tuple[float, float], p2: Tuple[float, float]):
        self.measurement_count += 1
        measurement_id = self.measurement_count

        if self.view_mode == "fft":
            dx_freq = float(p2[0] - p1[0])
            dy_freq = float(p2[1] - p1[1])
            dist_freq = float(np.hypot(dx_freq, dy_freq))

            if self._nyq_x and self._nyq_y and self._fft_region is not None:
                px_scale_x = (2.0 * float(self._nyq_x)) / float(self._fft_region.shape[1])
                px_scale_y = (2.0 * float(self._nyq_y)) / float(self._fft_region.shape[0])
                if px_scale_x != 0 and px_scale_y != 0:
                    dx_px = dx_freq / px_scale_x
                    dy_px = dy_freq / px_scale_y
                    dist_px = float(np.hypot(dx_px, dy_px))
                else:
                    dist_px = 0.0
            else:
                dist_px = 0.0

            result: Dict[str, float] = {
                "distance_pixels": dist_px,
                "distance_physical": dist_freq,
                "scale_x": float(self.ax_x.scale) if self.ax_x else 1.0,
                "scale_y": float(self.ax_y.scale) if self.ax_y else 1.0,
            }
            if dist_freq != 0:
                result["d_spacing"] = utils.calculate_d_spacing(dist_freq)

            line = pg.PlotDataItem([p1[0], p2[0]], [p1[1], p2[1]], pen=DRAWN_LINE_PEN)
            self.p1.addItem(line)

            label_text = self._format_measurement_label(result, measurement_id)
            mid_x = (p1[0] + p2[0]) / 2.0
            mid_y = (p1[1] + p2[1]) / 2.0

            text_item = MeasurementLabel(label_text, anchor=(0, 0), fill=LABEL_BRUSH_COLOR)
            text_item.setPos(mid_x, mid_y)
            text_item.sigLabelClicked.connect(self._on_measurement_label_clicked)
            self.p1.addItem(text_item)

            self.measurement_items.append((line, text_item))
            self._add_to_measurement_history(label_text)
            return

        scale_x = float(self.ax_x.scale) if self.ax_x is not None else 1.0
        scale_y = float(self.ax_y.scale) if self.ax_y is not None else scale_x

        dx_phys = float(p2[0] - p1[0])
        dy_phys = float(p2[1] - p1[1])

        dist_phys = float(np.hypot(dx_phys, dy_phys))

        if scale_x != 0 and scale_y != 0:
            dx_px = dx_phys / scale_x
            dy_px = dy_phys / scale_y
            dist_px = float(np.hypot(dx_px, dy_px))
        else:
            dist_px = 0.0

        result: Dict[str, float] = {
            "distance_pixels": dist_px,
            "distance_physical": dist_phys,
            "scale_x": scale_x,
            "scale_y": scale_y,
        }

        if self.is_reciprocal_space and dist_phys != 0:
            frequency = 1.0 / dist_phys
            result["d_spacing"] = utils.calculate_d_spacing(frequency)

        line = pg.PlotDataItem([p1[0], p2[0]], [p1[1], p2[1]], pen=DRAWN_LINE_PEN)
        self.p1.addItem(line)

        label_text = self._format_measurement_label(result, measurement_id)
        mid_x = (p1[0] + p2[0]) / 2.0
        mid_y = (p1[1] + p2[1]) / 2.0

        text_item = MeasurementLabel(label_text, anchor=(0, 0), fill=LABEL_BRUSH_COLOR)
        text_item.setPos(mid_x, mid_y)
        text_item.sigLabelClicked.connect(self._on_measurement_label_clicked)
        self.p1.addItem(text_item)

        self.measurement_items.append((line, text_item))
        self._add_to_measurement_history(label_text)

    def _on_measurement_label_clicked(self, label: pg.TextItem):
        selected_index = None
        for idx, (_line_item, text_item) in enumerate(self.measurement_items):
            if text_item is label:
                selected_index = idx
                break

        self.selected_measurement_index = selected_index

        for idx, (_line_item, text_item) in enumerate(self.measurement_items):
            if idx == selected_index:
                self._set_label_fill(text_item, pg.mkBrush(255, 200, 0, 255))
            else:
                self._set_label_fill(text_item, LABEL_BRUSH_COLOR)

    def _clear_measurements(self):
        for line_item, text_item in self.measurement_items:
            self.p1.removeItem(line_item)
            self.p1.removeItem(text_item)
        self.measurement_items.clear()
        self.selected_measurement_index = None

        if self.measurement_history_window is not None:
            self.measurement_history_window.clear_all()

    def _delete_selected_measurement(self):
        if self.selected_measurement_index is None:
            return

        if not (0 <= self.selected_measurement_index < len(self.measurement_items)):
            self.selected_measurement_index = None
            return

        line_item, text_item = self.measurement_items.pop(
            self.selected_measurement_index
        )
        self.p1.removeItem(line_item)
        self.p1.removeItem(text_item)

        self.selected_measurement_index = None
        for _line_item, text_item in self.measurement_items:
            self._set_label_fill(text_item, LABEL_BRUSH_COLOR)

    def _set_label_fill(self, text_item: pg.TextItem, brush: pg.QtGui.QBrush):
        if hasattr(text_item, "setFill"):
            text_item.setFill(brush)
        elif hasattr(text_item, "setBrush"):
            text_item.setBrush(brush)

    def delete_measurement_by_label(self, label_text: str):
        target_index = None
        for idx, (_line_item, text_item) in enumerate(self.measurement_items):
            if text_item.toPlainText() == label_text:
                target_index = idx
                break

        if target_index is None:
            return

        self.selected_measurement_index = target_index
        self._delete_selected_measurement()

    def _format_measurement_label(
        self, result: dict, measurement_id: Optional[int] = None
    ) -> str:
        if self.view_mode == "fft":
            scaled_dist, scaled_unit = utils.format_si_scale(
                result["distance_physical"], self.freq_axis_base_unit
            )
            prefix = f"#{measurement_id} " if measurement_id is not None else ""
            if "d_spacing" in result:
                return (
                    f"{prefix}d: {result['d_spacing']:.4f} Å\n"
                    f"({scaled_dist:.4f} {scaled_unit}⁻¹)"
                )
            return (
                f"{prefix}{scaled_dist:.4f} {scaled_unit}⁻¹\n"
                f"({result['distance_pixels']:.1f} px)"
            )

        scaled_dist, scaled_unit = utils.format_si_scale(
            result["distance_physical"], self.ax_x.units
        )
        prefix = f"#{measurement_id} " if measurement_id is not None else ""

        if self.is_reciprocal_space and "d_spacing" in result:
            return (
                f"{prefix}d: {result['d_spacing']:.4f} Å\n"
                f"({scaled_dist:.4f} {scaled_unit}⁻¹)"
            )
        return (
            f"{prefix}{scaled_dist:.4f} {scaled_unit}\n"
            f"({result['distance_pixels']:.1f} px)"
        )

    def _show_measurement_history(self):
        if self.measurement_history_window is None:
            self.measurement_history_window = MeasurementHistoryWindow(self)

        self.measurement_history_window.show()
        self.measurement_history_window.raise_()
        self.measurement_history_window.activateWindow()

    def _add_to_measurement_history(self, measurement_text: str):
        if self.measurement_history_window is None:
            self.measurement_history_window = MeasurementHistoryWindow(self)

        self.measurement_history_window.add_measurement(measurement_text)

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


class DirectoryFuzzyOpenDialog(QtWidgets.QDialog):
    """Simple fuzzy finder over files in a directory."""

    def __init__(self, parent: Optional[QtWidgets.QWidget], directory: Path):
        super().__init__(parent)
        self.directory = directory
        self._all_files: List[str] = []

        self.setWindowTitle(f"Open File - {directory}")
        self.resize(500, 400)

        layout = QtWidgets.QVBoxLayout(self)

        self.filter_edit = QtWidgets.QLineEdit(self)
        self.filter_edit.setPlaceholderText("Type to fuzzy-filter; Enter to open")
        layout.addWidget(self.filter_edit)

        self.list_widget = QtWidgets.QListWidget(self)
        layout.addWidget(self.list_widget)

        self.filter_edit.textChanged.connect(self._on_filter_changed)
        self.filter_edit.returnPressed.connect(self._on_return_pressed)
        self.list_widget.itemActivated.connect(self._on_item_activated)
        self.list_widget.itemDoubleClicked.connect(self._on_item_activated)

        self._populate_files()

        if self._all_files:
            self.list_widget.setCurrentRow(0)

        # Focus the filter line edit when shown
        QtCore.QTimer.singleShot(0, self.filter_edit.setFocus)

    def _populate_files(self) -> None:
        # Restrict to common image/signal extensions
        exts = {
            ".dm3",
            ".dm4",
            ".tif",
            ".tiff",
            ".mrc",
            ".ser",
            ".png",
            ".jpg",
            ".jpeg",
        }

        try:
            self._all_files = sorted(
                f.name
                for f in self.directory.iterdir()
                if f.is_file() and (not exts or f.suffix.lower() in exts)
            )
        except Exception:
            self._all_files = []

        self._update_list(self._all_files)

    def _update_list(self, names: List[str]) -> None:
        self.list_widget.clear()
        self.list_widget.addItems(names)

    @staticmethod
    def _fuzzy_match(pattern: str, text: str) -> bool:
        """Simple subsequence-based fuzzy matching."""
        it = iter(text)
        return all(ch in it for ch in pattern)

    def _on_filter_changed(self, text: str) -> None:
        pattern = text.strip().lower()
        if not pattern:
            self._update_list(self._all_files)
            if self._all_files:
                self.list_widget.setCurrentRow(0)
            return

        matches = [
            name
            for name in self._all_files
            if self._fuzzy_match(pattern, name.lower())
        ]
        self._update_list(matches)
        if matches:
            self.list_widget.setCurrentRow(0)

    def _on_return_pressed(self) -> None:
        current = self.list_widget.currentItem()
        if current is None and self.list_widget.count() > 0:
            current = self.list_widget.item(0)
        if current is not None:
            self._open_item(current)

    def _on_item_activated(self, item: QtWidgets.QListWidgetItem) -> None:  # type: ignore[override]
        self._open_item(item)

    def _open_item(self, item: QtWidgets.QListWidgetItem) -> None:
        name = item.text()
        path = self.directory / name
        if path.is_file():
            open_image_file(str(path))
        self.accept()


def open_image_file(file_path: str):
    """Open an image file; if it contains multiple images, open one window per image."""
    try:
        loaded = hs.load(file_path)

        signals = loaded if isinstance(loaded, list) else [loaded]

        for sig_index, signal in enumerate(signals):
            if signal.axes_manager.navigation_dimension == 0:
                suffix = f"[{sig_index}]" if len(signals) > 1 else None
                window = ImageViewerWindow(file_path, signal=signal, window_suffix=suffix)
                window.show()
            else:
                nav_shape = signal.axes_manager.navigation_shape
                for nav_index in np.ndindex(nav_shape):
                    sub_signal = signal.inav[nav_index]
                    idx_str = ",".join(str(i) for i in nav_index)
                    if len(signals) > 1:
                        suffix = f"[{sig_index}; {idx_str}]"
                    else:
                        suffix = f"[{idx_str}]"
                    window = ImageViewerWindow(
                        file_path, signal=sub_signal, window_suffix=suffix
                    )
                    window.show()

    except Exception as e:
        QtWidgets.QMessageBox.critical(None, "Error", f"Could not open file: {str(e)}")
