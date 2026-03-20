# SPDX-License-Identifier: GPL-2.0-only
# Copyright (C) 2026 Cooper Stuntz
# See LICENSE for full license terms.

"""FFT viewer window for a selected ROI."""
from __future__ import annotations

from typing import Optional, List, Tuple, Dict

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets, QtCore, QtGui

import utils
from dialogs import MeasurementHistoryWindow, ToneCurveDialog
from measurement_tools import (
    LineDrawingTool,
    MeasurementLabel,
    LABEL_BRUSH_COLOR,
    DRAWN_LINE_PEN,
)
from scale_bars import DynamicScaleBar


DEFAULT_FFT_WINDOW_SIZE = (700, 700)


class FFTViewerWindow(QtWidgets.QMainWindow):
    """Separate window displaying FFT for a specific FFT box."""

    def __init__(
        self,
        parent,
        region: np.ndarray,
        scale_x: float,
        scale_y: float,
        ax_x_name: str,
        ax_x_units: str,
        ax_y_name: str,
        ax_y_units: str,
        fft_name: str,
        parent_name: str = "",
    ):
        super().__init__()

        self.fft_name = fft_name
        self._update_title(fft_name, parent_name)
        self.resize(*DEFAULT_FFT_WINDOW_SIZE)

        self.scale_x = scale_x
        self.scale_y = scale_y
        self.ax_x_name = ax_x_name
        self.ax_x_units = ax_x_units
        self.ax_y_name = ax_y_name
        self.ax_y_units = ax_y_units
        self.region = region
        self.parent_image_window = parent

        # Base real-space units for the parent image; used to derive
        # reciprocal-space labels such as m⁻¹, nm⁻¹, or Å⁻¹.
        self.fft_unit_x = f"1/{self.ax_x_units}" if self.ax_x_units else "1/px"
        self.fft_unit_y = f"1/{self.ax_y_units}" if self.ax_y_units else "1/px"
        self.is_reciprocal_space = True
        self.freq_axis_base_unit = self.ax_x_units or "m"

        self._magnitude_spectrum = None
        self._fft_complex = None
        self._inverse_fft_cache = None
        self._nyq_x = None
        self._nyq_y = None
        self._last_region_id = None

        self.line_tool: Optional[LineDrawingTool] = None
        self.measurement_history_window: Optional[MeasurementHistoryWindow] = None
        self.measurement_count: int = 0
        self.btn_measure: Optional[QtWidgets.QPushButton] = None
        self.measurement_items: List[Tuple[pg.PlotDataItem, pg.TextItem]] = []
        self.selected_measurement_index: Optional[int] = None
        self.scale_bar: Optional[DynamicScaleBar] = None

        self.display_min: Optional[float] = None
        self.display_max: Optional[float] = None
        self.display_gamma: float = 1.0

        self._tone_dialog: Optional[ToneCurveDialog] = None

        self.setup_ui()
        self._compute_fft()
        self.update_display()
        self.setup_keyboard_shortcuts()

    def _update_title(self, fft_name: str, parent_name: str):
        title = f"FFT - {parent_name} - {fft_name}" if parent_name else f"FFT - {fft_name}"
        self.setWindowTitle(title)

    def setup_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QVBoxLayout(central)

        toolbar = QtWidgets.QHBoxLayout()
        self.chk_inverse = QtWidgets.QCheckBox("Show Inverse FFT")
        self.chk_inverse.stateChanged.connect(self.update_display)
        toolbar.addWidget(self.chk_inverse)

        btn_adjust = QtWidgets.QPushButton("Adjust Image")
        btn_adjust.clicked.connect(self._open_adjust_dialog)
        toolbar.addWidget(btn_adjust)

        self.btn_measure = QtWidgets.QPushButton("Measure Distance")
        self.btn_measure.setCheckable(True)
        self.btn_measure.clicked.connect(self._toggle_line_measurement)
        toolbar.addWidget(self.btn_measure)

        btn_clear_meas = QtWidgets.QPushButton("Clear Measurements")
        btn_clear_meas.clicked.connect(self._clear_measurements)
        toolbar.addWidget(btn_clear_meas)

        btn_history = QtWidgets.QPushButton("Measurement History")
        btn_history.clicked.connect(self._show_measurement_history)
        toolbar.addWidget(btn_history)

        toolbar.addStretch()
        layout.addLayout(toolbar)

        self.glw = pg.GraphicsLayoutWidget()
        self.glw.ci.setContentsMargins(0, 0, 0, 0)
        if hasattr(self.glw.ci, "layout"):
            self.glw.ci.layout.setContentsMargins(0, 0, 0, 0)
            self.glw.ci.layout.setSpacing(0)

        layout.addWidget(self.glw)

        self.plot = self.glw.addPlot()
        self.img_fft = pg.ImageItem(axisOrder="row-major")
        colormap = pg.colormap.get("magma")
        self.img_fft.setLookupTable(colormap.getLookupTable())
        self.plot.addItem(self.img_fft)

        self.plot.hideAxis("bottom")
        self.plot.hideAxis("left")
        self.plot.hideAxis("top")
        self.plot.hideAxis("right")
        self.plot.hideButtons()
        self.plot.setMenuEnabled(False)
        self.plot.invertY(True)
        if hasattr(self.plot.vb, "setPadding"):
            self.plot.vb.setPadding(0.0)

        # FFT / diffraction view: show a reciprocal-space scale bar
        # using the parent image's spatial units (e.g. m, nm, Å) and
        # render them as m⁻¹, nm⁻¹, or Å⁻¹ as appropriate.
        self.scale_bar = DynamicScaleBar(self.plot.vb, units=self.freq_axis_base_unit)
        self.scale_bar.reciprocal = True

        self.line_tool = LineDrawingTool(self.plot, self._on_line_drawn)

    def _compute_fft(self):
        current_region_id = id(self.region)
        if (
            self._last_region_id == current_region_id
            and self._magnitude_spectrum is not None
        ):
            return

        self._last_region_id = current_region_id

        self._magnitude_spectrum, self._nyq_x, self._nyq_y = utils.compute_fft(
            self.region, self.scale_x, self.scale_y
        )

        window = np.hanning(self.region.shape[0])[:, None] * np.hanning(
            self.region.shape[1]
        )[None, :]
        windowed = self.region * window
        self._fft_complex = np.fft.fftshift(np.fft.fft2(windowed))

        self._inverse_fft_cache = None

    def update_display(self):
        if self._magnitude_spectrum is None:
            return

        if self.chk_inverse.isChecked():
            if self._inverse_fft_cache is None:
                self._inverse_fft_cache = utils.compute_inverse_fft(self._fft_complex)
            display_data = self._inverse_fft_cache
        else:
            display_data = self._magnitude_spectrum

        if self.display_min is None or self.display_max is None:
            finite = display_data[np.isfinite(display_data)]
            if finite.size > 0:
                self.display_min = float(finite.min())
                self.display_max = float(finite.max())
            else:
                self.display_min = 0.0
                self.display_max = 1.0

        adjusted = utils.apply_intensity_transform(
            display_data,
            self.display_min,
            self.display_max,
            self.display_gamma,
        )
        if adjusted is None:
            return

        self.img_fft.setImage(adjusted, autoLevels=False, levels=(0.0, 1.0))
        self.img_fft.setRect(
            QtCore.QRectF(-self._nyq_x, -self._nyq_y, 2 * self._nyq_x, 2 * self._nyq_y)
        )

    def setup_keyboard_shortcuts(self):
        delete_shortcut = QtGui.QShortcut(QtGui.QKeySequence.Delete, self)
        delete_shortcut.activated.connect(self._delete_selected_measurement)

        backspace_shortcut = QtGui.QShortcut(
            QtGui.QKeySequence(QtCore.Qt.Key_Backspace), self
        )
        backspace_shortcut.activated.connect(self._delete_selected_measurement)

        escape_shortcut = QtGui.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key_Escape), self)
        escape_shortcut.activated.connect(self._exit_measure_mode)

    def _exit_measure_mode(self):
        if self.btn_measure is not None and self.btn_measure.isChecked():
            self.btn_measure.setChecked(False)

        if self.line_tool is not None:
            self.line_tool.disable()
        if self.btn_measure is not None:
            self.btn_measure.setStyleSheet("")

    def _toggle_line_measurement(self):
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

        dx_freq = float(p2[0] - p1[0])
        dy_freq = float(p2[1] - p1[1])
        dist_freq = float(np.hypot(dx_freq, dy_freq))

        if self._nyq_x and self._nyq_y and self.region is not None:
            px_scale_x = (2.0 * float(self._nyq_x)) / float(self.region.shape[1])
            px_scale_y = (2.0 * float(self._nyq_y)) / float(self.region.shape[0])
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
            "scale_x": float(self.scale_x) if self.scale_x else 1.0,
            "scale_y": float(self.scale_y) if self.scale_y else 1.0,
        }

        if dist_freq != 0:
            result["d_spacing"] = utils.calculate_d_spacing(dist_freq)

        line = pg.PlotDataItem(
            [p1[0], p2[0]], [p1[1], p2[1]], pen=DRAWN_LINE_PEN
        )
        self.plot.addItem(line)

        label_text = self._format_measurement_label(result, measurement_id)
        mid_x = (p1[0] + p2[0]) / 2.0
        mid_y = (p1[1] + p2[1]) / 2.0

        text_item = MeasurementLabel(label_text, anchor=(0, 0), fill=LABEL_BRUSH_COLOR)
        text_item.setPos(mid_x, mid_y)
        text_item.sigLabelClicked.connect(self._on_measurement_label_clicked)
        self.plot.addItem(text_item)

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
            self.plot.removeItem(line_item)
            self.plot.removeItem(text_item)
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
        self.plot.removeItem(line_item)
        self.plot.removeItem(text_item)

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

    def _open_adjust_dialog(self):
        if self._magnitude_spectrum is None:
            return

        if self._tone_dialog is not None and self._tone_dialog.isVisible():
            self._tone_dialog.raise_()
            self._tone_dialog.activateWindow()
            return

        if self.chk_inverse.isChecked():
            if self._inverse_fft_cache is None:
                self._inverse_fft_cache = utils.compute_inverse_fft(self._fft_complex)
            source = self._inverse_fft_cache
        else:
            source = self._magnitude_spectrum

        initial_min = self.display_min
        initial_max = self.display_max
        initial_gamma = self.display_gamma

        def on_params_changed(min_val: float, max_val: float, gamma: float):
            self.display_min = float(min_val)
            self.display_max = float(max_val)
            self.display_gamma = float(gamma)
            self.update_display()

        dialog = ToneCurveDialog(
            source,
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
            self.update_display()

        def handle_finished(_result):
            self._tone_dialog = None

        dialog.rejected.connect(handle_rejected)
        dialog.finished.connect(handle_finished)

        dialog.setModal(False)
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()
