# SPDX-License-Identifier: GPL-2.0-only
# Copyright (C) 2026 Cooper Stuntz
# See LICENSE for full license terms.

"""Measurement behavior controller for image-viewer windows."""

from __future__ import annotations

from typing import Any, Dict, Optional, Protocol, Tuple

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore

import utils
from dialogs import MeasurementHistoryWindow
from measurement_tools import MeasurementLabel, LABEL_BRUSH_COLOR, DRAWN_LINE_PEN


class _LoggerLike(Protocol):
    def debug(self, msg: str, *args) -> None: ...


class _MeasurementControllerOwner(Protocol):
    _line_draw_mode: str
    _on_calibration_pixels_selected: Any
    _calibration_dialog_state: dict[str, Any] | None
    btn_measure: QtWidgets.QPushButton | None
    line_tool: Any
    ax_x: Any
    ax_y: Any
    measurement_count: int
    view_mode: str
    _nyq_x: Any
    _nyq_y: Any
    _fft_region: Any
    is_reciprocal_space: bool
    p1: Any
    measurement_items: list[tuple[pg.PlotDataItem, pg.TextItem]]
    selected_measurement_index: int | None
    measurement_history_window: MeasurementHistoryWindow | None
    freq_axis_base_unit: str

    def _open_calibration_dialog(self, state: dict[str, Any]) -> None: ...
    def _on_measurement_label_clicked(self, label: pg.TextItem) -> None: ...


class MeasurementController:
    """Owns measurement and line-draw interaction state for a viewer."""

    def __init__(self, viewer: _MeasurementControllerOwner, logger: _LoggerLike):
        self.viewer = viewer
        self.logger = logger

    def exit_measure_mode(self) -> None:
        viewer = self.viewer
        if viewer._line_draw_mode == "calibration":
            viewer._line_draw_mode = "measurement"
            viewer._on_calibration_pixels_selected = None
            state = dict(viewer._calibration_dialog_state or {})
            viewer._calibration_dialog_state = None
            if state:
                QtCore.QTimer.singleShot(0, lambda: viewer._open_calibration_dialog(state))

        if viewer.btn_measure is not None and viewer.btn_measure.isChecked():
            viewer.btn_measure.setChecked(False)

        if viewer.line_tool is not None:
            viewer.line_tool.disable()
        if viewer.btn_measure is not None:
            viewer.btn_measure.setStyleSheet("")

    def toggle_line_measurement(self) -> None:
        viewer = self.viewer
        if viewer.line_tool is None:
            return

        if viewer.btn_measure is not None and viewer.btn_measure.isChecked():
            viewer._line_draw_mode = "measurement"
            viewer._on_calibration_pixels_selected = None
            viewer.line_tool.enable()
            viewer.btn_measure.setStyleSheet("background-color: #4caf50; color: white;")
        else:
            viewer.line_tool.disable()
            if viewer.btn_measure is not None:
                viewer.btn_measure.setStyleSheet("")

    def on_line_drawn(self, p1: Tuple[float, float], p2: Tuple[float, float]) -> None:
        viewer = self.viewer
        if viewer._line_draw_mode == "calibration":
            scale_x = float(viewer.ax_x.scale) if viewer.ax_x is not None else 1.0
            scale_y = float(viewer.ax_y.scale) if viewer.ax_y is not None else scale_x

            dx_phys = float(p2[0] - p1[0])
            dy_phys = float(p2[1] - p1[1])
            if scale_x != 0 and scale_y != 0:
                dx_px = dx_phys / scale_x
                dy_px = dy_phys / scale_y
                dist_px = float(np.hypot(dx_px, dy_px))
            else:
                dist_px = 0.0

            viewer._line_draw_mode = "measurement"
            if viewer.line_tool is not None:
                viewer.line_tool.disable()

            callback = viewer._on_calibration_pixels_selected
            viewer._on_calibration_pixels_selected = None
            if callback is not None:
                callback(dist_px)
            return

        viewer.measurement_count += 1
        measurement_id = viewer.measurement_count

        if viewer.view_mode == "fft":
            dx_freq = float(p2[0] - p1[0])
            dy_freq = float(p2[1] - p1[1])
            dist_freq = float(np.hypot(dx_freq, dy_freq))

            if viewer._nyq_x and viewer._nyq_y and viewer._fft_region is not None:
                px_scale_x = (2.0 * float(viewer._nyq_x)) / float(viewer._fft_region.shape[1])
                px_scale_y = (2.0 * float(viewer._nyq_y)) / float(viewer._fft_region.shape[0])
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
                "scale_x": float(viewer.ax_x.scale) if viewer.ax_x else 1.0,
                "scale_y": float(viewer.ax_y.scale) if viewer.ax_y else 1.0,
            }
            if dist_freq != 0:
                result["d_spacing"] = utils.calculate_d_spacing(dist_freq)

            self._add_measurement_graphics(p1, p2, result, measurement_id)
            return

        scale_x = float(viewer.ax_x.scale) if viewer.ax_x is not None else 1.0
        scale_y = float(viewer.ax_y.scale) if viewer.ax_y is not None else scale_x

        dx_phys = float(p2[0] - p1[0])
        dy_phys = float(p2[1] - p1[1])
        dist_phys = float(np.hypot(dx_phys, dy_phys))

        if scale_x != 0 and scale_y != 0:
            dx_px = dx_phys / scale_x
            dy_px = dy_phys / scale_y
            dist_px = float(np.hypot(dx_px, dy_px))
        else:
            dist_px = 0.0

        result = {
            "distance_pixels": dist_px,
            "distance_physical": dist_phys,
            "scale_x": scale_x,
            "scale_y": scale_y,
        }

        if viewer.is_reciprocal_space and dist_phys != 0:
            frequency = 1.0 / dist_phys
            result["d_spacing"] = utils.calculate_d_spacing(frequency)

        self._add_measurement_graphics(p1, p2, result, measurement_id)
        self.logger.debug(
            "Measurement #%s: p1=%s p2=%s distance_physical=%.6g distance_pixels=%.6g",
            measurement_id,
            p1,
            p2,
            dist_phys,
            dist_px,
        )

    def on_measurement_label_clicked(self, label: pg.TextItem) -> None:
        viewer = self.viewer
        selected_index = None
        for idx, (_line_item, text_item) in enumerate(viewer.measurement_items):
            if text_item is label:
                selected_index = idx
                break

        viewer.selected_measurement_index = selected_index

        for idx, (_line_item, text_item) in enumerate(viewer.measurement_items):
            if idx == selected_index:
                self.set_label_fill(text_item, pg.mkBrush(255, 200, 0, 255))
            else:
                self.set_label_fill(text_item, LABEL_BRUSH_COLOR)

    def clear_measurements(self) -> None:
        viewer = self.viewer
        for line_item, text_item in viewer.measurement_items:
            viewer.p1.removeItem(line_item)
            viewer.p1.removeItem(text_item)
        viewer.measurement_items.clear()
        viewer.selected_measurement_index = None

        if viewer.measurement_history_window is not None:
            viewer.measurement_history_window.clear_all(notify_parent=False)

    def clear_measurements_from_history(self) -> None:
        viewer = self.viewer
        for line_item, text_item in viewer.measurement_items:
            viewer.p1.removeItem(line_item)
            viewer.p1.removeItem(text_item)
        viewer.measurement_items.clear()
        viewer.selected_measurement_index = None

    def delete_selected_measurement(self) -> None:
        viewer = self.viewer
        if viewer.selected_measurement_index is None:
            return

        if not (0 <= viewer.selected_measurement_index < len(viewer.measurement_items)):
            viewer.selected_measurement_index = None
            return

        line_item, text_item = viewer.measurement_items.pop(viewer.selected_measurement_index)
        viewer.p1.removeItem(line_item)
        viewer.p1.removeItem(text_item)

        viewer.selected_measurement_index = None
        for _line_item, item in viewer.measurement_items:
            self.set_label_fill(item, LABEL_BRUSH_COLOR)

    @staticmethod
    def set_label_fill(text_item: pg.TextItem, brush: pg.QtGui.QBrush) -> None:
        if hasattr(text_item, "setFill"):
            text_item.setFill(brush)
        elif hasattr(text_item, "setBrush"):
            text_item.setBrush(brush)

    def delete_measurement_by_label(self, label_text: str) -> None:
        viewer = self.viewer
        target_index = None
        for idx, (_line_item, text_item) in enumerate(viewer.measurement_items):
            if text_item.toPlainText() == label_text:
                target_index = idx
                break

        if target_index is None:
            return

        viewer.selected_measurement_index = target_index
        self.delete_selected_measurement()

    def format_measurement_label(self, result: dict, measurement_id: Optional[int] = None) -> str:
        viewer = self.viewer
        if viewer.view_mode == "fft":
            scaled_dist, scaled_unit = utils.format_reciprocal_scale(
                result["distance_physical"], viewer.freq_axis_base_unit
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

        if viewer.is_reciprocal_space:
            scaled_dist, scaled_unit = utils.format_reciprocal_scale(
                result["distance_physical"], viewer.ax_x.units
            )
        else:
            scaled_dist, scaled_unit = utils.format_si_scale(
                result["distance_physical"], viewer.ax_x.units
            )
        prefix = f"#{measurement_id} " if measurement_id is not None else ""

        if viewer.is_reciprocal_space and "d_spacing" in result:
            return (
                f"{prefix}d: {result['d_spacing']:.4f} Å\n"
                f"({scaled_dist:.4f} {scaled_unit})"
            )
        return (
            f"{prefix}{scaled_dist:.4f} {scaled_unit}\n"
            f"({result['distance_pixels']:.1f} px)"
        )

    def show_measurement_history(self) -> None:
        viewer = self.viewer
        if viewer.measurement_history_window is None:
            viewer.measurement_history_window = MeasurementHistoryWindow(viewer)

        viewer.measurement_history_window.show()
        viewer.measurement_history_window.raise_()
        viewer.measurement_history_window.activateWindow()

    def add_to_measurement_history(self, measurement_text: str) -> None:
        viewer = self.viewer
        if viewer.measurement_history_window is None:
            viewer.measurement_history_window = MeasurementHistoryWindow(viewer)

        viewer.measurement_history_window.add_measurement(measurement_text)

    def _add_measurement_graphics(
        self,
        p1: Tuple[float, float],
        p2: Tuple[float, float],
        result: Dict[str, float],
        measurement_id: int,
    ) -> None:
        viewer = self.viewer
        line = pg.PlotDataItem([p1[0], p2[0]], [p1[1], p2[1]], pen=DRAWN_LINE_PEN)
        viewer.p1.addItem(line)

        label_text = self.format_measurement_label(result, measurement_id)
        mid_x = (p1[0] + p2[0]) / 2.0
        mid_y = (p1[1] + p2[1]) / 2.0

        text_item = MeasurementLabel(label_text, anchor=(0, 0), fill=LABEL_BRUSH_COLOR)
        text_item.setPos(mid_x, mid_y)
        text_item.sigLabelClicked.connect(viewer._on_measurement_label_clicked)
        viewer.p1.addItem(text_item)

        viewer.measurement_items.append((line, text_item))
        self.add_to_measurement_history(label_text)
