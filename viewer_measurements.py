# SPDX-License-Identifier: GPL-2.0-only
# Copyright (C) 2026 Cooper Stuntz
# See LICENSE for full license terms.

"""Measurement behavior controller for image-viewer windows."""

from __future__ import annotations

from typing import Any, Dict, Optional, Protocol, Tuple

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets

import unit_utils
import utils
from dialogs import MeasurementHistoryWindow, LineProfileWindow
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
    data: Any
    measurement_items: list[tuple[int, pg.PlotDataItem, pg.TextItem]]
    profile_measurement_count: int
    profile_measurement_items: dict[int, dict[str, Any]]
    selected_measurement_index: int | None
    measurement_history_window: MeasurementHistoryWindow | None
    freq_axis_base_unit: str

    def _open_calibration_dialog(self, state: dict[str, Any]) -> None: ...
    def _on_measurement_label_clicked(self, label: pg.TextItem) -> None: ...
    def _on_measurement_drawing_state_changed(self, is_drawing: bool) -> None: ...


class MeasurementController:
    """Owns measurement and line-draw interaction state for a viewer."""

    def __init__(self, viewer: _MeasurementControllerOwner, logger: _LoggerLike):
        self.viewer = viewer
        self.logger = logger

    def exit_measure_mode(self) -> None:
        viewer = self.viewer
        self.logger.debug("Exiting measurement mode (line_draw_mode=%s)", viewer._line_draw_mode)
        if viewer._line_draw_mode == "calibration":
            viewer._line_draw_mode = "measurement"
            viewer._on_calibration_pixels_selected = None
            state = dict(viewer._calibration_dialog_state or {})
            viewer._calibration_dialog_state = None
            if state:
                self.logger.debug("Restoring calibration dialog with pending state")
                QtCore.QTimer.singleShot(0, lambda: viewer._open_calibration_dialog(state))

        if viewer.btn_measure is not None and viewer.btn_measure.isChecked():
            viewer.btn_measure.setChecked(False)

        if viewer.line_tool is not None:
            viewer.line_tool.disable()
        viewer._on_measurement_drawing_state_changed(False)
        if viewer.btn_measure is not None:
            viewer.btn_measure.setStyleSheet("")
        self.logger.debug("Measurement mode exited")

    def toggle_line_measurement(self) -> None:
        viewer = self.viewer
        if viewer.line_tool is None:
            self.logger.debug("Ignoring measurement toggle: line tool is unavailable")
            return

        if viewer.btn_measure is not None and viewer.btn_measure.isChecked():
            viewer._line_draw_mode = "measurement"
            viewer._on_calibration_pixels_selected = None
            viewer.line_tool.enable()
            viewer._on_measurement_drawing_state_changed(False)
            viewer.btn_measure.setStyleSheet("background-color: #4caf50; color: white;")
            self.logger.debug("Measurement mode entered (distance)")
            self.logger.debug("Measurement tool enabled")
        else:
            viewer.line_tool.disable()
            viewer._on_measurement_drawing_state_changed(False)
            if viewer.btn_measure is not None:
                viewer.btn_measure.setStyleSheet("")
            self.logger.debug("Measurement mode exited (toggle)")
            self.logger.debug("Measurement tool disabled")

    def start_profile_measurement(self) -> None:
        viewer = self.viewer
        if viewer.line_tool is None:
            self.logger.debug("Profile measurement requested but line tool is unavailable")
            QtWidgets.QMessageBox.information(
                viewer,
                "Profile",
                "Line drawing tool is not available.",
            )
            return

        viewer._line_draw_mode = "profile"
        viewer._on_calibration_pixels_selected = None
        if viewer.btn_measure is not None and not viewer.btn_measure.isChecked():
            viewer.btn_measure.setChecked(True)
        if viewer.btn_measure is not None:
            viewer.btn_measure.setStyleSheet("background-color: #4caf50; color: white;")
        viewer.line_tool.enable()
        viewer._on_measurement_drawing_state_changed(False)
        self.logger.debug("Measurement mode entered (profile)")
        self.logger.debug("Profile measurement mode enabled")

    def on_line_drawn(self, p1: Tuple[float, float], p2: Tuple[float, float]) -> None:
        viewer = self.viewer
        self.logger.debug("Line drawn: mode=%s p1=%s p2=%s", viewer._line_draw_mode, p1, p2)
        if viewer._line_draw_mode == "profile":
            self._add_profile_measurement(p1, p2)
            return

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
                self.logger.debug("Calibration distance selected: %.6g px", dist_px)
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
            self.logger.debug(
                "FFT measurement #%s stored: distance_physical=%.6g distance_pixels=%.6g",
                measurement_id,
                dist_freq,
                dist_px,
            )
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
        for idx, (_measurement_id, _line_item, text_item) in enumerate(viewer.measurement_items):
            if text_item is label:
                selected_index = idx
                break

        viewer.selected_measurement_index = selected_index
        self.logger.debug("Measurement label selected: index=%s", selected_index)

        for idx, (_measurement_id, _line_item, text_item) in enumerate(viewer.measurement_items):
            if idx == selected_index:
                self.set_label_fill(text_item, pg.mkBrush(255, 200, 0, 255))
            else:
                self.set_label_fill(text_item, LABEL_BRUSH_COLOR)

    def clear_measurements(self) -> None:
        viewer = self.viewer
        distance_count = len(viewer.measurement_items)
        profile_count = len(viewer.profile_measurement_items)
        self.logger.debug(
            "Clearing all measurements from viewer: distances=%s profiles=%s",
            distance_count,
            profile_count,
        )
        for _measurement_id, line_item, text_item in viewer.measurement_items:
            viewer.p1.removeItem(line_item)
            viewer.p1.removeItem(text_item)
        viewer.measurement_items.clear()

        for profile_id in list(viewer.profile_measurement_items.keys()):
            self._delete_profile_measurement(profile_id)

        viewer.selected_measurement_index = None

        if viewer.measurement_history_window is not None:
            viewer.measurement_history_window.clear_all(notify_parent=False)
        self.logger.debug("Viewer measurements cleared")

    def clear_measurements_from_history(self) -> None:
        viewer = self.viewer
        distance_count = len(viewer.measurement_items)
        profile_count = len(viewer.profile_measurement_items)
        self.logger.debug(
            "Clearing viewer overlays from history window request: distances=%s profiles=%s",
            distance_count,
            profile_count,
        )
        for _measurement_id, line_item, text_item in viewer.measurement_items:
            viewer.p1.removeItem(line_item)
            viewer.p1.removeItem(text_item)
        viewer.measurement_items.clear()

        for profile_id in list(viewer.profile_measurement_items.keys()):
            self._delete_profile_measurement(profile_id)

        viewer.selected_measurement_index = None
        self.logger.debug("Viewer overlays cleared from history request")

    def delete_selected_measurement(self) -> None:
        viewer = self.viewer
        if viewer.selected_measurement_index is None:
            self.logger.debug("Delete selected measurement ignored: nothing selected")
            return

        if not (0 <= viewer.selected_measurement_index < len(viewer.measurement_items)):
            self.logger.debug(
                "Delete selected measurement ignored: invalid index=%s count=%s",
                viewer.selected_measurement_index,
                len(viewer.measurement_items),
            )
            viewer.selected_measurement_index = None
            return

        deleted_index = viewer.selected_measurement_index
        measurement_id, line_item, text_item = viewer.measurement_items.pop(viewer.selected_measurement_index)
        viewer.p1.removeItem(line_item)
        viewer.p1.removeItem(text_item)

        viewer.selected_measurement_index = None
        for _measurement_id, _line_item, item in viewer.measurement_items:
            self.set_label_fill(item, LABEL_BRUSH_COLOR)
        self.logger.debug(
            "Deleted selected measurement: index=%s id=%s remaining=%s",
            deleted_index,
            measurement_id,
            len(viewer.measurement_items),
        )

    @staticmethod
    def set_label_fill(text_item: pg.TextItem, brush: pg.QtGui.QBrush) -> None:
        if hasattr(text_item, "setFill"):
            text_item.setFill(brush)
        elif hasattr(text_item, "setBrush"):
            text_item.setBrush(brush)

    def delete_measurement_by_label(self, label_text: str) -> None:
        viewer = self.viewer
        self.logger.debug("Delete measurement by label requested: %s", label_text)
        target_index = None
        for idx, (_measurement_id, _line_item, text_item) in enumerate(viewer.measurement_items):
            if text_item.toPlainText() == label_text:
                target_index = idx
                break

        if target_index is not None:
            viewer.selected_measurement_index = target_index
            self.delete_selected_measurement()
            return

        if label_text.startswith("P#"):
            profile_id_text = label_text.split(" ", 1)[0][2:]
            try:
                profile_id = int(profile_id_text)
            except Exception:
                profile_id = None

            if profile_id is not None:
                deleted = self._delete_profile_measurement(profile_id)
                if deleted:
                    self.logger.debug("Deleted profile by label: id=%s", profile_id)
                    return

        self.logger.debug("Delete measurement by label ignored: not found")

    def delete_measurement_by_history_id(self, entry_id: int, entry_type: str) -> None:
        viewer = self.viewer
        self.logger.debug(
            "Delete measurement by history id requested: id=%s type=%s",
            entry_id,
            entry_type,
        )
        if entry_type == "profile":
            deleted = self._delete_profile_measurement(entry_id)
            self.logger.debug("Delete profile by history id result: deleted=%s", deleted)
            return

        target_index = None
        for idx, (measurement_id, _line_item, _text_item) in enumerate(viewer.measurement_items):
            if measurement_id == entry_id:
                target_index = idx
                break

        if target_index is None:
            self.logger.debug("Delete distance by history id ignored: id not found")
            return

        viewer.selected_measurement_index = target_index
        self.delete_selected_measurement()

    def open_measurement_by_history_id(self, entry_id: int, entry_type: str) -> None:
        viewer = self.viewer
        self.logger.debug(
            "Open measurement by history id requested: id=%s type=%s",
            entry_id,
            entry_type,
        )
        if entry_type != "profile":
            self.logger.debug("Open measurement ignored: non-profile entry")
            return

        record = viewer.profile_measurement_items.get(entry_id)
        if record is None:
            self.logger.debug("Open profile ignored: id not found")
            return

        window = record.get("window")
        if isinstance(window, LineProfileWindow):
            try:
                if window.isVisible():
                    window.raise_()
                    window.activateWindow()
                    self.logger.debug("Focused open profile window: id=%s", entry_id)
                    return
                window.show()
                window.raise_()
                window.activateWindow()
                self.logger.debug("Reopened existing profile window: id=%s", entry_id)
                return
            except RuntimeError:
                self.logger.debug("Profile window object invalid, recreating: id=%s", entry_id)

        distances = record.get("distances")
        intensities = record.get("intensities")
        x_axis_label = str(record.get("x_axis_label", "Distance (px)"))
        title = str(record.get("title", f"Profile Measurement #{entry_id}"))
        if not isinstance(distances, np.ndarray) or not isinstance(intensities, np.ndarray):
            self.logger.debug("Open profile failed: profile data missing for id=%s", entry_id)
            return

        profile_window = LineProfileWindow(
            title,
            distances,
            intensities,
            x_axis_label=x_axis_label,
            parent=viewer,
        )
        profile_window.show()
        profile_window.raise_()
        profile_window.activateWindow()
        record["window"] = profile_window
        self.logger.debug("Recreated and opened profile window: id=%s", entry_id)

    def rename_measurement_by_history_id(self, entry_id: int, entry_type: str, new_text: str) -> None:
        viewer = self.viewer
        self.logger.debug(
            "Rename measurement by history id requested: id=%s type=%s new_text=%s",
            entry_id,
            entry_type,
            new_text,
        )

        if entry_type == "profile":
            record = viewer.profile_measurement_items.get(entry_id)
            if record is None:
                self.logger.debug("Rename profile ignored: id not found")
                return

            window = record.get("window")
            record["history_text"] = new_text
            profile_title = f"Profile Measurement #{entry_id} - {new_text}"
            record["title"] = profile_title
            text_item = record.get("text_item")
            if isinstance(text_item, MeasurementLabel):
                if hasattr(text_item, "setText"):
                    text_item.setText(new_text)
                elif hasattr(text_item, "setPlainText"):
                    text_item.setPlainText(new_text)
            if isinstance(window, LineProfileWindow):
                try:
                    window.setWindowTitle(profile_title)
                except RuntimeError:
                    record["window"] = None
            self.logger.debug("Renamed profile measurement: id=%s", entry_id)
            return

        for idx, (measurement_id, _line_item, text_item) in enumerate(viewer.measurement_items):
            if measurement_id != entry_id:
                continue

            if hasattr(text_item, "setText"):
                text_item.setText(new_text)
            elif hasattr(text_item, "setPlainText"):
                text_item.setPlainText(new_text)
            self.logger.debug("Renamed distance measurement: id=%s index=%s", entry_id, idx)
            return

        self.logger.debug("Rename distance ignored: id not found")

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
            self.logger.debug("Creating measurement history window")
            viewer.measurement_history_window = MeasurementHistoryWindow(viewer)

        history_window = viewer.measurement_history_window
        if history_window is None:
            self.logger.debug("Measurement history window unavailable after creation attempt")
            return

        history_window.show()
        history_window.raise_()
        history_window.activateWindow()
        self.logger.debug("Measurement history window shown")

    def add_to_measurement_history(self, measurement_text: str) -> None:
        self.add_to_measurement_history_entry(measurement_text)

    def add_to_measurement_history_entry(
        self,
        measurement_text: str,
        *,
        measurement_id: int | None = None,
        measurement_type: str = "distance",
    ) -> None:
        viewer = self.viewer
        if viewer.measurement_history_window is None:
            self.logger.debug("Creating measurement history window for first measurement")
            viewer.measurement_history_window = MeasurementHistoryWindow(viewer)

        history_window = viewer.measurement_history_window
        if history_window is None:
            self.logger.debug("Measurement history add skipped: history window unavailable")
            return

        history_window.add_measurement(
            measurement_text,
            measurement_id=measurement_id,
            measurement_type=measurement_type,
        )
        self.logger.debug(
            "Measurement added to history: id=%s type=%s text=%s",
            measurement_id,
            measurement_type,
            measurement_text,
        )

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

        text_item = MeasurementLabel(
            label_text,
            color=pg.mkColor(0, 0, 0),
            anchor=(0, 0),
            fill=LABEL_BRUSH_COLOR,
        )
        text_item.setPos(mid_x, mid_y)
        text_item.sigLabelClicked.connect(viewer._on_measurement_label_clicked)
        viewer.p1.addItem(text_item)

        viewer.measurement_items.append((measurement_id, line, text_item))
        self.logger.debug(
            "Measurement graphics added: id=%s midpoint=(%.6g, %.6g)",
            measurement_id,
            mid_x,
            mid_y,
        )
        self.add_to_measurement_history_entry(
            label_text,
            measurement_id=measurement_id,
            measurement_type="distance",
        )

    def _add_profile_measurement(self, p1: Tuple[float, float], p2: Tuple[float, float]) -> None:
        viewer = self.viewer
        profile = self._extract_line_profile(p1, p2)
        if profile is None:
            self.logger.debug("Profile measurement ignored: failed to extract profile")
            QtWidgets.QMessageBox.information(
                viewer,
                "Profile",
                "Could not sample a profile from this line.",
            )
            return

        distances, intensities, line_item, x_axis_label = profile
        viewer.profile_measurement_count += 1
        profile_id = viewer.profile_measurement_count
        line_label = f"P#{profile_id} profile ({len(distances)} pts)"
        profile_annotation = f"P#{profile_id} profile"

        mid_x = (p1[0] + p2[0]) / 2.0
        mid_y = (p1[1] + p2[1]) / 2.0
        text_item = MeasurementLabel(
            profile_annotation,
            color=pg.mkColor(0, 0, 0),
            anchor=(0, 0),
            fill=LABEL_BRUSH_COLOR,
        )
        text_item.setPos(mid_x, mid_y)
        viewer.p1.addItem(text_item)

        profile_window = LineProfileWindow(
            f"Profile Measurement #{profile_id}",
            distances,
            intensities,
            x_axis_label=x_axis_label,
            parent=viewer,
        )
        profile_window.show()
        profile_window.raise_()
        profile_window.activateWindow()

        viewer.profile_measurement_items[profile_id] = {
            "line_item": line_item,
            "text_item": text_item,
            "window": profile_window,
            "distances": distances,
            "intensities": intensities,
            "title": f"Profile Measurement #{profile_id}",
            "history_text": line_label,
            "x_axis_label": x_axis_label,
        }
        self.add_to_measurement_history_entry(
            line_label,
            measurement_id=profile_id,
            measurement_type="profile",
        )
        self.logger.debug(
            "Profile measurement added: id=%s samples=%s midpoint=(%.6g, %.6g)",
            profile_id,
            len(distances),
            mid_x,
            mid_y,
        )

    def _extract_line_profile(
        self,
        p1: Tuple[float, float],
        p2: Tuple[float, float],
    ) -> tuple[np.ndarray, np.ndarray, pg.PlotDataItem, str] | None:
        viewer = self.viewer
        if viewer.data is None or viewer.ax_x is None or viewer.ax_y is None:
            self.logger.debug("Profile extraction failed: missing image data or axis calibration")
            return None

        image = np.asarray(viewer.data, dtype=float)
        if image.ndim != 2 or image.shape[0] < 2 or image.shape[1] < 2:
            self.logger.debug("Profile extraction failed: invalid image shape=%s", getattr(image, "shape", None))
            return None

        if float(viewer.ax_x.scale) == 0.0 or float(viewer.ax_y.scale) == 0.0:
            self.logger.debug("Profile extraction failed: axis scale is zero")
            return None

        x0 = (float(p1[0]) - float(viewer.ax_x.offset)) / float(viewer.ax_x.scale)
        y0 = (float(p1[1]) - float(viewer.ax_y.offset)) / float(viewer.ax_y.scale)
        x1 = (float(p2[0]) - float(viewer.ax_x.offset)) / float(viewer.ax_x.scale)
        y1 = (float(p2[1]) - float(viewer.ax_y.offset)) / float(viewer.ax_y.scale)

        dx = x1 - x0
        dy = y1 - y0
        sample_count = int(max(abs(dx), abs(dy))) + 1
        sample_count = max(sample_count, 2)

        xs = np.linspace(x0, x1, sample_count)
        ys = np.linspace(y0, y1, sample_count)

        width = image.shape[1]
        height = image.shape[0]
        xs_clipped = np.clip(xs, 0.0, float(width - 1))
        ys_clipped = np.clip(ys, 0.0, float(height - 1))

        intensities = self._sample_image_bilinear(image, xs_clipped, ys_clipped)
        distances_px = np.sqrt((xs_clipped - xs_clipped[0]) ** 2 + (ys_clipped - ys_clipped[0]) ** 2)

        axis_unit = unit_utils.normalize_axis_unit(getattr(viewer.ax_x, "units", None), default="")
        if axis_unit:
            dx_units = (xs_clipped - xs_clipped[0]) * float(viewer.ax_x.scale)
            dy_units = (ys_clipped - ys_clipped[0]) * float(viewer.ax_y.scale)
            distances_world = np.sqrt(dx_units**2 + dy_units**2)

            reference_distance = float(distances_world[-1]) if distances_world.size else 0.0
            if reference_distance > 0 and np.isfinite(reference_distance):
                if unit_utils.is_reciprocal_unit(axis_unit):
                    scaled_ref, display_unit = utils.format_reciprocal_scale(reference_distance, axis_unit)
                else:
                    scaled_ref, display_unit = utils.format_si_scale(reference_distance, axis_unit)

                scale_factor = float(scaled_ref) / reference_distance if reference_distance != 0 else 1.0
                distances = distances_world * scale_factor
                x_axis_label = f"Distance ({display_unit})"
                self.logger.debug(
                    "Profile x-axis uses calibrated units with SI formatting: axis_unit=%s display_unit=%s ref_world=%.6g ref_scaled=%.6g",
                    axis_unit,
                    display_unit,
                    reference_distance,
                    float(scaled_ref),
                )
            else:
                distances = distances_world
                x_axis_label = f"Distance ({axis_unit})"
                self.logger.debug(
                    "Profile x-axis uses calibrated units without scaling (degenerate span): axis_unit=%s",
                    axis_unit,
                )
        else:
            distances = distances_px
            x_axis_label = "Distance (px)"
            self.logger.debug("Profile x-axis uses pixel units (uncalibrated axes)")

        line_item = pg.PlotDataItem([p1[0], p2[0]], [p1[1], p2[1]], pen=DRAWN_LINE_PEN)
        viewer.p1.addItem(line_item)
        self.logger.debug(
            "Profile extracted: p1_px=(%.3f, %.3f) p2_px=(%.3f, %.3f) samples=%s",
            x0,
            y0,
            x1,
            y1,
            sample_count,
        )
        return distances, intensities, line_item, x_axis_label

    @staticmethod
    def _sample_image_bilinear(image: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
        x0 = np.floor(xs).astype(np.int64)
        y0 = np.floor(ys).astype(np.int64)
        x1 = np.minimum(x0 + 1, image.shape[1] - 1)
        y1 = np.minimum(y0 + 1, image.shape[0] - 1)

        wx = xs - x0
        wy = ys - y0

        i00 = image[y0, x0]
        i10 = image[y0, x1]
        i01 = image[y1, x0]
        i11 = image[y1, x1]

        return (
            i00 * (1.0 - wx) * (1.0 - wy)
            + i10 * wx * (1.0 - wy)
            + i01 * (1.0 - wx) * wy
            + i11 * wx * wy
        )

    def _delete_profile_measurement(self, profile_id: int) -> bool:
        viewer = self.viewer
        record = viewer.profile_measurement_items.pop(profile_id, None)
        if record is None:
            return False

        line_item = record.get("line_item")
        if line_item is not None:
            viewer.p1.removeItem(line_item)

        text_item = record.get("text_item")
        if text_item is not None:
            viewer.p1.removeItem(text_item)

        profile_window = record.get("window")
        if isinstance(profile_window, LineProfileWindow):
            try:
                profile_window.close()
            except RuntimeError:
                pass
        self.logger.debug("Profile measurement deleted: id=%s", profile_id)
        return True
