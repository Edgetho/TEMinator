# SPDX-License-Identifier: GPL-2.0-only
# Copyright (C) 2026 Cooper Stuntz
# See LICENSE for full license terms.

"""Measurement behavior controller for image-viewer windows."""

from __future__ import annotations

from typing import Any, Dict, Optional, Protocol, Tuple

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets

import line_profile_logic
import unit_utils
import utils
from dialogs import LineProfileWindow, MeasurementHistoryWindow
from measurement_tools import DRAWN_LINE_PEN, LABEL_BRUSH_COLOR, MeasurementLabel
from types_common import LoggerLike


class _MeasurementControllerOwner(Protocol):
    """Protocol for objects that own a MeasurementController."""

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
    img_orig: Any
    data: Any
    measurement_items: list[tuple[int, pg.PlotDataItem, pg.TextItem]]
    profile_measurement_count: int
    profile_measurement_items: dict[int, dict[str, Any]]
    selected_measurement_index: int | None
    measurement_history_window: MeasurementHistoryWindow | None
    freq_axis_base_unit: str
    edx_manager: Any

    def _prepare_for_measurement_input(self) -> None:
        """Prepare the viewer for measurement input."""
        ...

    def _open_calibration_dialog(self, state: dict[str, Any]) -> None:
        """Open the calibration dialog with saved state.

        Args:
            state: Persisted calibration-dialog state restored after distance picking.
        """
        ...

    def _on_measurement_drawing_state_changed(self, is_drawing: bool) -> None:
        """Handle measurement drawing mode state changes.

        Args:
            is_drawing: Boolean flag indicating whether drawing.

        """
        ...


class MeasurementController:
    """Owns measurement and line-draw interaction state for a viewer."""

    def __init__(self, viewer: _MeasurementControllerOwner, logger: LoggerLike):
        """Initialize the measurement controller.

        Args:
            viewer: The image viewer window that owns this controller.
            logger: Logger for debug output.
        """
        self.viewer = viewer
        self.logger = logger

    def exit_measure_mode(self) -> None:
        """Exit measurement mode and restore the line tool state.

        Disables the line drawing tool and updates the UI to reflect the change.
        If in calibration mode, restores the calibration dialog with pending state.
        """
        viewer = self.viewer
        self.logger.debug(
            "Exiting measurement mode (line_draw_mode=%s)", viewer._line_draw_mode
        )
        if viewer._line_draw_mode == "calibration":
            viewer._line_draw_mode = "measurement"
            viewer._on_calibration_pixels_selected = None
            state = dict(viewer._calibration_dialog_state or {})
            viewer._calibration_dialog_state = None
            if state:
                self.logger.debug("Restoring calibration dialog with pending state")
                QtCore.QTimer.singleShot(
                    0, lambda: viewer._open_calibration_dialog(state)
                )

        if viewer.btn_measure is not None and viewer.btn_measure.isChecked():
            viewer.btn_measure.setChecked(False)

        if viewer.line_tool is not None:
            viewer.line_tool.disable()
        viewer._on_measurement_drawing_state_changed(False)
        if viewer.btn_measure is not None:
            viewer.btn_measure.setStyleSheet("")
        self.logger.debug("Measurement mode exited")

    def toggle_line_measurement(self) -> None:
        """Toggle distance measurement mode on/off.

        Enables or disables the line-drawing tool for distance measurements.
        Updates the measurement mode button styling to indicate active state.
        """
        viewer = self.viewer
        if viewer.line_tool is None:
            self.logger.debug("Ignoring measurement toggle: line tool is unavailable")
            return

        if viewer.btn_measure is not None and viewer.btn_measure.isChecked():
            viewer._prepare_for_measurement_input()
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

    def start_distance_measurement(self) -> None:
        """Initiate a distance measurement operation.

        Activates the measurement mode button and enables the line drawing tool.
        """
        viewer = self.viewer
        if viewer.btn_measure is not None and not viewer.btn_measure.isChecked():
            viewer.btn_measure.setChecked(True)
        self.toggle_line_measurement()

    def start_profile_measurement(self) -> None:
        """Initiate a line profile measurement operation.

        Prepares the viewer for profile input and enables the line drawing tool.
        Shows an information dialog if the line tool is unavailable.
        """
        viewer = self.viewer
        if viewer.line_tool is None:
            self.logger.debug(
                "Profile measurement requested but line tool is unavailable"
            )
            QtWidgets.QMessageBox.information(
                viewer,
                "Profile",
                "Line drawing tool is not available.",
            )
            return

        viewer._prepare_for_measurement_input()
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
        """Handle a line being drawn for measurement.

        Processes the line based on the current drawing mode:
        - "profile": Adds a line profile measurement.
        - "calibration": Converts physical distance to pixel distance via callback.
        - "measurement": Adds a distance or FFT frequency measurement.

        Args:
            p1: Start point of the line (x, y).
            p2: End point of the line (x, y).
        """
        viewer = self.viewer
        self.logger.debug(
            "Line drawn: mode=%s p1=%s p2=%s", viewer._line_draw_mode, p1, p2
        )
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
                px_scale_x = (2.0 * float(viewer._nyq_x)) / float(
                    viewer._fft_region.shape[1]
                )
                px_scale_y = (2.0 * float(viewer._nyq_y)) / float(
                    viewer._fft_region.shape[0]
                )
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
        """Handle a measurement label being clicked.

        Selects the measurement and highlights its label with a yellow color.
        Other measurement labels are reset to their original color.

        Args:
            label: The TextItem label that was clicked.
        """
        viewer = self.viewer
        selected_index = None
        for idx, (_measurement_id, _line_item, text_item) in enumerate(
            viewer.measurement_items
        ):
            if text_item is label:
                selected_index = idx
                break

        viewer.selected_measurement_index = selected_index
        self.logger.debug("Measurement label selected: index=%s", selected_index)

        for idx, (_measurement_id, _line_item, text_item) in enumerate(
            viewer.measurement_items
        ):
            if idx == selected_index:
                self.set_label_fill(text_item, pg.mkBrush(255, 200, 0, 255))
            else:
                self.set_label_fill(text_item, LABEL_BRUSH_COLOR)

    def clear_measurements_from_history(self) -> None:
        """Remove all measurements from the viewer per history window request.

        Similar to clear_measurements but called from the measurement history window.
        Clears both distance and profile measurements from the display.
        """
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
        """Delete the currently selected measurement from the viewer.

        Removes the selected measurement's line and label from the plot.
        Does nothing if no measurement is selected or the index is invalid.
        """
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
        measurement_id, line_item, text_item = viewer.measurement_items.pop(
            viewer.selected_measurement_index
        )
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
        """Set the background fill color of a measurement label.

        Args:
            text_item: The TextItem label to modify.
            brush: The brush to use for the fill color.
        """
        if hasattr(text_item, "setFill"):
            text_item.setFill(brush)
        elif hasattr(text_item, "setBrush"):
            text_item.setBrush(brush)

    def delete_measurement_by_label(self, label_text: str) -> None:
        """Delete a measurement by its label text.

        Args:
            label_text: The label text to delete (e.g., "M#5" or "P#3").
        """
        viewer = self.viewer
        self.logger.debug("Delete measurement by label requested: %s", label_text)
        target_index = None
        for idx, (_measurement_id, _line_item, text_item) in enumerate(
            viewer.measurement_items
        ):
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
        """Delete a measurement by its history entry ID.

        Args:
            entry_id: The ID of the measurement from the history window.
            entry_type: Type of measurement ("distance" or "profile").
        """
        viewer = self.viewer
        self.logger.debug(
            "Delete measurement by history id requested: id=%s type=%s",
            entry_id,
            entry_type,
        )
        if entry_type == "profile":
            deleted = self._delete_profile_measurement(entry_id)
            self.logger.debug(
                "Delete profile by history id result: deleted=%s", deleted
            )
            return

        target_index = None
        for idx, (measurement_id, _line_item, _text_item) in enumerate(
            viewer.measurement_items
        ):
            if measurement_id == entry_id:
                target_index = idx
                break

        if target_index is None:
            self.logger.debug("Delete distance by history id ignored: id not found")
            return

        viewer.selected_measurement_index = target_index
        self.delete_selected_measurement()

    def open_measurement_by_history_id(self, entry_id: int, entry_type: str) -> None:
        """Open/display a measurement from the history window.

        Selects the measurement to display its profile plot (for profiles) or
        highlights the measurement on the main image (for distances).

        Args:
            entry_id: The ID of the measurement from the history window.
            entry_type: Type of measurement ("distance" or "profile").
        """
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
                self.logger.debug(
                    "Profile window object invalid, recreating: id=%s", entry_id
                )

        distances = record.get("distances")
        intensities = record.get("intensities")
        trace_colors = record.get("trace_colors")
        x_axis_label = str(record.get("x_axis_label", "Distance (px)"))
        title = str(record.get("title", f"Profile Measurement #{entry_id}"))
        if not isinstance(distances, np.ndarray):
            self.logger.debug(
                "Open profile failed: profile distances missing for id=%s", entry_id
            )
            return

        valid_intensities = isinstance(intensities, np.ndarray) or isinstance(
            intensities, dict
        )
        if not valid_intensities:
            self.logger.debug(
                "Open profile failed: profile data missing for id=%s", entry_id
            )
            return

        profile_window = LineProfileWindow(
            title,
            distances,
            intensities,
            x_axis_label=x_axis_label,
            trace_colors=trace_colors if isinstance(trace_colors, dict) else None,
            on_refresh=lambda pid=entry_id: self.refresh_profile_measurement(pid),
            parent=viewer,
        )
        profile_window.show()
        profile_window.raise_()
        profile_window.activateWindow()
        record["window"] = profile_window
        self.logger.debug("Recreated and opened profile window: id=%s", entry_id)

    def rename_measurement_by_history_id(
        self, entry_id: int, entry_type: str, new_text: str
    ) -> None:
        """Rename a measurement via the history window.

        Args:
            entry_id: The ID of the measurement from the history window.
            entry_type: Type of measurement ("distance" or "profile").
            new_text: The new measurement label text.
        """
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

        for idx, (measurement_id, _line_item, text_item) in enumerate(
            viewer.measurement_items
        ):
            if measurement_id != entry_id:
                continue

            if hasattr(text_item, "setText"):
                text_item.setText(new_text)
            elif hasattr(text_item, "setPlainText"):
                text_item.setPlainText(new_text)
            self.logger.debug(
                "Renamed distance measurement: id=%s index=%s", entry_id, idx
            )
            return

        self.logger.debug("Rename distance ignored: id not found")

    def format_measurement_label(
        self, result: dict, measurement_id: Optional[int] = None
    ) -> str:
        """Format a measurement result dictionary into a display label.

        Formats the label based on the current view mode (FFT vs image),
        including physical units and d-spacing if applicable.

        Args:
            result: Dictionary with measurement data (distance_physical, distance_pixels, scales, etc.).
            measurement_id: Optional ID number to include in the label prefix.

        Returns:
            Formatted label string with physical units and pixel distance.
        """
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
        """Display the measurement history window.

        Opens or focuses the history window that shows all measurements taken
        and allows reviewing, renaming, and deleting measurements.
        """
        viewer = self.viewer
        if viewer.measurement_history_window is None:
            self.logger.debug("Creating measurement history window")
            viewer.measurement_history_window = MeasurementHistoryWindow(viewer)

        history_window = viewer.measurement_history_window
        if history_window is None:
            self.logger.debug(
                "Measurement history window unavailable after creation attempt"
            )
            return

        history_window.show()
        history_window.raise_()
        history_window.activateWindow()
        self.logger.debug("Measurement history window shown")

    def add_to_measurement_history_entry(
        self,
        measurement_text: str,
        *,
        measurement_id: int | None = None,
        measurement_type: str = "distance",
    ) -> None:
        """Add a measurement entry to the history window.

        Creates the history window if it doesn't exist. Adds the measurement
        with its ID and type for later retrieval.

        Args:
            measurement_text: The formatted measurement label text.
            measurement_id: Optional ID number for the measurement.
            measurement_type: Type of measurement ("distance" or "profile").
        """
        viewer = self.viewer
        if viewer.measurement_history_window is None:
            self.logger.debug(
                "Creating measurement history window for first measurement"
            )
            viewer.measurement_history_window = MeasurementHistoryWindow(viewer)

        history_window = viewer.measurement_history_window
        if history_window is None:
            self.logger.debug(
                "Measurement history add skipped: history window unavailable"
            )
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
        """Add visual measurement line and label to the plot.

        Creates a PlotDataItem line between two points and a MeasurementLabel
        at the midpoint. Also adds the measurement to history.

        Args:
            p1: Start point of the measurement line (x, y).
            p2: End point of the measurement line (x, y).
            result: Dictionary with measurement data.
            measurement_id: The ID number for this measurement.
        """
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
        text_item.sigLabelClicked.connect(self.on_measurement_label_clicked)
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

    def _add_profile_measurement(
        self, p1: Tuple[float, float], p2: Tuple[float, float]
    ) -> None:
        """Add a line profile measurement between two points.

        Extracts intensities along the line, creates a plot window showing the
        profile, and adds a label to the main plot.

        Args:
            p1: Start point of the profile line (x, y).
            p2: End point of the profile line (x, y).
        """
        viewer = self.viewer
        initial_width = 0.0
        profile = self._extract_line_profile(p1, p2, integration_width_px=initial_width)
        if profile is None:
            self.logger.debug("Profile measurement ignored: failed to extract profile")
            QtWidgets.QMessageBox.information(
                viewer,
                "Profile",
                "Could not sample a profile from this line.",
            )
            return

        distances, intensities, line_item, x_axis_label, trace_colors = profile
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

        def on_width_changed(width: float) -> None:
            """Update profile when integration width changes."""
            self._update_profile_integration_width(profile_id, width)

        profile_window = LineProfileWindow(
            f"Profile Measurement #{profile_id}",
            distances,
            intensities,
            x_axis_label=x_axis_label,
            trace_colors=trace_colors,
            on_refresh=lambda pid=profile_id: self.refresh_profile_measurement(pid),
            on_integration_width_changed=on_width_changed,
            integration_width_px=initial_width,
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
            "trace_colors": trace_colors,
            "title": f"Profile Measurement #{profile_id}",
            "history_text": line_label,
            "x_axis_label": x_axis_label,
            "p1": (float(p1[0]), float(p1[1])),
            "p2": (float(p2[0]), float(p2[1])),
            "integration_width_px": initial_width,
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
        create_line_item: bool = True,
        integration_width_px: float = 0.0,
    ) -> tuple[
        np.ndarray,
        np.ndarray | Dict[str, np.ndarray],
        pg.PlotDataItem | None,
        str,
        Dict[str, Any] | None,
    ] | None:
        """Extract intensity values along a line in the image.

        Samples pixel intensities using bilinear interpolation along a line
        between two points. Returns the distances and intensities.

        Args:
            p1: Start point of the profile line (x, y).
            p2: End point of the profile line (x, y).
            create_line_item: Whether to create a line item on the plot.
            integration_width_px: Width in pixels for perpendicular integration.

        Returns:
            Tuple of (distances, intensities, line_item, x_axis_label),
            or None if extraction fails.
        """
        viewer = self.viewer
        if viewer.data is None:
            self.logger.debug("Profile extraction failed: missing image data")
            return None

        image = np.asarray(viewer.data, dtype=float)
        if image.ndim != 2 or image.shape[0] < 2 or image.shape[1] < 2:
            self.logger.debug(
                "Profile extraction failed: invalid image shape=%s",
                getattr(image, "shape", None),
            )
            return None

        width = image.shape[1]
        height = image.shape[0]

        rect_mapping = None
        image_item = getattr(viewer, "img_orig", None)
        if image_item is not None and hasattr(image_item, "rect"):
            try:
                image_rect = image_item.rect()
                rect_mapping = line_profile_logic.rect_mapping_from_rect(
                    image_width=width,
                    image_height=height,
                    rect_left=float(image_rect.left()),
                    rect_top=float(image_rect.top()),
                    rect_width=float(image_rect.width()),
                    rect_height=float(image_rect.height()),
                )
            except Exception:
                rect_mapping = None

        axis_calibration = None
        if viewer.ax_x is not None and viewer.ax_y is not None:
            try:
                axis_calibration = line_profile_logic.AxisCalibration(
                    scale_x=float(viewer.ax_x.scale),
                    scale_y=float(viewer.ax_y.scale),
                    offset_x=float(viewer.ax_x.offset),
                    offset_y=float(viewer.ax_y.offset),
                )
            except Exception:
                self.logger.debug(
                    "Profile extraction failed: invalid axis calibration values"
                )
                return None

        mapped = line_profile_logic.map_view_points_to_pixel(
            p1=p1,
            p2=p2,
            rect_mapping=rect_mapping,
            axis_calibration=axis_calibration,
        )
        if mapped is None:
            self.logger.debug("Profile extraction failed: axis scale is zero")
            return None
        x0, y0, x1, y1 = mapped

        if not line_profile_logic.endpoints_are_finite(x0, y0, x1, y1):
            self.logger.debug(
                "Profile extraction failed: non-finite profile endpoints x0=%s y0=%s x1=%s y1=%s",
                x0,
                y0,
                x1,
                y1,
            )
            return None

        x0, y0, x1, y1 = line_profile_logic.clamp_profile_endpoints(
            x0=x0,
            y0=y0,
            x1=x1,
            y1=y1,
            width=width,
            height=height,
        )

        sample_count = line_profile_logic.compute_sample_count(
            x0=x0,
            y0=y0,
            x1=x1,
            y1=y1,
            width=width,
            height=height,
        )

        xs_clipped, ys_clipped = line_profile_logic.sample_line_coordinates(
            x0=x0,
            y0=y0,
            x1=x1,
            y1=y1,
            sample_count=sample_count,
            width=width,
            height=height,
        )

        if integration_width_px > 0:
            intensities: np.ndarray | Dict[str, np.ndarray] = (
                line_profile_logic.sample_perpendicular_strip(
                    image=image,
                    xs=xs_clipped,
                    ys=ys_clipped,
                    integration_width_px=integration_width_px,
                    sampler=self._sample_image_bilinear,
                )
            )
        else:
            intensities = self._sample_image_bilinear(
                image, xs_clipped, ys_clipped
            )
        trace_colors: Dict[str, Any] | None = None

        if (
            viewer.view_mode == "image"
            and hasattr(viewer, "edx_manager")
            and viewer.edx_manager.get_has_edx_data()
            and viewer.edx_manager.active_elements
        ):
            active_profiles: Dict[str, np.ndarray] = {}
            active_colors: Dict[str, Any] = {}

            for element_name in viewer.edx_manager.elemental_maps.keys():
                if element_name not in viewer.edx_manager.active_elements:
                    continue

                map_data = viewer.edx_manager.elemental_maps.get(element_name)
                if map_data is None and hasattr(viewer.edx_manager, "_load_elemental_map"):
                    if viewer.edx_manager._load_elemental_map(element_name):
                        map_data = viewer.edx_manager.elemental_maps.get(element_name)

                if map_data is None:
                    continue

                map_image = np.asarray(map_data, dtype=float)
                if map_image.ndim != 2 or map_image.shape != image.shape:
                    continue

                if integration_width_px > 0:
                    active_profiles[element_name] = (
                        line_profile_logic.sample_perpendicular_strip(
                            image=map_image,
                            xs=xs_clipped,
                            ys=ys_clipped,
                            integration_width_px=integration_width_px,
                            sampler=self._sample_image_bilinear,
                        )
                    )
                else:
                    active_profiles[element_name] = self._sample_image_bilinear(
                        map_image, xs_clipped, ys_clipped
                    )
                active_colors[element_name] = viewer.edx_manager.element_colors.get(
                    element_name
                )

            if active_profiles:
                intensities = active_profiles
                trace_colors = active_colors
                self.logger.debug(
                    "Profile extracted with per-element traces: elements=%s samples=%s integration_width=%.1fpx",
                    list(active_profiles.keys()),
                    sample_count,
                    integration_width_px,
                )
        distances_px = line_profile_logic.pixel_distance_axis(xs_clipped, ys_clipped)

        axis_unit = line_profile_logic.resolve_profile_axis_unit(
            view_mode=viewer.view_mode,
            axis_units=getattr(getattr(viewer, "ax_x", None), "units", None),
            freq_axis_base_unit=getattr(viewer, "freq_axis_base_unit", None),
        )

        if axis_unit:
            distances_world = line_profile_logic.world_distance_axis(
                xs=xs_clipped,
                ys=ys_clipped,
                rect_mapping=rect_mapping,
                axis_calibration=axis_calibration,
            )

            distances, x_axis_label, display_unit, reference_distance, scaled_ref = (
                line_profile_logic.scaled_distance_axis(
                    distances_world=distances_world,
                    axis_unit=axis_unit,
                    view_mode=viewer.view_mode,
                    is_reciprocal_space=viewer.is_reciprocal_space,
                    format_reciprocal_scale=utils.format_reciprocal_scale,
                    format_si_scale=utils.format_si_scale,
                )
            )
            if (
                display_unit is not None
                and reference_distance is not None
                and scaled_ref is not None
            ):
                self.logger.debug(
                    "Profile x-axis uses calibrated units with SI formatting: axis_unit=%s display_unit=%s ref_world=%.6g ref_scaled=%.6g",
                    axis_unit,
                    display_unit,
                    reference_distance,
                    scaled_ref,
                )
            else:
                self.logger.debug(
                    "Profile x-axis uses calibrated units without scaling (degenerate span): axis_unit=%s",
                    axis_unit,
                )
        else:
            distances = distances_px
            x_axis_label = "Distance (px)"
            self.logger.debug("Profile x-axis uses pixel units (uncalibrated axes)")

        line_item: pg.PlotDataItem | None = None
        if create_line_item:
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
        return distances, intensities, line_item, x_axis_label, trace_colors

    def refresh_profile_measurement(self, profile_id: int) -> None:
        """Recompute and refresh a stored profile using current image-view state."""
        viewer = self.viewer
        record = viewer.profile_measurement_items.get(profile_id)
        if record is None:
            self.logger.debug("Profile refresh ignored: id=%s not found", profile_id)
            return

        p1 = record.get("p1")
        p2 = record.get("p2")
        if not (
            isinstance(p1, tuple)
            and isinstance(p2, tuple)
            and len(p1) == 2
            and len(p2) == 2
        ):
            self.logger.debug("Profile refresh ignored: missing endpoints for id=%s", profile_id)
            return

        integration_width = record.get("integration_width_px", 0.0)
        refreshed = self._extract_line_profile(
            (float(p1[0]), float(p1[1])),
            (float(p2[0]), float(p2[1])),
            create_line_item=False,
            integration_width_px=integration_width,
        )
        if refreshed is None:
            self.logger.debug("Profile refresh failed: extraction returned None for id=%s", profile_id)
            QtWidgets.QMessageBox.information(
                viewer,
                "Profile",
                "Could not refresh profile from the current view.",
            )
            return

        distances, intensities, _line_item, x_axis_label, trace_colors = refreshed
        record["distances"] = distances
        record["intensities"] = intensities
        record["x_axis_label"] = x_axis_label
        record["trace_colors"] = trace_colors

        history_text = f"P#{profile_id} profile ({len(distances)} pts)"
        record["history_text"] = history_text

        window = record.get("window")
        if isinstance(window, LineProfileWindow):
            try:
                window.update_profile_data(
                    distances=distances,
                    intensities=intensities,
                    x_axis_label=x_axis_label,
                    trace_colors=trace_colors,
                )
            except RuntimeError:
                record["window"] = None

        if viewer.measurement_history_window is not None:
            try:
                for row in range(viewer.measurement_history_window.list_widget.count()):
                    item = viewer.measurement_history_window.list_widget.item(row)
                    if item is None:
                        continue
                    metadata = item.data(QtCore.Qt.UserRole)
                    if (
                        isinstance(metadata, dict)
                        and metadata.get("id") == profile_id
                        and metadata.get("type") == "profile"
                    ):
                        metadata["text"] = history_text
                        item.setText(history_text)
                        item.setData(QtCore.Qt.UserRole, metadata)
                        if 0 <= row < len(viewer.measurement_history_window.measurements):
                            viewer.measurement_history_window.measurements[row] = metadata
                        break
            except Exception:
                pass

        self.logger.debug(
            "Profile measurement refreshed: id=%s samples=%s",
            profile_id,
            len(distances),
        )

    def _update_profile_integration_width(self, profile_id: int, width: float) -> None:
        """Update integration width and refresh profile data.

        Args:
            profile_id: ID of the profile to update.
            width: New integration width in pixels.
        """
        viewer = self.viewer
        record = viewer.profile_measurement_items.get(profile_id)
        if record is None:
            self.logger.debug("Integration width update ignored: id=%s not found", profile_id)
            return

        record["integration_width_px"] = width
        self.logger.debug("Integration width updated for profile id=%s: %.1fpx", profile_id, width)
        self.refresh_profile_measurement(profile_id)

    @staticmethod
    def _sample_image_bilinear(
        image: np.ndarray, xs: np.ndarray, ys: np.ndarray
    ) -> np.ndarray:
        """Sample image intensities using bilinear interpolation.

        Args:
            image: 2D image array.
            xs: X coordinates for sampling.
            ys: Y coordinates for sampling.

        Returns:
            Interpolated intensity values at the given coordinates.
        """
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
        """Delete a profile measurement by ID.

        Removes the profile line, label, and plot window from the viewer.

        Args:
            profile_id: The ID of the profile measurement to delete.

        Returns:
            True if successfully deleted; False if not found.
        """
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
