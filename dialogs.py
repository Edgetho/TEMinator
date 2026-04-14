# SPDX-License-Identifier: GPL-2.0-only
# Copyright (C) 2026 Cooper Stuntz
# See LICENSE for full license terms.

"""Dialog and auxiliary windows (history, metadata, tone curve)."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

from file_navigation import IMAGE_EXTENSIONS
from viewer_settings import (
    RESAMPLING_BALANCED,
    RESAMPLING_FAST,
    RESAMPLING_HIGH,
    RenderSettings,
)

logger = logging.getLogger(__name__)


class MeasurementHistoryWindow(QtWidgets.QMainWindow):
    """Window displaying measurement history."""

    def __init__(self, parent=None):
        """Initialize the measurement history window UI.

        Args:
            parent: Optional parent widget, typically the image viewer.
        """
        super().__init__(parent)
        self.setWindowTitle("Measurement History")
        self.resize(500, 400)
        self.measurements: list[dict[str, Any]] = []

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QVBoxLayout(central)

        self.list_widget = QtWidgets.QListWidget()
        self.list_widget.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        layout.addWidget(QtWidgets.QLabel("Measurements:"))
        layout.addWidget(self.list_widget)
        self.list_widget.itemDoubleClicked.connect(self._open_selected_measurement)
        self.list_widget.itemChanged.connect(self._on_history_item_changed)

        self._suppress_item_changed = False
        self._editing_item = False

        rename_shortcut_return = QtGui.QShortcut(
            QtGui.QKeySequence(QtCore.Qt.Key_Return), self
        )
        rename_shortcut_return.activated.connect(self._begin_inline_rename_selected)
        rename_shortcut_enter = QtGui.QShortcut(
            QtGui.QKeySequence(QtCore.Qt.Key_Enter), self
        )
        rename_shortcut_enter.activated.connect(self._begin_inline_rename_selected)

        btn_layout = QtWidgets.QHBoxLayout()
        btn_clear = QtWidgets.QPushButton("Clear All")
        btn_clear.clicked.connect(self.clear_all)
        btn_delete = QtWidgets.QPushButton("Delete Selected")
        btn_delete.clicked.connect(self.delete_selected)
        btn_rename = QtWidgets.QPushButton("Rename Selected")
        btn_rename.clicked.connect(self._begin_inline_rename_selected)
        btn_copy = QtWidgets.QPushButton("Copy Selected")
        btn_copy.clicked.connect(self.copy_selected)
        btn_export = QtWidgets.QPushButton("Export as XLSX")
        btn_export.clicked.connect(self.export_as_csv)
        btn_layout.addWidget(btn_clear)
        btn_layout.addWidget(btn_delete)
        btn_layout.addWidget(btn_rename)
        btn_layout.addWidget(btn_copy)
        btn_layout.addWidget(btn_export)
        btn_layout.addStretch()
        layout.addLayout(btn_layout)

    def _selected_item_with_metadata(self):
        """Return selected list item and parsed metadata.

        Returns:
            Tuple of (item, metadata_dict, row_index). Item/metadata are None when no selection exists.
        """
        row = self.list_widget.currentRow()
        if row < 0:
            return None, None, row

        item = self.list_widget.item(row)
        if item is None:
            return None, None, row

        metadata = item.data(QtCore.Qt.UserRole)
        if not isinstance(metadata, dict):
            metadata = {
                "id": None,
                "type": "distance",
                "text": item.text(),
            }
        return item, metadata, row

    def _measurement_controller(self):
        """Resolve the measurement controller from the parent window."""
        parent = self.parent()
        if parent is None:
            return None
        return getattr(parent, "measurements", None)

    def add_measurement(
        self,
        measurement_text: str,
        *,
        measurement_id: int | None = None,
        measurement_type: str = "distance",
        measurement_parent=None,
    ):
        """Add a measurement to the history.

        Args:
            measurement_text: Rendered measurement label text shown in history.
            measurement_id: Numeric identifier for the measurement history entry.
            measurement_type: Measurement category, typically distance or profile.
            measurement_parent: Optional measurement controller object used to
                resolve detailed profile data for export.
        """
        item = QtWidgets.QListWidgetItem(measurement_text)
        metadata = {
            "id": measurement_id,
            "type": measurement_type,
            "text": measurement_text,
            "parent": measurement_parent,
        }
        item.setFlags(item.flags() | QtCore.Qt.ItemIsEditable)
        item.setData(QtCore.Qt.UserRole, metadata)
        self._suppress_item_changed = True
        self.list_widget.addItem(item)
        self._suppress_item_changed = False
        self.measurements.append(metadata)
        self.list_widget.scrollToBottom()
        logger.debug(
            "MeasurementHistory add: count=%s id=%s type=%s text=%s",
            len(self.measurements),
            measurement_id,
            measurement_type,
            measurement_text,
        )

    def clear_all(self, _checked: bool = False, *, notify_parent: bool = True):
        """Clear all measurements.

        Args:
            _checked: Unused checkbox/button state from Qt signal signatures.
            notify_parent: If True, propagates clear actions to the parent measurement controller.
        """
        logger.debug(
            "MeasurementHistory clear all requested: count=%s notify_parent=%s",
            len(self.measurements),
            notify_parent,
        )
        self.list_widget.clear()
        self.measurements.clear()

        if notify_parent:
            parent = self.parent()
            measurements = self._measurement_controller()
            if measurements is not None and hasattr(
                measurements, "clear_measurements_from_history"
            ):
                measurements.clear_measurements_from_history()
            elif parent is not None and hasattr(
                parent, "clear_measurements_from_history"
            ):
                parent.clear_measurements_from_history()
        logger.debug("MeasurementHistory cleared")

    def delete_selected(self):
        """Delete the currently selected measurement from history."""
        item, metadata, row = self._selected_item_with_metadata()
        if item is None or metadata is None or row < 0:
            logger.debug("MeasurementHistory delete selected ignored: no row selected")
            QtWidgets.QMessageBox.information(
                self, "Delete", "No measurement selected."
            )
            return

        item = self.list_widget.takeItem(row)
        if item is None:
            return

        text = item.text()
        entry_id = metadata.get("id")
        entry_type = metadata.get("type", "distance")
        del item
        logger.debug(
            "MeasurementHistory deleting row=%s id=%s type=%s text=%s",
            row,
            entry_id,
            entry_type,
            text,
        )

        if 0 <= row < len(self.measurements):
            self.measurements.pop(row)

        parent = self.parent()
        measurements = self._measurement_controller()
        if (
            measurements is not None
            and entry_id is not None
            and hasattr(measurements, "delete_measurement_by_history_id")
        ):
            measurements.delete_measurement_by_history_id(
                int(entry_id), str(entry_type)
            )
        elif measurements is not None and hasattr(
            measurements, "delete_measurement_by_label"
        ):
            measurements.delete_measurement_by_label(text)
        elif (
            parent is not None
            and hasattr(parent, "delete_measurement_by_history_id")
            and entry_id is not None
        ):
            parent.delete_measurement_by_history_id(int(entry_id), str(entry_type))
        elif parent is not None and hasattr(parent, "delete_measurement_by_label"):
            parent.delete_measurement_by_label(text)
        logger.debug(
            "MeasurementHistory delete selected complete: remaining=%s",
            len(self.measurements),
        )

    def _begin_inline_rename_selected(self):
        """Start inline editing of selected history entry text."""
        if self.list_widget.state() == QtWidgets.QAbstractItemView.EditingState:
            return

        item, metadata, row = self._selected_item_with_metadata()
        if item is None or metadata is None or row < 0:
            logger.debug("MeasurementHistory inline rename ignored: no row selected")
            QtWidgets.QMessageBox.information(
                self, "Rename", "No measurement selected."
            )
            return

        self._editing_item = True
        self.list_widget.editItem(item)
        logger.debug("MeasurementHistory inline rename started: row=%s", row)

    def _on_history_item_changed(self, item: QtWidgets.QListWidgetItem) -> None:
        """Handle inline rename edits committed in the list widget.

        Args:
            item: The edited history list item.
        """
        if self._suppress_item_changed:
            return

        metadata = item.data(QtCore.Qt.UserRole)
        if not isinstance(metadata, dict):
            metadata = {
                "id": None,
                "type": "distance",
                "text": item.text(),
            }

        old_text = str(metadata.get("text", ""))
        new_text = item.text().strip()

        if not new_text:
            self._suppress_item_changed = True
            item.setText(old_text)
            self._suppress_item_changed = False
            logger.debug("MeasurementHistory inline rename rejected: empty text")
            return

        if new_text == old_text:
            self._editing_item = False
            return

        entry_id = metadata.get("id")
        entry_type = str(metadata.get("type", "distance"))

        parent = self.parent()
        measurements = self._measurement_controller()
        if (
            measurements is not None
            and entry_id is not None
            and hasattr(measurements, "rename_measurement_by_history_id")
        ):
            measurements.rename_measurement_by_history_id(
                int(entry_id), entry_type, new_text
            )
        elif (
            parent is not None
            and hasattr(parent, "rename_measurement_by_history_id")
            and entry_id is not None
        ):
            parent.rename_measurement_by_history_id(int(entry_id), entry_type, new_text)

        metadata["text"] = new_text
        self._suppress_item_changed = True
        item.setData(QtCore.Qt.UserRole, metadata)
        self._suppress_item_changed = False

        row = self.list_widget.row(item)
        if 0 <= row < len(self.measurements):
            self.measurements[row] = metadata

        self._editing_item = False
        logger.debug(
            "MeasurementHistory inline rename committed: row=%s id=%s type=%s old=%s new=%s",
            row,
            entry_id,
            entry_type,
            old_text,
            new_text,
        )

    def _open_selected_measurement(self, item: QtWidgets.QListWidgetItem):
        """Re-open or focus a measurement target (currently profile windows).

        Args:
            item: Selected list item carrying measurement metadata and display text.
        """
        metadata = item.data(QtCore.Qt.UserRole)
        if not isinstance(metadata, dict):
            logger.debug("MeasurementHistory open ignored: missing metadata")
            return

        entry_id = metadata.get("id")
        entry_type = str(metadata.get("type", "distance"))
        if entry_id is None:
            logger.debug("MeasurementHistory open ignored: missing entry id")
            return

        parent = self.parent()
        measurements = self._measurement_controller()
        if measurements is not None and hasattr(
            measurements, "open_measurement_by_history_id"
        ):
            logger.debug(
                "MeasurementHistory open requested: id=%s type=%s", entry_id, entry_type
            )
            measurements.open_measurement_by_history_id(int(entry_id), entry_type)
        elif parent is not None and hasattr(parent, "open_measurement_by_history_id"):
            logger.debug(
                "MeasurementHistory open requested: id=%s type=%s", entry_id, entry_type
            )
            parent.open_measurement_by_history_id(int(entry_id), entry_type)

    def copy_selected(self):
        """Copy selected measurement to clipboard."""
        current = self.list_widget.currentItem()
        if current:
            QtWidgets.QApplication.clipboard().setText(current.text())
            logger.debug("MeasurementHistory copied selected: %s", current.text())
            QtWidgets.QMessageBox.information(
                self, "Copied", "Measurement copied to clipboard!"
            )

    def export_as_csv(self):
        """Export measurements to XLSX workbook with one sheet per profile."""
        if not self.measurements:
            logger.debug("MeasurementHistory export requested with no measurements")
            QtWidgets.QMessageBox.warning(self, "No Data", "No measurements to export!")
            return

        file_path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export Measurements", "", "Excel Files (*.xlsx)"
        )

        if file_path:
            try:
                from openpyxl import Workbook

                workbook = Workbook()
                history_sheet = workbook.active
                history_sheet.title = "History"
                history_sheet.append(["Measurement"])
                for measurement in self.measurements:
                    history_sheet.append([str(measurement.get("text", ""))])

                profile_sheet_count = 0
                for measurement in self.measurements:
                    if str(measurement.get("type", "distance")) != "profile":
                        continue

                    profile_id = measurement.get("id")
                    sheet_name = self._build_profile_sheet_name(profile_id)
                    profile_sheet = workbook.create_sheet(title=sheet_name)
                    profile_sheet_count += 1

                    profile_record = self._resolve_profile_record(measurement)
                    if not isinstance(profile_record, dict):
                        profile_sheet.append(["note", "Profile data unavailable"])
                        profile_sheet.append(["profile_id", profile_id])
                        continue

                    title = str(profile_record.get("title", f"Profile Measurement #{profile_id}"))
                    x_axis_label = str(profile_record.get("x_axis_label", "Distance (px)"))
                    integration_width = profile_record.get("integration_width_px", 0.0)
                    p1 = profile_record.get("p1")
                    p2 = profile_record.get("p2")

                    profile_sheet.append(["profile_id", profile_id])
                    profile_sheet.append(["title", title])
                    profile_sheet.append(["x_axis_label", x_axis_label])
                    profile_sheet.append(["integration_width_px", integration_width])
                    profile_sheet.append(["p1", "" if p1 is None else str(p1)])
                    profile_sheet.append(["p2", "" if p2 is None else str(p2)])
                    profile_sheet.append([])

                    distances = np.asarray(profile_record.get("distances", []), dtype=float)
                    intensities = profile_record.get("intensities")
                    x_column_label = self._x_axis_column_name(x_axis_label)

                    if isinstance(intensities, dict):
                        trace_names = [str(name) for name in intensities.keys()]
                        trace_arrays = {
                            str(name): np.asarray(values, dtype=float)
                            for name, values in intensities.items()
                        }
                        profile_sheet.append([x_column_label, *trace_names])
                        row_count = max(
                            [len(distances)]
                            + [len(arr) for arr in trace_arrays.values()]
                        )
                        for idx in range(row_count):
                            row = [
                                float(distances[idx]) if idx < len(distances) else None,
                            ]
                            for trace_name in trace_names:
                                values = trace_arrays[trace_name]
                                row.append(float(values[idx]) if idx < len(values) else None)
                            profile_sheet.append(row)
                    else:
                        intensity_array = np.asarray(intensities, dtype=float)
                        profile_sheet.append([x_column_label, "Intensity"])
                        row_count = max(len(distances), len(intensity_array))
                        for idx in range(row_count):
                            row = [
                                float(distances[idx]) if idx < len(distances) else None,
                                float(intensity_array[idx]) if idx < len(intensity_array) else None,
                            ]
                            profile_sheet.append(row)

                if not file_path.lower().endswith(".xlsx"):
                    file_path = f"{file_path}.xlsx"
                workbook.save(file_path)

                logger.debug(
                    "MeasurementHistory exported workbook: path=%s count=%s profile_sheets=%s",
                    file_path,
                    len(self.measurements),
                    profile_sheet_count,
                )
                QtWidgets.QMessageBox.information(
                    self, "Success", f"Exported workbook to {file_path}"
                )
            except ImportError:
                QtWidgets.QMessageBox.critical(
                    self,
                    "Missing Dependency",
                    "openpyxl is required for XLSX export but is not installed.",
                )
            except Exception as e:  # pragma: no cover - I/O error path
                logger.debug(
                    "MeasurementHistory workbook export failed: path=%s error=%s",
                    file_path,
                    str(e),
                )
                QtWidgets.QMessageBox.critical(
                    self, "Error", f"Could not export workbook: {str(e)}"
                )

    def _resolve_profile_record(self, measurement: dict[str, Any]) -> dict[str, Any] | None:
        """Return stored profile record for a measurement history item."""
        measurement_parent = measurement.get("parent")
        profile_id = measurement.get("id")
        if measurement_parent is None or profile_id is None:
            logger.debug(
                "MeasurementHistory profile export missing parent/id: id=%s", profile_id
            )
            return None

        viewer = getattr(measurement_parent, "viewer", None)
        if viewer is None:
            logger.debug(
                "MeasurementHistory profile export missing viewer: id=%s", profile_id
            )
            return None

        profile_items = getattr(viewer, "profile_measurement_items", None)
        if not isinstance(profile_items, dict):
            logger.debug(
                "MeasurementHistory profile export missing profile mapping: id=%s",
                profile_id,
            )
            return None
        return profile_items.get(profile_id)

    @staticmethod
    def _build_profile_sheet_name(profile_id: Any) -> str:
        """Return a safe Excel worksheet name for profile data."""
        raw = f"Profile_{profile_id}"
        sanitized = re.sub(r"[\\/*?:\[\]]", "_", raw)
        return sanitized[:31] if sanitized else "Profile"

    @staticmethod
    def _x_axis_column_name(x_axis_label: str) -> str:
        """Derive concise column header from axis label."""
        unit_match = re.search(r"\(([^)]+)\)", x_axis_label)
        axis_lower = x_axis_label.strip().lower()
        axis_prefix = "angle" if axis_lower.startswith("angle") else "distance"
        if unit_match:
            return f"{axis_prefix}_{unit_match.group(1).strip()}"
        return axis_prefix


class LineProfileWindow(QtWidgets.QMainWindow):
    """Window displaying intensity profile along a measured line."""

    def __init__(
        self,
        title: str,
        distances: np.ndarray,
        intensities: np.ndarray | Dict[str, np.ndarray],
        x_axis_label: str = "Distance (px)",
        trace_colors: Optional[Dict[str, Any]] = None,
        on_refresh: Optional[Callable[[], None]] = None,
        on_integration_width_changed: Optional[Callable[[float], None]] = None,
        on_radial_length_changed: Optional[Callable[[float], None]] = None,
        on_azimuthal_span_changed: Optional[Callable[[float], None]] = None,
        integration_width_px: float = 0.0,
        radial_length_px: Optional[float] = None,
        azimuthal_span_deg: Optional[float] = None,
        parent: Optional[QtWidgets.QWidget] = None,
    ):
        """Initialize the line profile plotting window.

        Args:
            title: Window title.
            distances: Sample positions along the measured line.
            intensities: Intensity values at each sample position, or mapping of
                trace name -> intensity values.
            x_axis_label: Label for the X axis.
            trace_colors: Optional mapping of trace name to pyqtgraph-compatible color.
            on_refresh: Optional callback to refresh profile from current view state.
            on_integration_width_changed: Optional callback when integration width changes.
            on_radial_length_changed: Optional callback when radial length changes.
            on_azimuthal_span_changed: Optional callback when azimuthal span changes.
            integration_width_px: Initial integration width in pixels.
            radial_length_px: Optional initial radial profile length in pixels.
            azimuthal_span_deg: Optional initial azimuthal span in degrees.
            parent: Optional parent widget.
        """
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(640, 480)
        self._on_refresh = on_refresh
        self._on_integration_width_changed = on_integration_width_changed
        self._on_radial_length_changed = on_radial_length_changed
        self._on_azimuthal_span_changed = on_azimuthal_span_changed
        self.integration_width_px = integration_width_px
        self._pending_integration_width_px = float(integration_width_px)
        self._width_change_debounce_timer = QtCore.QTimer(self)
        self._width_change_debounce_timer.setSingleShot(True)
        self._width_change_debounce_timer.setInterval(250)
        self._width_change_debounce_timer.timeout.connect(self._emit_debounced_width_change)
        self.radial_length_px = radial_length_px
        self.azimuthal_span_deg = azimuthal_span_deg

        central = QtWidgets.QWidget(self)
        self.setCentralWidget(central)
        layout = QtWidgets.QVBoxLayout(central)

        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground("w")
        self.plot_widget.showGrid(x=True, y=True, alpha=0.2)
        self.plot_widget.setLabel("left", "Intensity")
        plot_item = self.plot_widget.getPlotItem()
        # Favor responsiveness for dense profile traces.
        plot_item.setClipToView(True)
        plot_item.setDownsampling(auto=True, mode="peak")

        self._render_profile(
            distances=distances,
            intensities=intensities,
            x_axis_label=x_axis_label,
            trace_colors=trace_colors,
        )

        control_row = QtWidgets.QHBoxLayout()

        integration_label = QtWidgets.QLabel("Integration Width (px):")
        control_row.addWidget(integration_label)

        self.integration_width_spinbox = QtWidgets.QDoubleSpinBox()
        self.integration_width_spinbox.setMinimum(0.0)
        self.integration_width_spinbox.setMaximum(100.0)
        self.integration_width_spinbox.setSingleStep(1.0)
        self.integration_width_spinbox.setValue(integration_width_px)
        self.integration_width_spinbox.setToolTip(
            "Width in pixels perpendicular to the profile line for integration"
        )
        self.integration_width_spinbox.valueChanged.connect(self._on_width_changed)
        control_row.addWidget(self.integration_width_spinbox)

        self.radial_length_spinbox: QtWidgets.QDoubleSpinBox | None = None
        if radial_length_px is not None:
            control_row.addWidget(QtWidgets.QLabel("Radial Length (px):"))
            self.radial_length_spinbox = QtWidgets.QDoubleSpinBox()
            self.radial_length_spinbox.setMinimum(1.0)
            self.radial_length_spinbox.setMaximum(5000.0)
            self.radial_length_spinbox.setSingleStep(5.0)
            self.radial_length_spinbox.setValue(float(radial_length_px))
            self.radial_length_spinbox.valueChanged.connect(self._on_radial_length_value_changed)
            control_row.addWidget(self.radial_length_spinbox)

        self.azimuthal_span_spinbox: QtWidgets.QDoubleSpinBox | None = None
        if azimuthal_span_deg is not None:
            control_row.addWidget(QtWidgets.QLabel("Azimuthal Span (deg):"))
            self.azimuthal_span_spinbox = QtWidgets.QDoubleSpinBox()
            self.azimuthal_span_spinbox.setMinimum(0.1)
            self.azimuthal_span_spinbox.setMaximum(180.0)
            self.azimuthal_span_spinbox.setSingleStep(0.5)
            self.azimuthal_span_spinbox.setValue(float(azimuthal_span_deg))
            self.azimuthal_span_spinbox.valueChanged.connect(self._on_azimuthal_span_value_changed)
            control_row.addWidget(self.azimuthal_span_spinbox)

        control_row.addStretch()

        button_row = QtWidgets.QHBoxLayout()
        button_row.addStretch()
        self.btn_refresh = QtWidgets.QPushButton("Refresh")
        self.btn_refresh.setToolTip(
            "Recompute this profile from the current image view"
        )
        self.btn_refresh.setEnabled(self._on_refresh is not None)
        self.btn_refresh.clicked.connect(self._refresh_requested)
        button_row.addWidget(self.btn_refresh)

        layout.addWidget(self.plot_widget)
        layout.addLayout(control_row)
        layout.addLayout(button_row)

        logger.debug(
            "LineProfileWindow created: title=%s points=%s x_axis=%s integration_width=%.1fpx",
            title,
            len(distances),
            x_axis_label,
            integration_width_px,
        )

    def _render_profile(
        self,
        distances: np.ndarray,
        intensities: np.ndarray | Dict[str, np.ndarray],
        x_axis_label: str,
        trace_colors: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Render one or more profile traces into the plot widget."""
        plot_item = self.plot_widget.getPlotItem()
        legend = plot_item.legend
        if legend is not None:
            plot_item.scene().removeItem(legend)
            plot_item.legend = None

        self.plot_widget.clear()
        self.plot_widget.setLabel("bottom", x_axis_label)
        self.plot_widget.setLabel("left", "Intensity")

        if isinstance(intensities, dict):
            if intensities:
                self.plot_widget.addLegend(offset=(10, 10))
            fallback_colors = ["r", "g", "b", "m", "c", "y", "k"]
            for idx, (trace_name, trace_values) in enumerate(intensities.items()):
                color = None
                if trace_colors is not None:
                    color = trace_colors.get(trace_name)
                if color is None:
                    color = fallback_colors[idx % len(fallback_colors)]
                self.plot_widget.plot(
                    distances,
                    trace_values,
                    pen=pg.mkPen(color, width=2),
                    name=str(trace_name),
                )
            return

        self.plot_widget.plot(distances, intensities, pen=pg.mkPen("b", width=2))

    def update_profile_data(
        self,
        distances: np.ndarray,
        intensities: np.ndarray | Dict[str, np.ndarray],
        x_axis_label: str,
        trace_colors: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Update existing window plot with freshly sampled profile data."""
        self._render_profile(
            distances=distances,
            intensities=intensities,
            x_axis_label=x_axis_label,
            trace_colors=trace_colors,
        )

    def _refresh_requested(self) -> None:
        """Invoke the refresh callback for this profile window."""
        if self._on_refresh is None:
            return
        self._on_refresh()

    def _on_width_changed(self, value: float) -> None:
        """Handle integration width spinbox value changes."""
        self.integration_width_px = float(value)
        self._pending_integration_width_px = float(value)
        self._width_change_debounce_timer.start()
        logger.debug("Integration width changed to %.1f px (debounced)", value)

    def _emit_debounced_width_change(self) -> None:
        """Emit integration-width callback after interaction settles."""
        if self._on_integration_width_changed is None:
            return
        self._on_integration_width_changed(float(self._pending_integration_width_px))

    def _on_radial_length_value_changed(self, value: float) -> None:
        """Handle radial length updates for linked peak-collection profiles."""
        self.radial_length_px = float(value)
        if self._on_radial_length_changed is not None:
            self._on_radial_length_changed(float(value))

    def _on_azimuthal_span_value_changed(self, value: float) -> None:
        """Handle azimuthal span updates for linked peak-collection profiles."""
        self.azimuthal_span_deg = float(value)
        if self._on_azimuthal_span_changed is not None:
            self._on_azimuthal_span_changed(float(value))

    def set_shared_parameters(
        self,
        *,
        integration_width_px: Optional[float] = None,
        radial_length_px: Optional[float] = None,
        azimuthal_span_deg: Optional[float] = None,
    ) -> None:
        """Synchronize optional shared control widgets without recursive callbacks."""
        if integration_width_px is not None:
            self.integration_width_px = float(integration_width_px)
            self.integration_width_spinbox.blockSignals(True)
            self.integration_width_spinbox.setValue(float(integration_width_px))
            self.integration_width_spinbox.blockSignals(False)

        if radial_length_px is not None and self.radial_length_spinbox is not None:
            self.radial_length_px = float(radial_length_px)
            self.radial_length_spinbox.blockSignals(True)
            self.radial_length_spinbox.setValue(float(radial_length_px))
            self.radial_length_spinbox.blockSignals(False)

        if azimuthal_span_deg is not None and self.azimuthal_span_spinbox is not None:
            self.azimuthal_span_deg = float(azimuthal_span_deg)
            self.azimuthal_span_spinbox.blockSignals(True)
            self.azimuthal_span_spinbox.setValue(float(azimuthal_span_deg))
            self.azimuthal_span_spinbox.blockSignals(False)


class MetadataWindow(QtWidgets.QMainWindow):
    """Window displaying both raw and cleaned HyperSpy metadata.

    The left tab shows the "raw" metadata (typically the original
    ``Signal.original_metadata`` / reader dictionary), while the right
    tab shows HyperSpy's cleaned ``Signal.metadata.as_dictionary()``
    view. Callers may provide either or both; missing inputs are shown
    as a short explanatory message.
    """

    def __init__(
        self,
        parent=None,
        title: str = "Image Metadata",
        raw_metadata: Optional[dict] = None,
        cleaned_metadata: Optional[dict] = None,
    ):
        """Initialize metadata viewer window with raw and cleaned tabs.

        Args:
            parent: Optional parent widget.
            title: Window title text.
            raw_metadata: Original/raw metadata dictionary.
            cleaned_metadata: Cleaned HyperSpy metadata dictionary.
        """
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(800, 600)

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QVBoxLayout(central)

        self.tabs = QtWidgets.QTabWidget()
        layout.addWidget(self.tabs)

        self.raw_edit = QtWidgets.QPlainTextEdit()
        self.raw_edit.setReadOnly(True)
        self.cleaned_edit = QtWidgets.QPlainTextEdit()
        self.cleaned_edit.setReadOnly(True)

        self.tabs.addTab(self.raw_edit, "Raw original metadata")
        self.tabs.addTab(self.cleaned_edit, "HyperSpy metadata")

        # Populate initial content if provided
        self.update_metadata(raw_metadata, cleaned_metadata)

    def _format_metadata(self, metadata: Optional[dict], fallback_message: str) -> str:
        """Serialize metadata dictionary into pretty text.

        Args:
            metadata: Metadata dictionary to format.
            fallback_message: Message used when metadata is unavailable.

        Returns:
            JSON-like text for display in the metadata pane.
        """
        if metadata is None:
            return fallback_message
        try:
            return json.dumps(metadata, indent=2, default=str)
        except TypeError:
            return str(metadata)

    def update_metadata(
        self,
        raw_metadata: Optional[dict] = None,
        cleaned_metadata: Optional[dict] = None,
    ) -> None:
        """Update displayed metadata for both tabs.

        Args:
            raw_metadata: Reader-level metadata dictionary from the original image file.
            cleaned_metadata: HyperSpy-normalized metadata dictionary for display.
        """

        raw_text = self._format_metadata(
            raw_metadata,
            "No original/raw metadata is available for this image.",
        )
        cleaned_text = self._format_metadata(
            cleaned_metadata,
            "No cleaned HyperSpy metadata dictionary is available for this image.",
        )

        self.raw_edit.setPlainText(raw_text)
        self.cleaned_edit.setPlainText(cleaned_text)


class ToneCurveDialog(QtWidgets.QDialog):
    """Interactive histogram and tone-curve adjustment dialog."""

    def __init__(
        self,
        image: np.ndarray,
        initial_min: Optional[float] = None,
        initial_max: Optional[float] = None,
        initial_gamma: float = 1.0,
        parent: Optional[QtWidgets.QWidget] = None,
        on_params_changed=None,
    ):
        """Initialize the tone-curve adjustment dialog.

        Args:
            image: Source image array used for histogram and preview mapping.
            initial_min: Initial black level.
            initial_max: Initial white level.
            initial_gamma: Initial gamma value.
            parent: Optional parent widget.
            on_params_changed: Optional callback invoked with (min, max, gamma).
        """
        super().__init__(parent)
        self.setWindowTitle("Adjust Image")
        self.resize(700, 500)

        self.image = np.asarray(image, dtype=float)
        self.on_params_changed = on_params_changed

        # Debounce parameter change callbacks to avoid excessive
        # full-image re-renders while the user is dragging handles.
        self._emit_delay_ms = 30  # ~33 fps worst case
        self._emit_timer = QtCore.QTimer(self)
        self._emit_timer.setSingleShot(True)
        self._emit_timer.timeout.connect(self._emit_params_now)
        self._pending_params = None

        finite_mask = np.isfinite(self.image)
        if np.any(finite_mask):
            finite_vals = self.image[finite_mask]
            self.data_min = float(finite_vals.min())
            self.data_max = float(finite_vals.max())
        else:
            self.data_min = 0.0
            self.data_max = 1.0

        if initial_min is None:
            initial_min = self.data_min
        if initial_max is None:
            initial_max = self.data_max

        self.min_val = float(initial_min)
        self.max_val = float(initial_max)
        self.gamma = float(initial_gamma) if initial_gamma > 0 else 1.0

        if self.max_val <= self.min_val:
            self.max_val = self.min_val + 1e-6

        self._building = True
        self._init_ui()
        self._update_all()
        self._building = False

    # UI construction ---------------------------------------------------

    def _init_ui(self):
        """Build dialog widgets, plot items, and signal bindings."""
        layout = QtWidgets.QVBoxLayout(self)

        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground("w")
        self.plot_widget.showGrid(x=True, y=True, alpha=0.2)
        self.plot_widget.setLabel("bottom", "Input intensity")
        self.plot_widget.setLabel("left", "Displayed value")

        plot_item = self.plot_widget.getPlotItem()
        plot_item.setMouseEnabled(x=False, y=False)
        plot_item.setMenuEnabled(False)
        plot_item.hideButtons()
        layout.addWidget(self.plot_widget)

        self.hist_curve = self.plot_widget.plot()
        self.curve_item = self.plot_widget.plot(pen=pg.mkPen("m", width=2))

        self.min_line = pg.InfiniteLine(
            angle=90, movable=True, pen=pg.mkPen("c", width=2)
        )
        self.max_line = pg.InfiniteLine(
            angle=90, movable=True, pen=pg.mkPen("c", width=2)
        )
        self.plot_widget.addItem(self.min_line)
        self.plot_widget.addItem(self.max_line)
        self.min_line.sigPositionChanged.connect(self._on_min_max_changed)
        self.max_line.sigPositionChanged.connect(self._on_min_max_changed)

        from pyqtgraph import CircleROI

        span_x = max(self.data_max - self.data_min, 1e-6)
        size_x = span_x * 0.06 or 1.0
        handle_height = 0.12

        x_mid = (self.min_val + self.max_val) * 0.5
        y_mid = 0.5
        self.gamma_roi = CircleROI(
            [x_mid - size_x * 0.5, y_mid - handle_height * 0.5],
            [size_x, handle_height],
            pen=pg.mkPen("r", width=2),
        )
        if hasattr(self.gamma_roi, "setBrush"):
            self.gamma_roi.setBrush(pg.mkBrush(255, 0, 0, 120))
        self.gamma_roi.addScaleHandle((0.5, 0.5), (0.5, 0.5))
        self.gamma_roi.sigRegionChanged.connect(self._on_gamma_changed)
        self.plot_widget.addItem(self.gamma_roi)

        params_layout = QtWidgets.QHBoxLayout()

        self.lbl_min = QtWidgets.QLabel("Min (black):")
        self.edit_min = QtWidgets.QLineEdit()
        self.edit_min.setFixedWidth(120)
        self.edit_min.setAlignment(QtCore.Qt.AlignRight)
        self.edit_min.editingFinished.connect(self._on_min_text_changed)

        self.lbl_max = QtWidgets.QLabel("Max (white):")
        self.edit_max = QtWidgets.QLineEdit()
        self.edit_max.setFixedWidth(120)
        self.edit_max.setAlignment(QtCore.Qt.AlignRight)
        self.edit_max.editingFinished.connect(self._on_max_text_changed)

        self.lbl_gamma = QtWidgets.QLabel("Gamma:")
        self.edit_gamma = QtWidgets.QLineEdit()
        self.edit_gamma.setFixedWidth(120)
        self.edit_gamma.setAlignment(QtCore.Qt.AlignRight)
        self.edit_gamma.editingFinished.connect(self._on_gamma_text_changed)

        params_layout.addWidget(self.lbl_min)
        params_layout.addWidget(self.edit_min)
        params_layout.addSpacing(20)
        params_layout.addWidget(self.lbl_max)
        params_layout.addWidget(self.edit_max)
        params_layout.addSpacing(20)
        params_layout.addWidget(self.lbl_gamma)
        params_layout.addWidget(self.edit_gamma)
        params_layout.addStretch()
        layout.addLayout(params_layout)

        btn_layout = QtWidgets.QHBoxLayout()
        self.btn_reset = QtWidgets.QPushButton("Reset")
        self.btn_reset.clicked.connect(self._on_reset)
        btn_layout.addWidget(self.btn_reset)

        btn_layout.addStretch()

        btn_ok = QtWidgets.QPushButton("OK")
        btn_cancel = QtWidgets.QPushButton("Cancel")
        btn_ok.clicked.connect(self.accept)
        btn_cancel.clicked.connect(self.reject)
        btn_layout.addWidget(btn_ok)
        btn_layout.addWidget(btn_cancel)
        layout.addLayout(btn_layout)

    # Helpers -----------------------------------------------------------

    def _update_all(self):
        """Refresh histogram, curve visuals, labels, and callback state."""
        self._update_histogram()
        self._update_curve_and_handles()
        self._update_labels()
        self._emit_params()

    def _update_histogram(self):
        """Compute and draw normalized image histogram data."""
        if self.image.size == 0:
            return
        finite = self.image[np.isfinite(self.image)]
        if finite.size == 0:
            return

        y, bin_edges = np.histogram(
            finite, bins=256, range=(self.data_min, self.data_max)
        )
        if y.max() > 0:
            y = y.astype(float) / float(y.max())

        x = bin_edges.astype(float)

        self.hist_curve.setData(
            x,
            y,
            stepMode=True,
            fillLevel=0.0,
            brush=(150, 150, 150, 80),
            pen=pg.mkPen(150, 150, 150, 180),
        )
        self.plot_widget.setXRange(self.data_min, self.data_max, padding=0.02)
        self.plot_widget.setYRange(0.0, 1.0, padding=0.02)

    def _update_curve_and_handles(self):
        """Recompute tone-curve line and reposition interactive controls."""
        if self.max_val <= self.min_val:
            self.max_val = self.min_val + 1e-6

        xs = np.linspace(self.data_min, self.data_max, 512)
        norm = (xs - self.min_val) / (self.max_val - self.min_val)
        norm = np.clip(norm, 0.0, 1.0)
        inv_gamma = 1.0 / max(self.gamma, 1e-6)
        ys = np.power(norm, inv_gamma)

        self.curve_item.setData(xs, ys)

        self.min_line.setPos(self.min_val)
        self.max_line.setPos(self.max_val)

        span_x = max(self.max_val - self.min_val, 1e-6)
        size_x = span_x * 0.06 or 1.0
        handle_height = 0.12

        x_mid = self.min_val + 0.5 * span_x
        norm_mid = 0.5
        y_mid = float(np.power(norm_mid, 1.0 / max(self.gamma, 1e-6)))

        self._building = True
        self.gamma_roi.setSize([size_x, handle_height])
        self.gamma_roi.setPos(x_mid - size_x * 0.5, y_mid - handle_height * 0.5)
        self._building = False

    def _update_labels(self):
        """Update parameter text fields from current numeric values."""
        self.edit_min.setText(f"{self.min_val:.6g}")
        self.edit_max.setText(f"{self.max_val:.6g}")
        self.edit_gamma.setText(f"{self.gamma:.6g}")

    def _emit_params(self):
        """Schedule a debounced callback with the latest parameters.

        This avoids triggering a full image re-render on every tiny drag
        step from the min/max lines or gamma handle while keeping the UI
        responsive.
        """

        if self.on_params_changed is None or self._building:
            return

        self._pending_params = (self.min_val, self.max_val, self.gamma)

        if self._emit_timer.isActive():
            self._emit_timer.stop()
        self._emit_timer.start(self._emit_delay_ms)

    def _emit_params_now(self):
        """Invoke the parameter-changed callback immediately."""
        if self.on_params_changed is None or self._pending_params is None:
            return

        min_val, max_val, gamma = self._pending_params
        self.on_params_changed(min_val, max_val, gamma)

    # Event handlers ----------------------------------------------------

    def _on_min_max_changed(self):
        """Handle drag updates of min/max guide lines."""
        if self._building:
            return

        self.min_val = float(self.min_line.value())
        self.max_val = float(self.max_line.value())

        if self.min_val < self.data_min:
            self.min_val = self.data_min
        if self.max_val > self.data_max:
            self.max_val = self.data_max
        if self.max_val <= self.min_val:
            self.max_val = self.min_val + 1e-6

        self._update_curve_and_handles()
        self._update_labels()
        self._emit_params()

    def _on_min_text_changed(self):
        """Handle edits to the minimum intensity text field."""
        text = self.edit_min.text().strip()
        try:
            value = float(text)
        except Exception:
            self._update_labels()
            return

        value = max(self.data_min, min(value, self.data_max))
        if value >= self.max_val:
            value = self.max_val - 1e-6

        self.min_val = value
        self._update_curve_and_handles()
        self._update_labels()
        self._emit_params()

    def _on_max_text_changed(self):
        """Handle edits to the maximum intensity text field."""
        text = self.edit_max.text().strip()
        try:
            value = float(text)
        except Exception:
            self._update_labels()
            return

        value = max(self.data_min, min(value, self.data_max))
        if value <= self.min_val:
            value = self.min_val + 1e-6

        self.max_val = value
        self._update_curve_and_handles()
        self._update_labels()
        self._emit_params()

    def _on_gamma_text_changed(self):
        """Handle edits to the gamma text field."""
        text = self.edit_gamma.text().strip()
        try:
            value = float(text)
        except Exception:
            self._update_labels()
            return

        if not np.isfinite(value) or value <= 0:
            value = 1.0

        self.gamma = float(value)
        self._update_curve_and_handles()
        self._update_labels()
        self._emit_params()

    def _on_gamma_changed(self):
        """Handle dragging of the gamma control point on the curve plot."""
        if self._building:
            return

        rect = (
            self.gamma_roi.parentBounds()
            if hasattr(self.gamma_roi, "parentBounds")
            else self.gamma_roi.boundingRect()
        )
        pos = self.gamma_roi.pos()
        cx = float(pos.x() + rect.width() * 0.5)
        cy = float(pos.y() + rect.height() * 0.5)

        span_x = max(self.max_val - self.min_val, 1e-6)
        x_n = (cx - self.min_val) / span_x
        x_n = float(np.clip(x_n, 1e-3, 1.0 - 1e-3))
        y_n = float(np.clip(cy, 1e-3, 1.0 - 1e-3))

        try:
            self.gamma = float(np.log(x_n) / np.log(y_n))
        except Exception:
            self.gamma = 1.0

        self._update_curve_and_handles()
        self._update_labels()
        self._emit_params()

    def _on_reset(self):
        """Reset min/max/gamma to full-range defaults."""
        self.min_val = self.data_min
        self.max_val = self.data_max
        self.gamma = 1.0
        self._update_all()


class DirectoryFuzzyOpenDialog(QtWidgets.QDialog):
    """Simple fuzzy finder over files in a directory."""

    def __init__(self, parent: Optional[QtWidgets.QWidget], directory: Path):
        """Initialize fuzzy file-open dialog for a directory.

        Args:
            parent: Optional parent widget.
            directory: Directory whose files are listed for fuzzy filtering.
        """
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

        QtCore.QTimer.singleShot(0, self.filter_edit.setFocus)

    def _populate_files(self) -> None:
        """Load candidate files from disk and refresh the list widget."""
        try:
            self._all_files = sorted(
                f.name
                for f in self.directory.iterdir()
                if f.is_file()
                and (not IMAGE_EXTENSIONS or f.suffix.lower() in IMAGE_EXTENSIONS)
            )
        except Exception:
            self._all_files = []

        self._update_list(self._all_files)

    def _update_list(self, names: List[str]) -> None:
        """Replace list widget contents with provided file names.

        Args:
            names: Candidate file names to display in the fuzzy-open list.
        """
        self.list_widget.clear()
        self.list_widget.addItems(names)

    @staticmethod
    def _fuzzy_match(pattern: str, text: str) -> bool:
        """Simple subsequence-based fuzzy matching.

        Args:
            pattern: Lowercased user search pattern used for subsequence matching.
            text: Current text value from editable UI controls.

        Returns:
            True when the pattern characters appear in order within the file name.
        """

        it = iter(text)
        return all(ch in it for ch in pattern)

    def _on_filter_changed(self, text: str) -> None:
        """Apply fuzzy filtering as the user types in the filter field.

        Args:
            text: User-facing text value for this operation.

        """
        pattern = text.strip().lower()
        if not pattern:
            self._update_list(self._all_files)
            if self._all_files:
                self.list_widget.setCurrentRow(0)
            return

        matches = [
            name for name in self._all_files if self._fuzzy_match(pattern, name.lower())
        ]
        self._update_list(matches)
        if matches:
            self.list_widget.setCurrentRow(0)

    def _on_return_pressed(self) -> None:
        """Open current match when Enter is pressed in the filter field."""
        current = self.list_widget.currentItem()
        if current is None and self.list_widget.count() > 0:
            current = self.list_widget.item(0)
        if current is not None:
            self._open_item(current)

    def _on_item_activated(self, item: QtWidgets.QListWidgetItem) -> None:  # type: ignore[override]
        """Open file when a list item is activated.

        Args:
            item: Selected list item carrying measurement metadata and display text.
        """
        self._open_item(item)

    def _open_item(self, item: QtWidgets.QListWidgetItem) -> None:
        """Open selected file in the image loader and close the dialog.

        Args:
            item: Selected list item carrying measurement metadata and display text.
        """
        from image_loader import open_image_file

        name = item.text()
        path = self.directory / name
        if path.is_file():
            open_image_file(str(path))
        self.accept()


class RenderSettingsDialog(QtWidgets.QDialog):
    """Dialog for display quality/performance rendering settings."""

    def __init__(self, parent: Optional[QtWidgets.QWidget], current: RenderSettings):
        """Initialize render settings selection dialog.

        Args:
            parent: Optional parent widget.
            current: Current render settings used to initialize controls.
        """
        super().__init__(parent)
        self.setWindowTitle("Parameters")
        self.resize(520, 220)

        layout = QtWidgets.QVBoxLayout(self)

        form = QtWidgets.QFormLayout()
        layout.addLayout(form)

        self.chk_hardware = QtWidgets.QCheckBox("Use hardware acceleration (OpenGL)")
        self.chk_hardware.setChecked(
            bool(current.get("use_hardware_acceleration", True))
        )
        form.addRow("Rendering backend:", self.chk_hardware)

        self.cmb_quality = QtWidgets.QComboBox()
        self.cmb_quality.addItem("Fast (nearest-neighbor)", RESAMPLING_FAST)
        self.cmb_quality.addItem("Balanced (linear + downsample)", RESAMPLING_BALANCED)
        self.cmb_quality.addItem("High quality (mipmap anti-aliasing)", RESAMPLING_HIGH)

        quality = (
            str(current.get("image_resampling_quality", RESAMPLING_HIGH))
            .strip()
            .lower()
        )
        index = max(0, self.cmb_quality.findData(quality))
        self.cmb_quality.setCurrentIndex(index)
        form.addRow("Image resampling:", self.cmb_quality)

        description = QtWidgets.QLabel(
            "High quality uses a multiscale area-averaged image pyramid to avoid aliasing while maintaining hardware-accelerated display."
        )
        description.setWordWrap(True)
        layout.addWidget(description)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel,
            parent=self,
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def selected_settings(self) -> RenderSettings:
        """Return settings selected in the dialog controls.

        Returns:
            Render-settings dictionary built from current dialog control values.
        """
        return {
            "use_hardware_acceleration": self.chk_hardware.isChecked(),
            "image_resampling_quality": str(
                self.cmb_quality.currentData() or RESAMPLING_HIGH
            ),
        }
