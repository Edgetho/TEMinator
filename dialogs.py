# SPDX-License-Identifier: GPL-2.0-only
# Copyright (C) 2026 Cooper Stuntz
# See LICENSE for full license terms.

"""Dialog and auxiliary windows (history, metadata, tone curve)."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List
from typing import Optional
from typing import Any

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets, QtCore, QtGui

from file_navigation import IMAGE_EXTENSIONS


logger = logging.getLogger(__name__)


class MeasurementHistoryWindow(QtWidgets.QMainWindow):
    """Window displaying measurement history."""

    def __init__(self, parent=None):
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

        rename_shortcut_return = QtGui.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key_Return), self)
        rename_shortcut_return.activated.connect(self._begin_inline_rename_selected)
        rename_shortcut_enter = QtGui.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key_Enter), self)
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
        btn_export = QtWidgets.QPushButton("Export as CSV")
        btn_export.clicked.connect(self.export_as_csv)
        btn_layout.addWidget(btn_clear)
        btn_layout.addWidget(btn_delete)
        btn_layout.addWidget(btn_rename)
        btn_layout.addWidget(btn_copy)
        btn_layout.addWidget(btn_export)
        btn_layout.addStretch()
        layout.addLayout(btn_layout)

    def _selected_item_with_metadata(self):
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

    def add_measurement(
        self,
        measurement_text: str,
        *,
        measurement_id: int | None = None,
        measurement_type: str = "distance",
    ):
        """Add a measurement to the history."""
        item = QtWidgets.QListWidgetItem(measurement_text)
        metadata = {
            "id": measurement_id,
            "type": measurement_type,
            "text": measurement_text,
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
        """Clear all measurements."""
        logger.debug(
            "MeasurementHistory clear all requested: count=%s notify_parent=%s",
            len(self.measurements),
            notify_parent,
        )
        self.list_widget.clear()
        self.measurements.clear()

        if notify_parent:
            parent = self.parent()
            if parent is not None and hasattr(parent, "clear_measurements_from_history"):
                parent.clear_measurements_from_history()
        logger.debug("MeasurementHistory cleared")

    def delete_selected(self):
        """Delete the currently selected measurement from history."""
        item, metadata, row = self._selected_item_with_metadata()
        if item is None or metadata is None or row < 0:
            logger.debug("MeasurementHistory delete selected ignored: no row selected")
            QtWidgets.QMessageBox.information(self, "Delete", "No measurement selected.")
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
        if (
            parent is not None
            and hasattr(parent, "delete_measurement_by_history_id")
            and entry_id is not None
        ):
            parent.delete_measurement_by_history_id(int(entry_id), str(entry_type))
        elif parent is not None and hasattr(parent, "delete_measurement_by_label"):
            parent.delete_measurement_by_label(text)
        logger.debug("MeasurementHistory delete selected complete: remaining=%s", len(self.measurements))

    def _begin_inline_rename_selected(self):
        """Start inline editing of selected history entry text."""
        if self.list_widget.state() == QtWidgets.QAbstractItemView.EditingState:
            return

        item, metadata, row = self._selected_item_with_metadata()
        if item is None or metadata is None or row < 0:
            logger.debug("MeasurementHistory inline rename ignored: no row selected")
            QtWidgets.QMessageBox.information(self, "Rename", "No measurement selected.")
            return

        self._editing_item = True
        self.list_widget.editItem(item)
        logger.debug("MeasurementHistory inline rename started: row=%s", row)

    def _on_history_item_changed(self, item: QtWidgets.QListWidgetItem) -> None:
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
        if (
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
        """Re-open or focus a measurement target (currently profile windows)."""
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
        if parent is not None and hasattr(parent, "open_measurement_by_history_id"):
            logger.debug("MeasurementHistory open requested: id=%s type=%s", entry_id, entry_type)
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
        """Export measurements to CSV file."""
        if not self.measurements:
            logger.debug("MeasurementHistory export requested with no measurements")
            QtWidgets.QMessageBox.warning(self, "No Data", "No measurements to export!")
            return

        file_path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export Measurements", "", "CSV Files (*.csv)"
        )

        if file_path:
            try:
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write("Measurement\n")
                    for measurement in self.measurements:
                        f.write(f"{measurement.get('text', '')}\n")
                logger.debug("MeasurementHistory exported CSV: path=%s count=%s", file_path, len(self.measurements))
                QtWidgets.QMessageBox.information(
                    self, "Success", f"Exported to {file_path}"
                )
            except Exception as e:  # pragma: no cover - I/O error path
                logger.debug("MeasurementHistory export failed: path=%s error=%s", file_path, str(e))
                QtWidgets.QMessageBox.critical(
                    self, "Error", f"Could not export: {str(e)}"
                )


class LineProfileWindow(QtWidgets.QMainWindow):
    """Window displaying intensity profile along a measured line."""

    def __init__(
        self,
        title: str,
        distances: np.ndarray,
        intensities: np.ndarray,
        x_axis_label: str = "Distance (px)",
        parent: Optional[QtWidgets.QWidget] = None,
    ):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(640, 420)

        central = QtWidgets.QWidget(self)
        self.setCentralWidget(central)
        layout = QtWidgets.QVBoxLayout(central)

        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground("w")
        self.plot_widget.showGrid(x=True, y=True, alpha=0.2)
        self.plot_widget.setLabel("bottom", x_axis_label)
        self.plot_widget.setLabel("left", "Intensity")
        self.plot_widget.plot(distances, intensities, pen=pg.mkPen("b", width=2))
        layout.addWidget(self.plot_widget)

        logger.debug(
            "LineProfileWindow created: title=%s points=%s x_axis=%s",
            title,
            len(distances),
            x_axis_label,
        )


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

        Either argument may be ``None``; in that case a short message is
        shown instead of JSON content.
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
        self._update_histogram()
        self._update_curve_and_handles()
        self._update_labels()
        self._emit_params()

    def _update_histogram(self):
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
        if self.on_params_changed is None or self._pending_params is None:
            return

        min_val, max_val, gamma = self._pending_params
        self.on_params_changed(min_val, max_val, gamma)

    # Event handlers ----------------------------------------------------

    def _on_min_max_changed(self):
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
        self.min_val = self.data_min
        self.max_val = self.data_max
        self.gamma = 1.0
        self._update_all()


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

        QtCore.QTimer.singleShot(0, self.filter_edit.setFocus)

    def _populate_files(self) -> None:
        try:
            self._all_files = sorted(
                f.name
                for f in self.directory.iterdir()
                if f.is_file() and (not IMAGE_EXTENSIONS or f.suffix.lower() in IMAGE_EXTENSIONS)
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
        from image_loader import open_image_file

        name = item.text()
        path = self.directory / name
        if path.is_file():
            open_image_file(str(path))
        self.accept()
