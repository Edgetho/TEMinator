"""Dialog and auxiliary windows (history, metadata, tone curve)."""
from __future__ import annotations

import json
from typing import Optional

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets, QtCore, QtGui


class MeasurementHistoryWindow(QtWidgets.QMainWindow):
    """Window displaying measurement history."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Measurement History")
        self.resize(500, 400)
        self.measurements = []  # Store all measurements

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QVBoxLayout(central)

        self.list_widget = QtWidgets.QListWidget()
        layout.addWidget(QtWidgets.QLabel("Measurements:"))
        layout.addWidget(self.list_widget)

        btn_layout = QtWidgets.QHBoxLayout()
        btn_clear = QtWidgets.QPushButton("Clear All")
        btn_clear.clicked.connect(self.clear_all)
        btn_delete = QtWidgets.QPushButton("Delete Selected")
        btn_delete.clicked.connect(self.delete_selected)
        btn_copy = QtWidgets.QPushButton("Copy Selected")
        btn_copy.clicked.connect(self.copy_selected)
        btn_export = QtWidgets.QPushButton("Export as CSV")
        btn_export.clicked.connect(self.export_as_csv)
        btn_layout.addWidget(btn_clear)
        btn_layout.addWidget(btn_delete)
        btn_layout.addWidget(btn_copy)
        btn_layout.addWidget(btn_export)
        btn_layout.addStretch()
        layout.addLayout(btn_layout)

    def add_measurement(self, measurement_text: str):
        """Add a measurement to the history."""
        self.list_widget.addItem(measurement_text)
        self.measurements.append(measurement_text)
        self.list_widget.scrollToBottom()

    def clear_all(self):
        """Clear all measurements."""
        self.list_widget.clear()
        self.measurements.clear()
        QtWidgets.QMessageBox.information(self, "Cleared", "All measurements cleared!")

    def delete_selected(self):
        """Delete the currently selected measurement from history."""
        row = self.list_widget.currentRow()
        if row < 0:
            QtWidgets.QMessageBox.information(self, "Delete", "No measurement selected.")
            return

        item = self.list_widget.takeItem(row)
        if item is None:
            return

        text = item.text()
        del item

        if 0 <= row < len(self.measurements):
            self.measurements.pop(row)

        parent = self.parent()
        if parent is not None and hasattr(parent, "delete_measurement_by_label"):
            parent.delete_measurement_by_label(text)

    def copy_selected(self):
        """Copy selected measurement to clipboard."""
        current = self.list_widget.currentItem()
        if current:
            QtWidgets.QApplication.clipboard().setText(current.text())
            QtWidgets.QMessageBox.information(
                self, "Copied", "Measurement copied to clipboard!"
            )

    def export_as_csv(self):
        """Export measurements to CSV file."""
        if not self.measurements:
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
                        f.write(f"{measurement}\n")
                QtWidgets.QMessageBox.information(
                    self, "Success", f"Exported to {file_path}"
                )
            except Exception as e:  # pragma: no cover - I/O error path
                QtWidgets.QMessageBox.critical(
                    self, "Error", f"Could not export: {str(e)}"
                )


class MetadataWindow(QtWidgets.QMainWindow):
    """Window displaying full image metadata extracted by HyperSpy."""

    def __init__(
        self,
        parent=None,
        title: str = "Image Metadata",
        metadata: Optional[dict] = None,
    ):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(600, 500)

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QVBoxLayout(central)

        self.text_edit = QtWidgets.QPlainTextEdit()
        self.text_edit.setReadOnly(True)
        layout.addWidget(self.text_edit)

        if metadata is not None:
            self.update_metadata(metadata)

    def update_metadata(self, metadata: dict):
        """Update displayed metadata."""
        try:
            text = json.dumps(metadata, indent=2, default=str)
        except TypeError:
            text = str(metadata)
        self.text_edit.setPlainText(text)


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
        if self.on_params_changed is not None and not self._building:
            self.on_params_changed(self.min_val, self.max_val, self.gamma)

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
