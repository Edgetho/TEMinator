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
from fft_viewer import FFTViewerWindow
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


class ImageViewerWindow(QtWidgets.QMainWindow):
    """Window for viewing and analyzing a single image with FFT boxes."""

    def __init__(self, file_path: str, signal=None, window_suffix: Optional[str] = None):
        super().__init__()
        self.setAcceptDrops(True)
        self.file_path = file_path
        self.signal = None
        self.data = None
        self.ax_x = None
        self.ax_y = None
        self.fft_boxes: List[pg.RectROI] = []
        self.fft_count = 0
        self.is_reciprocal_space = False
        self.line_tool: Optional[LineDrawingTool] = None
        self.fft_windows: List[FFTViewerWindow] = []
        self.fft_to_fft_window: Dict[pg.RectROI, FFTViewerWindow] = {}
        self.fft_box_meta: Dict[pg.RectROI, Dict[str, object]] = {}
        self.selected_fft_box: Optional[pg.RectROI] = None
        self.scale_bar: Optional[DynamicScaleBar] = None
        self.measurement_history_window: Optional[MeasurementHistoryWindow] = None
        self.metadata_window: Optional[MetadataWindow] = None
        self.measurement_count = 0
        self.btn_measure: Optional[QtWidgets.QPushButton] = None
        self.measurement_items: List[Tuple[pg.PlotDataItem, pg.TextItem]] = []
        self.selected_measurement_index: Optional[int] = None

        self.display_min: Optional[float] = None
        self.display_max: Optional[float] = None
        self.display_gamma: float = 1.0
        self._tone_dialog: Optional[ToneCurveDialog] = None

        if signal is not None:
            self._setup_from_signal(signal, window_suffix)
        else:
            self._load_and_setup()

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

        self.is_reciprocal_space = utils.is_diffraction_pattern(self.data)

        self._init_display_window()

        file_name = Path(self.file_path).stem
        if window_suffix:
            title = f"Image Viewer - {file_name} {window_suffix}"
        else:
            title = f"Image Viewer - {file_name}"
        self.setWindowTitle(title)

        self.setup_ui()
        self.resize(*DEFAULT_IMAGE_WINDOW_SIZE)

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

    @property
    def image_bounds(self) -> Tuple[float, float, float, float]:
        x_offset = self.ax_x.offset if self.ax_x else 0
        y_offset = self.ax_y.offset if self.ax_y else 0
        w = self.ax_x.size * self.ax_x.scale if self.ax_x else 1
        h = self.ax_y.size * self.ax_y.scale if self.ax_y else 1
        return x_offset, y_offset, w, h

    def _init_display_window(self):
        if self.data is None:
            self.display_min = 0.0
            self.display_max = 1.0
            self.display_gamma = 1.0
            return

        arr = np.asarray(self.data, dtype=float)
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

        toolbar = self._create_toolbar()
        main_layout.addLayout(toolbar)

        self.glw = pg.GraphicsLayoutWidget()
        self.glw.ci.setContentsMargins(0, 0, 0, 0)
        if hasattr(self.glw.ci, "layout"):
            self.glw.ci.layout.setContentsMargins(0, 0, 0, 0)
            self.glw.ci.layout.setSpacing(0)
        main_layout.addWidget(self.glw)

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
        self._update_image_display()

        x_offset, y_offset, w, h = self.image_bounds
        self.img_orig.setRect(QtCore.QRectF(x_offset, y_offset, w, h))

        if hasattr(self.p1.vb, "setPadding"):
            self.p1.vb.setPadding(0.0)
        self.p1.setXRange(x_offset, x_offset + w, padding=0)
        self.p1.setYRange(y_offset, y_offset + h, padding=0)

        if not self.is_reciprocal_space:
            units = self.ax_x.units if self.ax_x is not None and self.ax_x.units else "m"
            self.scale_bar = DynamicScaleBar(self.p1.vb, units=units)

        self.line_tool = LineDrawingTool(self.p1, self._on_line_drawn)

        self.setup_keyboard_shortcuts()

    def _update_image_display(self):
        if self.data is None or self.img_orig is None:
            return

        if self.display_min is None or self.display_max is None:
            self._init_display_window()

        adjusted = utils.apply_intensity_transform(
            self.data,
            self.display_min,
            self.display_max,
            self.display_gamma,
        )
        if adjusted is None:
            return

        self.img_orig.setImage(adjusted, autoLevels=False, levels=(0.0, 1.0))

    def setup_keyboard_shortcuts(self):
        delete_shortcut = QtGui.QShortcut(QtGui.QKeySequence.Delete, self)
        delete_shortcut.activated.connect(self._delete_selected_roi)

        backspace_shortcut = QtGui.QShortcut(
            QtGui.QKeySequence(QtCore.Qt.Key_Backspace), self
        )
        backspace_shortcut.activated.connect(self._delete_selected_roi)

        escape_shortcut = QtGui.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key_Escape), self)
        escape_shortcut.activated.connect(self._exit_measure_mode)

    def _open_adjust_dialog(self):
        if self.data is None:
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

        dialog = ToneCurveDialog(
            self.data,
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

    def _create_toolbar(self) -> QtWidgets.QHBoxLayout:
        layout = QtWidgets.QHBoxLayout()

        btn_add_fft = QtWidgets.QPushButton("Add New FFT Box")
        btn_add_fft.clicked.connect(self._add_new_fft)
        layout.addWidget(btn_add_fft)

        self.btn_measure = QtWidgets.QPushButton("Measure Distance")
        self.btn_measure.setCheckable(True)
        self.btn_measure.clicked.connect(self._toggle_line_measurement)
        layout.addWidget(self.btn_measure)

        btn_clear_meas = QtWidgets.QPushButton("Clear Measurements")
        btn_clear_meas.clicked.connect(self._clear_measurements)
        layout.addWidget(btn_clear_meas)

        btn_delete_selected = QtWidgets.QPushButton("Delete Selected")
        btn_delete_selected.clicked.connect(self._delete_selected_roi)
        layout.addWidget(btn_delete_selected)

        btn_metadata = QtWidgets.QPushButton("Image Metadata")
        btn_metadata.clicked.connect(self._show_metadata_window)
        layout.addWidget(btn_metadata)

        btn_history = QtWidgets.QPushButton("Measurement History")
        btn_history.clicked.connect(self._show_measurement_history)
        layout.addWidget(btn_history)

        btn_adjust = QtWidgets.QPushButton("Adjust Image")
        btn_adjust.clicked.connect(self._open_adjust_dialog)
        layout.addWidget(btn_adjust)

        layout.addStretch()
        return layout

    def _add_new_fft(self, x_offset=None, y_offset=None, w=None, h=None):
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
            fft_window.region = region
            fft_window._compute_fft()
            fft_window.update_display()
            fft_window.show()
            fft_window.raise_()
            fft_window.activateWindow()
        else:
            fft_name = f"FFT {fft_id}"
            parent_title = self.windowTitle().replace("Image Viewer - ", "")
            fft_window = FFTViewerWindow(
                self,
                region,
                self.ax_x.scale,
                self.ax_y.scale,
                self.ax_x.name,
                self.ax_x.units,
                self.ax_y.name,
                self.ax_y.units,
                fft_name,
                parent_title,
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

    def _show_metadata_window(self):
        if self.signal is None:
            QtWidgets.QMessageBox.information(
                self, "Metadata", "No metadata available for this image."
            )
            return

        metadata_dict = self._get_original_metadata_dict()
        if metadata_dict is None:
            metadata_dict = self.signal.metadata.as_dictionary()

        file_name = Path(self.file_path).name
        title = f"Metadata - {file_name}"

        if self.metadata_window is None:
            self.metadata_window = MetadataWindow(
                self, title=title, metadata=metadata_dict
            )
        else:
            self.metadata_window.setWindowTitle(title)
            self.metadata_window.update_metadata(metadata_dict)

        self.metadata_window.show()
        self.metadata_window.raise_()
        self.metadata_window.activateWindow()


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
