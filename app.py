"""Individual image viewer window with FFT analysis."""
import sys
import json
import numpy as np
import hyperspy.api as hs
import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets, QtCore, QtGui
from pathlib import Path
from typing import Optional, List, Tuple, Dict
import utils

# Qt signal compatibility (PyQt5 vs PySide)
Signal = getattr(QtCore, "pyqtSignal", getattr(QtCore, "Signal", None))

# Module-level constants
FFT_COLORS = ['r', 'g', 'b', 'y', 'c', 'm']
PREVIEW_LINE_PEN = pg.mkPen('y', width=2, style=QtCore.Qt.DashLine)
DRAWN_LINE_PEN = pg.mkPen('w', width=2)
LABEL_BRUSH_COLOR = pg.mkBrush(255, 255, 100, 220)  # Yellow background
DEFAULT_FFT_WINDOW_SIZE = (700, 700)
DEFAULT_IMAGE_WINDOW_SIZE = (1000, 900)
DEFAULT_MAIN_WINDOW_SIZE = (600, 400)


class ScaleBarItem(pg.GraphicsObject):
    """Microscopy-style scale bar for images."""
    
    def __init__(self, scale_per_pixel: float, units: str = "px"):
        super().__init__()
        self.scale_per_pixel = scale_per_pixel
        self.base_units = units
        self.bar_length_physical = 1.0  # Physical length in units
        self.display_unit = units
        self.setPos(0, 0)
        self.update_length()
        
    def update_length(self):
        """Update scale bar physical length using SI units."""
        # Choose a "nice" physical length so the bar is a reasonable
        # size on screen (targeting ~100 pixels long).
        reference = self.scale_per_pixel
        if reference <= 0 or not np.isfinite(reference):
            self.bar_length_physical = 1.0
            self.display_value = 1.0
            self.display_unit = self.base_units
            return

        # Physical length corresponding to ~100 pixels
        target_phys = reference * 100.0
        if target_phys <= 0 or not np.isfinite(target_phys):
            target_phys = reference

        magnitude = 10 ** np.floor(np.log10(target_phys))
        mantissa = target_phys / magnitude

        if mantissa < 1.5:
            nice = 1.0
        elif mantissa < 3.5:
            nice = 2.0
        elif mantissa < 7.5:
            nice = 5.0
        else:
            nice = 10.0

        # Store physical length in base units for geometry
        self.bar_length_physical = nice * magnitude

        # Use SI formatting only for the label text
        scaled_val, si_unit = utils.format_si_scale(self.bar_length_physical, self.base_units, precision=2)
        self.display_value = scaled_val
        self.display_unit = si_unit
        
    def set_scale(self, scale_per_pixel: float):
        """Update scale per pixel and recalculate bar length."""
        self.scale_per_pixel = scale_per_pixel
        self.update_length()
        self.update()
    
    def boundingRect(self):
        """Return bounding rect of scale bar."""
        if self.scale_per_pixel <= 0 or not np.isfinite(self.scale_per_pixel):
            bar_pixels = 100
        else:
            bar_pixels = self.bar_length_physical / self.scale_per_pixel
            # Clamp to avoid absurdly large rects
            bar_pixels = max(10, min(bar_pixels, 500))
        return QtCore.QRectF(0, 0, bar_pixels + 30, 40)
    
    def paint(self, p, *args):
        """Draw the scale bar."""
        if self.scale_per_pixel <= 0:
            return
        
        bar_pixels = self.bar_length_physical / self.scale_per_pixel
        # Guard against extreme values that can overflow the painter
        bar_pixels = max(10, min(bar_pixels, 500))
        
        # Draw horizontal line
        p.setPen(QtGui.QPen(QtCore.Qt.red, 2))
        p.drawLine(10, 15, 10 + bar_pixels, 15)
        
        # Draw end caps
        p.drawLine(10, 10, 10, 20)
        p.drawLine(10 + bar_pixels, 10, 10 + bar_pixels, 20)
        
        # Draw text label
        # display_value is the SI-scaled number for the label
        text = f"{self.display_value:.2g} {self.display_unit}"
        p.setPen(QtGui.QPen(QtCore.Qt.red))
        font = QtGui.QFont()
        font.setPointSize(8)
        p.setFont(font)
        p.drawText(10, 28, text)


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
        
        # Measurements list
        self.list_widget = QtWidgets.QListWidget()
        layout.addWidget(QtWidgets.QLabel("Measurements:"))
        layout.addWidget(self.list_widget)
        
        # Buttons
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

        # Keep internal list in sync
        if 0 <= row < len(self.measurements):
            self.measurements.pop(row)

        # Notify parent viewer so the annotation is removed from the image
        parent = self.parent()
        if parent is not None and hasattr(parent, "delete_measurement_by_label"):
            parent.delete_measurement_by_label(text)
    
    def copy_selected(self):
        """Copy selected measurement to clipboard."""
        current = self.list_widget.currentItem()
        if current:
            QtWidgets.QApplication.clipboard().setText(current.text())
            QtWidgets.QMessageBox.information(self, "Copied", "Measurement copied to clipboard!")
    
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
                with open(file_path, 'w') as f:
                    f.write("Measurement\n")
                    for measurement in self.measurements:
                        f.write(f"{measurement}\n")
                QtWidgets.QMessageBox.information(self, "Success", f"Exported to {file_path}")
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "Error", f"Could not export: {str(e)}")


class MetadataWindow(QtWidgets.QMainWindow):
    """Window displaying full image metadata extracted by HyperSpy."""

    def __init__(self, parent=None, title: str = "Image Metadata", metadata: Optional[dict] = None):
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
            # Fallback: simple string representation
            text = str(metadata)
        self.text_edit.setPlainText(text)


class FFTViewerWindow(QtWidgets.QMainWindow):
    """Separate window displaying FFT for a specific FFT box."""
    
    def __init__(self, parent, region: np.ndarray, scale_x: float, scale_y: float, 
                 ax_x_name: str, ax_x_units: str, ax_y_name: str, ax_y_units: str, 
                 fft_name: str, parent_name: str = ""):
        super().__init__()
        
        # Set title
        self._update_title(fft_name, parent_name)
        self.resize(*DEFAULT_FFT_WINDOW_SIZE)
        
        # Store parameters
        self.scale_x = scale_x
        self.scale_y = scale_y
        self.ax_x_name = ax_x_name
        self.ax_x_units = ax_x_units
        self.ax_y_name = ax_y_name
        self.ax_y_units = ax_y_units
        self.region = region
        
        # Cache for FFT computations
        self._magnitude_spectrum = None
        self._fft_complex = None
        self._inverse_fft_cache = None
        self._nyq_x = None
        self._nyq_y = None
        self._last_region_id = None  # Track if region changed
        
        self.setup_ui()
        self._compute_fft()
        self.update_display()
        
    def _update_title(self, fft_name: str, parent_name: str):
        """Update window title based on image and FFT name."""
        title = f"FFT - {parent_name} - {fft_name}" if parent_name else f"FFT - {fft_name}"
        self.setWindowTitle(title)
        
    def setup_ui(self):
        """Setup UI with FFT display and controls."""
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QVBoxLayout(central)
        
        # Toolbar
        toolbar = QtWidgets.QHBoxLayout()
        self.chk_inverse = QtWidgets.QCheckBox("Show Inverse FFT")
        self.chk_inverse.stateChanged.connect(self.update_display)
        toolbar.addWidget(self.chk_inverse)
        toolbar.addStretch()
        layout.addLayout(toolbar)
        
        # Plot
        self.glw = pg.GraphicsLayoutWidget()
        layout.addWidget(self.glw)
        
        self.plot = self.glw.addPlot(title="2D FFT")
        self.img_fft = pg.ImageItem(axisOrder='row-major')
        colormap = pg.colormap.get('magma')
        self.img_fft.setLookupTable(colormap.getLookupTable())
        self.plot.addItem(self.img_fft)
        
    def _compute_fft(self):
        """Compute FFT lazily only if region has changed."""
        # Check if region actually changed to avoid redundant computation
        current_region_id = id(self.region)
        if self._last_region_id == current_region_id and self._magnitude_spectrum is not None:
            return
        
        self._last_region_id = current_region_id
        
        # Compute windowed FFT
        self._magnitude_spectrum, self._nyq_x, self._nyq_y = utils.compute_fft(
            self.region, self.scale_x, self.scale_y
        )
        
        # Compute complex FFT for inverse (only once)
        window = np.hanning(self.region.shape[0])[:, None] * np.hanning(self.region.shape[1])[None, :]
        windowed = self.region * window
        self._fft_complex = np.fft.fftshift(np.fft.fft2(windowed))
        
        # Clear inverse cache when region changes
        self._inverse_fft_cache = None
        
    def update_display(self):
        """Update FFT display, optionally showing inverse (with caching)."""
        if self._magnitude_spectrum is None:
            return
        
        # Determine which data to display with caching
        if self.chk_inverse.isChecked():
            # Use cached inverse FFT if available
            if self._inverse_fft_cache is None:
                self._inverse_fft_cache = utils.compute_inverse_fft(self._fft_complex)
            display_data = self._inverse_fft_cache
        else:
            display_data = self._magnitude_spectrum
        
        self.img_fft.setImage(display_data)
        self.img_fft.setRect(QtCore.QRectF(-self._nyq_x, -self._nyq_y, 2*self._nyq_x, 2*self._nyq_y))
        
        # Set axis labels
        unit_x = f"1/{self.ax_x_units}" if self.ax_x_units else "1/px"
        unit_y = f"1/{self.ax_y_units}" if self.ax_y_units else "1/px"
        self.plot.setLabel('bottom', "Frequency X", units=unit_x)
        self.plot.setLabel('left', "Frequency Y", units=unit_y)


class LineDrawingTool:
    """Tool for drawing measurement lines on a plot."""
    
    def __init__(self, plot: pg.PlotItem, on_line_drawn_callback):
        self.plot = plot
        self.on_line_drawn_callback = on_line_drawn_callback
        self.drawing = False
        self.start_point = None
        self.line_item = None
        self.is_enabled = False
        self.vb = plot.vb
        
        # Store original mouse event methods
        self.original_mouse_press = self.vb.mousePressEvent
        self.original_mouse_move = self.vb.mouseMoveEvent
        self.original_mouse_release = self.vb.mouseReleaseEvent
        
    def enable(self):
        """Enable line drawing mode."""
        self.is_enabled = True
        self.drawing = False
        self.start_point = None
        
        # Replace mouse event handlers
        self.vb.mousePressEvent = self._on_mouse_press
        self.vb.mouseMoveEvent = self._on_mouse_move
        self.vb.mouseReleaseEvent = self._on_mouse_release
        
    def disable(self):
        """Disable line drawing mode."""
        self.is_enabled = False
        
        # Restore original mouse event handlers
        self.vb.mousePressEvent = self.original_mouse_press
        self.vb.mouseMoveEvent = self.original_mouse_move
        self.vb.mouseReleaseEvent = self.original_mouse_release
        
        self._clear_preview_line()
        self.drawing = False
        self.start_point = None
        
    def _clear_preview_line(self):
        """Remove preview line from plot."""
        if self.line_item:
            self.plot.removeItem(self.line_item)
            self.line_item = None
    
    def _on_mouse_press(self, event):
        """Handle mouse press for line drawing."""
        if not self.is_enabled:
            self.original_mouse_press(event)
            return
        
        scene_pos = event.scenePos()
        if not self.plot.sceneBoundingRect().contains(scene_pos):
            self.original_mouse_press(event)
            return
        
        view_pos = self.vb.mapSceneToView(scene_pos)
        # Start a new line on mouse press
        self.drawing = True
        self.start_point = (view_pos.x(), view_pos.y())
        self._clear_preview_line()

        event.accept()
    
    def _on_mouse_move(self, event):
        """Handle mouse move for line drawing preview."""
        if not self.is_enabled:
            self.original_mouse_move(event)
            return
        
        if not self.drawing or self.start_point is None:
            self.original_mouse_move(event)
            return
        
        scene_pos = event.scenePos()
        if not self.plot.sceneBoundingRect().contains(scene_pos):
            self.original_mouse_move(event)
            return
        
        view_pos = self.vb.mapSceneToView(scene_pos)
        
        # Update preview line
        self._clear_preview_line()
        self.line_item = pg.PlotDataItem(
            [self.start_point[0], view_pos.x()],
            [self.start_point[1], view_pos.y()],
            pen=PREVIEW_LINE_PEN
        )
        self.plot.addItem(self.line_item)
        event.accept()
    
    def _on_mouse_release(self, event):
        """Handle mouse release."""
        if not self.is_enabled:
            self.original_mouse_release(event)
            return
        
        # If we weren't drawing, let the original handler process the event
        if not self.drawing or self.start_point is None:
            self.original_mouse_release(event)
            return

        scene_pos = event.scenePos()
        if not self.plot.sceneBoundingRect().contains(scene_pos):
            # Outside plot area; cancel the drawing and delegate
            self._clear_preview_line()
            self.drawing = False
            self.start_point = None
            self.original_mouse_release(event)
            return

        # Finish the line on mouse release and trigger callback
        view_pos = self.vb.mapSceneToView(scene_pos)
        end_point = (view_pos.x(), view_pos.y())
        self.drawing = False
        self._clear_preview_line()
        self.on_line_drawn_callback(self.start_point, end_point)
        self.start_point = None

        event.accept()


class FFTBoxROI(pg.RectROI):
    """Custom RectROI for FFT boxes with click and double-click signals."""

    sigBoxClicked = Signal(object)
    sigBoxDoubleClicked = Signal(object)

    def mouseClickEvent(self, ev):
        """Emit signals on single and double clicks while preserving default behavior."""
        # Call base implementation if available (version-safe)
        try:
            super().mouseClickEvent(ev)  # type: ignore[attr-defined]
        except AttributeError:
            pass

        if ev.button() == QtCore.Qt.LeftButton:
            if ev.double():
                self.sigBoxDoubleClicked.emit(self)
            else:
                self.sigBoxClicked.emit(self)


class MeasurementLabel(pg.TextItem):
    """Clickable label for measurement annotations."""

    sigLabelClicked = Signal(object)

    def mouseClickEvent(self, ev):
        """Emit a signal when the label is clicked."""
        super().mouseClickEvent(ev)

        if ev.button() == QtCore.Qt.LeftButton:
            self.sigLabelClicked.emit(self)


class ImageViewerWindow(QtWidgets.QMainWindow):
    """Window for viewing and analyzing a single image with FFT boxes."""
    
    def __init__(self, file_path: str, signal=None, window_suffix: Optional[str] = None):
        super().__init__()
        self.file_path = file_path
        self.signal = None  # HyperSpy signal object (may be provided or loaded)
        self.data = None
        self.ax_x = None
        self.ax_y = None
        self.fft_boxes = []  # List of all FFT boxes
        self.fft_count = 0
        self.is_reciprocal_space = False
        self.line_tool = None
        self.fft_windows = []  # Track open FFT windows
        self.fft_to_fft_window: Dict[pg.RectROI, FFTViewerWindow] = {}  # Map FFT box to FFT window
        self.fft_box_meta: Dict[pg.RectROI, Dict[str, object]] = {}  # Store metadata per FFT box
        self.selected_fft_box: Optional[pg.RectROI] = None
        self.scale_bar = None  # Scale bar item
        self.measurement_history_window = None  # Measurement history window
        self.metadata_window = None  # Metadata display window
        self.measurement_count = 0  # Sequential counter for measurements
        self.btn_measure = None  # Reference to Measure Distance button
        self.measurement_items: List[Tuple[pg.PlotDataItem, pg.TextItem]] = []  # Stored measurement graphics
        self.selected_measurement_index: Optional[int] = None  # Index of selected measurement

        # If a HyperSpy signal is provided, use it directly; otherwise load from file
        if signal is not None:
            self._setup_from_signal(signal, window_suffix)
        else:
            self._load_and_setup()
        
    def _load_and_setup(self):
        """Load image file and setup UI."""
        try:
            s = hs.load(self.file_path)
            if s.axes_manager.navigation_dimension != 0:
                s = s.inav[0, 0]
            self._setup_from_signal(s)
            
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error Loading File", str(e))

    def _setup_from_signal(self, signal, window_suffix: Optional[str] = None):
        """Initialize viewer state from a HyperSpy signal instance."""
        self.signal = signal
        self.data = signal.data
        self.ax_x = signal.axes_manager[0]
        self.ax_y = signal.axes_manager[1]

        # Try to apply calibration from the original metadata (e.g., CalibrationDeltaX/Y in meters)
        calibrated = self._apply_calibration_from_original_metadata()
        if not calibrated:
            QtWidgets.QMessageBox.warning(
                self,
                "Calibration",
                "This image could not be calibrated successfully from metadata; "
                "using default pixel scaling instead.",
            )
        
        # Check if this looks like a diffraction pattern
        self.is_reciprocal_space = utils.is_diffraction_pattern(self.data)
        
        # Set window title from filename (optionally with suffix/index)
        file_name = Path(self.file_path).stem
        if window_suffix:
            title = f"Image Viewer - {file_name} {window_suffix}"
        else:
            title = f"Image Viewer - {file_name}"
        self.setWindowTitle(title)
        
        self.setup_ui()
        self.resize(*DEFAULT_IMAGE_WINDOW_SIZE)

    def _get_original_metadata_dict(self) -> Optional[dict]:
        """Return the original (raw) metadata as a plain dictionary if available."""
        if self.signal is None:
            return None

        original_meta = None

        # Newer HyperSpy: signal.original_metadata
        if hasattr(self.signal, "original_metadata"):
            original_meta = self.signal.original_metadata
        # Older style: metadata.original_metadata
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
        """Use CalibrationDeltaX/Y (meters) from original metadata to set axis scales.

        Returns True if calibration was successfully applied, False otherwise.
        """
        meta = self._get_original_metadata_dict()
        if not meta:
            return False

        # Look for SER-style header parameters containing CalibrationDeltaX/Y
        ser_params = None
        if "ser_header_parameters" in meta:
            ser_params = meta["ser_header_parameters"]
        else:
            # Fallback: case-insensitive search for similar key
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

        # Apply calibration in meters to both axes
        try:
            self.ax_x.scale = float(dx)
            self.ax_y.scale = float(dy)
            self.ax_x.units = "m"
            self.ax_y.units = "m"

            # Optional offsets, if provided
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
        """Get image bounds (x_offset, y_offset, width, height)."""
        x_offset = self.ax_x.offset if self.ax_x else 0
        y_offset = self.ax_y.offset if self.ax_y else 0
        w = self.ax_x.size * self.ax_x.scale if self.ax_x else 1
        h = self.ax_y.size * self.ax_y.scale if self.ax_y else 1
        return x_offset, y_offset, w, h
        
    def setup_ui(self):
        """Setup UI with image and FFT controls."""
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        main_layout = QtWidgets.QVBoxLayout(central)
        
        # Toolbar
        toolbar = self._create_toolbar()
        main_layout.addLayout(toolbar)
        
        # Main graphics widget with plot
        self.glw = pg.GraphicsLayoutWidget()
        main_layout.addWidget(self.glw)
        
        # Plot: Original Image with FFT boxes
        self.p1 = self.glw.addPlot(title="Original Image")
        
        # Lock aspect ratio to 1:1 (square pixels)
        self.p1.vb.setAspectLocked(True, ratio=1.0)
        
        self.img_orig = pg.ImageItem(axisOrder='row-major')
        self.p1.addItem(self.img_orig)
        self.p1.invertY(True)
        self.img_orig.setImage(self.data)
        
        # Set image position and scale
        x_offset, y_offset, w, h = self.image_bounds
        self.img_orig.setRect(QtCore.QRectF(x_offset, y_offset, w, h))
        
        # Set axis labels and units - adjust for reciprocal space
        if self.is_reciprocal_space:
            ax_x_label = f"{self.ax_x.name} (reciprocal)"
            ax_y_label = f"{self.ax_y.name} (reciprocal)"
            ax_x_units = f"1/{self.ax_x.units}" if self.ax_x.units else "1/px"
            ax_y_units = f"1/{self.ax_y.units}" if self.ax_y.units else "1/px"
        else:
            ax_x_label = self.ax_x.name
            ax_y_label = self.ax_y.name
            ax_x_units = self.ax_x.units
            ax_y_units = self.ax_y.units
        
        self.p1.setLabel('bottom', ax_x_label, units=ax_x_units)
        self.p1.setLabel('left', ax_y_label, units=ax_y_units)
        
        # Ensure the full image is visible on load
        self.p1.setXRange(x_offset, x_offset + w, padding=0)
        self.p1.setYRange(y_offset, y_offset + h, padding=0)
        
        # Initialize line tool
        self.line_tool = LineDrawingTool(self.p1, self._on_line_drawn)
        
        # Setup keyboard shortcuts
        self.setup_keyboard_shortcuts()
    
    def setup_keyboard_shortcuts(self):
        """Setup keyboard shortcuts for ROI management."""
        # Add shortcuts for deleting selected measurement or ROI
        delete_shortcut = QtGui.QShortcut(QtGui.QKeySequence.Delete, self)
        delete_shortcut.activated.connect(self._delete_selected_roi)

        backspace_shortcut = QtGui.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key_Backspace), self)
        backspace_shortcut.activated.connect(self._delete_selected_roi)
    
    def _delete_selected_roi(self):
        """Delete the currently selected ROI."""
        # Prefer deleting a selected measurement if present
        if self.selected_measurement_index is not None:
            self._delete_selected_measurement()
            return

        # Otherwise, delete the selected FFT ROI if any
        fft_box = self.selected_fft_box
        if fft_box is None or fft_box not in self.fft_boxes:
            QtWidgets.QMessageBox.information(self, "Delete", "No measurement or ROI selected.")
            return

        index = self.fft_boxes.index(fft_box)

        # Remove ROI graphic
        self.p1.removeItem(fft_box)

        # Remove associated label
        meta = self.fft_box_meta.pop(fft_box, None)
        if meta is not None:
            text_item = meta.get("text_item")
            if text_item is not None:
                self.p1.removeItem(text_item)

        # Close and remove associated FFT window if it exists
        fft_window = self.fft_to_fft_window.pop(fft_box, None)
        if fft_window is not None:
            fft_window.close()
            if fft_window in self.fft_windows:
                self.fft_windows.remove(fft_window)

        # Remove from list and clear selection
        self.fft_boxes.pop(index)
        self.selected_fft_box = None

        QtWidgets.QMessageBox.information(self, "Deleted", f"FFT Box {index} deleted.")
        
    def _create_toolbar(self) -> QtWidgets.QHBoxLayout:
        """Create toolbar with controls."""
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
        
        layout.addStretch()
        return layout
        
    def _add_new_fft(self, x_offset=None, y_offset=None, w=None, h=None):
        """Add a new FFT box to the image."""
        # Ensure we have valid dimensions
        if x_offset is None or y_offset is None or w is None or h is None:
            x_offset, y_offset, w, h = self.image_bounds
        
        # Ensure non-zero dimensions
        if w is None or h is None or w <= 0 or h <= 0:
            QtWidgets.QMessageBox.warning(self, "Error", "Invalid image bounds")
            return
        
        # Create FFT box with unique color, ensuring valid positions
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
        
        # Add text label to FFT box
        text_item = pg.TextItem(f"FFT {fft_id}", anchor=(0, 0), fill=pg.mkBrush(color))
        text_item.setPos(fft_box.pos()[0], fft_box.pos()[1])
        self.p1.addItem(text_item)
        
        # Connect signal for when FFT box finishes moving
        fft_box.sigRegionChangeFinished.connect(lambda: self._on_fft_finished(fft_box, fft_id, text_item))
        fft_box.sigBoxClicked.connect(lambda roi=fft_box: self._on_fft_box_clicked(roi))
        fft_box.sigBoxDoubleClicked.connect(lambda roi=fft_box: self._on_fft_box_double_clicked(roi, fft_id, text_item))
        self.fft_boxes.append(fft_box)
        self.fft_box_meta[fft_box] = {"id": fft_id, "text_item": text_item}
        
    def _on_fft_finished(self, fft_box: pg.RectROI, fft_id: int, text_item: pg.TextItem):
        """Handle FFT box finished moving - open or update FFT window."""
        region = fft_box.getArrayRegion(self.data, self.img_orig)
        
        if region is None or region.shape[0] < 2 or region.shape[1] < 2:
            return
        
        # Update text label position
        text_item.setPos(fft_box.pos()[0], fft_box.pos()[1])

        # Open or update the associated FFT window
        self._open_or_update_fft_window(fft_box, fft_id, text_item, region)

    def _open_or_update_fft_window(self, fft_box: pg.RectROI, fft_id: int, text_item: pg.TextItem, region: np.ndarray):
        """Open a new FFT window or update an existing one for a given ROI."""
        if fft_box in self.fft_to_fft_window:
            # Update existing window with new region and bring it to front
            fft_window = self.fft_to_fft_window[fft_box]
            fft_window.region = region
            fft_window._compute_fft()
            fft_window.update_display()
            fft_window.show()
            fft_window.raise_()
            fft_window.activateWindow()
        else:
            # Create new FFT window
            fft_name = f"FFT {fft_id}"
            parent_title = self.windowTitle().replace("Image Viewer - ", "")
            fft_window = FFTViewerWindow(
                self, region,
                self.ax_x.scale, self.ax_y.scale,
                self.ax_x.name, self.ax_x.units,
                self.ax_y.name, self.ax_y.units,
                fft_name,
                parent_title
            )
            fft_window.show()
            self.fft_windows.append(fft_window)
            self.fft_to_fft_window[fft_box] = fft_window

    def _on_fft_box_clicked(self, fft_box: pg.RectROI):
        """Track the most recently clicked FFT box for selection/deletion."""
        if fft_box in self.fft_boxes:
            self.selected_fft_box = fft_box

    def _on_fft_box_double_clicked(self, fft_box: pg.RectROI, fft_id: int, text_item: pg.TextItem):
        """Reopen or update the FFT window when its ROI is double-clicked."""
        region = fft_box.getArrayRegion(self.data, self.img_orig)
        if region is None or region.shape[0] < 2 or region.shape[1] < 2:
            return

        text_item.setPos(fft_box.pos()[0], fft_box.pos()[1])
        self._open_or_update_fft_window(fft_box, fft_id, text_item, region)
        
    def _toggle_line_measurement(self):
        """Toggle line drawing tool for distance measurement."""
        if self.btn_measure is not None and self.btn_measure.isChecked():
            # Enable measurement mode and highlight button
            self.line_tool.enable()
            self.btn_measure.setStyleSheet("background-color: #4caf50; color: white;")
        else:
            # Disable measurement mode and reset button appearance
            self.line_tool.disable()
            if self.btn_measure is not None:
                self.btn_measure.setStyleSheet("")
        
    def _on_line_drawn(self, p1: Tuple[float, float], p2: Tuple[float, float]):
        """Handle completed line measurement."""
        # Increment measurement counter for sequential labeling
        self.measurement_count += 1
        measurement_id = self.measurement_count
        
        # Coordinates from LineDrawingTool are in axis (physical) units,
        # so compute distances directly in those units and derive pixels.
        scale_x = float(self.ax_x.scale) if self.ax_x is not None else 1.0
        scale_y = float(self.ax_y.scale) if self.ax_y is not None else scale_x

        dx_phys = float(p2[0] - p1[0])
        dy_phys = float(p2[1] - p1[1])

        dist_phys = np.hypot(dx_phys, dy_phys)

        # Convert back to pixel distances using the calibration
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

        # For diffraction patterns, also compute d-spacing from reciprocal distance
        if self.is_reciprocal_space and dist_phys != 0:
            frequency = 1.0 / dist_phys
            result["d_spacing"] = utils.calculate_d_spacing(frequency)
        
        # Draw the line on the plot (in axis coordinates)
        line = pg.PlotDataItem([p1[0], p2[0]], [p1[1], p2[1]], pen=DRAWN_LINE_PEN)
        self.p1.addItem(line)
        
        # Create label for the line
        label_text = self._format_measurement_label(result, measurement_id)
        mid_x = (p1[0] + p2[0]) / 2
        mid_y = (p1[1] + p2[1]) / 2
        
        text_item = MeasurementLabel(label_text, anchor=(0, 0), fill=LABEL_BRUSH_COLOR)
        text_item.setPos(mid_x, mid_y)
        text_item.sigLabelClicked.connect(self._on_measurement_label_clicked)
        self.p1.addItem(text_item)

        # Track measurement graphics for later deletion
        self.measurement_items.append((line, text_item))
        
        # Add to measurement history
        self._add_to_measurement_history(label_text)

    def _on_measurement_label_clicked(self, label: pg.TextItem):
        """Select a measurement when its label is clicked."""
        selected_index = None
        for idx, (line_item, text_item) in enumerate(self.measurement_items):
            if text_item is label:
                selected_index = idx
                break

        self.selected_measurement_index = selected_index

        # Visually highlight the selected label
        for idx, (line_item, text_item) in enumerate(self.measurement_items):
            if idx == selected_index:
                text_item.setFill(pg.mkBrush(255, 200, 0, 255))  # brighter highlight
            else:
                text_item.setFill(LABEL_BRUSH_COLOR)

    def _clear_measurements(self):
        """Remove all measurement lines and labels from the image and history."""
        for line_item, text_item in self.measurement_items:
            self.p1.removeItem(line_item)
            self.p1.removeItem(text_item)
        self.measurement_items.clear()
        self.selected_measurement_index = None

        if self.measurement_history_window is not None:
            self.measurement_history_window.clear_all()

    def _delete_selected_measurement(self):
        """Delete the currently selected measurement annotation from the image."""
        if self.selected_measurement_index is None:
            return

        if not (0 <= self.selected_measurement_index < len(self.measurement_items)):
            self.selected_measurement_index = None
            return

        line_item, text_item = self.measurement_items.pop(self.selected_measurement_index)
        self.p1.removeItem(line_item)
        self.p1.removeItem(text_item)

        # Clear selection and reset label styles
        self.selected_measurement_index = None
        for _, t in self.measurement_items:
            t.setFill(LABEL_BRUSH_COLOR)

    def delete_measurement_by_label(self, label_text: str):
        """Delete the first measurement whose label text matches the given string."""
        target_index = None
        for idx, (_, text_item) in enumerate(self.measurement_items):
            if text_item.toPlainText() == label_text:
                target_index = idx
                break

        if target_index is None:
            return

        self.selected_measurement_index = target_index
        self._delete_selected_measurement()
        
    def _format_measurement_label(self, result: dict, measurement_id: Optional[int] = None) -> str:
        """Format measurement result as text label."""
        scaled_dist, scaled_unit = utils.format_si_scale(result['distance_physical'], self.ax_x.units)
        prefix = f"#{measurement_id} " if measurement_id is not None else ""
        
        if self.is_reciprocal_space and 'd_spacing' in result:
            return f"{prefix}d: {result['d_spacing']:.4f} Å\n({scaled_dist:.4f} {scaled_unit}⁻¹)"
        else:
            return f"{prefix}{scaled_dist:.4f} {scaled_unit}\n({result['distance_pixels']:.1f} px)"
    
    def _show_measurement_history(self):
        """Show or create the measurement history window."""
        if self.measurement_history_window is None:
            self.measurement_history_window = MeasurementHistoryWindow(self)
        
        self.measurement_history_window.show()
        self.measurement_history_window.raise_()
        self.measurement_history_window.activateWindow()
    
    def _add_to_measurement_history(self, measurement_text: str):
        """Add a measurement to history."""
        if self.measurement_history_window is None:
            self.measurement_history_window = MeasurementHistoryWindow(self)
        
        self.measurement_history_window.add_measurement(measurement_text)

    def _show_metadata_window(self):
        """Show a window with the original metadata parsed by HyperSpy."""
        if self.signal is None:
            QtWidgets.QMessageBox.information(self, "Metadata", "No metadata available for this image.")
            return

        metadata_dict = self._get_original_metadata_dict()
        if metadata_dict is None:
            # Last resort: use the standard HyperSpy metadata dictionary
            metadata_dict = self.signal.metadata.as_dictionary()

        file_name = Path(self.file_path).name
        title = f"Metadata - {file_name}"

        if self.metadata_window is None:
            self.metadata_window = MetadataWindow(self, title=title, metadata=metadata_dict)
        else:
            self.metadata_window.setWindowTitle(title)
            self.metadata_window.update_metadata(metadata_dict)

        self.metadata_window.show()
        self.metadata_window.raise_()
        self.metadata_window.activateWindow()
        

class MainWindow(QtWidgets.QMainWindow):
    """Main application window with drag-and-drop support."""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Fast FFT Image Analyzer")
        self.resize(*DEFAULT_MAIN_WINDOW_SIZE)
        self.setAcceptDrops(True)
        
        # Create central widget with instructions
        self._setup_ui()
        
    def _setup_ui(self):
        """Setup main window UI."""
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QVBoxLayout(central)
        
        label = QtWidgets.QLabel(
            "Drag and drop an image file here to open it\n"
            "(Supports DM3, DM4, TIFF, and other HyperSpy formats)"
        )
        label.setAlignment(QtCore.Qt.AlignCenter)
        label.setStyleSheet("font-size: 16px; color: #666;")
        layout.addStretch()
        layout.addWidget(label)
        layout.addStretch()
        
    def dragEnterEvent(self, event: QtGui.QDragEnterEvent):
        """Accept drag events with files."""
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            
    def dropEvent(self, event: QtGui.QDropEvent):
        """Handle dropped files."""
        for url in event.mimeData().urls():
            file_path = url.toLocalFile()
            self._open_image(file_path)
            
    def _open_image(self, file_path: str):
        """Open image file; if it contains multiple images, open one window per image."""
        try:
            loaded = hs.load(file_path)

            # HyperSpy may return a single signal or a list of signals
            signals = loaded if isinstance(loaded, list) else [loaded]

            for sig_index, signal in enumerate(signals):
                if signal.axes_manager.navigation_dimension == 0:
                    # Single image in this signal
                    suffix = f"[{sig_index}]" if len(signals) > 1 else None
                    window = ImageViewerWindow(file_path, signal=signal, window_suffix=suffix)
                    window.show()
                else:
                    # Multiple images along navigation axes - open one window per navigation position
                    nav_shape = signal.axes_manager.navigation_shape
                    for nav_index in np.ndindex(nav_shape):
                        sub_signal = signal.inav[nav_index]
                        # Build a readable suffix like [0,1] or [sig0,0,1]
                        idx_str = ",".join(str(i) for i in nav_index)
                        if len(signals) > 1:
                            suffix = f"[{sig_index}; {idx_str}]"
                        else:
                            suffix = f"[{idx_str}]"
                        window = ImageViewerWindow(file_path, signal=sub_signal, window_suffix=suffix)
                        window.show()

        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"Could not open file: {str(e)}")


def main():
    """Main entry point for the application."""
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
