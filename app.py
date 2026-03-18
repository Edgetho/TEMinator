"""Individual image viewer window with FFT analysis."""
import sys
import numpy as np
import hyperspy.api as hs
import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets, QtCore, QtGui
from pathlib import Path
from typing import Optional, List, Tuple, Dict
import utils

# Module-level constants
ROI_COLORS = ['r', 'g', 'b', 'y', 'c', 'm']
PREVIEW_LINE_PEN = pg.mkPen('y', width=2, style=QtCore.Qt.DashLine)
DRAWN_LINE_PEN = pg.mkPen('w', width=2)
LABEL_BRUSH_COLOR = pg.mkBrush(255, 255, 100, 220)  # Yellow background
DEFAULT_FFT_WINDOW_SIZE = (700, 700)
DEFAULT_IMAGE_WINDOW_SIZE = (1000, 900)
DEFAULT_MAIN_WINDOW_SIZE = (600, 400)


class FFTViewerWindow(QtWidgets.QMainWindow):
    """Separate window displaying FFT for a specific ROI."""
    
    def __init__(self, parent, region: np.ndarray, scale_x: float, scale_y: float, 
                 ax_x_name: str, ax_x_units: str, ax_y_name: str, ax_y_units: str, 
                 roi_name: str, parent_name: str = ""):
        super().__init__()
        
        # Set title
        self._update_title(roi_name, parent_name)
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
        
    def _update_title(self, roi_name: str, parent_name: str):
        """Update window title based on image and ROI name."""
        title = f"FFT - {parent_name} - {roi_name}" if parent_name else f"FFT - {roi_name}"
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
        
        if not self.drawing:
            # Start new line
            self.drawing = True
            self.start_point = (view_pos.x(), view_pos.y())
            self._clear_preview_line()
        else:
            # End line and trigger callback
            end_point = (view_pos.x(), view_pos.y())
            self.drawing = False
            self._clear_preview_line()
            self.on_line_drawn_callback(self.start_point, end_point)
            self.start_point = None
        
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
        
        event.accept()


class ImageViewerWindow(QtWidgets.QMainWindow):
    """Window for viewing and analyzing a single image with ROIs."""
    
    def __init__(self, file_path: str):
        super().__init__()
        self.file_path = file_path
        self.data = None
        self.ax_x = None
        self.ax_y = None
        self.rois = []  # List of all ROI boxes
        self.roi_count = 0
        self.is_reciprocal_space = False
        self.line_tool = None
        self.fft_windows = []  # Track open FFT windows
        self.roi_to_fft_window: Dict[pg.RectROI, FFTViewerWindow] = {}  # Map ROI to FFT window
        
        self._load_and_setup()
        
    def _load_and_setup(self):
        """Load image file and setup UI."""
        try:
            s = hs.load(self.file_path)
            if s.axes_manager.navigation_dimension != 0:
                s = s.inav[0, 0]
            
            self.data = s.data
            self.ax_x = s.axes_manager[0]
            self.ax_y = s.axes_manager[1]
            
            # Check if this looks like a diffraction pattern
            self.is_reciprocal_space = utils.is_diffraction_pattern(self.data)
            
            # Set window title from filename
            file_name = Path(self.file_path).stem
            self.setWindowTitle(f"Image Viewer - {file_name}")
            
            self.setup_ui()
            self.resize(*DEFAULT_IMAGE_WINDOW_SIZE)
            
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error Loading File", str(e))
            
    @property
    def image_bounds(self) -> Tuple[float, float, float, float]:
        """Get image bounds (x_offset, y_offset, width, height)."""
        x_offset = self.ax_x.offset
        y_offset = self.ax_y.offset
        w = self.ax_x.size * self.ax_x.scale
        h = self.ax_y.size * self.ax_y.scale
        return x_offset, y_offset, w, h
        
    def setup_ui(self):
        """Setup UI with image and ROI controls."""
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        main_layout = QtWidgets.QVBoxLayout(central)
        
        # Toolbar
        toolbar = self._create_toolbar()
        main_layout.addLayout(toolbar)
        
        # Main graphics widget with plot
        self.glw = pg.GraphicsLayoutWidget()
        main_layout.addWidget(self.glw)
        
        # Plot: Original Image with ROI boxes
        self.p1 = self.glw.addPlot(title="Original Image")
        self.img_orig = pg.ImageItem(axisOrder='row-major')
        self.p1.addItem(self.img_orig)
        self.p1.invertY(True)
        self.img_orig.setImage(self.data)
        
        # Set image position and scale
        x_offset, y_offset, w, h = self.image_bounds
        self.img_orig.setRect(QtCore.QRectF(x_offset, y_offset, w, h))
        
        # Set axis labels and units
        self.p1.setLabel('bottom', f"{self.ax_x.name}", units=self.ax_x.units)
        self.p1.setLabel('left', f"{self.ax_y.name}", units=self.ax_y.units)
        
        # Create initial ROI and initialize line tool
        self._add_new_roi(*self.image_bounds)
        self.line_tool = LineDrawingTool(self.p1, self._on_line_drawn)
        
    def _create_toolbar(self) -> QtWidgets.QHBoxLayout:
        """Create toolbar with controls."""
        layout = QtWidgets.QHBoxLayout()
        
        btn_add_roi = QtWidgets.QPushButton("Add New ROI")
        btn_add_roi.clicked.connect(self._add_new_roi)
        layout.addWidget(btn_add_roi)
        
        btn_measure = QtWidgets.QPushButton("Measure Distance")
        btn_measure.clicked.connect(self._toggle_line_measurement)
        layout.addWidget(btn_measure)
        
        layout.addStretch()
        return layout
        
    def _add_new_roi(self, x_offset=None, y_offset=None, w=None, h=None):
        """Add a new ROI box to the image."""
        if x_offset is None:
            x_offset, y_offset, w, h = self.image_bounds
        
        # Create ROI with unique color
        color = ROI_COLORS[self.roi_count % len(ROI_COLORS)]
        roi = pg.RectROI([x_offset + w/4, y_offset + h/4], 
                         [w/2, h/2], 
                         pen=pg.mkPen(color, width=2))
        roi.addScaleHandle([1, 1], [0, 0])
        roi.addScaleHandle([0, 0], [1, 1])
        self.p1.addItem(roi)
        
        roi_id = self.roi_count
        self.roi_count += 1
        
        # Add text label to ROI
        text_item = pg.TextItem(f"ROI {roi_id}", anchor=(0, 0), fill=pg.mkBrush(color))
        text_item.setPos(roi.pos()[0], roi.pos()[1])
        self.p1.addItem(text_item)
        
        # Connect signal for when ROI finishes moving
        roi.sigRegionChangeFinished.connect(lambda: self._on_roi_finished(roi, roi_id, text_item))
        self.rois.append(roi)
        
    def _on_roi_finished(self, roi: pg.RectROI, roi_id: int, text_item: pg.TextItem):
        """Handle ROI finished moving - open or update FFT window."""
        region = roi.getArrayRegion(self.data, self.img_orig)
        
        if region is None or region.shape[0] < 2 or region.shape[1] < 2:
            return
        
        # Update text label position
        text_item.setPos(roi.pos()[0], roi.pos()[1])
        
        # Check if this ROI already has an FFT window
        if roi in self.roi_to_fft_window:
            # Update existing window with new region
            fft_window = self.roi_to_fft_window[roi]
            fft_window.region = region
            fft_window._compute_fft()
            fft_window.update_display()
        else:
            # Create new FFT window
            roi_name = f"ROI {roi_id}"
            parent_title = self.windowTitle().replace("Image Viewer - ", "")
            fft_window = FFTViewerWindow(
                self, region,
                self.ax_x.scale, self.ax_y.scale,
                self.ax_x.name, self.ax_x.units,
                self.ax_y.name, self.ax_y.units,
                roi_name,
                parent_title
            )
            fft_window.show()
            self.fft_windows.append(fft_window)
            self.roi_to_fft_window[roi] = fft_window
        
    def _toggle_line_measurement(self):
        """Toggle line drawing tool for distance measurement."""
        if self.line_tool.is_enabled:
            self.line_tool.disable()
            QtWidgets.QMessageBox.information(self, "Measure Distance", "Line measurement disabled.")
        else:
            self.line_tool.enable()
            QtWidgets.QMessageBox.information(self, "Measure Distance",
                "Click once to start a line, click again to end it.\n"
                "(For diffraction patterns, shows d-spacing; for real-space, shows physical distance)")
        
    def _on_line_drawn(self, p1: Tuple[float, float], p2: Tuple[float, float]):
        """Handle completed line measurement."""
        result = utils.measure_line_distance(
            p1, p2, self.ax_x.scale, self.ax_y.scale, self.is_reciprocal_space
        )
        
        # Build message
        msg = f"Distance: {result['distance_physical']:.4f} {self.ax_x.units}\n"
        msg += f"Pixels: {result['distance_pixels']:.1f}\n"
        if self.is_reciprocal_space and 'd_spacing' in result:
            msg += f"d-spacing: {result['d_spacing']:.4f} Å"
        
        QtWidgets.QMessageBox.information(self, "Measurement Result", msg)
        
        # Draw the line on the plot
        line = pg.PlotDataItem([p1[0], p2[0]], [p1[1], p2[1]], pen=DRAWN_LINE_PEN)
        self.p1.addItem(line)
        
        # Create label for the line
        label_text = self._format_measurement_label(result)
        mid_x = (p1[0] + p2[0]) / 2
        mid_y = (p1[1] + p2[1]) / 2
        
        text_item = pg.TextItem(label_text, anchor=(0, 0), fill=LABEL_BRUSH_COLOR)
        text_item.setPos(mid_x, mid_y)
        self.p1.addItem(text_item)
        
    def _format_measurement_label(self, result: dict) -> str:
        """Format measurement result as text label."""
        if self.is_reciprocal_space and 'd_spacing' in result:
            return f"d: {result['d_spacing']:.4f} Å\n({result['distance_physical']:.4f} {self.ax_x.units}⁻¹)"
        else:
            return f"{result['distance_physical']:.4f} {self.ax_x.units}\n({result['distance_pixels']:.1f} px)"
        

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
        """Open image in a new window."""
        try:
            window = ImageViewerWindow(file_path)
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
