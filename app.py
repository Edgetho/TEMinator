"""Individual image viewer window with FFT analysis."""
import sys
import numpy as np
import hyperspy.api as hs
import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets, QtCore, QtGui
from pathlib import Path
from typing import Optional, List, Tuple
import utils


class FFTViewerWindow(QtWidgets.QMainWindow):
    """Separate window displaying FFT for a specific ROI."""
    
    def __init__(self, parent, region: np.ndarray, scale_x: float, scale_y: float, 
                 ax_x_name: str, ax_x_units: str, ax_y_name: str, ax_y_units: str, roi_name: str, parent_name: str = ""):
        super().__init__()
        title = f"FFT - {parent_name} - {roi_name}" if parent_name else f"FFT - {roi_name}"
        self.setWindowTitle(title)
        self.resize(700, 700)
        
        self.scale_x = scale_x
        self.scale_y = scale_y
        self.ax_x_name = ax_x_name
        self.ax_x_units = ax_x_units
        self.ax_y_name = ax_y_name
        self.ax_y_units = ax_y_units
        self.region = region
        self.show_inverse = False
        self.fft_complex = None
        self.magnitude_spectrum = None
        self.nyq_x = None
        self.nyq_y = None
        
        self.setup_ui()
        self.compute_and_display_fft()
        
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
        
    def compute_and_display_fft(self):
        """Compute FFT and display it."""
        self.magnitude_spectrum, self.nyq_x, self.nyq_y = utils.compute_fft(
            self.region, self.scale_x, self.scale_y
        )
        
        # Also compute complex FFT for inverse
        window = np.hanning(self.region.shape[0])[:, None] * np.hanning(self.region.shape[1])[None, :]
        windowed = self.region * window
        self.fft_complex = np.fft.fftshift(np.fft.fft2(windowed))
        
        self.update_display()
        
    def update_display(self):
        """Update FFT display, optionally showing inverse."""
        if self.magnitude_spectrum is None:
            return
        
        if self.chk_inverse.isChecked():
            # Show inverse FFT
            display_data = utils.compute_inverse_fft(self.fft_complex)
        else:
            display_data = self.magnitude_spectrum
        
        self.img_fft.setImage(display_data)
        self.img_fft.setRect(QtCore.QRectF(-self.nyq_x, -self.nyq_y, 2*self.nyq_x, 2*self.nyq_y))
        
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
        self.original_mouse_enabled = True
        
        # Store original mouse event methods
        self.original_mouse_press_event = self.vb.mousePressEvent
        self.original_mouse_move_event = self.vb.mouseMoveEvent
        self.original_mouse_release_event = self.vb.mouseReleaseEvent
        
    def enable(self):
        """Enable line drawing mode."""
        self.is_enabled = True
        self.drawing = False
        self.start_point = None
        
        # Replace mouse event handlers with our own
        self.vb.mousePressEvent = self._on_mouse_press
        self.vb.mouseMoveEvent = self._on_mouse_move
        self.vb.mouseReleaseEvent = self._on_mouse_release
        
    def disable(self):
        """Disable line drawing mode."""
        self.is_enabled = False
        
        # Restore original mouse event handlers
        self.vb.mousePressEvent = self.original_mouse_press_event
        self.vb.mouseMoveEvent = self.original_mouse_move_event
        self.vb.mouseReleaseEvent = self.original_mouse_release_event
        
        # Clean up preview line
        if self.line_item:
            self.plot.removeItem(self.line_item)
            self.line_item = None
        
        self.drawing = False
        self.start_point = None
    
    def _on_mouse_press(self, event):
        """Handle mouse press for line drawing."""
        if not self.is_enabled:
            self.original_mouse_press_event(event)
            return
        
        # Get position in view coordinates
        scene_pos = event.scenePos()
        if not self.plot.sceneBoundingRect().contains(scene_pos):
            self.original_mouse_press_event(event)
            return
        
        view_pos = self.vb.mapSceneToView(scene_pos)
        
        if not self.drawing:
            # Start new line
            self.drawing = True
            self.start_point = (view_pos.x(), view_pos.y())
            if self.line_item:
                self.plot.removeItem(self.line_item)
                self.line_item = None
        else:
            # End line and emit callback
            end_point = (view_pos.x(), view_pos.y())
            self.drawing = False
            
            # Remove preview line
            if self.line_item:
                self.plot.removeItem(self.line_item)
                self.line_item = None
            
            # Call the measurement callback
            self.on_line_drawn_callback(self.start_point, end_point)
            self.start_point = None
        
        event.accept()
    
    def _on_mouse_move(self, event):
        """Handle mouse move for line drawing preview."""
        if not self.is_enabled:
            self.original_mouse_move_event(event)
            return
        
        if not self.drawing or self.start_point is None:
            self.original_mouse_move_event(event)
            return
        
        scene_pos = event.scenePos()
        if not self.plot.sceneBoundingRect().contains(scene_pos):
            self.original_mouse_move_event(event)
            return
        
        view_pos = self.vb.mapSceneToView(scene_pos)
        
        # Update preview line
        if self.line_item:
            self.plot.removeItem(self.line_item)
        
        self.line_item = pg.PlotDataItem(
            [self.start_point[0], view_pos.x()],
            [self.start_point[1], view_pos.y()],
            pen=pg.mkPen('y', width=2, style=QtCore.Qt.DashLine)
        )
        self.plot.addItem(self.line_item)
        event.accept()
    
    def _on_mouse_release(self, event):
        """Handle mouse release."""
        if not self.is_enabled:
            self.original_mouse_release_event(event)
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
        self.roi_to_fft_window = {}  # Map ROI to its FFT window
        
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
            self.resize(1000, 900)
            
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error Loading File", str(e))
            
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
        
        x_offset, x_scale = self.ax_x.offset, self.ax_x.scale
        y_offset, y_scale = self.ax_y.offset, self.ax_y.scale
        w = self.ax_x.size * x_scale
        h = self.ax_y.size * y_scale
        self.img_orig.setRect(QtCore.QRectF(x_offset, y_offset, w, h))
        
        self.p1.setLabel('bottom', f"{self.ax_x.name}", units=self.ax_x.units)
        self.p1.setLabel('left', f"{self.ax_y.name}", units=self.ax_y.units)
        
        # Create initial ROI
        self._add_new_roi(x_offset, y_offset, w, h)
        
        # Initialize line drawing tool
        self.line_tool = LineDrawingTool(self.p1, self._on_line_drawn)
        
    def _create_toolbar(self) -> QtWidgets.QHBoxLayout:
        """Create toolbar with controls."""
        layout = QtWidgets.QHBoxLayout()
        
        btn_add_roi = QtWidgets.QPushButton("Add New ROI")
        btn_add_roi.clicked.connect(lambda: self._add_new_roi())
        layout.addWidget(btn_add_roi)
        
        btn_measure = QtWidgets.QPushButton("Measure Distance")
        btn_measure.clicked.connect(self._enable_line_measurement)
        layout.addWidget(btn_measure)
        
        layout.addStretch()
        return layout
        
    def _add_new_roi(self, x_offset=None, y_offset=None, w=None, h=None):
        """Add a new ROI box to the image."""
        if x_offset is None:
            x_offset, y_offset = self.ax_x.offset, self.ax_y.offset
            w = self.ax_x.size * self.ax_x.scale
            h = self.ax_y.size * self.ax_y.scale
        
        # Create ROI with unique color
        colors = ['r', 'g', 'b', 'y', 'c', 'm']
        color = colors[self.roi_count % len(colors)]
        
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
        
        # Create a closure to capture roi_id and text_item
        def on_roi_finished():
            self._on_roi_finished(roi, roi_id, text_item)
        
        roi.sigRegionChangeFinished.connect(on_roi_finished)
        self.rois.append(roi)
        
    def _on_roi_finished(self, roi: pg.RectROI, roi_id: int, text_item: pg.TextItem = None):
        """Handle ROI finished moving - open or update FFT window."""
        region = roi.getArrayRegion(self.data, self.img_orig)
        
        if region is None or region.shape[0] < 2 or region.shape[1] < 2:
            return
        
        # Update text label position
        if text_item:
            text_item.setPos(roi.pos()[0], roi.pos()[1])
        
        # Check if this ROI already has an FFT window
        if roi in self.roi_to_fft_window:
            # Update existing window
            fft_window = self.roi_to_fft_window[roi]
            fft_window.region = region
            fft_window.compute_and_display_fft()
        else:
            # Create new FFT window
            roi_name = f"ROI {roi_id}"
            # Get parent window title
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
        
    def _enable_line_measurement(self):
        """Enable line drawing tool for distance measurement."""
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
        # Get scales for both axes
        scale_x = self.ax_x.scale
        scale_y = self.ax_y.scale
        
        result = utils.measure_line_distance(p1, p2, scale_x, scale_y, self.is_reciprocal_space)
        
        msg = f"Distance: {result['distance_physical']:.4f} {self.ax_x.units}\n"
        msg += f"Pixels: {result['distance_pixels']:.1f}\n"
        
        if self.is_reciprocal_space and 'd_spacing' in result:
            msg += f"d-spacing: {result['d_spacing']:.4f} Å"
        
        QtWidgets.QMessageBox.information(self, "Measurement Result", msg)
        
        # Draw the line on the plot
        line = pg.PlotDataItem(
            [p1[0], p2[0]],
            [p1[1], p2[1]],
            pen=pg.mkPen('w', width=2)
        )
        self.p1.addItem(line)
        
        # Format the distance label with units
        if self.is_reciprocal_space and 'd_spacing' in result:
            label_text = f"d: {result['d_spacing']:.4f} Å\n({result['distance_physical']:.4f} {self.ax_x.units}⁻¹)"
        else:
            label_text = f"{result['distance_physical']:.4f} {self.ax_x.units}\n({result['distance_pixels']:.1f} px)"
        
        # Calculate midpoint with small offset for better visibility
        mid_x = (p1[0] + p2[0]) / 2
        mid_y = (p1[1] + p2[1]) / 2
        
        # Create text annotation similar to ROI labels
        text_item = pg.TextItem(
            label_text,
            anchor=(0, 0),
            fill=pg.mkBrush(255, 255, 100, 220)  # Yellow background
        )
        text_item.setPos(mid_x, mid_y)
        self.p1.addItem(text_item)
        

class MainWindow(QtWidgets.QMainWindow):
    """Main application window with drag-and-drop support."""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Fast FFT Image Analyzer")
        self.resize(600, 400)
        
        # Setup drop zone
        self.setAcceptDrops(True)
        
        # Create central widget with instructions
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
        
        # Recent files list
        self.recent_files = []
        
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
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
