"""Application entrypoint and main window wiring.

This module now provides only the top-level ``main`` function and the
minimal Qt bootstrap; all substantial UI classes live in:

- dialogs.py            – history, metadata, and tone-curve dialogs
- measurement_tools.py  – line drawing, FFT ROI, measurement labels
- scale_bars.py         – static and dynamic scale-bar items
- fft_viewer.py         – FFTViewerWindow for individual ROIs
- image_viewer.py       – ImageViewerWindow and ``open_image_file`` helper
"""

import sys

from pyqtgraph.Qt import QtWidgets, QtCore

from image_viewer import open_image_file


DEFAULT_MAIN_WINDOW_SIZE = (600, 400)


class MainWindow(QtWidgets.QMainWindow):
    """Main application window with drag-and-drop support."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Fast FFT Image Analyzer")
        self.resize(*DEFAULT_MAIN_WINDOW_SIZE)
        self.setAcceptDrops(True)

        self._setup_ui()

    def _setup_ui(self):
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

    def dragEnterEvent(self, event):  # type: ignore[override]
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):  # type: ignore[override]
        for url in event.mimeData().urls():
            file_path = url.toLocalFile()
            if file_path:
                self._open_image(file_path)

    def _open_image(self, file_path: str):
        open_image_file(file_path)


def main() -> None:
    """Main entry point for the application."""
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

    def _on_measurement_label_clicked(self, label: pg.TextItem):
        """Select a measurement when its label is clicked."""
        selected_index = None
        for idx, (line_item, text_item) in enumerate(self.measurement_items):
            if text_item is label:
                selected_index = idx
                break

        self.selected_measurement_index = selected_index

        # Visually highlight the selected label (if supported by this pg version)
        for idx, (line_item, text_item) in enumerate(self.measurement_items):
            if idx == selected_index:
                self._set_label_fill(text_item, pg.mkBrush(255, 200, 0, 255))  # brighter highlight
            else:
                self._set_label_fill(text_item, LABEL_BRUSH_COLOR)

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
            self._set_label_fill(t, LABEL_BRUSH_COLOR)

    def _set_label_fill(self, text_item: pg.TextItem, brush: pg.QtGui.QBrush):
        """Set the background/fill color of a label safely across pyqtgraph versions."""
        if hasattr(text_item, "setFill"):
            text_item.setFill(brush)
        elif hasattr(text_item, "setBrush"):
            text_item.setBrush(brush)

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
        

def open_image_file(file_path: str):
    """Open an image file; if it contains multiple images, open one window per image.

    This helper is used by both the main window and individual
    ImageViewerWindows so that dragging a supported file onto *any*
    window opens it in a new viewer.
    """
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
        QtWidgets.QMessageBox.critical(None, "Error", f"Could not open file: {str(e)}")


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
        """Open image file in one or more new ImageViewerWindows."""
        open_image_file(file_path)


def main():
    """Main entry point for the application."""
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()


def open_image_file(file_path: str):
    """Open an image file; if it contains multiple images, open one window per image.

    This helper is used by both the main window and individual
    ImageViewerWindows so that dragging a supported file onto *any*
    window opens it in a new viewer.
    """
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
        QtWidgets.QMessageBox.critical(None, "Error", f"Could not open file: {str(e)}")
