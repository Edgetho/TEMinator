"""Application entrypoint and main window wiring.

This module provides the top-level ``main`` function and the
minimal Qt bootstrap; all substantial UI classes live in:

- dialogs.py            – history, metadata, and tone-curve dialogs
- measurement_tools.py  – line drawing, FFT ROI, measurement labels
- scale_bars.py         – static and dynamic scale-bar items
- fft_viewer.py         – FFTViewerWindow for individual ROIs
- image_viewer.py       – ImageViewerWindow and helpers
"""

import sys
from pathlib import Path

from pyqtgraph.Qt import QtWidgets, QtCore, QtGui

from image_viewer import open_image_file, DirectoryFuzzyOpenDialog


DEFAULT_MAIN_WINDOW_SIZE = (600, 400)


class MainWindow(QtWidgets.QMainWindow):
    """Main application window with drag-and-drop support."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Fast FFT Image Analyzer")
        self.resize(*DEFAULT_MAIN_WINDOW_SIZE)
        self.setAcceptDrops(True)

        # Vim-style command line (":" commands available even before an image is open)
        self.command_edit: QtWidgets.QLineEdit | None = None

        self._setup_ui()

        # Install global event filter so ':' works regardless of focus
        app = QtWidgets.QApplication.instance()
        if app is not None:
            app.installEventFilter(self)

        # Direct ':' shortcut as a backup so command mode is easy to enter
        colon_shortcut = QtGui.QShortcut(QtGui.QKeySequence(":"), self)
        colon_shortcut.activated.connect(self._enter_command_mode)

    def _setup_ui(self) -> None:
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

        # Hidden vim-style command line at the bottom
        self.command_edit = QtWidgets.QLineEdit()
        self.command_edit.setPlaceholderText("Vim command (:e <file>, :E)")
        self.command_edit.returnPressed.connect(self._execute_command_from_line)
        self.command_edit.hide()
        layout.addWidget(self.command_edit)

    def dragEnterEvent(self, event):  # type: ignore[override]
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):  # type: ignore[override]
        for url in event.mimeData().urls():
            file_path = url.toLocalFile()
            if file_path:
                self._open_image(file_path)

    def _open_image(self, file_path: str) -> None:
        open_image_file(file_path)

    # Vim-style command handling -------------------------------------

    def eventFilter(self, obj, event):  # type: ignore[override]
        """Capture ':' globally when this window is active."""
        if self.isActiveWindow() and event.type() == QtCore.QEvent.KeyPress:
            key_event = event  # QKeyEvent
            # Enter command mode on ':' with no modifiers
            if getattr(key_event, "text", lambda: "")() == ":" and not key_event.modifiers():
                self._enter_command_mode()
                return True

            # Allow Esc to cancel command mode
            if (
                self.command_edit is not None
                and self.command_edit.isVisible()
                and getattr(key_event, "key", lambda: None)() == QtCore.Qt.Key_Escape
            ):
                self._exit_command_mode()
                return True

        return super().eventFilter(obj, event)

    def _enter_command_mode(self) -> None:
        if self.command_edit is None:
            return
        self.command_edit.show()
        self.command_edit.clear()
        self.command_edit.setText(":")
        self.command_edit.setFocus()
        self.command_edit.setCursorPosition(len(self.command_edit.text()))

    def _exit_command_mode(self) -> None:
        if self.command_edit is None:
            return
        self.command_edit.clear()
        self.command_edit.hide()
        # Return focus to the main window
        self.setFocus()

    def _execute_command_from_line(self) -> None:
        if self.command_edit is None:
            return

        text = self.command_edit.text().strip()
        if text.startswith(":"):
            text = text[1:]
        text = text.strip()
        if not text:
            self._exit_command_mode()
            return

        parts = text.split(maxsplit=1)
        cmd = parts[0]
        arg = parts[1] if len(parts) > 1 else ""

        handled = self._run_vim_command(cmd, arg)
        if not handled:
            QtWidgets.QMessageBox.information(
                self,
                "Command",
                f"Unknown command: {cmd}",
            )

        self._exit_command_mode()

    def _run_vim_command(self, cmd: str, arg: str) -> bool:
        """Dispatch vim-like commands available on the initial window.

        Supported commands here:
          :e <filename>   – open a file (relative to current working directory by default)
          :E              – fuzzy-open a file in the current working directory
        """

        cmd_str = cmd.strip()
        if not cmd_str:
            return False

        upper_cmd = cmd_str.upper()
        lower_cmd = cmd_str.lower()

        # :E – directory fuzzy finder rooted at CWD
        if upper_cmd == "E" and not arg:
            self._open_directory_fuzzy_view()
            return True

        # :e <filename> – open file, default base is CWD
        if lower_cmd == "e":
            if not arg:
                QtWidgets.QMessageBox.information(
                    self,
                    "Command",
                    "Usage: :e <filename>",
                )
                return True
            self._open_file_by_name(arg)
            return True

        return False

    def _open_file_by_name(self, filename: str) -> None:
        name = filename.strip().strip("\"").strip("'")
        if not name:
            return

        path = Path(name)
        if not path.is_absolute():
            base = Path.cwd()
            path = base / name

        if not path.is_file():
            QtWidgets.QMessageBox.warning(
                self,
                "Open File",
                f"File not found: {path}",
            )
            return

        open_image_file(str(path))

    def _open_directory_fuzzy_view(self) -> None:
        directory = Path.cwd()
        if not directory.is_dir():
            QtWidgets.QMessageBox.warning(
                self,
                "Directory",
                f"Directory not found: {directory}",
            )
            return

        dialog = DirectoryFuzzyOpenDialog(self, directory)
        dialog.exec_()


def main() -> None:
    """Main entry point for the application."""
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
