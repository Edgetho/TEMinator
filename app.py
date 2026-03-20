# TEMinator
# Copyright (C) 2026 Cooper Stuntz
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; version 2 of the License.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, see <https://www.gnu.org/licenses/>.

"""Application entrypoint and main window wiring.

This module provides the top-level ``main`` function and the
minimal Qt bootstrap; all substantial UI classes live in:

- dialogs.py            – history, metadata, and tone-curve dialogs
- measurement_tools.py  – line drawing, FFT ROI, measurement labels
- scale_bars.py         – static and dynamic scale-bar items
- image_viewer.py       – ImageViewerWindow and helpers
"""

import argparse
import sys
from pathlib import Path

from pyqtgraph.Qt import QtWidgets, QtCore, QtGui

from image_viewer import open_image_file, DirectoryFuzzyOpenDialog


DEFAULT_MAIN_WINDOW_SIZE = (600, 400)


class MainWindow(QtWidgets.QMainWindow):
    """Main application window with drag-and-drop support."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("TEMinator!")
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
        self._setup_menu_bar()

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QVBoxLayout(central)

        label = QtWidgets.QLabel(
            "<div style='text-align:center;'>"
            "<p style='margin:0;'>Drag and drop an image file here to open it.</p>"
            "<p style='margin:6px 0 12px 0;'>(Supports DM3, DM4, TIFF, and other HyperSpy formats)</p>"
            "<p style='margin:6px;'> Provided without warranty under the GNU GPLv2.<.</p>"
            "<p style='margin:6px 0 0 0;'>Source: "
            "<a href='https://github.com/Edgetho/TEMinator'>github.com/Edgetho/TEMinator</a></p>"
            "<p style='margin:6px 0 0 0;'>&copy; 2026 Cooper Stuntz</p>"
            "</div>"
        )
        label.setAlignment(QtCore.Qt.AlignCenter)
        label.setTextFormat(QtCore.Qt.RichText)
        label.setOpenExternalLinks(True)
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

    def _setup_menu_bar(self) -> None:
        menu_bar = self.menuBar()
        menu_bar.clear()

        file_menu = menu_bar.addMenu("File")
        act_open = file_menu.addAction("Open")
        act_open.triggered.connect(self._open_file_dialog)
        act_save_view = file_menu.addAction("Save View", lambda: self._show_not_implemented("Save View"))
        act_build_figure = file_menu.addAction("Build Figure", lambda: self._show_not_implemented("Build Figure"))
        act_calibrate_image = file_menu.addAction("Calibrate Image", lambda: self._show_not_implemented("Calibrate Image"))
        act_parameters = file_menu.addAction("Parameters", lambda: self._show_not_implemented("Parameters"))

        manipulate_menu = menu_bar.addMenu("Manipulate")
        act_fft = manipulate_menu.addAction("FFT", lambda: self._show_not_implemented("FFT"))
        act_inverse_fft = manipulate_menu.addAction("Inverse FFT", lambda: self._show_not_implemented("Inverse FFT"))

        measure_menu = menu_bar.addMenu("Measure")
        act_distance = measure_menu.addAction("Distance", lambda: self._show_not_implemented("Distance"))
        act_history = measure_menu.addAction("History", lambda: self._show_not_implemented("History"))
        act_intensity = measure_menu.addAction("Intensity", lambda: self._show_not_implemented("Intensity"))
        act_profile = measure_menu.addAction("Profile", lambda: self._show_not_implemented("Profile"))

        display_menu = menu_bar.addMenu("Display")
        act_adjust = display_menu.addAction("Adjust", lambda: self._show_not_implemented("Adjust"))
        act_metadata = display_menu.addAction("Metadata", lambda: self._show_not_implemented("Metadata"))

        # Startup state: only File/Open and File/Parameters are enabled.
        for action in (
            act_save_view,
            act_build_figure,
            act_fft,
            act_inverse_fft,
            act_distance,
            act_history,
            act_intensity,
            act_profile,
            act_adjust,
            act_metadata,
            act_calibrate_image
        ):
            action.setEnabled(False)

        act_open.setEnabled(True)
        act_parameters.setEnabled(True)

    def _show_not_implemented(self, feature_name: str) -> None:
        QtWidgets.QMessageBox.information(
            self,
            feature_name,
            f"{feature_name} is planned but not implemented yet.",
        )

    def _open_file_dialog(self) -> None:
        selected_file, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Open Image",
            str(Path.cwd()),
            "Image files (*.dm3 *.dm4 *.emi *.tif *.tiff *.mrc *.ser *.png *.jpg *.jpeg);;All files (*)",
        )
        if selected_file:
            self._open_image(selected_file)

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
        if self.isVisible():
            self.close()

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

        self._open_image(str(path))

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


def _parse_cli_args(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    """Parse TEMinator CLI arguments and return remaining Qt arguments."""
    parser = argparse.ArgumentParser(
        prog="teminator",
        description="Launch TEMinator and optionally open an image file.",
    )
    parser.add_argument(
        "image",
        nargs="?",
        help="Optional image path to open on startup.",
    )
    return parser.parse_known_args(argv)


def main() -> None:
    """Main entry point for the application."""
    cli_args, qt_args = _parse_cli_args(sys.argv[1:])
    app = QtWidgets.QApplication([sys.argv[0], *qt_args])

    # Optional: set a fun custom application icon if available.
    # Place your icon file next to this app.py as "app_icon.png"
    # (or adjust the filename below).
    icon_path = Path(__file__).with_name("app_icon.png")
    if icon_path.is_file():
        app.setWindowIcon(QtGui.QIcon(str(icon_path)))

    window = MainWindow()
    window.show()

    if cli_args.image:
        startup_path = Path(cli_args.image).expanduser()
        if not startup_path.is_absolute():
            startup_path = (Path.cwd() / startup_path).resolve()
        if startup_path.is_file():
            window._open_image(str(startup_path))
        else:
            QtWidgets.QMessageBox.warning(
                window,
                "Open File",
                f"File not found: {startup_path}",
            )

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
