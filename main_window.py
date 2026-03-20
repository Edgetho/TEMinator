# SPDX-License-Identifier: GPL-2.0-only
# Copyright (C) 2026 Cooper Stuntz
# See LICENSE for full license terms.

"""Startup/main application window implementation."""

from __future__ import annotations

import logging
from pathlib import Path

import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets, QtCore, QtGui

from command_utils import enter_command_mode, exit_command_mode, parse_command_input
from dialogs import DirectoryFuzzyOpenDialog, RenderSettingsDialog
from file_navigation import (
    IMAGE_FILE_FILTER,
)
from image_loader import open_image_file
from main_window_commands import MainWindowCommandRouter
from menu_manager import MenuBuilder, MenuItemConfig, create_shared_menu_config
from utils import (
    show_about_dialog,
    show_keyboard_shortcuts_dialog,
    show_readme_dialog,
    open_parameters_dialog,
    open_file_dialog,
)
from viewer_settings import (
    load_render_settings,
    save_render_settings,
    global_render_config_options,
    hardware_acceleration_available,
)

logger = logging.getLogger(__name__)


DEFAULT_MAIN_WINDOW_SIZE = (600, 400)


class MainWindow(QtWidgets.QMainWindow):
    """Main application window with drag-and-drop support."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("TEMinator!")
        self.resize(*DEFAULT_MAIN_WINDOW_SIZE)
        self.setAcceptDrops(True)

        self.command_edit: QtWidgets.QLineEdit | None = None
        self._render_settings = load_render_settings()
        self.commands = MainWindowCommandRouter(self)

        self._setup_ui()
        self._update_render_status_indicator()

        app = QtWidgets.QApplication.instance()
        if app is not None:
            app.installEventFilter(self)

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

        self.command_edit = QtWidgets.QLineEdit()
        self.command_edit.setPlaceholderText("Vim command (:e <file>, :E)")
        self.command_edit.returnPressed.connect(self._execute_command_from_line)
        self.command_edit.hide()
        layout.addWidget(self.command_edit)

    def _setup_menu_bar(self) -> None:
        # Get the comprehensive menu configuration
        config = create_shared_menu_config()
        
        # Create callbacks mapping for this window
        callbacks_map = {
            "Open": self._open_file_dialog,
            "Parameters": self._open_parameters_dialog,
            "Keyboard Shortcuts": self._show_keyboard_shortcuts,
            "About": self._show_about_dialog,
            "README": self._show_readme,
            "Save View": lambda: self._show_not_implemented("Save View"),
            "Build Figure": lambda: self._show_not_implemented("Build Figure"),
            "Calibrate": lambda: self._show_not_implemented("Calibrate"),
            "FFT": lambda: self._show_not_implemented("FFT"),
            "Inverse FFT": lambda: self._show_not_implemented("Inverse FFT"),
            "Distance": lambda: self._show_not_implemented("Distance"),
            "History": lambda: self._show_not_implemented("History"),
            "Intensity": lambda: self._show_not_implemented("Intensity"),
            "Profile": lambda: self._show_not_implemented("Profile"),
            "Metadata": lambda: self._show_not_implemented("Metadata"),
            "Render Diagnostics": lambda: self._show_not_implemented("Render Diagnostics"),
            "Cycle Colormap Forward": lambda: self._show_not_implemented("Cycle Colormap Forward"),
            "Cycle Colormap Backward": lambda: self._show_not_implemented("Cycle Colormap Backward"),
        }
        
        # Update config with actual callbacks
        for item in config:
            if item.title in callbacks_map:
                item.callback = callbacks_map[item.title]
        
        # Build menus using the builder - no image is available in main window
        # The MenuBuilder will automatically grey out items that require_image
        self.menu_builder = MenuBuilder(self, logger)
        self.menu_builder.build_from_config(config, image_available=False)
        
        logger.debug("Main window menu bar setup complete with keyboard shortcuts")


    def _show_not_implemented(self, feature_name: str) -> None:
        QtWidgets.QMessageBox.information(
            self,
            feature_name,
            f"{feature_name} is planned but not implemented yet.",
        )

    def _show_keyboard_shortcuts(self) -> None:
        """Display keyboard shortcuts help dialog."""
        config = create_shared_menu_config()
        extra_shortcuts = {
            "Enter command mode": ":",
            "Exit command mode": "Esc",
        }
        show_keyboard_shortcuts_dialog(self, config, extra_shortcuts)

    def _show_readme(self) -> None:
        """Display README content in a scrollable dialog."""
        show_readme_dialog(self)

    def _show_about_dialog(self) -> None:
        """Display the About dialog with app information."""
        show_about_dialog(self)
        logger.debug("Requested about dialog")


    def _render_status_text(self) -> str:
        quality = str(self._render_settings.get("image_resampling_quality", "high")).strip().lower()
        requested_hw = bool(self._render_settings.get("use_hardware_acceleration", True))
        available_hw = hardware_acceleration_available()

        if requested_hw and available_hw:
            backend = "OpenGL"
        elif requested_hw and not available_hw:
            backend = "Raster (OpenGL unavailable)"
        else:
            backend = "Raster (OpenGL disabled)"

        return f"Graphics settings loaded: backend={backend}, resampling={quality}"

    def _update_render_status_indicator(self) -> None:
        self.statusBar().showMessage(self._render_status_text())

    def _open_parameters_dialog(self) -> None:
        current = load_render_settings()
        updated = open_parameters_dialog(self, current)
        if updated is None:
            return

        save_render_settings(updated)
        gl_available = hardware_acceleration_available()
        pg.setConfigOptions(
            **global_render_config_options(updated, hardware_available=gl_available)
        )
        self._render_settings = updated
        self._update_render_status_indicator()

    def _open_file_dialog(self) -> None:
        selected_file = open_file_dialog(self, str(Path.cwd()))
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

    def eventFilter(self, obj, event):  # type: ignore[override]
        if self.isActiveWindow() and event.type() == QtCore.QEvent.KeyPress:
            key_event = event
            if getattr(key_event, "text", lambda: "")() == ":" and not key_event.modifiers():
                self._enter_command_mode()
                return True

            if (
                self.command_edit is not None
                and self.command_edit.isVisible()
                and getattr(key_event, "key", lambda: None)() == QtCore.Qt.Key_Escape
            ):
                self._exit_command_mode()
                return True

        return super().eventFilter(obj, event)

    def _enter_command_mode(self) -> None:
        enter_command_mode(self.command_edit)

    def _exit_command_mode(self) -> None:
        exit_command_mode(self.command_edit, focus_target=self)

    def _execute_command_from_line(self) -> None:
        if self.command_edit is None:
            return

        parsed = parse_command_input(self.command_edit.text())
        if parsed is None:
            self._exit_command_mode()
            return

        cmd, arg = parsed
        handled = self._run_vim_command(cmd, arg)
        if not handled:
            QtWidgets.QMessageBox.information(self, "Command", f"Unknown command: {cmd}")

        self._exit_command_mode()

    def _run_vim_command(self, cmd: str, arg: str) -> bool:
        return self.commands.run_vim_command(cmd, arg)

    def _open_file_by_name(self, filename: str) -> None:
        self.commands.open_file_by_name(filename)

    def _open_directory_fuzzy_view(self) -> None:
        self.commands.open_directory_fuzzy_view()
