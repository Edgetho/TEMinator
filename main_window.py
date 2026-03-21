# SPDX-License-Identifier: GPL-2.0-only
# Copyright (C) 2026 Cooper Stuntz
# See LICENSE for full license terms.

"""Startup/main application window implementation."""

from __future__ import annotations

import logging
from pathlib import Path

import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets, QtCore, QtGui

from command_utils import CommandModeController
from dialogs import DirectoryFuzzyOpenDialog, RenderSettingsDialog
from file_navigation import (
    IMAGE_FILE_FILTER,
)
from image_loader import open_image_file
from main_window_commands import MainWindowCommandRouter
from menu_manager import MenuBuilder, build_menu_config_for_role
from utils import (
    HelpDialogActions,
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
        """Initialize the main application window with UI and event handlers."""
        super().__init__()
        self.setWindowTitle("TEMinator!")
        self.resize(*DEFAULT_MAIN_WINDOW_SIZE)
        self.setAcceptDrops(True)

        self.command_edit: QtWidgets.QLineEdit | None = None
        self._render_settings = load_render_settings()
        self.commands = MainWindowCommandRouter(self)
        self._command_mode = CommandModeController(
            command_edit_getter=lambda: self.command_edit,
            run_command=self.commands.run_vim_command,
            on_unknown_command=lambda cmd: QtWidgets.QMessageBox.information(
                self, "Command", f"Unknown command: {cmd}"
            ),
            focus_target_getter=lambda: self,
        )
        self.help_actions = HelpDialogActions(
            parent_widget=self,
            menu_config_provider=lambda: build_menu_config_for_role(role="main", callbacks_map={}),
            extra_shortcuts_provider=lambda: {
                "Enter command mode": ":",
                "Exit command mode": "Esc",
            },
            logger_instance=logger,
        )

        self._setup_ui()
        self._update_render_status_indicator()

        app = QtWidgets.QApplication.instance()
        if app is not None:
            app.installEventFilter(self)

        colon_shortcut = QtGui.QShortcut(QtGui.QKeySequence(":"), self)
        colon_shortcut.activated.connect(self._enter_command_mode)

    def _setup_ui(self) -> None:
        """Set up the user interface including central widget and command line."""
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
        """Configure the application menu bar with all actions and commands."""
        def not_implemented(feature_name: str):
            """Create a callback for an unimplemented feature action.
            
            Args:
                feature_name: Display name of the feature.
                
            Returns:
                A callable that shows the not-implemented dialog when invoked.
            """
            return lambda: self._show_not_implemented(feature_name)

        config = build_menu_config_for_role(
            role="main",
            callbacks_map={
            "Open": self._open_file_dialog,
            "Parameters": self._open_parameters_dialog,
            "Keyboard Shortcuts": self.help_actions.show_keyboard_shortcuts,
            "About": self.help_actions.show_about,
            "README": self.help_actions.show_readme,
            },
            not_implemented_factory=not_implemented,
        )
        
        # Build menus using the builder - no image is available in main window
        # The MenuBuilder will automatically grey out items that require_image
        self.menu_builder = MenuBuilder(self, logger)
        self.menu_builder.build_from_config(config, image_available=False)
        
        logger.debug("Main window menu bar setup complete with keyboard shortcuts")


    def _show_not_implemented(self, feature_name: str) -> None:
        """Display a dialog indicating that a feature is not yet implemented.
        
        Args:
            feature_name: Name of the feature that is not implemented.
        """
        QtWidgets.QMessageBox.information(
            self,
            feature_name,
            f"{feature_name} is planned but not implemented yet.",
        )

    def _render_status_text(self) -> str:
        """Generate the status bar text displaying render settings and hardware info.
        
        Returns:
            Formatted string showing hardware acceleration and resampling quality.
        """
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
        """Update the status bar to show current render settings."""
        self.statusBar().showMessage(self._render_status_text())

    def _open_parameters_dialog(self) -> None:
        """Open the parameters/settings dialog to configure render quality and hardware acceleration."""
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
        """Open a file selection dialog and load the selected image file."""
        selected_file = open_file_dialog(self, str(Path.cwd()))
        if selected_file:
            self._open_image(selected_file)

    def dragEnterEvent(self, event):  # type: ignore[override]
        """Accept drag events for image files dropped onto the window.

                        Args:
                            event: Qt event object carrying user interaction details.
                    
        """
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):  # type: ignore[override]
        """Handle dropped files by opening them as images.

                        Args:
                            event: Qt event object carrying user interaction details.
                    
        """
        for url in event.mimeData().urls():
            file_path = url.toLocalFile()
            if file_path:
                self._open_image(file_path)

    def _open_image(self, file_path: str) -> None:
        """Load and display an image file.
        
        Args:
            file_path: Path to the image file to open.
        """
        open_image_file(file_path)
        if self.isVisible():
            self.close()

    def eventFilter(self, obj, event):  # type: ignore[override]
        """Handle keyboard shortcuts including vim-style command mode (colon key).

                        Args:
                            obj: Input value for obj.
                            event: Qt event object carrying user interaction details.
                    
        """
        if self._command_mode.handle_key_event(self.isActiveWindow(), event):
            return True

        return super().eventFilter(obj, event)

    def _enter_command_mode(self) -> None:
        """Enter vim-style command mode, showing the command line edit."""
        self._command_mode.enter_mode()

    def _exit_command_mode(self) -> None:
        """Exit vim-style command mode and hide the command line edit."""
        self._command_mode.exit_mode()

    def _execute_command_from_line(self) -> None:
        """Parse and execute a command from the command line, or display an error."""
        self._command_mode.execute_from_line()
