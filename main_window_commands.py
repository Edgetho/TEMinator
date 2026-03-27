# SPDX-License-Identifier: GPL-2.0-only
# Copyright (C) 2026 Cooper Stuntz
# See LICENSE for full license terms.

"""Command routing controller for the startup main window."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Protocol, cast

from pyqtgraph.Qt import QtWidgets

from dialogs import DirectoryFuzzyOpenDialog
from file_navigation import open_directory_fuzzy_dialog, open_image_by_name


class _MainWindowCommandsOwner(Protocol):
    """Protocol for objects that own the main window command router."""

    def _open_image(self, file_path: str) -> None:
        """Open an image file.

        Args:
            file_path: Path to the target file on disk.

        """
        ...


class MainWindowCommandRouter:
    """Owns vim-style command dispatch for the startup window."""

    def __init__(self, window: _MainWindowCommandsOwner):
        """Initialize the command router for the main window.

        Args:
            window: The main window instance that will execute commands.
        """
        self.window = window

    def run_vim_command(self, cmd: str, arg: str) -> bool:
        """Execute a vim-style command.

        Supported commands:
        - E: Open directory browser
        - e <filename>: Open a specific file

        Args:
            cmd: Command name (case-insensitive).
            arg: Optional argument for the command.

        Returns:
            True if command was recognized and executed, False otherwise.
        """
        cmd_str = cmd.strip()
        if not cmd_str:
            return False

        upper_cmd = cmd_str.upper()
        lower_cmd = cmd_str.lower()

        if upper_cmd == "E" and not arg:
            self.open_directory_fuzzy_view()
            return True

        if lower_cmd == "e":
            if not arg:
                QtWidgets.QMessageBox.information(
                    self.window, "Command", "Usage: :e <filename>"
                )
                return True
            self.open_file_by_name(arg)
            return True

        return False

    def open_file_by_name(self, filename: str) -> None:
        """Open an image file by name relative to the current working directory.

        Args:
            filename: Filename or relative path to open.
        """
        open_image_by_name(
            parent=cast(QtWidgets.QWidget, self.window),
            filename=filename,
            base_directory=Path.cwd(),
            open_callback=cast(Callable[[str], None], self.window._open_image),
        )

    def open_directory_fuzzy_view(self) -> None:
        """Open a fuzzy file browser starting from the current working directory."""
        open_directory_fuzzy_dialog(
            parent=cast(QtWidgets.QWidget, self.window),
            directory=Path.cwd(),
            dialog_cls=DirectoryFuzzyOpenDialog,
        )
