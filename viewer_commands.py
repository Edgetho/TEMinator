# SPDX-License-Identifier: GPL-2.0-only
# Copyright (C) 2026 Cooper Stuntz
# See LICENSE for full license terms.

"""Command routing controller for image-viewer windows."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol, cast

from pyqtgraph.Qt import QtWidgets

from dialogs import DirectoryFuzzyOpenDialog
from file_navigation import open_directory_fuzzy_dialog, open_image_by_name
from image_loader import open_image_file
from types_common import LoggerLike


class _ImageViewerCommandsOwner(Protocol):
    """Protocol for image viewer windows managed by ViewerCommandRouter."""

    file_path: str
    btn_measure: QtWidgets.QPushButton | None
    fft_manager: Any
    measurements: Any

    def _open_adjust_dialog(self) -> None:
        """Open the tone curve adjustment dialog."""
        ...


class ViewerCommandRouter:
    """Owns vim-style command dispatch for a single image-viewer window."""

    def __init__(self, viewer: _ImageViewerCommandsOwner, logger: LoggerLike):
        """Initialize the command router for an image viewer window.

        Args:
            viewer: The image viewer window that will execute commands.
            logger: A logger instance for debug output.
        """
        self.viewer = viewer
        self.logger = logger

    def open_file_by_name(self, filename: str) -> None:
        """Open an image relative to this viewer's current file directory.

        Args:
            filename: Input value for filename.

        """

        viewer = self.viewer
        try:
            base = Path(viewer.file_path).parent
        except Exception:
            base = Path.cwd()

        open_image_by_name(
            parent=cast(QtWidgets.QWidget, viewer),
            filename=filename,
            base_directory=base,
            open_callback=open_image_file,
        )

    def open_directory_fuzzy_view(self) -> None:
        """Open fuzzy directory browser rooted at this viewer's file directory."""
        viewer = self.viewer
        try:
            directory = Path(viewer.file_path).parent
        except Exception:
            directory = Path.cwd()

        open_directory_fuzzy_dialog(
            parent=cast(QtWidgets.QWidget, viewer),
            directory=directory,
            dialog_cls=DirectoryFuzzyOpenDialog,
        )

    def run_vim_command(self, cmd: str, arg: str) -> bool:
        """Dispatch image-viewer vim-like commands.

        Args:
            cmd: Parsed command name to execute.
            arg: Optional command argument value.

        Returns:
            Detailed parameter description.

        """

        viewer = self.viewer
        cmd_str = cmd.strip()
        if not cmd_str:
            return False

        self.logger.debug("Executing vim command: cmd=%s arg=%s", cmd, arg)

        upper_cmd = cmd_str.upper()
        lower_cmd = cmd_str.lower()

        if upper_cmd == "F":
            viewer.fft_manager.add_new_fft()
            return True

        if upper_cmd == "D":
            viewer.measurements.start_distance_measurement()
            return True

        if upper_cmd == "A":
            viewer._open_adjust_dialog()
            return True

        if upper_cmd == "E" and not arg:
            self.open_directory_fuzzy_view()
            return True

        if lower_cmd == "e":
            if not arg:
                QtWidgets.QMessageBox.information(
                    cast(QtWidgets.QWidget, viewer),
                    "Command",
                    "Usage: :e <filename>",
                )
                return True
            self.open_file_by_name(arg)
            return True

        return False
