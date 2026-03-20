# SPDX-License-Identifier: GPL-2.0-only
# Copyright (C) 2026 Cooper Stuntz
# See LICENSE for full license terms.

"""Command routing controller for image-viewer windows."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, cast

from pyqtgraph.Qt import QtWidgets

from dialogs import DirectoryFuzzyOpenDialog
from file_navigation import open_directory_fuzzy_dialog, open_image_by_name
from image_loader import open_image_file


class _LoggerLike(Protocol):
    def debug(self, msg: str, *args) -> None: ...


class _ImageViewerCommandsOwner(Protocol):
    file_path: str
    btn_measure: QtWidgets.QPushButton | None

    def _add_new_fft(self) -> None: ...
    def _toggle_line_measurement(self) -> None: ...
    def _open_adjust_dialog(self) -> None: ...


class ViewerCommandRouter:
    """Owns vim-style command dispatch for a single image-viewer window."""

    def __init__(self, viewer: _ImageViewerCommandsOwner, logger: _LoggerLike):
        self.viewer = viewer
        self.logger = logger

    def open_file_by_name(self, filename: str) -> None:
        """Open an image relative to this viewer's current file directory."""

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
        """Dispatch image-viewer vim-like commands."""

        viewer = self.viewer
        cmd_str = cmd.strip()
        if not cmd_str:
            return False

        self.logger.debug("Executing vim command: cmd=%s arg=%s", cmd, arg)

        upper_cmd = cmd_str.upper()
        lower_cmd = cmd_str.lower()

        if upper_cmd == "F":
            viewer._add_new_fft()
            return True

        if upper_cmd == "D":
            if viewer.btn_measure is not None:
                if not viewer.btn_measure.isChecked():
                    viewer.btn_measure.setChecked(True)
                    viewer._toggle_line_measurement()
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
