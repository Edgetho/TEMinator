# SPDX-License-Identifier: GPL-2.0-only
# Copyright (C) 2026 Cooper Stuntz
# See LICENSE for full license terms.

"""Shared file-open helpers for image viewer windows."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional, Type

from pyqtgraph.Qt import QtWidgets

IMAGE_EXTENSIONS = (
    ".dm3",
    ".dm4",
    ".emi",
    ".tif",
    ".tiff",
    ".mrc",
    ".ser",
    ".png",
    ".jpg",
    ".jpeg",
)

IMAGE_FILE_FILTER = "Image files (*.dm3 *.dm4 *.emi *.tif *.tiff *.mrc *.ser *.png *.jpg *.jpeg);;All files (*)"


def resolve_image_path(filename: str, base_directory: Path) -> Optional[Path]:
    """Resolve a user-provided image filename against a base directory.

    Args:
        filename: Input value for filename.
        base_directory: Input value for base directory.

    Returns:
        Detailed parameter description.

    """

    name = (filename or "").strip().strip('"').strip("'")
    if not name:
        return None

    candidate = Path(name)
    if not candidate.is_absolute():
        candidate = base_directory / candidate

    return candidate


def open_image_by_name(
    parent: QtWidgets.QWidget,
    filename: str,
    base_directory: Path,
    open_callback: Callable[[str], None],
) -> bool:
    """Resolve and open an image filename, displaying user-facing errors.

    Args:
        parent: Optional parent widget or owning object.
        filename: Input value for filename.
        base_directory: Input value for base directory.
        open_callback: Input value for open callback.

    Returns:
        Detailed parameter description.

    """

    path = resolve_image_path(filename, base_directory)
    if path is None:
        return False

    if not path.is_file():
        QtWidgets.QMessageBox.warning(parent, "Open File", f"File not found: {path}")
        return False

    open_callback(str(path))
    return True


def open_directory_fuzzy_dialog(
    parent: QtWidgets.QWidget,
    directory: Path,
    dialog_cls: Type[QtWidgets.QDialog],
) -> bool:
    """Open a directory-backed fuzzy-open dialog with validation.

    Args:
        parent: Optional parent widget or owning object.
        directory: Input value for directory.
        dialog_cls: Input value for dialog cls.

    Returns:
        Detailed parameter description.

    """

    if not directory.is_dir():
        QtWidgets.QMessageBox.warning(
            parent,
            "Directory",
            f"Directory not found: {directory}",
        )
        return False

    dialog = dialog_cls(parent, directory)
    dialog.exec_()
    return True
