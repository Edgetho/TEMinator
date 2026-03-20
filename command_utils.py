# SPDX-License-Identifier: GPL-2.0-only
# Copyright (C) 2026 Cooper Stuntz
# See LICENSE for full license terms.

"""Shared helpers for vim-style command-line handling in Qt windows."""

from __future__ import annotations

from typing import Optional, Tuple

from pyqtgraph.Qt import QtWidgets


def parse_command_input(text: str) -> Optional[Tuple[str, str]]:
    """Parse command-line text into ``(command, argument)``.

    Accepts optional leading ``:`` and trims surrounding whitespace.
    Returns ``None`` when the input does not contain a command.
    """

    clean = (text or "").strip()
    if clean.startswith(":"):
        clean = clean[1:]
    clean = clean.strip()
    if not clean:
        return None

    parts = clean.split(maxsplit=1)
    cmd = parts[0]
    arg = parts[1] if len(parts) > 1 else ""
    return cmd, arg


def enter_command_mode(command_edit: Optional[QtWidgets.QLineEdit]) -> None:
    """Show and focus a command line edit prefilled with ':'."""

    if command_edit is None:
        return

    command_edit.show()
    command_edit.clear()
    command_edit.setText(":")
    command_edit.setFocus()
    command_edit.setCursorPosition(len(command_edit.text()))


def exit_command_mode(
    command_edit: Optional[QtWidgets.QLineEdit],
    focus_target: Optional[QtWidgets.QWidget] = None,
) -> None:
    """Hide and clear a command line edit and optionally restore focus."""

    if command_edit is None:
        return

    command_edit.clear()
    command_edit.hide()

    if focus_target is not None and hasattr(focus_target, "setFocus"):
        focus_target.setFocus()
