# SPDX-License-Identifier: GPL-2.0-only
# Copyright (C) 2026 Cooper Stuntz
# See LICENSE for full license terms.

"""Shared helpers for vim-style command-line handling in Qt windows."""

from __future__ import annotations

from typing import Callable, Optional, Tuple

from pyqtgraph.Qt import QtCore, QtWidgets


def parse_command_input(text: str) -> Optional[Tuple[str, str]]:
    """Parse command-line text into ``(command, argument)``.

                Accepts optional leading ``:`` and trims surrounding whitespace.
                Returns ``None`` when the input does not contain a command.

                Args:
                    text: User-facing text value for this operation.

                Returns:
                    Detailed parameter description.
            
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
    """Show and focus a command line edit prefilled with ':'.

                Args:
                    command_edit: Input value for command edit.
            
    """

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
    """Hide and clear a command line edit and optionally restore focus.

                Args:
                    command_edit: Input value for command edit.
                    focus_target: Input value for focus target.
            
    """

    if command_edit is None:
        return

    command_edit.clear()
    command_edit.hide()

    if focus_target is not None and hasattr(focus_target, "setFocus"):
        focus_target.setFocus()


class CommandModeController:
    """Own command-mode key handling, visibility, and command dispatch flow."""

    def __init__(
        self,
        *,
        command_edit_getter: Callable[[], Optional[QtWidgets.QLineEdit]],
        run_command: Callable[[str, str], bool],
        on_unknown_command: Callable[[str], None],
        focus_target_getter: Optional[Callable[[], Optional[QtWidgets.QWidget]]] = None,
    ) -> None:
        """Initialize command-mode orchestration callbacks.

        Args:
            command_edit_getter: Returns the command line edit instance for this window.
            run_command: Executes a parsed ``(cmd, arg)`` pair and returns ``True`` if handled.
            on_unknown_command: Called when command dispatch returns ``False``.
            focus_target_getter: Optional provider for the widget that should regain focus when
                exiting command mode.
        """
        self._command_edit_getter = command_edit_getter
        self._run_command = run_command
        self._on_unknown_command = on_unknown_command
        self._focus_target_getter = focus_target_getter

    def _command_edit(self) -> Optional[QtWidgets.QLineEdit]:
        """Return the current command line edit widget, if available."""
        return self._command_edit_getter()

    def _focus_target(self) -> Optional[QtWidgets.QWidget]:
        """Return the preferred focus target widget for command-mode exit."""
        if self._focus_target_getter is None:
            return None
        return self._focus_target_getter()

    def handle_key_event(self, is_active_window: bool, event: object) -> bool:
        """Handle ``:`` and ``Esc`` key presses for command mode.

        Args:
            is_active_window: Whether the owning window is currently active.
            event: Incoming Qt event object from an event filter.

        Returns:
            ``True`` if the event was consumed by command-mode logic, otherwise ``False``.
        """
        if not is_active_window:
            return False

        if getattr(event, "type", lambda: None)() != QtCore.QEvent.KeyPress:
            return False

        key_event = event
        if getattr(key_event, "text", lambda: "")() == ":" and not key_event.modifiers():
            self.enter_mode()
            return True

        command_edit = self._command_edit()
        if (
            command_edit is not None
            and command_edit.isVisible()
            and getattr(key_event, "key", lambda: None)() == QtCore.Qt.Key_Escape
        ):
            self.exit_mode()
            return True

        return False

    def enter_mode(self) -> None:
        """Show and focus the command line in vim-style command mode."""
        enter_command_mode(self._command_edit())

    def exit_mode(self) -> None:
        """Hide command mode and restore focus target if provided."""
        exit_command_mode(self._command_edit(), focus_target=self._focus_target())

    def execute_from_line(self) -> None:
        """Parse and dispatch command-line input, handling unknown commands.

        When parsing fails (empty input), command mode is exited with no dispatch.
        """
        command_edit = self._command_edit()
        if command_edit is None:
            return

        parsed = parse_command_input(command_edit.text())
        if parsed is None:
            self.exit_mode()
            return

        cmd, arg = parsed
        handled = self._run_command(cmd, arg)
        if not handled:
            self._on_unknown_command(cmd)

        self.exit_mode()
