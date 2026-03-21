# SPDX-License-Identifier: GPL-2.0-only
# Copyright (C) 2026 Cooper Stuntz
# See LICENSE for full license terms.

"""Shared typing protocols used across TEMinator modules."""

from __future__ import annotations

from typing import Protocol


class LoggerLike(Protocol):
    """Minimal logger protocol for injected logging dependencies."""

    def debug(self, msg: str, *args) -> None:
        """Log a debug message with optional formatting arguments.

        Args:
            msg: Format string or message text.
            *args: Positional values used by logger formatting.
        """
        ...
