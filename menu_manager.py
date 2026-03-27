# SPDX-License-Identifier: GPL-2.0-only
# Copyright (C) 2026 Cooper Stuntz
# See LICENSE for full license terms.

"""Menu creation and keyboard shortcut management for TEMinator."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Literal, Mapping, Optional

from pyqtgraph.Qt import QtGui, QtWidgets

logger = logging.getLogger(__name__)


MenuRole = Literal["main", "viewer"]


@dataclass
class MenuItemConfig:
    """Configuration for a single menu item.

    Attributes:
        title: Display name of the menu item
        shortcut: Keyboard shortcut (e.g., "Ctrl+O"), empty string for none
        callback: Function to call when item is triggered
        is_implemented: Whether this feature is fully implemented
        requires_image: Whether this requires an active image to be useful
        requires_edx: Whether this requires EDX data to be available
        menu_path: Menu path like "File" or "Display"
    """

    title: str
    shortcut: str
    callback: Callable
    is_implemented: bool = True
    requires_image: bool = False
    requires_edx: bool = False
    menu_path: str = ""


class MenuBuilder:
    """Builder class for creating menus with consistent handling of shortcuts and state."""

    def __init__(
        self,
        parent: QtWidgets.QMainWindow,
        logger_instance: logging.Logger | None = None,
    ):
        """Initialize the menu builder.

        Args:
            parent: The QMainWindow to attach menus to
            logger_instance: Optional logger instance for debug messages
        """
        self.parent = parent
        self.logger = logger_instance or logger
        self.menu_bar = parent.menuBar()
        self.actions: Dict[str, QtWidgets.QAction] = {}
        self.menus: Dict[str, QtWidgets.QMenu] = {}

    def clear(self) -> None:
        """Clear all menus from the menu bar."""
        self.menu_bar.clear()
        self.actions.clear()
        self.menus.clear()

    def add_menu_item(
        self,
        menu: QtWidgets.QMenu,
        title: str,
        callback: Callable,
        shortcut: str = "",
        is_enabled: bool = True,
    ) -> QtWidgets.QAction:
        """Add a menu action with optional keyboard shortcut.

        Args:
            menu: The QMenu to add the action to
            title: The text label for the menu item
            callback: The function to call when the action is triggered
            shortcut: Optional keyboard shortcut string (e.g., "Ctrl+O")
            is_enabled: Whether the action should be enabled

        Returns:
            The created QAction
        """
        action = menu.addAction(title, callback)
        action.setEnabled(is_enabled)

        if shortcut:
            action.setShortcut(QtGui.QKeySequence(shortcut))
            self.logger.debug(f"Menu item '{title}' assigned shortcut: {shortcut}")

        # Store action with a unique key for later reference
        key = f"{menu.title()}::{title}"
        self.actions[key] = action

        return action

    def build_from_config(
        self,
        config: List[MenuItemConfig],
        image_available: bool = False,
        edx_available: bool = False,
    ) -> Dict[str, QtWidgets.QAction]:
        """Build menus from a list of MenuItemConfigs.

        This method:
        1. Groups items by menu_path
        2. Creates menu structure
        3. Adds items with proper enabled/disabled state based on:
           - Whether the item is implemented
           - Whether an image is available (if required)
           - Whether EDX data is available (if required)

        Args:
            config: List of MenuItemConfig items
            image_available: Whether an active image is available
            edx_available: Whether EDX data is available

        Returns:
            Dictionary of all created actions, keyed by menu path and title
        """
        self.clear()

        # Group items by menu
        menu_items: Dict[str, List[MenuItemConfig]] = {}
        for item in config:
            if item.menu_path not in menu_items:
                menu_items[item.menu_path] = []
            menu_items[item.menu_path].append(item)

        # Create menus and add items
        created_actions: Dict[str, QtWidgets.QAction] = {}

        for menu_name, items in menu_items.items():
            menu = self.menu_bar.addMenu(menu_name)
            self.menus[menu_name] = menu

            for item in items:
                # Determine if item should be enabled
                is_enabled = item.is_implemented and (
                    (not item.requires_image or image_available) and
                    (not item.requires_edx or edx_available)
                )

                action = self.add_menu_item(
                    menu,
                    item.title,
                    item.callback,
                    shortcut=item.shortcut,
                    is_enabled=is_enabled,
                )

                created_actions[f"{menu_name}::{item.title}"] = action

                if not is_enabled:
                    if not item.is_implemented:
                        reason = "not implemented"
                    elif item.requires_edx and not edx_available:
                        reason = "requires EDX data"
                    else:
                        reason = "requires active image"
                    self.logger.debug(f"Disabled menu item '{item.title}' ({reason})")

        return created_actions

    def set_action_enabled(
        self,
        menu_name: str,
        item_title: str,
        enabled: bool,
    ) -> bool:
        """Enable or disable a specific menu action.

        Args:
            menu_name: The menu name (e.g., "File", "Edit")
            item_title: The menu item title
            enabled: Whether to enable or disable

        Returns:
            True if the action was found and updated, False otherwise
        """
        key = f"{menu_name}::{item_title}"
        if key in self.actions:
            self.actions[key].setEnabled(enabled)
            return True
        return False


def create_shared_menu_config() -> List[MenuItemConfig]:
    """Create the comprehensive menu configuration for TEMinator.

    This configuration includes all menu items from both main_window and image_viewer.
    Items are marked with flags to indicate:
    - Whether they're fully implemented
    - Whether they require an active image

    Windows will use the appropriate subset and state for their context.

    Returns:
        List of MenuItemConfig items for all standard menus
    """
    # Note: callback functions will be replaced with actual methods from the window class
    # For now, using dummy lambdas - subclasses will override these
    return [
        # File menu
        MenuItemConfig(
            title="Open",
            shortcut="Ctrl+O",
            callback=lambda: None,
            is_implemented=True,
            requires_image=False,
            menu_path="File",
        ),
        MenuItemConfig(
            title="Save View",
            shortcut="Ctrl+S",
            callback=lambda: None,
            is_implemented=True,
            requires_image=True,
            menu_path="File",
        ),
        MenuItemConfig(
            title="Build Figure",
            shortcut="Ctrl+B",
            callback=lambda: None,
            is_implemented=False,
            requires_image=True,
            menu_path="File",
        ),
        MenuItemConfig(
            title="Calibrate",
            shortcut="",
            callback=lambda: None,
            is_implemented=True,
            requires_image=True,
            menu_path="File",
        ),
        MenuItemConfig(
            title="Parameters",
            shortcut="Ctrl+,",
            callback=lambda: None,
            is_implemented=True,
            requires_image=False,
            menu_path="File",
        ),
        # Manipulate menu
        MenuItemConfig(
            title="FFT",
            shortcut="f",
            callback=lambda: None,
            is_implemented=True,
            requires_image=True,
            menu_path="Manipulate",
        ),
        MenuItemConfig(
            title="Inverse FFT",
            shortcut="Shift+F",
            callback=lambda: None,
            is_implemented=True,
            requires_image=True,
            menu_path="Manipulate",
        ),
        MenuItemConfig(
            title="Adjust",
            shortcut="a",
            callback=lambda: None,
            is_implemented=True,
            requires_image=True,
            menu_path="Manipulate",
        ),
        # Measure menu
        MenuItemConfig(
            title="Distance",
            shortcut="d",
            callback=lambda: None,
            is_implemented=True,
            requires_image=True,
            menu_path="Measure",
        ),
        MenuItemConfig(
            title="History",
            shortcut="h",
            callback=lambda: None,
            is_implemented=True,
            requires_image=True,
            menu_path="Measure",
        ),
        MenuItemConfig(
            title="Intensity",
            shortcut="i",
            callback=lambda: None,
            is_implemented=False,
            requires_image=True,
            menu_path="Measure",
        ),
        MenuItemConfig(
            title="Profile",
            shortcut="p",
            callback=lambda: None,
            is_implemented=True,
            requires_image=True,
            menu_path="Measure",
        ),
        # Display menu - shared items
        MenuItemConfig(
            title="Metadata",
            shortcut="m",
            callback=lambda: None,
            is_implemented=True,
            requires_image=True,
            menu_path="Display",
        ),
        # Display menu - image viewer specific (but included for completeness)
        MenuItemConfig(
            title="Render Diagnostics",
            shortcut="",
            callback=lambda: None,
            is_implemented=True,
            requires_image=True,
            menu_path="Display",
        ),
        MenuItemConfig(
            title="Cycle Colormap Forward",
            shortcut="+",
            callback=lambda: None,
            is_implemented=True,
            requires_image=True,
            menu_path="Display",
        ),
        MenuItemConfig(
            title="Cycle Colormap Backward",
            shortcut="-",
            callback=lambda: None,
            is_implemented=True,
            requires_image=True,
            menu_path="Display",
        ),
        # EDS menu
        MenuItemConfig(
            title="Toggle Spectra Panel",
            shortcut="",
            callback=lambda: None,
            is_implemented=True,
            requires_image=True,
            requires_edx=True,
            menu_path="EDS",
        ),
        MenuItemConfig(
            title="Select Integration Region",
            shortcut="",
            callback=lambda: None,
            is_implemented=True,
            requires_image=True,
            requires_edx=True,
            menu_path="EDS",
        ),
        MenuItemConfig(
            title="Export EDS Results",
            shortcut="",
            callback=lambda: None,
            is_implemented=True,
            requires_image=True,
            requires_edx=True,
            menu_path="EDS",
        ),
        # Help menu
        MenuItemConfig(
            title="Keyboard Shortcuts",
            shortcut="?",
            callback=lambda: None,
            is_implemented=True,
            requires_image=False,
            menu_path="Help",
        ),
        MenuItemConfig(
            title="README",
            shortcut="",
            callback=lambda: None,
            is_implemented=True,
            requires_image=False,
            menu_path="Help",
        ),
        MenuItemConfig(
            title="About",
            shortcut="",
            callback=lambda: None,
            is_implemented=True,
            requires_image=False,
            menu_path="Help",
        ),
    ]


def build_menu_config_for_role(
    *,
    role: MenuRole,
    callbacks_map: Mapping[str, Callable],
    not_implemented_factory: Optional[Callable[[str], Callable]] = None,
) -> List[MenuItemConfig]:
    """Create shared menu config and bind callbacks for a specific window role.

    Args:
        role: Window role that determines default callback policy.
        callbacks_map: Role-specific callback map keyed by menu item title.
        not_implemented_factory: Optional factory for fallback callbacks used
            by the main window when no explicit callback is provided.

    Returns:
        Fully bound list of menu item configs.
    """
    config = create_shared_menu_config()

    for item in config:
        callback = callbacks_map.get(item.title)
        if callback is not None:
            item.callback = callback
            continue

        if role == "main" and not_implemented_factory is not None:
            item.callback = not_implemented_factory(item.title)

    return config
