# TEMinator
# Copyright (C) 2026 Cooper Stuntz
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 2 as
# published by the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY, even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, see <https://www.gnu.org/licenses/>.

"""Application entrypoint and Qt bootstrap."""

import argparse
import logging
import os
import sys
from pathlib import Path

import pyqtgraph as pg
from pyqtgraph.Qt import QtGui, QtWidgets

from main_window import MainWindow
from viewer_settings import (
    global_render_config_options,
    hardware_acceleration_available,
    load_render_settings,
    set_effective_render_settings,
)


def _parse_cli_args(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    """Parse TEMinator CLI arguments and return remaining Qt arguments.

    Args:
        argv: Input value for argv.

    Returns:
        Detailed parameter description.

    """

    parser = argparse.ArgumentParser(
        prog="teminator",
        description="Launch TEMinator and optionally open an image file.",
    )
    parser.add_argument(
        "image",
        nargs="?",
        help="Optional image path to open on startup.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose debug logging.",
    )
    parser.add_argument(
        "--force-x11",
        action="store_true",
        help="Force Qt to use the X11/xcb platform backend (Linux only).",
    )
    parser.add_argument(
        "--force-software",
        action="store_true",
        help="Force software rendering (disable hardware acceleration).",
    )
    return parser.parse_known_args(argv)


def main() -> None:
    """Main entry point for the application."""

    cli_args, qt_args = _parse_cli_args(sys.argv[1:])

    logging.basicConfig(
        level=logging.DEBUG if cli_args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
    logger = logging.getLogger(__name__)
    logger.debug("Verbose mode enabled")
    logger.debug("Qt passthrough args: %s", qt_args)

    if cli_args.force_x11 and sys.platform.startswith("linux"):
        os.environ["QT_QPA_PLATFORM"] = "xcb"
        # Also ensure QT_OPENGL is set for consistent X11/xcb rendering
        if "QT_OPENGL" not in os.environ:
            os.environ["QT_OPENGL"] = "desktop"
        logger.info(
            "Forcing Qt platform backend to X11/xcb (--force-x11); QT_OPENGL=%s",
            os.environ.get("QT_OPENGL"),
        )
    logger.debug(
        "Qt environment: QT_QPA_PLATFORM=%r QT_OPENGL=%r QT_XCB_GL_INTEGRATION=%r",
        os.environ.get("QT_QPA_PLATFORM"),
        os.environ.get("QT_OPENGL"),
        os.environ.get("QT_XCB_GL_INTEGRATION"),
    )

    app = QtWidgets.QApplication(["TEMinator", *qt_args])
    app.setOrganizationName("TEMinator")
    app.setApplicationName("TEMinator")
    app.setApplicationDisplayName("TEMinator")

    if (
        sys.platform.startswith("linux")
        and os.environ.get("XDG_SESSION_TYPE") == "wayland"
    ):
        if hasattr(app, "setDesktopFileName"):
            app.setDesktopFileName("teminator")

    settings = load_render_settings()

    # Override hardware acceleration if --force-software is specified
    if cli_args.force_software:
        settings["use_hardware_acceleration"] = False
        logger.info(
            "Force software rendering (--force-software); hardware acceleration disabled"
        )

    # Cache the effective render settings for use by ImageViewer diagnostics
    set_effective_render_settings(settings)

    gl_available = hardware_acceleration_available()
    logger.debug(
        "Render settings at startup: requested_hw=%s gl_available=%s",
        bool(settings.get("use_hardware_acceleration", True)),
        gl_available,
    )
    pg.setConfigOptions(
        **global_render_config_options(settings, hardware_available=gl_available)
    )
    logger.debug("pyqtgraph config useOpenGL=%s", pg.getConfigOption("useOpenGL"))

    if bool(settings.get("use_hardware_acceleration", True)) and not gl_available:
        logger.warning(
            "Hardware acceleration was requested but no OpenGL context is available; "
            "falling back to non-OpenGL rendering."
        )

    icon_path = Path(__file__).with_name("app_icon.png")
    app_icon: QtGui.QIcon | None = None
    if icon_path.is_file():
        loaded_icon = QtGui.QIcon(str(icon_path))
        if not loaded_icon.isNull():
            app_icon = loaded_icon
            app.setWindowIcon(loaded_icon)

    window = MainWindow()
    if app_icon is not None and not app_icon.isNull():
        window.setWindowIcon(app_icon)
    window.show()

    if cli_args.image:
        startup_path = Path(cli_args.image).expanduser()
        if not startup_path.is_absolute():
            startup_path = (Path.cwd() / startup_path).resolve()
        logger.debug("Startup image argument resolved to: %s", startup_path)
        if startup_path.is_file():
            window._open_image(str(startup_path))
        else:
            QtWidgets.QMessageBox.warning(
                window,
                "Open File",
                f"File not found: {startup_path}",
            )

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
