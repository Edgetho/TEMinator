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
import sys
from pathlib import Path

from pyqtgraph.Qt import QtWidgets, QtGui

from main_window import MainWindow


def _parse_cli_args(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    """Parse TEMinator CLI arguments and return remaining Qt arguments."""

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

    app = QtWidgets.QApplication([sys.argv[0], *qt_args])

    icon_path = Path(__file__).with_name("app_icon.png")
    if icon_path.is_file():
        app.setWindowIcon(QtGui.QIcon(str(icon_path)))

    window = MainWindow()
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
