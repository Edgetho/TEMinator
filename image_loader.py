# SPDX-License-Identifier: GPL-2.0-only
# Copyright (C) 2026 Cooper Stuntz
# See LICENSE for full license terms.

"""Image loading entrypoints for launching viewer windows."""

from __future__ import annotations

import logging

import hyperspy.api as hs
import numpy as np
from pyqtgraph.Qt import QtWidgets


logger = logging.getLogger(__name__)


def open_image_file(file_path: str) -> None:
    """Open an image file; if it contains multiple images, open one window per image.

                Args:
                    file_path: Path to the target file on disk.
            
    """

    try:
        from image_viewer import ImageViewerWindow

        logger.debug("Opening image file: %s", file_path)
        loaded = hs.load(file_path)

        signals = loaded if isinstance(loaded, list) else [loaded]
        logger.debug("Loaded %d signal(s) from %s", len(signals), file_path)

        for sig_index, signal in enumerate(signals):
            if signal.axes_manager.navigation_dimension == 0:
                suffix = f"[{sig_index}]" if len(signals) > 1 else None
                window = ImageViewerWindow(file_path, signal=signal, window_suffix=suffix)
                window.show()
                logger.debug("Opened viewer for signal index %s suffix=%s", sig_index, suffix)
            else:
                nav_shape = signal.axes_manager.navigation_shape
                for nav_index in np.ndindex(nav_shape):
                    sub_signal = signal.inav[nav_index]
                    idx_str = ",".join(str(i) for i in nav_index)
                    if len(signals) > 1:
                        suffix = f"[{sig_index}; {idx_str}]"
                    else:
                        suffix = f"[{idx_str}]"
                    window = ImageViewerWindow(
                        file_path,
                        signal=sub_signal,
                        window_suffix=suffix,
                    )
                    window.show()
                    logger.debug(
                        "Opened viewer for signal index %s navigation index %s suffix=%s",
                        sig_index,
                        nav_index,
                        suffix,
                    )

    except Exception as exc:
        logger.exception("Could not open file: %s", file_path)
        QtWidgets.QMessageBox.critical(None, "Error", f"Could not open file: {str(exc)}")
