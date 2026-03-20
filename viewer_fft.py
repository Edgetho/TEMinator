# SPDX-License-Identifier: GPL-2.0-only
# Copyright (C) 2026 Cooper Stuntz
# See LICENSE for full license terms.

"""FFT ROI and FFT-window lifecycle controller for image-viewer windows."""

from __future__ import annotations

from typing import Any, Protocol, Sequence, cast

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets

from measurement_tools import FFTBoxROI


class _LoggerLike(Protocol):
    def debug(self, msg: str, *args) -> None: ...


class _FFTWindowManagerOwner(Protocol):
    view_mode: str
    p1: Any
    image_bounds: tuple[float, float, float, float]
    fft_count: int
    fft_boxes: list[pg.RectROI]
    fft_box_meta: dict[pg.RectROI, dict[str, object]]
    data: Any
    img_orig: Any
    file_path: str
    fft_windows: list[Any]
    fft_to_fft_window: dict[pg.RectROI, Any]
    selected_fft_box: pg.RectROI | None
    selected_measurement_index: int | None

    def _on_fft_finished(self, fft_box: pg.RectROI, fft_id: int, text_item: pg.TextItem) -> None: ...
    def _on_fft_box_clicked(self, fft_box: pg.RectROI) -> None: ...
    def _on_fft_box_double_clicked(self, fft_box: pg.RectROI, fft_id: int, text_item: pg.TextItem) -> None: ...
    def _delete_selected_measurement(self) -> None: ...


class FFTWindowManager:
    """Owns FFT ROI and child FFT-window lifecycle for a viewer."""

    def __init__(self, viewer: _FFTWindowManagerOwner, logger: _LoggerLike, fft_colors: Sequence[str]):
        self.viewer = viewer
        self.logger = logger
        self.fft_colors = fft_colors

    def add_new_fft(self, x_offset=None, y_offset=None, w=None, h=None) -> None:
        viewer = self.viewer
        if viewer.view_mode != "image":
            return

        if x_offset is None or y_offset is None or w is None or h is None:
            try:
                (x_range, y_range) = viewer.p1.vb.viewRange()
                x0, x1 = x_range
                y0, y1 = y_range
                x_offset = float(x0)
                y_offset = float(y0)
                w = float(x1 - x0)
                h = float(y1 - y0)
            except Exception:
                x_offset, y_offset, w, h = viewer.image_bounds

        if w is None or h is None or w <= 0 or h <= 0:
            QtWidgets.QMessageBox.warning(cast(QtWidgets.QWidget, viewer), "Error", "Invalid image bounds")
            return

        color = self.fft_colors[viewer.fft_count % len(self.fft_colors)]
        roi_x = float(x_offset) + float(w) / 4.0
        roi_y = float(y_offset) + float(h) / 4.0
        roi_w = float(w) / 2.0
        roi_h = float(h) / 2.0
        fft_box = FFTBoxROI([roi_x, roi_y], [roi_w, roi_h], pen=pg.mkPen(color, width=2))
        fft_box.addScaleHandle([1, 1], [0, 0])
        fft_box.addScaleHandle([0, 0], [1, 1])
        viewer.p1.addItem(fft_box)

        fft_id = viewer.fft_count
        viewer.fft_count += 1

        text_item = pg.TextItem(f"FFT {fft_id}", anchor=(0, 0), fill=pg.mkBrush(color))
        text_item.setPos(fft_box.pos()[0], fft_box.pos()[1])
        viewer.p1.addItem(text_item)

        fft_box.sigRegionChangeFinished.connect(lambda: viewer._on_fft_finished(fft_box, fft_id, text_item))
        fft_box.sigBoxClicked.connect(lambda roi=fft_box: viewer._on_fft_box_clicked(roi))
        fft_box.sigBoxDoubleClicked.connect(lambda roi=fft_box: viewer._on_fft_box_double_clicked(roi, fft_id, text_item))
        viewer.fft_boxes.append(fft_box)
        viewer.fft_box_meta[fft_box] = {"id": fft_id, "text_item": text_item}
        self.logger.debug(
            "Added FFT ROI id=%s at (%.3f, %.3f) size=(%.3f, %.3f)",
            fft_id,
            roi_x,
            roi_y,
            roi_w,
            roi_h,
        )

    def on_fft_finished(self, fft_box: pg.RectROI, fft_id: int, text_item: pg.TextItem) -> None:
        viewer = self.viewer
        region = fft_box.getArrayRegion(viewer.data, viewer.img_orig)

        if region is None or region.shape[0] < 2 or region.shape[1] < 2:
            return

        text_item.setPos(fft_box.pos()[0], fft_box.pos()[1])
        self.logger.debug("FFT ROI finalized id=%s region_shape=%s", fft_id, getattr(region, "shape", None))

        self.open_or_update_fft_window(fft_box, fft_id, text_item, region)

    def open_or_update_fft_window(
        self,
        fft_box: pg.RectROI,
        fft_id: int,
        text_item: pg.TextItem,
        region: np.ndarray,
    ) -> None:
        viewer = self.viewer
        if fft_box in viewer.fft_to_fft_window:
            self.logger.debug("Updating FFT window for id=%s", fft_id)
            fft_window = viewer.fft_to_fft_window[fft_box]
            fft_window._fft_region = region
            fft_window._compute_fft()
            fft_window._init_display_window()
            fft_window._update_image_display()
            fft_window.show()
            fft_window.raise_()
            fft_window.activateWindow()
        else:
            self.logger.debug("Creating FFT window for id=%s", fft_id)
            from image_viewer import ImageViewerWindow

            fft_name = f"FFT {fft_id}"
            fft_window = ImageViewerWindow(
                viewer.file_path,
                view_mode="fft",
                fft_region=region,
                fft_name=fft_name,
                parent_image_window=viewer,
            )
            fft_window.show()
            viewer.fft_windows.append(fft_window)
            viewer.fft_to_fft_window[fft_box] = fft_window

    def on_fft_box_clicked(self, fft_box: pg.RectROI) -> None:
        viewer = self.viewer
        if fft_box in viewer.fft_boxes:
            viewer.selected_fft_box = fft_box

    def on_fft_box_double_clicked(
        self,
        fft_box: pg.RectROI,
        fft_id: int,
        text_item: pg.TextItem,
    ) -> None:
        viewer = self.viewer
        region = fft_box.getArrayRegion(viewer.data, viewer.img_orig)
        if region is None or region.shape[0] < 2 or region.shape[1] < 2:
            return

        text_item.setPos(fft_box.pos()[0], fft_box.pos()[1])
        self.open_or_update_fft_window(fft_box, fft_id, text_item, region)

    def delete_selected_roi(self) -> bool:
        viewer = self.viewer
        if viewer.selected_measurement_index is not None:
            viewer._delete_selected_measurement()
            return True

        fft_box = viewer.selected_fft_box
        if fft_box is None or fft_box not in viewer.fft_boxes:
            QtWidgets.QMessageBox.information(cast(QtWidgets.QWidget, viewer), "Delete", "No measurement or ROI selected.")
            return False

        index = viewer.fft_boxes.index(fft_box)
        viewer.p1.removeItem(fft_box)

        meta = viewer.fft_box_meta.pop(fft_box, None)
        if meta is not None:
            text_item = meta.get("text_item")
            if text_item is not None:
                viewer.p1.removeItem(text_item)

        fft_window = viewer.fft_to_fft_window.pop(fft_box, None)
        if fft_window is not None:
            fft_window.close()
            if fft_window in viewer.fft_windows:
                viewer.fft_windows.remove(fft_window)

        viewer.fft_boxes.pop(index)
        viewer.selected_fft_box = None

        QtWidgets.QMessageBox.information(cast(QtWidgets.QWidget, viewer), "Deleted", f"FFT Box {index} deleted.")
        return True
