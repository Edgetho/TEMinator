# SPDX-License-Identifier: GPL-2.0-only
# Copyright (C) 2026 Cooper Stuntz
# See LICENSE for full license terms.

"""FFT ROI and FFT-window lifecycle controller for image-viewer windows."""

from __future__ import annotations

from typing import Any, Literal, Protocol, Sequence, cast

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
    inverse_fft_count: int
    inverse_fft_boxes: list[pg.RectROI]
    inverse_fft_box_meta: dict[pg.RectROI, dict[str, object]]
    data: Any
    img_orig: Any
    file_path: str
    fft_windows: list[Any]
    fft_to_fft_window: dict[pg.RectROI, Any]
    inverse_fft_windows: list[Any]
    inverse_fft_to_window: dict[pg.RectROI, Any]
    selected_fft_box: pg.RectROI | None
    selected_inverse_fft_box: pg.RectROI | None
    selected_measurement_index: int | None

    def _on_fft_finished(self, fft_box: pg.RectROI, fft_id: int, text_item: pg.TextItem) -> None: ...
    def _on_fft_box_clicked(self, fft_box: pg.RectROI) -> None: ...
    def _on_fft_box_double_clicked(self, fft_box: pg.RectROI, fft_id: int, text_item: pg.TextItem) -> None: ...
    def _on_inverse_fft_finished(self, fft_box: pg.RectROI, fft_id: int, text_item: pg.TextItem) -> None: ...
    def _on_inverse_fft_box_clicked(self, fft_box: pg.RectROI) -> None: ...
    def _on_inverse_fft_box_double_clicked(self, fft_box: pg.RectROI, fft_id: int, text_item: pg.TextItem) -> None: ...
    def _delete_selected_measurement(self) -> None: ...


TransformKind = Literal["fft", "inverse_fft"]


class FFTWindowManager:
    """Owns FFT ROI and child FFT-window lifecycle for a viewer."""

    def __init__(self, viewer: _FFTWindowManagerOwner, logger: _LoggerLike, fft_colors: Sequence[str]):
        self.viewer = viewer
        self.logger = logger
        self.fft_colors = fft_colors

    @staticmethod
    def _label_for_kind(kind: TransformKind) -> str:
        return "FFT" if kind == "fft" else "iFFT"

    def _storage_for_kind(
        self,
        kind: TransformKind,
    ) -> tuple[str, str, str, str, str, str]:
        if kind == "fft":
            return (
                "fft_count",
                "fft_boxes",
                "fft_box_meta",
                "fft_windows",
                "fft_to_fft_window",
                "selected_fft_box",
            )
        return (
            "inverse_fft_count",
            "inverse_fft_boxes",
            "inverse_fft_box_meta",
            "inverse_fft_windows",
            "inverse_fft_to_window",
            "selected_inverse_fft_box",
        )

    def add_new_fft(self, x_offset=None, y_offset=None, w=None, h=None) -> None:
        self._add_new_transform("fft", x_offset, y_offset, w, h)

    def add_new_inverse_fft(self, x_offset=None, y_offset=None, w=None, h=None) -> None:
        self._add_new_transform("inverse_fft", x_offset, y_offset, w, h)

    def _add_new_transform(self, kind: TransformKind, x_offset=None, y_offset=None, w=None, h=None) -> None:
        viewer = self.viewer
        if viewer.view_mode not in {"image", "fft", "inverse_fft"}:
            self.logger.debug("Ignoring %s ROI add: viewer mode is %s", self._label_for_kind(kind), viewer.view_mode)
            return

        count_attr, boxes_attr, meta_attr, _windows_attr, _map_attr, _selected_attr = self._storage_for_kind(kind)
        label_prefix = self._label_for_kind(kind)
        self.logger.debug(
            "Adding new %s ROI: incoming_bounds=(%s,%s,%s,%s)",
            label_prefix,
            x_offset,
            y_offset,
            w,
            h,
        )

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
            self.logger.debug("Rejected %s ROI add due to invalid bounds", label_prefix)
            return

        transform_count = cast(int, getattr(viewer, count_attr))
        color = self.fft_colors[transform_count % len(self.fft_colors)]
        roi_x = float(x_offset) + float(w) / 4.0
        roi_y = float(y_offset) + float(h) / 4.0
        roi_w = float(w) / 2.0
        roi_h = float(h) / 2.0
        fft_box = FFTBoxROI([roi_x, roi_y], [roi_w, roi_h], pen=pg.mkPen(color, width=2))
        fft_box.addScaleHandle([1, 1], [0, 0])
        fft_box.addScaleHandle([0, 0], [1, 1])
        viewer.p1.addItem(fft_box)

        fft_id = transform_count
        setattr(viewer, count_attr, transform_count + 1)

        text_item = pg.TextItem(f"{label_prefix} {fft_id}", anchor=(0, 0), fill=pg.mkBrush(color))
        text_item.setPos(fft_box.pos()[0], fft_box.pos()[1])
        viewer.p1.addItem(text_item)

        if kind == "fft":
            fft_box.sigRegionChangeFinished.connect(lambda: viewer._on_fft_finished(fft_box, fft_id, text_item))
            fft_box.sigBoxClicked.connect(lambda roi=fft_box: viewer._on_fft_box_clicked(roi))
            fft_box.sigBoxDoubleClicked.connect(
                lambda roi=fft_box: viewer._on_fft_box_double_clicked(roi, fft_id, text_item)
            )
        else:
            fft_box.sigRegionChangeFinished.connect(lambda: viewer._on_inverse_fft_finished(fft_box, fft_id, text_item))
            fft_box.sigBoxClicked.connect(lambda roi=fft_box: viewer._on_inverse_fft_box_clicked(roi))
            fft_box.sigBoxDoubleClicked.connect(
                lambda roi=fft_box: viewer._on_inverse_fft_box_double_clicked(roi, fft_id, text_item)
            )

        boxes = cast(list[pg.RectROI], getattr(viewer, boxes_attr))
        box_meta = cast(dict[pg.RectROI, dict[str, object]], getattr(viewer, meta_attr))
        boxes.append(fft_box)
        box_meta[fft_box] = {"id": fft_id, "text_item": text_item}
        self.logger.debug(
            "Added %s ROI id=%s at (%.3f, %.3f) size=(%.3f, %.3f)",
            label_prefix,
            fft_id,
            roi_x,
            roi_y,
            roi_w,
            roi_h,
        )

    def on_inverse_fft_finished(self, fft_box: pg.RectROI, fft_id: int, text_item: pg.TextItem) -> None:
        self._on_transform_finished("inverse_fft", fft_box, fft_id, text_item)

    def on_fft_finished(self, fft_box: pg.RectROI, fft_id: int, text_item: pg.TextItem) -> None:
        self._on_transform_finished("fft", fft_box, fft_id, text_item)

    def _on_transform_finished(
        self,
        kind: TransformKind,
        fft_box: pg.RectROI,
        fft_id: int,
        text_item: pg.TextItem,
    ) -> None:
        viewer = self.viewer
        region = fft_box.getArrayRegion(viewer.data, viewer.img_orig)

        if region is None or region.shape[0] < 2 or region.shape[1] < 2:
            self.logger.debug(
                "%s ROI finalize ignored: invalid region shape=%s",
                self._label_for_kind(kind),
                getattr(region, "shape", None),
            )
            return

        text_item.setPos(fft_box.pos()[0], fft_box.pos()[1])
        self.logger.debug(
            "%s ROI finalized id=%s region_shape=%s",
            self._label_for_kind(kind),
            fft_id,
            getattr(region, "shape", None),
        )

        self.open_or_update_fft_window(kind, fft_box, fft_id, text_item, region)

    
    def _sanitize_region_for_render(self, region: np.ndarray) -> np.ndarray:
        arr = np.asarray(region)
        if np.iscomplexobj(arr):
            arr = np.abs(arr)
        arr = arr.astype(np.float32, copy=False)
        if hasattr(self.logger, "debug"):
            with np.errstate(invalid="ignore"):
                finite = np.isfinite(arr)
                if finite.any():
                    vmin = float(arr[finite].min())
                    vmax = float(arr[finite].max())
                else:
                    vmin = vmax = 0.0
            self.logger.debug(
                "[fft sanitize][after] dtype=%s shape=%s contiguous=%s min=%g max=%g",
                arr.dtype,
                arr.shape,
                arr.flags["C_CONTIGUOUS"],
                vmin,
                vmax,
            )
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        np.clip(arr, -1.0e30, 1.0e30, out=arr)
        return np.ascontiguousarray(arr)
    
    def open_or_update_fft_window(
        self,
        kind: TransformKind,
        fft_box: pg.RectROI,
        fft_id: int,
        text_item: pg.TextItem,
        region: np.ndarray,
    ) -> None:
        viewer = self.viewer
        _count_attr, _boxes_attr, _meta_attr, windows_attr, map_attr, _selected_attr = self._storage_for_kind(kind)
        windows = cast(list[Any], getattr(viewer, windows_attr))
        transform_to_window = cast(dict[pg.RectROI, Any], getattr(viewer, map_attr))

        region = self._sanitize_region_for_render(region)
        if fft_box in transform_to_window:
            self.logger.debug("Updating %s window for id=%s", self._label_for_kind(kind), fft_id)
            fft_window = transform_to_window[fft_box]
            fft_window._source_region = region
            fft_window._fft_region = region
            fft_window._refresh_transform_data()
            fft_window._init_display_window()
            fft_window._update_image_display()
            fft_window.show()
            fft_window.raise_()
            fft_window.activateWindow()
        else:
            self.logger.debug("Creating %s window for id=%s", self._label_for_kind(kind), fft_id)
            from image_viewer import ImageViewerWindow

            fft_name = f"{self._label_for_kind(kind)} {fft_id}"
            fft_window = ImageViewerWindow(
                viewer.file_path,
                view_mode=kind,
                source_region=region,
                fft_name=fft_name,
                parent_image_window=cast(Any, viewer),
            )
            fft_window.show()
            windows.append(fft_window)
            transform_to_window[fft_box] = fft_window
            self.logger.debug("Registered new %s child window: id=%s open_windows=%s", self._label_for_kind(kind), fft_id, len(windows))

    def on_fft_box_clicked(self, fft_box: pg.RectROI) -> None:
        self._on_transform_box_clicked("fft", fft_box)

    def on_inverse_fft_box_clicked(self, fft_box: pg.RectROI) -> None:
        self._on_transform_box_clicked("inverse_fft", fft_box)

    def _on_transform_box_clicked(self, kind: TransformKind, fft_box: pg.RectROI) -> None:
        viewer = self.viewer
        _count_attr, boxes_attr, _meta_attr, _windows_attr, _map_attr, selected_attr = self._storage_for_kind(kind)
        boxes = cast(list[pg.RectROI], getattr(viewer, boxes_attr))
        if fft_box in boxes:
            setattr(viewer, selected_attr, fft_box)
            self.logger.debug("Selected %s ROI", self._label_for_kind(kind))

            if kind == "fft":
                viewer.selected_inverse_fft_box = None
            else:
                viewer.selected_fft_box = None

    def on_fft_box_double_clicked(
        self,
        fft_box: pg.RectROI,
        fft_id: int,
        text_item: pg.TextItem,
    ) -> None:
        self._on_transform_box_double_clicked("fft", fft_box, fft_id, text_item)

    def on_inverse_fft_box_double_clicked(
        self,
        fft_box: pg.RectROI,
        fft_id: int,
        text_item: pg.TextItem,
    ) -> None:
        self._on_transform_box_double_clicked("inverse_fft", fft_box, fft_id, text_item)

    def _on_transform_box_double_clicked(
        self,
        kind: TransformKind,
        fft_box: pg.RectROI,
        fft_id: int,
        text_item: pg.TextItem,
    ) -> None:
        viewer = self.viewer
        region = fft_box.getArrayRegion(viewer.data, viewer.img_orig)
        if region is None or region.shape[0] < 2 or region.shape[1] < 2:
            self.logger.debug("%s ROI double-click ignored: invalid region shape=%s", self._label_for_kind(kind), getattr(region, "shape", None))
            return

        text_item.setPos(fft_box.pos()[0], fft_box.pos()[1])
        self.open_or_update_fft_window(kind, fft_box, fft_id, text_item, region)

    def delete_selected_roi(self) -> bool:
        viewer = self.viewer
        if viewer.selected_measurement_index is not None:
            viewer._delete_selected_measurement()
            return True

        fft_box = viewer.selected_fft_box
        if fft_box is not None:
            return self._delete_transform_roi("fft", fft_box)

        inverse_fft_box = viewer.selected_inverse_fft_box
        if inverse_fft_box is not None:
            return self._delete_transform_roi("inverse_fft", inverse_fft_box)

        QtWidgets.QMessageBox.information(cast(QtWidgets.QWidget, viewer), "Delete", "No measurement or ROI selected.")
        return False

    def _delete_transform_roi(self, kind: TransformKind, fft_box: pg.RectROI) -> bool:
        viewer = self.viewer
        _count_attr, boxes_attr, meta_attr, windows_attr, map_attr, selected_attr = self._storage_for_kind(kind)
        boxes = cast(list[pg.RectROI], getattr(viewer, boxes_attr))
        box_meta = cast(dict[pg.RectROI, dict[str, object]], getattr(viewer, meta_attr))
        windows = cast(list[Any], getattr(viewer, windows_attr))
        transform_to_window = cast(dict[pg.RectROI, Any], getattr(viewer, map_attr))

        if fft_box not in boxes:
            QtWidgets.QMessageBox.information(cast(QtWidgets.QWidget, viewer), "Delete", "No measurement or ROI selected.")
            self.logger.debug("Delete %s ROI ignored: selected ROI not found", self._label_for_kind(kind))
            return False

        index = boxes.index(fft_box)
        viewer.p1.removeItem(fft_box)

        meta = box_meta.pop(fft_box, None)
        if meta is not None:
            text_item = meta.get("text_item")
            if text_item is not None:
                viewer.p1.removeItem(text_item)

        fft_window = transform_to_window.pop(fft_box, None)
        if fft_window is not None:
            fft_window.close()
            if fft_window in windows:
                windows.remove(fft_window)

        boxes.pop(index)
        setattr(viewer, selected_attr, None)
        self.logger.debug("Deleted %s ROI index=%s remaining=%s", self._label_for_kind(kind), index, len(boxes))

        return True
