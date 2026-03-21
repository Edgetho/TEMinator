# SPDX-License-Identifier: GPL-2.0-only
# Copyright (C) 2026 Cooper Stuntz
# See LICENSE for full license terms.

"""Measurement-related tools: line drawing, FFT ROI, and labels."""
import logging
from typing import Tuple, Optional

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore


# Qt signal compatibility (PyQt5 vs PySide)
Signal = getattr(QtCore, "pyqtSignal", getattr(QtCore, "Signal", None))
logger = logging.getLogger(__name__)

# Pens and brushes used for measurement drawing
PREVIEW_LINE_PEN = pg.mkPen(255, 165, 0, width=2, style=QtCore.Qt.DashLine)
DRAWN_LINE_PEN = pg.mkPen(0, 255, 255, width=2)
LABEL_BRUSH_COLOR = pg.mkBrush(240, 240, 255, 235)


class LineDrawingTool:
    """Tool for drawing measurement lines on a plot."""

    def __init__(self, plot: pg.PlotItem, on_line_drawn_callback, on_drawing_state_changed=None):
        self.plot = plot
        self.on_line_drawn_callback = on_line_drawn_callback
        self.on_drawing_state_changed = on_drawing_state_changed
        self.drawing = False
        self._last_drawing_state = False
        self.start_point: Optional[Tuple[float, float]] = None
        self.line_item: Optional[pg.PlotDataItem] = None
        self.is_enabled = False
        self.vb = plot.vb

        # Store original mouse event methods
        self.original_mouse_press = self.vb.mousePressEvent
        self.original_mouse_move = self.vb.mouseMoveEvent
        self.original_mouse_release = self.vb.mouseReleaseEvent

    def _set_fft_roi_interaction_toggle(self, enabled: bool) -> None:
        """Enable/disable mouse interaction on FFT ROI items while measuring."""
        items = getattr(self.plot, "items", [])
        for item in items:
            if isinstance(item, FFTBoxROI):
                try:
                    item.setAcceptedMouseButtons(
                        QtCore.Qt.LeftButton if enabled else QtCore.Qt.NoButton
                    )
                except Exception:
                    pass
                try:
                    item.setAcceptHoverEvents(enabled)
                except Exception:
                    pass

    def enable(self):
        """Enable line drawing mode."""
        self.is_enabled = True
        self._set_drawing_state(False)
        self.start_point = None
        logger.debug("LineDrawingTool enabled")

        # Disable ROI interaction so ROI items don't steal mouse events.
        self._set_fft_roi_interaction_toggle(False)

        # Replace mouse event handlers
        self.vb.mousePressEvent = self._on_mouse_press
        self.vb.mouseMoveEvent = self._on_mouse_move
        self.vb.mouseReleaseEvent = self._on_mouse_release

    def disable(self):
        """Disable line drawing mode."""
        self.is_enabled = False

        # Restore original mouse event handlers
        self.vb.mousePressEvent = self.original_mouse_press
        self.vb.mouseMoveEvent = self.original_mouse_move
        self.vb.mouseReleaseEvent = self.original_mouse_release

        # Re-enable ROI interaction when not measuring.
        self._set_fft_roi_interaction_toggle(True)

        self._clear_preview_line()
        self._set_drawing_state(False)
        self.start_point = None
        logger.debug("LineDrawingTool disabled")

    def _set_drawing_state(self, is_drawing: bool) -> None:
        self.drawing = is_drawing
        if self._last_drawing_state == is_drawing:
            return
        self._last_drawing_state = is_drawing
        logger.debug("LineDrawingTool drawing state changed: drawing=%s", is_drawing)
        if self.on_drawing_state_changed is not None:
            self.on_drawing_state_changed(is_drawing)

    def _clear_preview_line(self):
        """Remove preview line from plot."""
        if self.line_item is not None:
            self.plot.removeItem(self.line_item)
            self.line_item = None

    def _on_mouse_press(self, event):
        """Handle mouse press for line drawing."""
        if not self.is_enabled:
            self.original_mouse_press(event)
            return

        scene_pos = event.scenePos()
        if not self.plot.sceneBoundingRect().contains(scene_pos):
            self.original_mouse_press(event)
            return

        view_pos = self.vb.mapSceneToView(scene_pos)
        # Start a new line on mouse press
        self._set_drawing_state(True)
        self.start_point = (view_pos.x(), view_pos.y())
        logger.debug("LineDrawingTool start point: %s", self.start_point)
        self._clear_preview_line()

        event.accept()

    def _on_mouse_move(self, event):
        """Handle mouse move for line drawing preview."""
        if not self.is_enabled:
            self.original_mouse_move(event)
            return

        if not self.drawing or self.start_point is None:
            self.original_mouse_move(event)
            return

        scene_pos = event.scenePos()
        if not self.plot.sceneBoundingRect().contains(scene_pos):
            self.original_mouse_move(event)
            return

        view_pos = self.vb.mapSceneToView(scene_pos)

        # Update preview line
        self._clear_preview_line()
        self.line_item = pg.PlotDataItem(
            [self.start_point[0], view_pos.x()],
            [self.start_point[1], view_pos.y()],
            pen=PREVIEW_LINE_PEN,
        )
        self.plot.addItem(self.line_item)
        event.accept()

    def _on_mouse_release(self, event):
        """Handle mouse release."""
        if not self.is_enabled:
            self.original_mouse_release(event)
            return

        # If we weren't drawing, let the original handler process the event
        if not self.drawing or self.start_point is None:
            self.original_mouse_release(event)
            return

        scene_pos = event.scenePos()
        if not self.plot.sceneBoundingRect().contains(scene_pos):
            # Outside plot area; cancel the drawing and delegate
            self._clear_preview_line()
            self._set_drawing_state(False)
            self.start_point = None
            logger.debug("LineDrawingTool draw cancelled: mouse release outside plot area")
            self.original_mouse_release(event)
            return

        # Finish the line on mouse release and trigger callback
        view_pos = self.vb.mapSceneToView(scene_pos)
        end_point = (view_pos.x(), view_pos.y())
        logger.debug("LineDrawingTool end point: %s", end_point)
        self._set_drawing_state(False)
        self._clear_preview_line()
        self.on_line_drawn_callback(self.start_point, end_point)
        self.start_point = None

        event.accept()


class FFTBoxROI(pg.RectROI):
    """Custom RectROI for FFT boxes with click and double-click signals."""

    sigBoxClicked = Signal(object)
    sigBoxDoubleClicked = Signal(object)

    def mouseClickEvent(self, ev):  # type: ignore[override]
        """Emit signals on single and double clicks while preserving default behavior."""
        # Call base implementation if available (version-safe)
        try:
            super().mouseClickEvent(ev)  # type: ignore[misc]
        except AttributeError:
            pass

        if ev.button() == QtCore.Qt.LeftButton:
            if ev.double():
                self.sigBoxDoubleClicked.emit(self)
            else:
                self.sigBoxClicked.emit(self)


class MeasurementLabel(pg.TextItem):
    """Clickable label for measurement annotations."""

    sigLabelClicked = Signal(object)

    def mouseClickEvent(self, ev):  # type: ignore[override]
        """Emit a signal when the label is clicked."""
        super().mouseClickEvent(ev)

        if ev.button() == QtCore.Qt.LeftButton:
            self.sigLabelClicked.emit(self)
