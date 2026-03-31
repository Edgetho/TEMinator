# SPDX-License-Identifier: GPL-2.0-only
# Copyright (C) 2026 Cooper Stuntz
# See LICENSE for full license terms.

"""Measurement-related tools: line drawing, FFT ROI, and labels."""

import logging
from typing import Optional, Tuple

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

    def __init__(
        self, plot: pg.PlotItem, on_line_drawn_callback, on_drawing_state_changed=None
    ):
        """Initialize line-drawing tool state and event handler references.

        Args:
            plot: Plot item where measurement lines are drawn.
            on_line_drawn_callback: Callback invoked with (start_point, end_point).
            on_drawing_state_changed: Optional callback receiving drawing-active state.
        """
        self.plot = plot
        self.on_line_drawn_callback = on_line_drawn_callback
        self.on_drawing_state_changed = on_drawing_state_changed
        self.drawing = False
        self._last_drawing_state = False
        self.start_point: Optional[Tuple[float, float]] = None
        self.line_item: Optional[pg.PlotDataItem] = None
        self.is_enabled = False
        self.vb = plot.vb
        self._previous_mouse_enabled: Optional[Tuple[bool, bool]] = None

        # Store original mouse event methods
        self.original_mouse_press = self.vb.mousePressEvent
        self.original_mouse_move = self.vb.mouseMoveEvent
        self.original_mouse_release = self.vb.mouseReleaseEvent
        self.original_mouse_drag = getattr(self.vb, "mouseDragEvent", None)

    def _set_fft_roi_interaction_toggle(self, enabled: bool) -> None:
        """Enable/disable mouse interaction on FFT ROI items while measuring.

        Args:
            enabled: Input value for enabled.

        """
        items = getattr(self.plot, "items", [])
        for item in items:
            if isinstance(item, FFTBoxROI):
                try:
                    item.set_interaction_enabled(enabled)
                except Exception:
                    pass

    def enable(self):
        """Enable line drawing mode."""
        self.is_enabled = True
        self._set_drawing_state(False)
        self.start_point = None
        logger.debug("LineDrawingTool enabled")

        # Disable ViewBox panning/zoom interaction while measuring.
        try:
            enabled = getattr(self.vb, "state", {}).get("mouseEnabled", [True, True])
            if isinstance(enabled, (list, tuple)) and len(enabled) >= 2:
                self._previous_mouse_enabled = (bool(enabled[0]), bool(enabled[1]))
            else:
                self._previous_mouse_enabled = (True, True)
            self.vb.setMouseEnabled(x=False, y=False)
        except Exception:
            self._previous_mouse_enabled = None

        # Disable ROI interaction so ROI items don't steal mouse events.
        self._set_fft_roi_interaction_toggle(False)

        # Replace mouse event handlers
        self.vb.mousePressEvent = self._on_mouse_press
        self.vb.mouseMoveEvent = self._on_mouse_move
        self.vb.mouseReleaseEvent = self._on_mouse_release
        if self.original_mouse_drag is not None:
            self.vb.mouseDragEvent = self._on_mouse_drag

    def disable(self):
        """Disable line drawing mode."""
        self.is_enabled = False

        # Restore original mouse event handlers
        self.vb.mousePressEvent = self.original_mouse_press
        self.vb.mouseMoveEvent = self.original_mouse_move
        self.vb.mouseReleaseEvent = self.original_mouse_release
        if self.original_mouse_drag is not None:
            self.vb.mouseDragEvent = self.original_mouse_drag

        # Restore ViewBox panning/zoom interaction when leaving measurement mode.
        if self._previous_mouse_enabled is not None:
            try:
                self.vb.setMouseEnabled(
                    x=self._previous_mouse_enabled[0],
                    y=self._previous_mouse_enabled[1],
                )
            except Exception:
                pass
            self._previous_mouse_enabled = None

        # Re-enable ROI interaction when not measuring.
        self._set_fft_roi_interaction_toggle(True)

        self._clear_preview_line()
        self._set_drawing_state(False)
        self.start_point = None
        logger.debug("LineDrawingTool disabled")

    def _set_drawing_state(self, is_drawing: bool) -> None:
        """Update drawing-state flag and notify listeners when it changes.

        Args:
            is_drawing: Boolean flag indicating whether drawing.

        """
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

    def _begin_line(self, view_pos) -> None:
        """Start a new preview line from the given view position.

        Args:
            view_pos: Input value for view pos.

        """
        self._set_drawing_state(True)
        self.start_point = (view_pos.x(), view_pos.y())
        logger.debug("LineDrawingTool start point: %s", self.start_point)
        self._clear_preview_line()

    def _update_line_preview(self, view_pos) -> None:
        """Redraw the temporary preview segment to the current cursor point.

        Args:
            view_pos: Input value for view pos.

        """
        if self.start_point is None:
            return
        self._clear_preview_line()
        self.line_item = pg.PlotDataItem(
            [self.start_point[0], view_pos.x()],
            [self.start_point[1], view_pos.y()],
            pen=PREVIEW_LINE_PEN,
        )
        self.plot.addItem(self.line_item)

    def _finish_line(self, view_pos) -> None:
        """Finalize the current line and trigger the line-drawn callback.

        Args:
            view_pos: Input value for view pos.

        """
        if self.start_point is None:
            return
        end_point = (view_pos.x(), view_pos.y())
        logger.debug("LineDrawingTool end point: %s", end_point)
        self._set_drawing_state(False)
        self._clear_preview_line()
        self.on_line_drawn_callback(self.start_point, end_point)
        self.start_point = None

    def _on_mouse_press(self, event):
        """Handle mouse press for line drawing.

        Args:
            event: Qt event object carrying user interaction details.

        """
        if not self.is_enabled:
            self.original_mouse_press(event)
            return

        scene_pos = event.scenePos()
        if not self.plot.sceneBoundingRect().contains(scene_pos):
            self.original_mouse_press(event)
            return

        view_pos = self.vb.mapSceneToView(scene_pos)
        # Start a new line on mouse press
        self._begin_line(view_pos)

        event.accept()

    def _on_mouse_move(self, event):
        """Handle mouse move for line drawing preview.

        Args:
            event: Qt event object carrying user interaction details.

        """
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
        self._update_line_preview(view_pos)
        event.accept()

    def _on_mouse_release(self, event):
        """Handle mouse release.

        Args:
            event: Qt event object carrying user interaction details.

        """
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
            logger.debug(
                "LineDrawingTool draw cancelled: mouse release outside plot area"
            )
            self.original_mouse_release(event)
            return

        # Finish the line on mouse release and trigger callback
        view_pos = self.vb.mapSceneToView(scene_pos)
        self._finish_line(view_pos)

        event.accept()

    def _on_mouse_drag(self, event):
        """Handle drag events directly so measurement works after prior pan/ROI drags.

        Args:
            event: Qt event object carrying user interaction details.

        """
        if not self.is_enabled:
            if self.original_mouse_drag is not None:
                self.original_mouse_drag(event)
            return

        scene_pos = event.scenePos()
        if not self.plot.sceneBoundingRect().contains(scene_pos):
            if event.isFinish() and self.drawing:
                self._clear_preview_line()
                self._set_drawing_state(False)
                self.start_point = None
                logger.debug(
                    "LineDrawingTool draw cancelled: drag finished outside plot area"
                )
            event.accept()
            return

        view_pos = self.vb.mapSceneToView(scene_pos)
        if event.isStart() or self.start_point is None:
            self._begin_line(view_pos)
        elif event.isFinish():
            self._finish_line(view_pos)
        else:
            self._update_line_preview(view_pos)

        event.accept()


class PointSelectionTool:
    """Tool for selecting individual points on a plot."""

    def __init__(self, plot: pg.PlotItem, on_point_selected_callback):
        """Initialize point-selection tool state and event handler references.

        Args:
            plot: Plot item where points are selected.
            on_point_selected_callback: Callback invoked with (x, y) in view coords.
        """
        self.plot = plot
        self.on_point_selected_callback = on_point_selected_callback
        self.is_enabled = False
        self.vb = plot.vb
        self._previous_mouse_enabled: Optional[Tuple[bool, bool]] = None

        self.original_mouse_press = self.vb.mousePressEvent

    def _set_fft_roi_interaction_toggle(self, enabled: bool) -> None:
        """Enable/disable mouse interaction on FFT ROI items while selecting.

        Args:
            enabled: Whether ROI interaction should be enabled.
        """
        items = getattr(self.plot, "items", [])
        for item in items:
            if isinstance(item, FFTBoxROI):
                try:
                    item.set_interaction_enabled(enabled)
                except Exception:
                    pass

    def enable(self) -> None:
        """Enable point selection mode."""
        self.is_enabled = True
        logger.debug("PointSelectionTool enabled")

        try:
            enabled = getattr(self.vb, "state", {}).get("mouseEnabled", [True, True])
            if isinstance(enabled, (list, tuple)) and len(enabled) >= 2:
                self._previous_mouse_enabled = (bool(enabled[0]), bool(enabled[1]))
            else:
                self._previous_mouse_enabled = (True, True)
            self.vb.setMouseEnabled(x=False, y=False)
        except Exception:
            self._previous_mouse_enabled = None

        self._set_fft_roi_interaction_toggle(False)
        self.vb.mousePressEvent = self._on_mouse_press

    def disable(self) -> None:
        """Disable point selection mode."""
        self.is_enabled = False
        self.vb.mousePressEvent = self.original_mouse_press

        if self._previous_mouse_enabled is not None:
            try:
                self.vb.setMouseEnabled(
                    x=self._previous_mouse_enabled[0],
                    y=self._previous_mouse_enabled[1],
                )
            except Exception:
                pass
            self._previous_mouse_enabled = None

        self._set_fft_roi_interaction_toggle(True)
        logger.debug("PointSelectionTool disabled")

    def _on_mouse_press(self, event):
        """Handle mouse press for selecting a point.

        Args:
            event: Qt event object carrying user interaction details.
        """
        if not self.is_enabled:
            self.original_mouse_press(event)
            return

        if event.button() != QtCore.Qt.LeftButton:
            self.original_mouse_press(event)
            return

        scene_pos = event.scenePos()
        if not self.plot.sceneBoundingRect().contains(scene_pos):
            self.original_mouse_press(event)
            return

        view_pos = self.vb.mapSceneToView(scene_pos)
        self.on_point_selected_callback((float(view_pos.x()), float(view_pos.y())))
        event.accept()


class FFTBoxROI(pg.RectROI):
    """Custom RectROI for FFT boxes with click and double-click signals."""

    sigBoxClicked = Signal(object)
    sigBoxDoubleClicked = Signal(object)

    def set_interaction_enabled(self, enabled: bool) -> None:
        """Toggle interaction for ROI and its resize/rotate handle items.

        Args:
            enabled: Input value for enabled.

        """
        mouse_buttons = QtCore.Qt.LeftButton if enabled else QtCore.Qt.NoButton

        try:
            self.setAcceptedMouseButtons(mouse_buttons)
        except Exception:
            pass

        try:
            self.setAcceptHoverEvents(enabled)
        except Exception:
            pass

        if not enabled:
            try:
                self.setSelected(False)
            except Exception:
                pass

        for child in list(self.childItems()):
            try:
                child.setAcceptedMouseButtons(mouse_buttons)
            except Exception:
                pass
            try:
                child.setAcceptHoverEvents(enabled)
            except Exception:
                pass

    def mouseClickEvent(self, ev):  # type: ignore[override]
        """Emit signals on single and double clicks while preserving default behavior.

        Args:
            ev: Input value for ev.

        """
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
        """Emit a signal when the label is clicked.

        Args:
            ev: Input value for ev.

        """
        super().mouseClickEvent(ev)

        if ev.button() == QtCore.Qt.LeftButton:
            self.sigLabelClicked.emit(self)
