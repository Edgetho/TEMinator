# SPDX-License-Identifier: GPL-2.0-only
# Copyright (C) 2026 Cooper Stuntz
# See LICENSE for full license terms.

"""Scale bar graphics items for image and FFT views."""

from typing import List, Optional, Tuple

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

import utils


class ScaleBarItem(pg.GraphicsObject):
    """Microscopy-style scale bar for images."""

    def __init__(self, scale_per_pixel: float, units: str = "px"):
        """Initialize a microscopy-style scale bar.

        Args:
            scale_per_pixel: Physical distance per pixel in base units.
            units: Unit label for the scale bar (default: "px").
        """
        super().__init__()
        self.scale_per_pixel = scale_per_pixel
        self.base_units = units
        self.bar_length_physical = 1.0  # Physical length in units
        self.display_unit = units
        self.setPos(0, 0)
        self.update_length()

    def update_length(self):
        """Update scale bar physical length using SI units."""
        reference = self.scale_per_pixel
        if reference <= 0 or not np.isfinite(reference):
            self.bar_length_physical = 1.0
            self.display_value = 1.0
            self.display_unit = self.base_units
            return

        # Physical length corresponding to ~100 pixels
        target_phys = reference * 100.0
        if target_phys <= 0 or not np.isfinite(target_phys):
            target_phys = reference

        magnitude = 10 ** np.floor(np.log10(target_phys))
        mantissa = target_phys / magnitude

        if mantissa < 1.5:
            nice = 1.0
        elif mantissa < 3.5:
            nice = 2.0
        elif mantissa < 7.5:
            nice = 5.0
        else:
            nice = 10.0

        # Store physical length in base units for geometry
        self.bar_length_physical = nice * magnitude

        # Use SI formatting only for the label text
        scaled_val, si_unit = utils.format_si_scale(
            self.bar_length_physical, self.base_units, precision=2
        )
        self.display_value = scaled_val
        self.display_unit = si_unit

    def boundingRect(self):  # type: ignore[override]
        """Return bounding rect of scale bar graphics item.

        Returns:
            QRectF covering the scale bar and label text area.
        """
        if self.scale_per_pixel <= 0 or not np.isfinite(self.scale_per_pixel):
            bar_pixels = 100
        else:
            bar_pixels = self.bar_length_physical / self.scale_per_pixel
            bar_pixels = max(10, min(bar_pixels, 500))
        return QtCore.QRectF(0, 0, bar_pixels + 30, 40)

    def paint(self, p, *args):  # type: ignore[override]
        """Paint the scale bar with line and label text.

        Args:
            p: QPainter to draw with.
        """  # pragma: no cover - pure UI
        if self.scale_per_pixel <= 0:
            return

        bar_pixels = self.bar_length_physical / self.scale_per_pixel
        bar_pixels = max(10, min(bar_pixels, 500))

        # Draw horizontal line
        p.setPen(QtGui.QPen(QtCore.Qt.red, 2))
        p.drawLine(10, 15, 10 + bar_pixels, 15)

        # Draw end caps
        p.drawLine(10, 10, 10, 20)
        p.drawLine(10 + bar_pixels, 10, 10 + bar_pixels, 20)

        # Draw text label
        text = f"{self.display_value:.2g} {self.display_unit}"
        p.setPen(QtGui.QPen(QtCore.Qt.red))
        font = QtGui.QFont()
        font.setPointSize(8)
        p.setFont(font)
        p.drawText(10, 28, text)


class DynamicScaleBar(pg.GraphicsObject):
    """Dynamic overlay scale bar drawn in screen coordinates.

    - Length in physical units is always 2, 5, or 10 × 10^n of the
      base units for the axis (typically metres).
    - On-screen length stays between ``min_frac`` and ``max_frac`` of
      the current view width.
    - Drawn as an overlay (ignores data transforms), so text size is
      stable while the bar length and label update with zoom.
    """

    _length_px = 0.0
    _label_text = ""
    _rect = QtCore.QRectF()
    _extra_label: Optional[str] = None
    _status_tag: Optional[str] = None

    def __init__(
        self,
        viewbox: pg.ViewBox,
        units: str = "m",
        min_frac: float = 0.15,
        max_frac: float = 0.30,
        margin: int = 20,
    ):
        """Initialize a dynamic scale bar overlay.

        Args:
            viewbox: The plot ViewBox to attach to.
            units: Base unit name for the scale bar (default: "m").
            min_frac: Minimum scale bar width as fraction of view width (default: 0.15).
            max_frac: Maximum scale bar width as fraction of view width (default: 0.30).
            margin: Pixel margin from view edges (default: 20).
        """
        super().__init__()

        self.vb = viewbox
        self.units = units or ""
        # When True, the label is rendered as reciprocal units (e.g. m⁻¹)
        # by appending a "⁻¹" exponent to the formatted SI unit string.
        self.reciprocal: bool = False
        self.min_frac = float(min_frac)
        self.max_frac = float(max_frac)
        self.margin = margin

        self._length_px = 0.0
        self._label_text = ""
        self._rect = QtCore.QRectF()
        self._extra_label: str | None = None
        self._status_tag: str | None = None

        # Keep this item fixed in screen space
        self.setFlag(QtWidgets.QGraphicsItem.ItemIgnoresTransformations, True)
        self.setAcceptedMouseButtons(QtCore.Qt.NoButton)
        self.setAcceptHoverEvents(False)
        self.setZValue(1000)

        # Parent directly to the ViewBox
        self.setParentItem(self.vb)

        # Update whenever view range or size changes
        self.vb.sigRangeChanged.connect(self._update_geometry)
        self.vb.sigResized.connect(self._update_geometry)

        self._update_geometry()

    def set_extra_label(self, text: str | None) -> None:
        """Set an optional extra label drawn above the scale bar.

        Intended primarily for export (e.g., file name and ROI number).

        Args:
            text: User-facing text value for this operation.

        """

        # Normalize empty strings to None
        clean = text.strip() if isinstance(text, str) else None
        self._extra_label = clean or None
        self.update()

    def set_status_tag(self, text: str | None) -> None:
        """Set a persistent status tag appended to the scale-bar label.

        Args:
            text: User-facing text value for this operation.

        """

        clean = text.strip() if isinstance(text, str) else None
        self._status_tag = clean or None
        self.update()

    def _choose_length(
        self, target_val: float, world_per_px: float, width_px: float
    ) -> Tuple[float, float]:
        """Return (length_val, length_px) using 2/5/10 × 10^n close to target.

        Args:
            target_val: Input value for target val.
            world_per_px: Input value for world per px.
            width_px: Input value for width px.

        Returns:
            Detailed parameter description.

        """

        if target_val <= 0 or not np.isfinite(target_val):
            return 0.0, 0.0

        min_px = self.min_frac * width_px
        max_px = self.max_frac * width_px
        target_px = (self.min_frac + self.max_frac) / 2.0 * width_px

        exp0 = int(np.floor(np.log10(target_val)))
        best_val = None
        best_px = None
        best_score = None

        for e in range(exp0 - 6, exp0 + 7):
            for factor in (2.0, 5.0, 10.0):
                val = factor * (10.0**e)
                if val <= 0 or not np.isfinite(val):
                    continue

                px = val / world_per_px
                if px <= 0 or not np.isfinite(px):
                    continue

                if not (min_px <= px <= max_px):
                    continue

                score = abs(np.log(px / target_px))
                if best_score is None or score < best_score:
                    best_score = score
                    best_val = val
                    best_px = px

        if best_val is None or best_px is None:
            best_px = min(max(target_px, min_px), max_px)
            best_val = best_px * world_per_px

        return float(best_val), float(best_px)

    def _update_geometry(self, *args):  # pragma: no cover - pure UI
        """Update scale bar geometry for current view range and size.

        Called when the view is resized or panned. Recalculates the bar length
        and label based on the current physical coordinates and view dimensions.

        Args:
            *args: Input value for args.

        """
        width_px = max(float(self.vb.width()), 1.0)

        try:
            (x_range, _y_range) = self.vb.viewRange()
        except Exception:
            return

        x0, x1 = x_range
        width_world = float(x1 - x0)
        if width_world <= 0 or not np.isfinite(width_world):
            return

        world_per_px = width_world / width_px
        if world_per_px <= 0 or not np.isfinite(world_per_px):
            return

        min_px = self.min_frac * width_px
        max_px = self.max_frac * width_px
        target_px = (self.min_frac + self.max_frac) / 2.0 * width_px

        target_val = target_px * world_per_px
        length_val, bar_px = self._choose_length(target_val, world_per_px, width_px)

        if self.reciprocal:
            scaled, unit_str = utils.format_reciprocal_scale(length_val, self.units)
        else:
            scaled, unit_str = utils.format_si_scale(length_val, self.units)

        if unit_str:
            label = f"{scaled:.3g} {unit_str}"
        else:
            label = f"{scaled:.3g}"

        bar_px = max(min_px, min(bar_px, max_px))

        self.prepareGeometryChange()
        self._length_px = bar_px
        self._label_text = label
        self._rect = QtCore.QRectF(-2, -20, self._length_px + 4, 30)

        x = self.margin
        y = self.vb.height() - self.margin
        self.setPos(x, y)
        self.update()

    def boundingRect(self):  # type: ignore[override]
        """Return bounding rect of the scale bar graphics item.

        Returns:
            QRectF bounding the scale bar and labels.
        """
        return self._rect

    def paint(self, p, *args):  # type: ignore[override]
        """Paint scale bar with line, caps, and labels.

        Args:
            p: QPainter used for drawing.
        """
        if self._length_px <= 0 or not np.isfinite(self._length_px):
            return

        p.setPen(QtGui.QPen(QtCore.Qt.red, 2))
        cap = 5.0
        length = float(self._length_px)

        p.drawLine(QtCore.QLineF(0, 0, self._length_px, 0))
        p.drawLine(QtCore.QLineF(0, -cap, 0, cap))
        p.drawLine(QtCore.QLineF(self._length_px, -cap, self._length_px, cap))

        p.setPen(QtGui.QPen(QtCore.Qt.red))
        font = QtGui.QFont()
        font.setPointSize(8)
        p.setFont(font)
        label_text = self._label_text
        if self._status_tag:
            label_text = f"{label_text} ({self._status_tag})"
        p.drawText(0, -int(cap) - 3, label_text)

        # Optional extra label (e.g. file name and ROI number) drawn above
        # the standard scale-bar label. Kept in the same red color.
        if self._extra_label:
            p.drawText(0, -int(cap) - 3 - 12, self._extra_label)


class DynamicLegendBox(pg.GraphicsObject):
    """Dynamic bottom-right legend overlay for selected EDX maps."""

    def __init__(
        self,
        viewbox: pg.ViewBox,
        margin: int = 20,
        min_frac: float = 0.14,
        max_frac: float = 0.32,
    ):
        """Initialize a dynamic legend overlay attached to a ViewBox.

        Args:
            viewbox: The plot ViewBox to attach to.
            margin: Pixel margin from the view edges.
            min_frac: Minimum box width as fraction of view width.
            max_frac: Maximum box width as fraction of view width.
        """
        super().__init__()
        self.vb = viewbox
        self.margin = int(margin)
        self.min_frac = float(min_frac)
        self.max_frac = float(max_frac)

        self._entries: List[Tuple[str, Tuple[int, int, int]]] = []
        self._rect = QtCore.QRectF(0, 0, 0, 0)
        self._font = QtGui.QFont()
        self._font.setPointSize(8)
        self._swatch_size = 12.0
        self._line_height = 14.0
        self._padding = 8.0
        self._spacing = 6.0

        self.setFlag(QtWidgets.QGraphicsItem.ItemIgnoresTransformations, True)
        self.setAcceptedMouseButtons(QtCore.Qt.NoButton)
        self.setAcceptHoverEvents(False)
        self.setZValue(1000)
        self.setParentItem(self.vb)

        self.vb.sigRangeChanged.connect(self._update_geometry)
        self.vb.sigResized.connect(self._update_geometry)

        self.hide()
        self._update_geometry()

    def set_entries(self, entries: List[Tuple[str, Tuple[int, int, int]]]) -> None:
        """Set legend entries as ``(name, rgb)`` tuples."""
        self._entries = list(entries)
        if self._entries:
            self.show()
        else:
            self.hide()
        self._update_geometry()
        self.update()

    def _update_geometry(self, *args):  # pragma: no cover - pure UI
        """Recalculate legend geometry and pin it to the bottom-right corner."""
        _ = args
        width_px = max(float(self.vb.width()), 1.0)
        height_px = max(float(self.vb.height()), 1.0)

        font_size = int(max(8, min(12, round(width_px * 0.013))))
        self._font.setPointSize(font_size)
        metrics = QtGui.QFontMetrics(self._font)

        self._swatch_size = float(max(10, min(18, round(width_px * 0.016))))
        self._padding = float(max(6, min(12, round(width_px * 0.010))))
        self._spacing = float(max(4, min(10, round(width_px * 0.008))))
        self._line_height = float(max(self._swatch_size, metrics.height()))

        if not self._entries:
            self.prepareGeometryChange()
            self._rect = QtCore.QRectF(0, 0, 0, 0)
            self.setPos(
                width_px - self.margin,
                height_px - self.margin,
            )
            return

        max_text_width = 0.0
        for name, _rgb in self._entries:
            max_text_width = max(max_text_width, float(metrics.horizontalAdvance(name)))

        content_width = self._swatch_size + self._spacing + max_text_width
        desired_width = content_width + 2.0 * self._padding
        min_width = self.min_frac * width_px
        max_width = self.max_frac * width_px
        box_width = max(min_width, min(desired_width, max_width))

        row_count = float(len(self._entries))
        content_height = row_count * self._line_height + max(0.0, (row_count - 1.0) * self._spacing)
        box_height = content_height + 2.0 * self._padding

        self.prepareGeometryChange()
        self._rect = QtCore.QRectF(0, 0, box_width, box_height)
        self.setPos(
            width_px - self.margin - box_width,
            height_px - self.margin - box_height,
        )

    def boundingRect(self):  # type: ignore[override]
        """Return legend box bounding rectangle."""
        return self._rect

    def paint(self, p, *args):  # type: ignore[override]
        """Paint the legend box and all map color entries."""
        _ = args
        if not self._entries:
            return

        palette = QtWidgets.QApplication.palette()
        bg = QtGui.QColor(palette.color(QtGui.QPalette.Window))
        bg.setAlpha(180)
        border = QtGui.QColor(palette.color(QtGui.QPalette.Mid))
        text_color = QtGui.QColor(palette.color(QtGui.QPalette.WindowText))

        p.setPen(QtGui.QPen(border, 1))
        p.setBrush(QtGui.QBrush(bg))
        p.drawRoundedRect(self._rect, 4.0, 4.0)

        p.setFont(self._font)
        p.setPen(QtGui.QPen(text_color))

        x0 = self._padding
        y = self._padding

        for name, rgb in self._entries:
            swatch_rect = QtCore.QRectF(x0, y, self._swatch_size, self._swatch_size)
            swatch_color = QtGui.QColor(int(rgb[0]), int(rgb[1]), int(rgb[2]))
            p.setPen(QtGui.QPen(border, 1))
            p.setBrush(QtGui.QBrush(swatch_color))
            p.drawRect(swatch_rect)

            text_x = x0 + self._swatch_size + self._spacing
            baseline = y + (self._line_height + QtGui.QFontMetrics(self._font).ascent()) / 2.0 - 1.0
            p.setPen(QtGui.QPen(text_color))
            p.drawText(QtCore.QPointF(text_x, baseline), name)

            y += self._line_height + self._spacing
