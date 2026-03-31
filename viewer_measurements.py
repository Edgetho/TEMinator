# SPDX-License-Identifier: GPL-2.0-only
# Copyright (C) 2026 Cooper Stuntz
# See LICENSE for full license terms.

"""Measurement behavior controller for image-viewer windows."""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
import subprocess
from typing import Any, Dict, Optional, Protocol, Tuple

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets
from scipy import ndimage, optimize

import line_profile_logic
import unit_utils
import utils
from dialogs import LineProfileWindow, MeasurementHistoryWindow
from measurement_tools import (
    DRAWN_LINE_PEN,
    LABEL_BRUSH_COLOR,
    MeasurementLabel,
    PointSelectionTool,
)
from types_common import LoggerLike


class _MeasurementControllerOwner(Protocol):
    """Protocol for objects that own a MeasurementController."""

    _line_draw_mode: str
    _on_calibration_pixels_selected: Any
    _calibration_dialog_state: dict[str, Any] | None
    btn_measure: QtWidgets.QPushButton | None
    line_tool: Any
    ax_x: Any
    ax_y: Any
    measurement_count: int
    view_mode: str
    _nyq_x: Any
    _nyq_y: Any
    _fft_region: Any
    is_reciprocal_space: bool
    p1: Any
    img_orig: Any
    data: Any
    measurement_items: list[tuple[int, pg.PlotDataItem, pg.TextItem]]
    profile_measurement_count: int
    profile_measurement_items: dict[int, dict[str, Any]]
    selected_measurement_index: int | None
    measurement_history_window: MeasurementHistoryWindow | None
    freq_axis_base_unit: str
    file_path: str
    signal: Any

    def _prepare_for_measurement_input(self) -> None:
        """Prepare the viewer for measurement input."""
        ...

    def _open_calibration_dialog(self, state: dict[str, Any]) -> None:
        """Open the calibration dialog with saved state.

        Args:
            state: Persisted calibration-dialog state restored after distance picking.
        """
        ...

    def _on_measurement_drawing_state_changed(self, is_drawing: bool) -> None:
        """Handle measurement drawing mode state changes.

        Args:
            is_drawing: Boolean flag indicating whether drawing.

        """
        ...

    def _get_original_metadata_dict(self) -> Optional[dict]:
        """Return original metadata dictionary for the current signal."""
        ...

    def grab(self) -> Any:
        """Return a pixmap snapshot of the widget."""
        ...


class MeasurementController:
    """Owns measurement and line-draw interaction state for a viewer."""

    def __init__(self, viewer: _MeasurementControllerOwner, logger: LoggerLike):
        """Initialize the measurement controller.

        Args:
            viewer: The image viewer window that owns this controller.
            logger: Logger for debug output.
        """
        self.viewer = viewer
        self.logger = logger
        self._peak_tool: PointSelectionTool | None = None
        self._peak_points: list[dict[str, Any]] = []
        self._peak_markers: list[tuple[pg.ScatterPlotItem, pg.TextItem]] = []

    def _ensure_peak_tool(self) -> PointSelectionTool | None:
        """Create the point-selection tool on first use."""
        viewer = self.viewer
        if self._peak_tool is not None:
            return self._peak_tool
        if viewer.p1 is None:
            return None
        self._peak_tool = PointSelectionTool(viewer.p1, self.on_peak_selected)
        return self._peak_tool

    def _disable_peak_selection(self) -> None:
        """Disable peak selection mode without clearing selected peaks."""
        if self._peak_tool is not None and self._peak_tool.is_enabled:
            self._peak_tool.disable()

    def start_peak_selection(self) -> None:
        """Enable interactive peak picking by clicking points on the image."""
        viewer = self.viewer
        peak_tool = self._ensure_peak_tool()
        if peak_tool is None:
            self.logger.debug("Peak selection requested but plot is unavailable")
            QtWidgets.QMessageBox.information(
                viewer,
                "Peak Selection",
                "Peak selection is not available for this view.",
            )
            return

        viewer._prepare_for_measurement_input()
        viewer._line_draw_mode = "peaks"
        viewer._on_calibration_pixels_selected = None

        if viewer.btn_measure is not None:
            viewer.btn_measure.blockSignals(True)
            viewer.btn_measure.setChecked(False)
            viewer.btn_measure.blockSignals(False)
            viewer.btn_measure.setStyleSheet("")

        if viewer.line_tool is not None:
            viewer.line_tool.disable()

        peak_tool.enable()
        viewer._on_measurement_drawing_state_changed(False)
        self.logger.debug("Peak selection mode enabled")

        QtWidgets.QMessageBox.information(
            viewer,
            "Peak Selection",
            "Click to add peak markers. Use Measure → Export Peaks CSV when done.",
        )

    def exit_measure_mode(self) -> None:
        """Exit measurement mode and restore the line tool state.

        Disables the line drawing tool and updates the UI to reflect the change.
        If in calibration mode, restores the calibration dialog with pending state.
        """
        viewer = self.viewer
        self.logger.debug(
            "Exiting measurement mode (line_draw_mode=%s)", viewer._line_draw_mode
        )
        if viewer._line_draw_mode == "calibration":
            viewer._line_draw_mode = "measurement"
            viewer._on_calibration_pixels_selected = None
            state = dict(viewer._calibration_dialog_state or {})
            viewer._calibration_dialog_state = None
            if state:
                self.logger.debug("Restoring calibration dialog with pending state")
                QtCore.QTimer.singleShot(
                    0, lambda: viewer._open_calibration_dialog(state)
                )

        if viewer.btn_measure is not None and viewer.btn_measure.isChecked():
            viewer.btn_measure.setChecked(False)

        self._disable_peak_selection()
        if viewer._line_draw_mode == "peaks":
            viewer._line_draw_mode = "measurement"

        if viewer.line_tool is not None:
            viewer.line_tool.disable()
        viewer._on_measurement_drawing_state_changed(False)
        if viewer.btn_measure is not None:
            viewer.btn_measure.setStyleSheet("")
        self.logger.debug("Measurement mode exited")

    def toggle_line_measurement(self) -> None:
        """Toggle distance measurement mode on/off.

        Enables or disables the line-drawing tool for distance measurements.
        Updates the measurement mode button styling to indicate active state.
        """
        viewer = self.viewer
        if viewer.line_tool is None:
            self.logger.debug("Ignoring measurement toggle: line tool is unavailable")
            return

        self._disable_peak_selection()

        if viewer.btn_measure is not None and viewer.btn_measure.isChecked():
            viewer._prepare_for_measurement_input()
            viewer._line_draw_mode = "measurement"
            viewer._on_calibration_pixels_selected = None
            viewer.line_tool.enable()
            viewer._on_measurement_drawing_state_changed(False)
            viewer.btn_measure.setStyleSheet("background-color: #4caf50; color: white;")
            self.logger.debug("Measurement mode entered (distance)")
            self.logger.debug("Measurement tool enabled")
        else:
            viewer.line_tool.disable()
            viewer._on_measurement_drawing_state_changed(False)
            if viewer.btn_measure is not None:
                viewer.btn_measure.setStyleSheet("")
            self.logger.debug("Measurement mode exited (toggle)")
            self.logger.debug("Measurement tool disabled")

    def start_distance_measurement(self) -> None:
        """Initiate a distance measurement operation.

        Activates the measurement mode button and enables the line drawing tool.
        """
        viewer = self.viewer
        if viewer.btn_measure is not None and not viewer.btn_measure.isChecked():
            viewer.btn_measure.setChecked(True)
        self.toggle_line_measurement()

    def start_profile_measurement(self) -> None:
        """Initiate a line profile measurement operation.

        Prepares the viewer for profile input and enables the line drawing tool.
        Shows an information dialog if the line tool is unavailable.
        """
        viewer = self.viewer
        if viewer.line_tool is None:
            self.logger.debug(
                "Profile measurement requested but line tool is unavailable"
            )
            QtWidgets.QMessageBox.information(
                viewer,
                "Profile",
                "Line drawing tool is not available.",
            )
            return

        self._disable_peak_selection()

        viewer._prepare_for_measurement_input()
        viewer._line_draw_mode = "profile"
        viewer._on_calibration_pixels_selected = None
        if viewer.btn_measure is not None and not viewer.btn_measure.isChecked():
            viewer.btn_measure.setChecked(True)
        if viewer.btn_measure is not None:
            viewer.btn_measure.setStyleSheet("background-color: #4caf50; color: white;")
        viewer.line_tool.enable()
        viewer._on_measurement_drawing_state_changed(False)
        self.logger.debug("Measurement mode entered (profile)")
        self.logger.debug("Profile measurement mode enabled")

    def on_line_drawn(self, p1: Tuple[float, float], p2: Tuple[float, float]) -> None:
        """Handle a line being drawn for measurement.

        Processes the line based on the current drawing mode:
        - "profile": Adds a line profile measurement.
        - "calibration": Converts physical distance to pixel distance via callback.
        - "measurement": Adds a distance or FFT frequency measurement.

        Args:
            p1: Start point of the line (x, y).
            p2: End point of the line (x, y).
        """
        viewer = self.viewer
        self.logger.debug(
            "Line drawn: mode=%s p1=%s p2=%s", viewer._line_draw_mode, p1, p2
        )
        if viewer._line_draw_mode == "profile":
            self._add_profile_measurement(p1, p2)
            return

        if viewer._line_draw_mode == "calibration":
            scale_x = float(viewer.ax_x.scale) if viewer.ax_x is not None else 1.0
            scale_y = float(viewer.ax_y.scale) if viewer.ax_y is not None else scale_x

            dx_phys = float(p2[0] - p1[0])
            dy_phys = float(p2[1] - p1[1])
            if scale_x != 0 and scale_y != 0:
                dx_px = dx_phys / scale_x
                dy_px = dy_phys / scale_y
                dist_px = float(np.hypot(dx_px, dy_px))
            else:
                dist_px = 0.0

            viewer._line_draw_mode = "measurement"
            if viewer.line_tool is not None:
                viewer.line_tool.disable()

            callback = viewer._on_calibration_pixels_selected
            viewer._on_calibration_pixels_selected = None
            if callback is not None:
                self.logger.debug("Calibration distance selected: %.6g px", dist_px)
                callback(dist_px)
            return

        viewer.measurement_count += 1
        measurement_id = viewer.measurement_count

        if viewer.view_mode == "fft":
            dx_freq = float(p2[0] - p1[0])
            dy_freq = float(p2[1] - p1[1])
            dist_freq = float(np.hypot(dx_freq, dy_freq))

            if viewer._nyq_x and viewer._nyq_y and viewer._fft_region is not None:
                px_scale_x = (2.0 * float(viewer._nyq_x)) / float(
                    viewer._fft_region.shape[1]
                )
                px_scale_y = (2.0 * float(viewer._nyq_y)) / float(
                    viewer._fft_region.shape[0]
                )
                if px_scale_x != 0 and px_scale_y != 0:
                    dx_px = dx_freq / px_scale_x
                    dy_px = dy_freq / px_scale_y
                    dist_px = float(np.hypot(dx_px, dy_px))
                else:
                    dist_px = 0.0
            else:
                dist_px = 0.0

            result: Dict[str, float] = {
                "distance_pixels": dist_px,
                "distance_physical": dist_freq,
                "scale_x": float(viewer.ax_x.scale) if viewer.ax_x else 1.0,
                "scale_y": float(viewer.ax_y.scale) if viewer.ax_y else 1.0,
            }
            if dist_freq != 0:
                result["d_spacing"] = utils.calculate_d_spacing(dist_freq)

            self._add_measurement_graphics(p1, p2, result, measurement_id)
            self.logger.debug(
                "FFT measurement #%s stored: distance_physical=%.6g distance_pixels=%.6g",
                measurement_id,
                dist_freq,
                dist_px,
            )
            return

        scale_x = float(viewer.ax_x.scale) if viewer.ax_x is not None else 1.0
        scale_y = float(viewer.ax_y.scale) if viewer.ax_y is not None else scale_x

        dx_phys = float(p2[0] - p1[0])
        dy_phys = float(p2[1] - p1[1])
        dist_phys = float(np.hypot(dx_phys, dy_phys))

        if scale_x != 0 and scale_y != 0:
            dx_px = dx_phys / scale_x
            dy_px = dy_phys / scale_y
            dist_px = float(np.hypot(dx_px, dy_px))
        else:
            dist_px = 0.0

        result = {
            "distance_pixels": dist_px,
            "distance_physical": dist_phys,
            "scale_x": scale_x,
            "scale_y": scale_y,
        }

        if viewer.is_reciprocal_space and dist_phys != 0:
            frequency = 1.0 / dist_phys
            result["d_spacing"] = utils.calculate_d_spacing(frequency)

        self._add_measurement_graphics(p1, p2, result, measurement_id)
        self.logger.debug(
            "Measurement #%s: p1=%s p2=%s distance_physical=%.6g distance_pixels=%.6g",
            measurement_id,
            p1,
            p2,
            dist_phys,
            dist_px,
        )

    def _image_array(self) -> np.ndarray | None:
        """Return current image data as a finite 2D float array when possible."""
        viewer = self.viewer
        if viewer.data is None:
            return None
        image = np.asarray(viewer.data, dtype=float)
        if image.ndim != 2 or image.shape[0] < 3 or image.shape[1] < 3:
            return None
        return image

    def _map_view_point_to_pixel(self, point: Tuple[float, float]) -> Tuple[float, float]:
        """Map a view-space point to image pixel coordinates."""
        viewer = self.viewer
        image = self._image_array()
        if image is None:
            return float(point[0]), float(point[1])

        width = int(image.shape[1])
        height = int(image.shape[0])

        rect_mapping = None
        image_item = getattr(viewer, "img_orig", None)
        if image_item is not None and hasattr(image_item, "rect"):
            try:
                image_rect = image_item.rect()
                rect_mapping = line_profile_logic.rect_mapping_from_rect(
                    image_width=width,
                    image_height=height,
                    rect_left=float(image_rect.left()),
                    rect_top=float(image_rect.top()),
                    rect_width=float(image_rect.width()),
                    rect_height=float(image_rect.height()),
                )
            except Exception:
                rect_mapping = None

        axis_calibration = None
        if viewer.ax_x is not None and viewer.ax_y is not None:
            try:
                axis_calibration = line_profile_logic.AxisCalibration(
                    scale_x=float(viewer.ax_x.scale),
                    scale_y=float(viewer.ax_y.scale),
                    offset_x=float(viewer.ax_x.offset),
                    offset_y=float(viewer.ax_y.offset),
                )
            except Exception:
                axis_calibration = None

        mapped = line_profile_logic.map_view_points_to_pixel(
            p1=point,
            p2=point,
            rect_mapping=rect_mapping,
            axis_calibration=axis_calibration,
        )
        if mapped is None:
            return float(point[0]), float(point[1])

        x_px, y_px = float(mapped[0]), float(mapped[1])
        x_px = float(np.clip(x_px, 0.0, float(width - 1)))
        y_px = float(np.clip(y_px, 0.0, float(height - 1)))
        return x_px, y_px

    def _map_pixel_to_view(self, x_px: float, y_px: float) -> Tuple[float, float]:
        """Map a pixel coordinate back to the viewer coordinate system."""
        viewer = self.viewer
        image = self._image_array()
        if image is None:
            return x_px, y_px

        width = int(image.shape[1])
        height = int(image.shape[0])
        image_item = getattr(viewer, "img_orig", None)

        if image_item is not None and hasattr(image_item, "rect"):
            try:
                image_rect = image_item.rect()
                if width > 1 and height > 1:
                    x_view = float(image_rect.left()) + (
                        float(x_px) * float(image_rect.width()) / float(width - 1)
                    )
                    y_view = float(image_rect.top()) + (
                        float(y_px) * float(image_rect.height()) / float(height - 1)
                    )
                    return x_view, y_view
            except Exception:
                pass

        if viewer.ax_x is not None and viewer.ax_y is not None:
            try:
                x_view = float(viewer.ax_x.offset) + float(x_px) * float(viewer.ax_x.scale)
                y_view = float(viewer.ax_y.offset) + float(y_px) * float(viewer.ax_y.scale)
                return x_view, y_view
            except Exception:
                pass

        return x_px, y_px

    @staticmethod
    def _gaussian2d_mesh(
        xy: tuple[np.ndarray, np.ndarray],
        amplitude: float,
        x0: float,
        y0: float,
        sigma_x: float,
        sigma_y: float,
        offset: float,
    ) -> np.ndarray:
        """Evaluate an axis-aligned 2D Gaussian on mesh coordinates."""
        xx, yy = xy
        exponent = -(
            ((xx - x0) ** 2) / (2.0 * sigma_x * sigma_x)
            + ((yy - y0) ** 2) / (2.0 * sigma_y * sigma_y)
        )
        return offset + amplitude * np.exp(exponent)

    def _fit_gaussian_peak(
        self,
        image: np.ndarray,
        peak_x_px: int,
        peak_y_px: int,
        fit_radius: int = 6,
    ) -> dict[str, Any]:
        """Fit a 2D Gaussian around a candidate peak pixel."""
        height, width = image.shape
        x_min = max(0, int(peak_x_px) - fit_radius)
        x_max = min(width - 1, int(peak_x_px) + fit_radius)
        y_min = max(0, int(peak_y_px) - fit_radius)
        y_max = min(height - 1, int(peak_y_px) + fit_radius)

        patch = image[y_min : y_max + 1, x_min : x_max + 1]
        yy, xx = np.mgrid[y_min : y_max + 1, x_min : x_max + 1]

        finite_patch = np.asarray(patch, dtype=float)
        if not np.isfinite(finite_patch).any():
            return {
                "fit_success": False,
                "center_x_px": float(peak_x_px),
                "center_y_px": float(peak_y_px),
                "amplitude": float("nan"),
                "offset": float("nan"),
                "sigma_x_px": float("nan"),
                "sigma_y_px": float("nan"),
                "fwhm_x_px": float("nan"),
                "fwhm_y_px": float("nan"),
                "r2": float("nan"),
                "fit_window_radius_px": int(fit_radius),
            }

        patch_min = float(np.nanmin(finite_patch))
        patch_max = float(np.nanmax(finite_patch))
        amp_guess = max(1e-12, patch_max - patch_min)
        offset_guess = patch_min
        sigma_guess = max(1.0, float(fit_radius) / 2.0)

        p0 = [
            amp_guess,
            float(peak_x_px),
            float(peak_y_px),
            sigma_guess,
            sigma_guess,
            offset_guess,
        ]
        lower = [0.0, float(x_min), float(y_min), 0.4, 0.4, -np.inf]
        upper = [np.inf, float(x_max), float(y_max), 20.0, 20.0, np.inf]

        try:
            params, _ = optimize.curve_fit(
                self._gaussian2d_mesh,
                (xx.ravel(), yy.ravel()),
                finite_patch.ravel(),
                p0=p0,
                bounds=(lower, upper),
                maxfev=12000,
            )
            amp, cx, cy, sx, sy, off = [float(v) for v in params]
            model = self._gaussian2d_mesh((xx, yy), amp, cx, cy, sx, sy, off)
            residual = finite_patch - model
            ss_res = float(np.sum(residual * residual))
            centered = finite_patch - float(np.mean(finite_patch))
            ss_tot = float(np.sum(centered * centered))
            r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
            return {
                "fit_success": True,
                "center_x_px": cx,
                "center_y_px": cy,
                "amplitude": amp,
                "offset": off,
                "sigma_x_px": abs(sx),
                "sigma_y_px": abs(sy),
                "fwhm_x_px": 2.354820045 * abs(sx),
                "fwhm_y_px": 2.354820045 * abs(sy),
                "r2": r2,
                "fit_window_radius_px": int(fit_radius),
            }
        except Exception:
            return {
                "fit_success": False,
                "center_x_px": float(peak_x_px),
                "center_y_px": float(peak_y_px),
                "amplitude": float("nan"),
                "offset": float("nan"),
                "sigma_x_px": float("nan"),
                "sigma_y_px": float("nan"),
                "fwhm_x_px": float("nan"),
                "fwhm_y_px": float("nan"),
                "r2": float("nan"),
                "fit_window_radius_px": int(fit_radius),
            }

    def _snap_click_to_nearest_peak(
        self,
        image: np.ndarray,
        click_x_px: float,
        click_y_px: float,
        search_radius: int = 14,
    ) -> tuple[int, int]:
        """Find nearest local-maximum pixel to the click position."""
        height, width = image.shape
        x0 = int(np.clip(round(click_x_px), 0, width - 1))
        y0 = int(np.clip(round(click_y_px), 0, height - 1))

        x_min = max(0, x0 - search_radius)
        x_max = min(width - 1, x0 + search_radius)
        y_min = max(0, y0 - search_radius)
        y_max = min(height - 1, y0 + search_radius)

        patch = image[y_min : y_max + 1, x_min : x_max + 1]
        if patch.size == 0:
            return x0, y0

        finite_patch = np.asarray(patch, dtype=float)
        if not np.isfinite(finite_patch).any():
            return x0, y0

        local_max = ndimage.maximum_filter(finite_patch, size=3, mode="nearest")
        maxima_mask = np.isfinite(finite_patch) & (finite_patch == local_max)
        candidates = np.argwhere(maxima_mask)
        if candidates.size == 0:
            best_rel = np.unravel_index(np.nanargmax(finite_patch), finite_patch.shape)
            return int(x_min + best_rel[1]), int(y_min + best_rel[0])

        click_rel_x = float(click_x_px - x_min)
        click_rel_y = float(click_y_px - y_min)
        d2 = (candidates[:, 1] - click_rel_x) ** 2 + (candidates[:, 0] - click_rel_y) ** 2
        nearest_idx = int(np.argmin(d2))
        nearest_rc = candidates[nearest_idx]
        return int(x_min + nearest_rc[1]), int(y_min + nearest_rc[0])

    def on_peak_selected(self, point: Tuple[float, float]) -> None:
        """Handle a newly selected peak point."""
        viewer = self.viewer
        image = self._image_array()
        if image is None:
            QtWidgets.QMessageBox.information(
                viewer,
                "Peak Selection",
                "Peak selection requires a 2D image.",
            )
            return

        click_x_px, click_y_px = self._map_view_point_to_pixel(point)
        peak_x_px, peak_y_px = self._snap_click_to_nearest_peak(
            image, click_x_px, click_y_px
        )
        fit = self._fit_gaussian_peak(image, peak_x_px, peak_y_px)

        center_x_px = float(fit.get("center_x_px", peak_x_px))
        center_y_px = float(fit.get("center_y_px", peak_y_px))
        x_view, y_view = self._map_pixel_to_view(center_x_px, center_y_px)

        peak_index = len(self._peak_points) + 1

        marker = pg.ScatterPlotItem(
            [x_view],
            [y_view],
            symbol="o",
            size=10,
            pen=pg.mkPen(255, 255, 0, width=2),
            brush=pg.mkBrush(255, 255, 0, 80),
        )
        viewer.p1.addItem(marker)

        label = MeasurementLabel(
            f"Pk#{peak_index}",
            color=pg.mkColor(0, 0, 0),
            anchor=(0, 0),
            fill=LABEL_BRUSH_COLOR,
        )
        label.setPos(x_view, y_view)
        viewer.p1.addItem(label)

        self._peak_markers.append((marker, label))
        self._peak_points.append(
            {
                "index": float(peak_index),
                "click_x_px": float(click_x_px),
                "click_y_px": float(click_y_px),
                "x_view": x_view,
                "y_view": y_view,
                "x_px": center_x_px,
                "y_px": center_y_px,
                "nearest_local_max_x_px": float(peak_x_px),
                "nearest_local_max_y_px": float(peak_y_px),
                "fit_success": bool(fit.get("fit_success", False)),
                "gaussian_amplitude": float(fit.get("amplitude", float("nan"))),
                "gaussian_offset": float(fit.get("offset", float("nan"))),
                "gaussian_sigma_x_px": float(fit.get("sigma_x_px", float("nan"))),
                "gaussian_sigma_y_px": float(fit.get("sigma_y_px", float("nan"))),
                "gaussian_fwhm_x_px": float(fit.get("fwhm_x_px", float("nan"))),
                "gaussian_fwhm_y_px": float(fit.get("fwhm_y_px", float("nan"))),
                "gaussian_fit_r2": float(fit.get("r2", float("nan"))),
                "gaussian_fit_window_radius_px": int(
                    fit.get("fit_window_radius_px", 0)
                ),
            }
        )
        self.logger.debug(
            "Peak selected: index=%s click_px=(%.3f,%.3f) snapped_px=(%.3f,%.3f) fit_success=%s",
            peak_index,
            click_x_px,
            click_y_px,
            center_x_px,
            center_y_px,
            bool(fit.get("fit_success", False)),
        )

    def clear_selected_peaks(self) -> None:
        """Clear all selected peak markers from the view."""
        viewer = self.viewer
        for marker, label in self._peak_markers:
            try:
                viewer.p1.removeItem(marker)
            except Exception:
                pass
            try:
                viewer.p1.removeItem(label)
            except Exception:
                pass
        self._peak_markers.clear()
        self._peak_points.clear()
        self.logger.debug("Cleared selected peaks")

    def export_peaks_to_csv(self) -> None:
        """Export selected peak coordinates and spacing metrics to CSV."""
        viewer = self.viewer
        if not self._peak_points:
            QtWidgets.QMessageBox.information(
                viewer,
                "Export Peaks CSV",
                "No peaks selected. Use Measure → Select Peaks first.",
            )
            return

        axis_units = (
            str(getattr(viewer.ax_x, "units", "") or "")
            if viewer.ax_x is not None
            else ""
        )
        units_label = axis_units or "axis_units"

        default_stem = Path(viewer.file_path).stem if viewer.file_path else "peaks"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_name = f"{default_stem}_peaks_{timestamp}.csv"
        default_path = str(Path(viewer.file_path).with_name(default_name))

        selected_path, _ = QtWidgets.QFileDialog.getSaveFileName(
            viewer,
            "Export Peaks CSV",
            default_path,
            "CSV Files (*.csv)",
        )
        if not selected_path:
            return

        rows: list[dict[str, Any]] = []

        scale_x = float(viewer.ax_x.scale) if viewer.ax_x is not None else float("nan")
        scale_y = float(viewer.ax_y.scale) if viewer.ax_y is not None else float("nan")

        def px_to_cal(x_px: float, y_px: float) -> tuple[float, float]:
            return self._map_pixel_to_view(float(x_px), float(y_px))

        for peak in self._peak_points:
            x_cal, y_cal = px_to_cal(float(peak["x_px"]), float(peak["y_px"]))
            nearest_x_cal, nearest_y_cal = px_to_cal(
                float(peak["nearest_local_max_x_px"]),
                float(peak["nearest_local_max_y_px"]),
            )
            sigma_x_cal = (
                float(peak["gaussian_sigma_x_px"]) * scale_x
                if np.isfinite(float(peak["gaussian_sigma_x_px"])) and np.isfinite(scale_x)
                else float("nan")
            )
            sigma_y_cal = (
                float(peak["gaussian_sigma_y_px"]) * scale_y
                if np.isfinite(float(peak["gaussian_sigma_y_px"])) and np.isfinite(scale_y)
                else float("nan")
            )

            rows.append(
                {
                    "peak_index": int(peak["index"]),
                    "x_px": f"{float(peak['x_px']):.12g}",
                    "y_px": f"{float(peak['y_px']):.12g}",
                    "x_cal": f"{x_cal:.12g}",
                    "y_cal": f"{y_cal:.12g}",
                    "nearest_x_px": f"{float(peak['nearest_local_max_x_px']):.12g}",
                    "nearest_y_px": f"{float(peak['nearest_local_max_y_px']):.12g}",
                    "nearest_x_cal": f"{nearest_x_cal:.12g}",
                    "nearest_y_cal": f"{nearest_y_cal:.12g}",
                    "gaussian_sigma_y_px": f"{float(peak['gaussian_sigma_y_px']):.12g}",
                    "gaussian_sigma_x_px": f"{float(peak['gaussian_sigma_x_px']):.12g}",
                    "gaussian_sigma_y_cal": f"{sigma_y_cal:.12g}",
                    "gaussian_sigma_x_cal": f"{sigma_x_cal:.12g}",
                    "gaussian_goodness_of_fit": f"{float(peak['gaussian_fit_r2']):.12g}",
                }
            )

        headers = [
            "peak_index",
            "x_px",
            "y_px",
            "x_cal",
            "y_cal",
            "nearest_x_px",
            "nearest_y_px",
            "nearest_x_cal",
            "nearest_y_cal",
            "gaussian_sigma_y_px",
            "gaussian_sigma_x_px",
            "gaussian_sigma_y_cal",
            "gaussian_sigma_x_cal",
            "gaussian_goodness_of_fit",
        ]

        repo_root = Path(__file__).resolve().parent
        git_hash = "unknown"
        try:
            git_hash = (
                subprocess.check_output(
                    ["git", "-C", str(repo_root), "rev-parse", "--short", "HEAD"],
                    text=True,
                    stderr=subprocess.DEVNULL,
                )
                .strip()
                .strip("\n")
            )
        except Exception:
            git_hash = "unknown"

        metadata_header_lines = [
            f"# teminator_version_git_hash,{git_hash}",
            f"# export_timestamp_iso,{datetime.now().isoformat()}",
            f"# source_file,{viewer.file_path}",
            f"# distance_units,{units_label}",
            "",
        ]

        try:
            with open(selected_path, "w", newline="", encoding="utf-8") as csv_file:
                for line in metadata_header_lines:
                    csv_file.write(f"{line}\n")
                writer = csv.DictWriter(csv_file, fieldnames=headers)
                writer.writeheader()
                writer.writerows(rows)
        except OSError as exc:
            self.logger.debug("Failed to export peaks CSV: %s", exc)
            QtWidgets.QMessageBox.critical(
                viewer,
                "Export Peaks CSV",
                f"Could not write CSV file:\n{exc}",
            )
            return

        csv_path = Path(selected_path)
        png_path = csv_path.with_suffix(".png")
        png_saved = False
        png_error: str | None = None
        try:
            capture_widget = getattr(viewer, "glw", None)
            if capture_widget is not None and hasattr(capture_widget, "grab"):
                pixmap = capture_widget.grab()
            else:
                pixmap = viewer.grab()

            png_saved = bool(pixmap.save(str(png_path), "PNG"))
            if not png_saved:
                png_error = "Qt could not encode PNG output."
        except Exception as exc:
            png_saved = False
            png_error = str(exc)

        self.logger.debug(
            "Exported peaks CSV: path=%s peaks=%s",
            selected_path,
            len(self._peak_points),
        )
        if png_saved:
            self.logger.debug("Exported peaks view PNG: path=%s", str(png_path))
            QtWidgets.QMessageBox.information(
                viewer,
                "Export Peaks CSV",
                (
                    f"Exported {len(self._peak_points)} peaks to:\n{selected_path}\n\n"
                    f"Exported current labeled view to:\n{png_path}"
                ),
            )
        else:
            if png_error:
                self.logger.debug("Failed to export peaks view PNG: %s", png_error)
            QtWidgets.QMessageBox.warning(
                viewer,
                "Export Peaks CSV",
                (
                    f"Exported {len(self._peak_points)} peaks to:\n{selected_path}\n\n"
                    "Could not export PNG view snapshot."
                    + (f"\nReason: {png_error}" if png_error else "")
                ),
            )

    def on_measurement_label_clicked(self, label: pg.TextItem) -> None:
        """Handle a measurement label being clicked.

        Selects the measurement and highlights its label with a yellow color.
        Other measurement labels are reset to their original color.

        Args:
            label: The TextItem label that was clicked.
        """
        viewer = self.viewer
        selected_index = None
        for idx, (_measurement_id, _line_item, text_item) in enumerate(
            viewer.measurement_items
        ):
            if text_item is label:
                selected_index = idx
                break

        viewer.selected_measurement_index = selected_index
        self.logger.debug("Measurement label selected: index=%s", selected_index)

        for idx, (_measurement_id, _line_item, text_item) in enumerate(
            viewer.measurement_items
        ):
            if idx == selected_index:
                self.set_label_fill(text_item, pg.mkBrush(255, 200, 0, 255))
            else:
                self.set_label_fill(text_item, LABEL_BRUSH_COLOR)

    def clear_measurements_from_history(self) -> None:
        """Remove all measurements from the viewer per history window request.

        Similar to clear_measurements but called from the measurement history window.
        Clears both distance and profile measurements from the display.
        """
        viewer = self.viewer
        distance_count = len(viewer.measurement_items)
        profile_count = len(viewer.profile_measurement_items)
        self.logger.debug(
            "Clearing viewer overlays from history window request: distances=%s profiles=%s",
            distance_count,
            profile_count,
        )
        for _measurement_id, line_item, text_item in viewer.measurement_items:
            viewer.p1.removeItem(line_item)
            viewer.p1.removeItem(text_item)
        viewer.measurement_items.clear()

        for profile_id in list(viewer.profile_measurement_items.keys()):
            self._delete_profile_measurement(profile_id)

        self.clear_selected_peaks()

        viewer.selected_measurement_index = None
        self.logger.debug("Viewer overlays cleared from history request")

    def delete_selected_measurement(self) -> None:
        """Delete the currently selected measurement from the viewer.

        Removes the selected measurement's line and label from the plot.
        Does nothing if no measurement is selected or the index is invalid.
        """
        viewer = self.viewer
        if viewer.selected_measurement_index is None:
            self.logger.debug("Delete selected measurement ignored: nothing selected")
            return

        if not (0 <= viewer.selected_measurement_index < len(viewer.measurement_items)):
            self.logger.debug(
                "Delete selected measurement ignored: invalid index=%s count=%s",
                viewer.selected_measurement_index,
                len(viewer.measurement_items),
            )
            viewer.selected_measurement_index = None
            return

        deleted_index = viewer.selected_measurement_index
        measurement_id, line_item, text_item = viewer.measurement_items.pop(
            viewer.selected_measurement_index
        )
        viewer.p1.removeItem(line_item)
        viewer.p1.removeItem(text_item)

        viewer.selected_measurement_index = None
        for _measurement_id, _line_item, item in viewer.measurement_items:
            self.set_label_fill(item, LABEL_BRUSH_COLOR)
        self.logger.debug(
            "Deleted selected measurement: index=%s id=%s remaining=%s",
            deleted_index,
            measurement_id,
            len(viewer.measurement_items),
        )

    @staticmethod
    def set_label_fill(text_item: pg.TextItem, brush: pg.QtGui.QBrush) -> None:
        """Set the background fill color of a measurement label.

        Args:
            text_item: The TextItem label to modify.
            brush: The brush to use for the fill color.
        """
        if hasattr(text_item, "setFill"):
            text_item.setFill(brush)
        elif hasattr(text_item, "setBrush"):
            text_item.setBrush(brush)

    def delete_measurement_by_label(self, label_text: str) -> None:
        """Delete a measurement by its label text.

        Args:
            label_text: The label text to delete (e.g., "M#5" or "P#3").
        """
        viewer = self.viewer
        self.logger.debug("Delete measurement by label requested: %s", label_text)
        target_index = None
        for idx, (_measurement_id, _line_item, text_item) in enumerate(
            viewer.measurement_items
        ):
            if text_item.toPlainText() == label_text:
                target_index = idx
                break

        if target_index is not None:
            viewer.selected_measurement_index = target_index
            self.delete_selected_measurement()
            return

        if label_text.startswith("P#"):
            profile_id_text = label_text.split(" ", 1)[0][2:]
            try:
                profile_id = int(profile_id_text)
            except Exception:
                profile_id = None

            if profile_id is not None:
                deleted = self._delete_profile_measurement(profile_id)
                if deleted:
                    self.logger.debug("Deleted profile by label: id=%s", profile_id)
                    return

        self.logger.debug("Delete measurement by label ignored: not found")

    def delete_measurement_by_history_id(self, entry_id: int, entry_type: str) -> None:
        """Delete a measurement by its history entry ID.

        Args:
            entry_id: The ID of the measurement from the history window.
            entry_type: Type of measurement ("distance" or "profile").
        """
        viewer = self.viewer
        self.logger.debug(
            "Delete measurement by history id requested: id=%s type=%s",
            entry_id,
            entry_type,
        )
        if entry_type == "profile":
            deleted = self._delete_profile_measurement(entry_id)
            self.logger.debug(
                "Delete profile by history id result: deleted=%s", deleted
            )
            return

        target_index = None
        for idx, (measurement_id, _line_item, _text_item) in enumerate(
            viewer.measurement_items
        ):
            if measurement_id == entry_id:
                target_index = idx
                break

        if target_index is None:
            self.logger.debug("Delete distance by history id ignored: id not found")
            return

        viewer.selected_measurement_index = target_index
        self.delete_selected_measurement()

    def open_measurement_by_history_id(self, entry_id: int, entry_type: str) -> None:
        """Open/display a measurement from the history window.

        Selects the measurement to display its profile plot (for profiles) or
        highlights the measurement on the main image (for distances).

        Args:
            entry_id: The ID of the measurement from the history window.
            entry_type: Type of measurement ("distance" or "profile").
        """
        viewer = self.viewer
        self.logger.debug(
            "Open measurement by history id requested: id=%s type=%s",
            entry_id,
            entry_type,
        )
        if entry_type != "profile":
            self.logger.debug("Open measurement ignored: non-profile entry")
            return

        record = viewer.profile_measurement_items.get(entry_id)
        if record is None:
            self.logger.debug("Open profile ignored: id not found")
            return

        window = record.get("window")
        if isinstance(window, LineProfileWindow):
            try:
                if window.isVisible():
                    window.raise_()
                    window.activateWindow()
                    self.logger.debug("Focused open profile window: id=%s", entry_id)
                    return
                window.show()
                window.raise_()
                window.activateWindow()
                self.logger.debug("Reopened existing profile window: id=%s", entry_id)
                return
            except RuntimeError:
                self.logger.debug(
                    "Profile window object invalid, recreating: id=%s", entry_id
                )

        distances = record.get("distances")
        intensities = record.get("intensities")
        x_axis_label = str(record.get("x_axis_label", "Distance (px)"))
        title = str(record.get("title", f"Profile Measurement #{entry_id}"))
        if not isinstance(distances, np.ndarray) or not isinstance(
            intensities, np.ndarray
        ):
            self.logger.debug(
                "Open profile failed: profile data missing for id=%s", entry_id
            )
            return

        profile_window = LineProfileWindow(
            title,
            distances,
            intensities,
            x_axis_label=x_axis_label,
            parent=viewer,
        )
        profile_window.show()
        profile_window.raise_()
        profile_window.activateWindow()
        record["window"] = profile_window
        self.logger.debug("Recreated and opened profile window: id=%s", entry_id)

    def rename_measurement_by_history_id(
        self, entry_id: int, entry_type: str, new_text: str
    ) -> None:
        """Rename a measurement via the history window.

        Args:
            entry_id: The ID of the measurement from the history window.
            entry_type: Type of measurement ("distance" or "profile").
            new_text: The new measurement label text.
        """
        viewer = self.viewer
        self.logger.debug(
            "Rename measurement by history id requested: id=%s type=%s new_text=%s",
            entry_id,
            entry_type,
            new_text,
        )

        if entry_type == "profile":
            record = viewer.profile_measurement_items.get(entry_id)
            if record is None:
                self.logger.debug("Rename profile ignored: id not found")
                return

            window = record.get("window")
            record["history_text"] = new_text
            profile_title = f"Profile Measurement #{entry_id} - {new_text}"
            record["title"] = profile_title
            text_item = record.get("text_item")
            if isinstance(text_item, MeasurementLabel):
                if hasattr(text_item, "setText"):
                    text_item.setText(new_text)
                elif hasattr(text_item, "setPlainText"):
                    text_item.setPlainText(new_text)
            if isinstance(window, LineProfileWindow):
                try:
                    window.setWindowTitle(profile_title)
                except RuntimeError:
                    record["window"] = None
            self.logger.debug("Renamed profile measurement: id=%s", entry_id)
            return

        for idx, (measurement_id, _line_item, text_item) in enumerate(
            viewer.measurement_items
        ):
            if measurement_id != entry_id:
                continue

            if hasattr(text_item, "setText"):
                text_item.setText(new_text)
            elif hasattr(text_item, "setPlainText"):
                text_item.setPlainText(new_text)
            self.logger.debug(
                "Renamed distance measurement: id=%s index=%s", entry_id, idx
            )
            return

        self.logger.debug("Rename distance ignored: id not found")

    def format_measurement_label(
        self, result: dict, measurement_id: Optional[int] = None
    ) -> str:
        """Format a measurement result dictionary into a display label.

        Formats the label based on the current view mode (FFT vs image),
        including physical units and d-spacing if applicable.

        Args:
            result: Dictionary with measurement data (distance_physical, distance_pixels, scales, etc.).
            measurement_id: Optional ID number to include in the label prefix.

        Returns:
            Formatted label string with physical units and pixel distance.
        """
        viewer = self.viewer
        if viewer.view_mode == "fft":
            scaled_dist, scaled_unit = utils.format_reciprocal_scale(
                result["distance_physical"], viewer.freq_axis_base_unit
            )
            prefix = f"#{measurement_id} " if measurement_id is not None else ""
            if "d_spacing" in result:
                return (
                    f"{prefix}d: {result['d_spacing']:.4f} Å\n"
                    f"({scaled_dist:.4f} {scaled_unit}⁻¹)"
                )
            return (
                f"{prefix}{scaled_dist:.4f} {scaled_unit}⁻¹\n"
                f"({result['distance_pixels']:.1f} px)"
            )

        if viewer.is_reciprocal_space:
            scaled_dist, scaled_unit = utils.format_reciprocal_scale(
                result["distance_physical"], viewer.ax_x.units
            )
        else:
            scaled_dist, scaled_unit = utils.format_si_scale(
                result["distance_physical"], viewer.ax_x.units
            )
        prefix = f"#{measurement_id} " if measurement_id is not None else ""

        if viewer.is_reciprocal_space and "d_spacing" in result:
            return (
                f"{prefix}d: {result['d_spacing']:.4f} Å\n"
                f"({scaled_dist:.4f} {scaled_unit})"
            )
        return (
            f"{prefix}{scaled_dist:.4f} {scaled_unit}\n"
            f"({result['distance_pixels']:.1f} px)"
        )

    def show_measurement_history(self) -> None:
        """Display the measurement history window.

        Opens or focuses the history window that shows all measurements taken
        and allows reviewing, renaming, and deleting measurements.
        """
        viewer = self.viewer
        if viewer.measurement_history_window is None:
            self.logger.debug("Creating measurement history window")
            viewer.measurement_history_window = MeasurementHistoryWindow(viewer)

        history_window = viewer.measurement_history_window
        if history_window is None:
            self.logger.debug(
                "Measurement history window unavailable after creation attempt"
            )
            return

        history_window.show()
        history_window.raise_()
        history_window.activateWindow()
        self.logger.debug("Measurement history window shown")

    def add_to_measurement_history_entry(
        self,
        measurement_text: str,
        *,
        measurement_id: int | None = None,
        measurement_type: str = "distance",
    ) -> None:
        """Add a measurement entry to the history window.

        Creates the history window if it doesn't exist. Adds the measurement
        with its ID and type for later retrieval.

        Args:
            measurement_text: The formatted measurement label text.
            measurement_id: Optional ID number for the measurement.
            measurement_type: Type of measurement ("distance" or "profile").
        """
        viewer = self.viewer
        if viewer.measurement_history_window is None:
            self.logger.debug(
                "Creating measurement history window for first measurement"
            )
            viewer.measurement_history_window = MeasurementHistoryWindow(viewer)

        history_window = viewer.measurement_history_window
        if history_window is None:
            self.logger.debug(
                "Measurement history add skipped: history window unavailable"
            )
            return

        history_window.add_measurement(
            measurement_text,
            measurement_id=measurement_id,
            measurement_type=measurement_type,
        )
        self.logger.debug(
            "Measurement added to history: id=%s type=%s text=%s",
            measurement_id,
            measurement_type,
            measurement_text,
        )

    def _add_measurement_graphics(
        self,
        p1: Tuple[float, float],
        p2: Tuple[float, float],
        result: Dict[str, float],
        measurement_id: int,
    ) -> None:
        """Add visual measurement line and label to the plot.

        Creates a PlotDataItem line between two points and a MeasurementLabel
        at the midpoint. Also adds the measurement to history.

        Args:
            p1: Start point of the measurement line (x, y).
            p2: End point of the measurement line (x, y).
            result: Dictionary with measurement data.
            measurement_id: The ID number for this measurement.
        """
        viewer = self.viewer
        line = pg.PlotDataItem([p1[0], p2[0]], [p1[1], p2[1]], pen=DRAWN_LINE_PEN)
        viewer.p1.addItem(line)

        label_text = self.format_measurement_label(result, measurement_id)
        mid_x = (p1[0] + p2[0]) / 2.0
        mid_y = (p1[1] + p2[1]) / 2.0

        text_item = MeasurementLabel(
            label_text,
            color=pg.mkColor(0, 0, 0),
            anchor=(0, 0),
            fill=LABEL_BRUSH_COLOR,
        )
        text_item.setPos(mid_x, mid_y)
        text_item.sigLabelClicked.connect(self.on_measurement_label_clicked)
        viewer.p1.addItem(text_item)

        viewer.measurement_items.append((measurement_id, line, text_item))
        self.logger.debug(
            "Measurement graphics added: id=%s midpoint=(%.6g, %.6g)",
            measurement_id,
            mid_x,
            mid_y,
        )
        self.add_to_measurement_history_entry(
            label_text,
            measurement_id=measurement_id,
            measurement_type="distance",
        )

    def _add_profile_measurement(
        self, p1: Tuple[float, float], p2: Tuple[float, float]
    ) -> None:
        """Add a line profile measurement between two points.

        Extracts intensities along the line, creates a plot window showing the
        profile, and adds a label to the main plot.

        Args:
            p1: Start point of the profile line (x, y).
            p2: End point of the profile line (x, y).
        """
        viewer = self.viewer
        profile = self._extract_line_profile(p1, p2)
        if profile is None:
            self.logger.debug("Profile measurement ignored: failed to extract profile")
            QtWidgets.QMessageBox.information(
                viewer,
                "Profile",
                "Could not sample a profile from this line.",
            )
            return

        distances, intensities, line_item, x_axis_label = profile
        viewer.profile_measurement_count += 1
        profile_id = viewer.profile_measurement_count
        line_label = f"P#{profile_id} profile ({len(distances)} pts)"
        profile_annotation = f"P#{profile_id} profile"

        mid_x = (p1[0] + p2[0]) / 2.0
        mid_y = (p1[1] + p2[1]) / 2.0
        text_item = MeasurementLabel(
            profile_annotation,
            color=pg.mkColor(0, 0, 0),
            anchor=(0, 0),
            fill=LABEL_BRUSH_COLOR,
        )
        text_item.setPos(mid_x, mid_y)
        viewer.p1.addItem(text_item)

        profile_window = LineProfileWindow(
            f"Profile Measurement #{profile_id}",
            distances,
            intensities,
            x_axis_label=x_axis_label,
            parent=viewer,
        )
        profile_window.show()
        profile_window.raise_()
        profile_window.activateWindow()

        viewer.profile_measurement_items[profile_id] = {
            "line_item": line_item,
            "text_item": text_item,
            "window": profile_window,
            "distances": distances,
            "intensities": intensities,
            "title": f"Profile Measurement #{profile_id}",
            "history_text": line_label,
            "x_axis_label": x_axis_label,
        }
        self.add_to_measurement_history_entry(
            line_label,
            measurement_id=profile_id,
            measurement_type="profile",
        )
        self.logger.debug(
            "Profile measurement added: id=%s samples=%s midpoint=(%.6g, %.6g)",
            profile_id,
            len(distances),
            mid_x,
            mid_y,
        )

    def _extract_line_profile(
        self,
        p1: Tuple[float, float],
        p2: Tuple[float, float],
    ) -> tuple[np.ndarray, np.ndarray, pg.PlotDataItem, str] | None:
        """Extract intensity values along a line in the image.

        Samples pixel intensities using bilinear interpolation along a line
        between two points. Returns the distances and intensities.

        Args:
            p1: Start point of the profile line (x, y).
            p2: End point of the profile line (x, y).

        Returns:
            Tuple of (distances, intensities, line_item, x_axis_label),
            or None if extraction fails.
        """
        viewer = self.viewer
        if viewer.data is None:
            self.logger.debug("Profile extraction failed: missing image data")
            return None

        image = np.asarray(viewer.data, dtype=float)
        if image.ndim != 2 or image.shape[0] < 2 or image.shape[1] < 2:
            self.logger.debug(
                "Profile extraction failed: invalid image shape=%s",
                getattr(image, "shape", None),
            )
            return None

        width = image.shape[1]
        height = image.shape[0]

        rect_mapping = None
        image_item = getattr(viewer, "img_orig", None)
        if image_item is not None and hasattr(image_item, "rect"):
            try:
                image_rect = image_item.rect()
                rect_mapping = line_profile_logic.rect_mapping_from_rect(
                    image_width=width,
                    image_height=height,
                    rect_left=float(image_rect.left()),
                    rect_top=float(image_rect.top()),
                    rect_width=float(image_rect.width()),
                    rect_height=float(image_rect.height()),
                )
            except Exception:
                rect_mapping = None

        axis_calibration = None
        if viewer.ax_x is not None and viewer.ax_y is not None:
            try:
                axis_calibration = line_profile_logic.AxisCalibration(
                    scale_x=float(viewer.ax_x.scale),
                    scale_y=float(viewer.ax_y.scale),
                    offset_x=float(viewer.ax_x.offset),
                    offset_y=float(viewer.ax_y.offset),
                )
            except Exception:
                self.logger.debug(
                    "Profile extraction failed: invalid axis calibration values"
                )
                return None

        mapped = line_profile_logic.map_view_points_to_pixel(
            p1=p1,
            p2=p2,
            rect_mapping=rect_mapping,
            axis_calibration=axis_calibration,
        )
        if mapped is None:
            self.logger.debug("Profile extraction failed: axis scale is zero")
            return None
        x0, y0, x1, y1 = mapped

        if not line_profile_logic.endpoints_are_finite(x0, y0, x1, y1):
            self.logger.debug(
                "Profile extraction failed: non-finite profile endpoints x0=%s y0=%s x1=%s y1=%s",
                x0,
                y0,
                x1,
                y1,
            )
            return None

        x0, y0, x1, y1 = line_profile_logic.clamp_profile_endpoints(
            x0=x0,
            y0=y0,
            x1=x1,
            y1=y1,
            width=width,
            height=height,
        )

        sample_count = line_profile_logic.compute_sample_count(
            x0=x0,
            y0=y0,
            x1=x1,
            y1=y1,
            width=width,
            height=height,
        )

        xs_clipped, ys_clipped = line_profile_logic.sample_line_coordinates(
            x0=x0,
            y0=y0,
            x1=x1,
            y1=y1,
            sample_count=sample_count,
            width=width,
            height=height,
        )

        intensities = self._sample_image_bilinear(image, xs_clipped, ys_clipped)
        distances_px = line_profile_logic.pixel_distance_axis(xs_clipped, ys_clipped)

        axis_unit = line_profile_logic.resolve_profile_axis_unit(
            view_mode=viewer.view_mode,
            axis_units=getattr(getattr(viewer, "ax_x", None), "units", None),
            freq_axis_base_unit=getattr(viewer, "freq_axis_base_unit", None),
        )

        if axis_unit:
            distances_world = line_profile_logic.world_distance_axis(
                xs=xs_clipped,
                ys=ys_clipped,
                rect_mapping=rect_mapping,
                axis_calibration=axis_calibration,
            )

            distances, x_axis_label, display_unit, reference_distance, scaled_ref = (
                line_profile_logic.scaled_distance_axis(
                    distances_world=distances_world,
                    axis_unit=axis_unit,
                    view_mode=viewer.view_mode,
                    is_reciprocal_space=viewer.is_reciprocal_space,
                    format_reciprocal_scale=utils.format_reciprocal_scale,
                    format_si_scale=utils.format_si_scale,
                )
            )
            if (
                display_unit is not None
                and reference_distance is not None
                and scaled_ref is not None
            ):
                self.logger.debug(
                    "Profile x-axis uses calibrated units with SI formatting: axis_unit=%s display_unit=%s ref_world=%.6g ref_scaled=%.6g",
                    axis_unit,
                    display_unit,
                    reference_distance,
                    scaled_ref,
                )
            else:
                self.logger.debug(
                    "Profile x-axis uses calibrated units without scaling (degenerate span): axis_unit=%s",
                    axis_unit,
                )
        else:
            distances = distances_px
            x_axis_label = "Distance (px)"
            self.logger.debug("Profile x-axis uses pixel units (uncalibrated axes)")

        line_item = pg.PlotDataItem([p1[0], p2[0]], [p1[1], p2[1]], pen=DRAWN_LINE_PEN)
        viewer.p1.addItem(line_item)
        self.logger.debug(
            "Profile extracted: p1_px=(%.3f, %.3f) p2_px=(%.3f, %.3f) samples=%s",
            x0,
            y0,
            x1,
            y1,
            sample_count,
        )
        return distances, intensities, line_item, x_axis_label

    @staticmethod
    def _sample_image_bilinear(
        image: np.ndarray, xs: np.ndarray, ys: np.ndarray
    ) -> np.ndarray:
        """Sample image intensities using bilinear interpolation.

        Args:
            image: 2D image array.
            xs: X coordinates for sampling.
            ys: Y coordinates for sampling.

        Returns:
            Interpolated intensity values at the given coordinates.
        """
        x0 = np.floor(xs).astype(np.int64)
        y0 = np.floor(ys).astype(np.int64)
        x1 = np.minimum(x0 + 1, image.shape[1] - 1)
        y1 = np.minimum(y0 + 1, image.shape[0] - 1)

        wx = xs - x0
        wy = ys - y0

        i00 = image[y0, x0]
        i10 = image[y0, x1]
        i01 = image[y1, x0]
        i11 = image[y1, x1]

        return (
            i00 * (1.0 - wx) * (1.0 - wy)
            + i10 * wx * (1.0 - wy)
            + i01 * (1.0 - wx) * wy
            + i11 * wx * wy
        )

    def _delete_profile_measurement(self, profile_id: int) -> bool:
        """Delete a profile measurement by ID.

        Removes the profile line, label, and plot window from the viewer.

        Args:
            profile_id: The ID of the profile measurement to delete.

        Returns:
            True if successfully deleted; False if not found.
        """
        viewer = self.viewer
        record = viewer.profile_measurement_items.pop(profile_id, None)
        if record is None:
            return False

        line_item = record.get("line_item")
        if line_item is not None:
            viewer.p1.removeItem(line_item)

        text_item = record.get("text_item")
        if text_item is not None:
            viewer.p1.removeItem(text_item)

        profile_window = record.get("window")
        if isinstance(profile_window, LineProfileWindow):
            try:
                profile_window.close()
            except RuntimeError:
                pass
        self.logger.debug("Profile measurement deleted: id=%s", profile_id)
        return True
