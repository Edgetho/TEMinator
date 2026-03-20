# SPDX-License-Identifier: GPL-2.0-only
# Copyright (C) 2026 Cooper Stuntz
# See LICENSE for full license terms.

"""Utility functions for FFT, measurements, image analysis, and dialogs."""
import numpy as np
from typing import Tuple, Optional, List, Dict
import subprocess
import os
import logging
import html
from pathlib import Path

import unit_utils
from pyqtgraph.Qt import QtWidgets, QtCore, QtGui

logger = logging.getLogger(__name__)


# Cache for window functions to avoid recomputation
_window_cache = {}


def get_git_commit_date() -> str:
    """
    Get the commit date of the current git branch.
    
    Deprecated: Use get_git_commit_info() instead.
    
    Returns:
        A string with the commit date in format "YYYY-MM-DD" or "Version 1.0" if git info unavailable
    """
    commit_date, _, _ = get_git_commit_info()
    return commit_date


def get_git_commit_info() -> Tuple[str, str, str]:
    """
    Get the commit date, short hash, and branch of the current git branch.
    
    Returns:
        A tuple of (commit_date, short_hash, branch_name) where commit_date is in format "YYYY-MM-DD"
        Falls back to ("Version 1.0", "", "") if git info unavailable
    """
    try:
        repo_dir = os.path.dirname(os.path.abspath(__file__))
        result = subprocess.run(
            ['git', 'log', '-1', '--format=%cI %h %D'],
            cwd=repo_dir,
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0 and result.stdout.strip():
            parts = result.stdout.strip().split(maxsplit=2)
            if len(parts) >= 2:
                # Extract date part from ISO format (YYYY-MM-DDTHH:MM:SS+ZZ:ZZ)
                commit_date = parts[0].split('T')[0]
                short_hash = parts[1]
                branch_name = ""
                if len(parts) >= 3:
                    decorations = parts[2]
                    if "HEAD -> " in decorations:
                        head_part = decorations.split("HEAD -> ", 1)[1]
                        branch_name = head_part.split(",", 1)[0].strip()
                    elif "origin/" in decorations:
                        origin_part = decorations.split("origin/", 1)[1]
                        branch_name = origin_part.split(",", 1)[0].strip()
                return commit_date, short_hash, branch_name
    except Exception:
        pass
    
    return "Version 1.0", "", ""


def show_about_dialog(parent_widget: QtWidgets.QWidget) -> None:
    """
    Display the About TEMinator dialog.
    
    Args:
        parent_widget: The parent widget for the dialog
    """
    commit_date, short_hash, branch_name = get_git_commit_info()
    version_str = commit_date
    if short_hash:
        version_str += f" ({short_hash})"
    if branch_name:
        version_str += f" [{branch_name}]"
    
    about_text = (
        "<b>TEMinator</b><br>"
        f"{version_str}<br>"
        "<br>"
        "Desktop viewer for electron microscopy images with fast, "
        "interactive FFT analysis, distance measurements, and "
        "metadata-aware scaling.<br>"
        "<br>"
        "<b>Copyright:</b> © 2026 Cooper Stuntz<br>"
        "<b>License:</b> GNU General Public License v2.0<br>"
        "<br>"
        "<a href='https://github.com/Edgetho/TEMinator'>GitHub Repository</a>"
    )
    
    dialog = QtWidgets.QDialog(parent_widget)
    dialog.setWindowTitle("About TEMinator")
    dialog.setSizeGripEnabled(False)
    
    main_layout = QtWidgets.QVBoxLayout()
    main_layout.setSizeConstraint(QtWidgets.QLayout.SetFixedSize)
    
    # Top layout with icon and text
    top_layout = QtWidgets.QHBoxLayout()
    
    # Add app icon on the left
    icon_label = QtWidgets.QLabel()
    icon_pixmap = QtGui.QPixmap("app_icon.png")
    if not icon_pixmap.isNull():
        icon_label.setPixmap(icon_pixmap.scaledToWidth(80, QtCore.Qt.SmoothTransformation))
        top_layout.addWidget(icon_label)
    
    # Add text on the right
    text_label = QtWidgets.QLabel(about_text)
    text_label.setOpenExternalLinks(True)
    text_label.setTextFormat(QtCore.Qt.RichText)
    text_label.setWordWrap(True)
    text_label.setAlignment(QtCore.Qt.AlignTop | QtCore.Qt.AlignLeft)
    top_layout.addWidget(text_label)
    
    main_layout.addLayout(top_layout)
    
    # Add close button
    button_box = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Close)
    button_box.rejected.connect(dialog.close)
    main_layout.addWidget(button_box)
    
    dialog.setLayout(main_layout)
    dialog.adjustSize()
    dialog.exec_()
    logger.debug("Displayed About dialog")


def _plain_text_to_pre_html(content: str) -> str:
    """Convert plain text content into escaped HTML wrapped in a styled <pre> block."""
    escaped_content = html.escape(content)
    return f"<pre class='mono-block'>{escaped_content}</pre>"


def _dialog_content_stylesheet() -> str:
    """Return a shared rich-text stylesheet for app-rendered dialogs."""
    return """
        body {
            margin: 0;
            padding: 8px;
            line-height: 1.45;
            font-size: 10pt;
        }
        h1, h2, h3 {
            margin: 0.4em 0 0.3em 0;
            font-weight: 600;
        }
        p, ul, ol {
            margin: 0.25em 0 0.5em 0;
        }
        a {
            text-decoration: none;
            color: #2f6db3;
        }
        a:hover {
            text-decoration: underline;
        }
        code {
            font-family: monospace;
            font-size: 0.96em;
            background: #f2f4f7;
            border: 1px solid #d8dde6;
            border-radius: 3px;
            padding: 1px 4px;
        }
        pre {
            font-family: monospace;
            white-space: pre-wrap;
            border: 1px solid #d8dde6;
            border-radius: 6px;
            background: #f7f9fc;
            padding: 8px;
        }
        pre.mono-block {
            white-space: pre;
            font-size: 9.8pt;
            background: #f8fafd;
        }
    """


def _wrap_html_document(body_html: str) -> str:
    """Wrap content in a complete HTML document with shared stylesheet."""
    return (
        "<html><head><style>"
        f"{_dialog_content_stylesheet()}"
        "</style></head><body>"
        f"{body_html}"
        "</body></html>"
    )


def _show_text_content_dialog(
    parent_widget: QtWidgets.QWidget,
    title: str,
    content: str,
    content_format: str = "html",
    width: int = 800,
    height: int = 600,
) -> None:
    """Render HTML/Markdown content in a consistent app-controlled dialog."""
    dialog = QtWidgets.QDialog(parent_widget)
    dialog.setWindowTitle(title)
    dialog.resize(width, height)

    layout = QtWidgets.QVBoxLayout(dialog)
    layout.setContentsMargins(10, 10, 10, 10)
    layout.setSpacing(8)

    text_view = QtWidgets.QTextBrowser(dialog)
    text_view.setReadOnly(True)
    text_view.setOpenExternalLinks(True)

    fixed_font = QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.FixedFont)
    if fixed_font.pointSize() > 0:
        fixed_font.setPointSize(max(9, fixed_font.pointSize()))
    text_view.setFont(fixed_font)

    text_view.document().setDefaultStyleSheet(_dialog_content_stylesheet())

    if content_format == "markdown":
        if hasattr(text_view, "setMarkdown"):
            text_view.setMarkdown(content)
        else:
            text_view.setHtml(_wrap_html_document(_plain_text_to_pre_html(content)))
    else:
        text_view.setHtml(_wrap_html_document(content))
    layout.addWidget(text_view)

    button_box = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Close)
    button_box.rejected.connect(dialog.close)
    layout.addWidget(button_box)

    dialog.exec_()


def show_keyboard_shortcuts_dialog(
    parent_widget: QtWidgets.QWidget,
    menu_config: List,
    extra_shortcuts: Optional[Dict[str, str]] = None,
    additional_colormaps: Optional[List[str]] = None,
) -> None:
    """
    Display keyboard shortcuts dialog.
    
    This is a centralized dialog for showing all keyboard shortcuts, with
    support for additional shortcuts and colormaps specific to certain windows.
    
    Args:
        parent_widget: The parent widget for the dialog
        menu_config: Menu configuration list from create_shared_menu_config()
        extra_shortcuts: Optional dict of extra keyboard shortcuts not in main menu
        additional_colormaps: Optional list of colormap names to display
    """
    logger.debug("Displaying keyboard shortcuts dialog")

    dialog = QtWidgets.QDialog(parent_widget)
    dialog.setWindowTitle("Keyboard Shortcuts")
    dialog.setStyleSheet(
        """
        QDialog {
            background: #f7f8fb;
        }
        QGroupBox {
            background: transparent;
            border: none;
            margin-top: 10px;
            font-weight: 600;
            padding-top: 2px;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 0px;
            padding: 0 4px;
            color: #2b3442;
        }
        QLabel {
            color: #1f2937;
        }
        QScrollArea {
            background: transparent;
        }
        QDialogButtonBox QPushButton {
            min-width: 88px;
            padding: 6px 14px;
        }
        """
    )

    main_layout = QtWidgets.QVBoxLayout(dialog)
    main_layout.setContentsMargins(12, 12, 12, 12)
    main_layout.setSpacing(10)

    header_label = QtWidgets.QLabel(
        "<b style='font-size:14pt;'>TEMinator Keyboard Shortcuts</b><br>"
    )
    header_label.setTextFormat(QtCore.Qt.RichText)
    header_label.setWordWrap(True)
    main_layout.addWidget(header_label)

    scroll = QtWidgets.QScrollArea(dialog)
    scroll.setWidgetResizable(True)
    scroll.setFrameShape(QtWidgets.QFrame.NoFrame)

    content_widget = QtWidgets.QWidget(scroll)
    content_layout = QtWidgets.QVBoxLayout(content_widget)
    content_layout.setContentsMargins(0, 0, 0, 0)
    content_layout.setSpacing(10)

    def add_section(title: str, rows: List[tuple[str, str]]) -> None:
        if not rows:
            return
        group = QtWidgets.QGroupBox(title, content_widget)
        section_layout = QtWidgets.QVBoxLayout(group)
        section_layout.setContentsMargins(10, 10, 10, 10)
        section_layout.setSpacing(6)

        grid = QtWidgets.QGridLayout()
        grid.setHorizontalSpacing(18)
        grid.setVerticalSpacing(4)
        grid.setColumnStretch(0, 3)
        grid.setColumnStretch(1, 1)

        for row_idx, (name, shortcut) in enumerate(rows):
            name_label = QtWidgets.QLabel(name)
            name_label.setTextFormat(QtCore.Qt.PlainText)
            name_label.setWordWrap(True)

            shortcut_label = QtWidgets.QLabel(f"<code>{html.escape(shortcut)}</code>")
            shortcut_label.setTextFormat(QtCore.Qt.RichText)
            shortcut_label.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
            shortcut_label.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)

            grid.addWidget(name_label, row_idx, 0)
            grid.addWidget(shortcut_label, row_idx, 1)

        section_layout.addLayout(grid)
        content_layout.addWidget(group)

    menu_order = ["File", "Manipulate", "Measure", "Display", "Help"]
    grouped_rows: Dict[str, List[tuple[str, str]]] = {}
    for item in menu_config:
        if not getattr(item, "shortcut", ""):
            continue
        menu_name = getattr(item, "menu_path", "") or "Other"
        grouped_rows.setdefault(menu_name, []).append((item.title, item.shortcut))

    for menu_name in menu_order:
        add_section(menu_name, grouped_rows.pop(menu_name, []))

    for menu_name in sorted(grouped_rows.keys()):
        add_section(menu_name, grouped_rows[menu_name])

    if extra_shortcuts:
        add_section("Special", list(extra_shortcuts.items()))

    if additional_colormaps:
        colormap_rows = [(f"Colormap: {cmap.capitalize()}", "via menu") for cmap in additional_colormaps]
        add_section("Display Colormaps", colormap_rows)

    content_layout.addStretch(1)
    scroll.setWidget(content_widget)
    main_layout.addWidget(scroll)

    button_box = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Close)
    button_box.rejected.connect(dialog.close)
    main_layout.addWidget(button_box)

    # Grow dialog to fit all shortcut rows so scrollbars are not needed on typical screens.
    content_widget.adjustSize()
    margins = main_layout.contentsMargins()
    estimated_width = content_widget.sizeHint().width() + margins.left() + margins.right() + 32
    estimated_height = (
        header_label.sizeHint().height()
        + content_widget.sizeHint().height()
        + button_box.sizeHint().height()
        + margins.top()
        + margins.bottom()
        + 36
    )

    screen = QtWidgets.QApplication.primaryScreen()
    if screen is not None:
        available = screen.availableGeometry()
        max_width = int(available.width() * 0.9)
        max_height = int(available.height() * 0.9)
        dialog.resize(min(estimated_width, max_width), min(estimated_height, max_height))
    else:
        dialog.resize(max(620, estimated_width), max(620, estimated_height))

    dialog.exec_()


def show_readme_dialog(parent_widget: QtWidgets.QWidget) -> None:
    """
    Display README content in a scrollable dialog.
    
    This is a centralized dialog for displaying the README file.
    
    Args:
        parent_widget: The parent widget for the dialog
    """
    readme_path = Path(__file__).parent / "README.md"
    
    if not readme_path.exists():
        QtWidgets.QMessageBox.warning(
            parent_widget,
            "README",
            f"README file not found at {readme_path}",
        )
        logger.warning(f"README file not found: {readme_path}")
        return
    
    try:
        with open(readme_path, "r") as f:
            readme_content = f.read()

        _show_text_content_dialog(
            parent_widget=parent_widget,
            title="README",
            content=readme_content,
            content_format="markdown",
            width=900,
            height=700,
        )
        logger.debug("Displayed README content in scrollable dialog")
    except Exception as e:
        QtWidgets.QMessageBox.warning(
            parent_widget,
            "README",
            f"Error reading README: {e}",
        )
        logger.error(f"Error reading README: {e}")


def open_parameters_dialog(
    parent_widget: QtWidgets.QWidget,
    current_settings: Dict,
    on_backend_available: bool = True,
) -> Optional[Dict]:
    """
    Display render parameters dialog and return updated settings if accepted.
    
    This is a centralized dialog for adjusting render settings. The caller
    is responsible for applying the settings if a dict is returned.
    
    Args:
        parent_widget: The parent widget for the dialog
        current_settings: Current render settings dict
        on_backend_available: Whether OpenGL backend is available
        
    Returns:
        Updated settings dict if user clicked OK, None otherwise
    """
    from dialogs import RenderSettingsDialog
    from viewer_settings import hardware_acceleration_available
    
    dialog = RenderSettingsDialog(parent_widget, current=current_settings)
    if dialog.exec_() != QtWidgets.QDialog.Accepted:
        return None
    
    updated = dialog.selected_settings()
    
    # Show warnings if applicable
    gl_available = hardware_acceleration_available()
    updated_hw = bool(updated.get("use_hardware_acceleration", True))
    current_hw = bool(current_settings.get("use_hardware_acceleration", True))
    
    if updated_hw and not gl_available:
        QtWidgets.QMessageBox.warning(
            parent_widget,
            "Parameters",
            "Hardware acceleration is enabled in settings, but no OpenGL context is available on this system/session. "
            "The viewer will use non-OpenGL rendering until hardware OpenGL becomes available.",
        )
    elif current_hw != updated_hw:
        QtWidgets.QMessageBox.information(
            parent_widget,
            "Parameters",
            "Hardware acceleration backend change will apply fully to newly opened windows.",
        )
    
    return updated


def open_file_dialog(
    parent_widget: QtWidgets.QWidget,
    start_dir: Optional[str] = None,
) -> Optional[str]:
    """
    Display file open dialog and return selected file path.
    
    This is a centralized dialog for opening image files.
    
    Args:
        parent_widget: The parent widget for the dialog
        start_dir: Starting directory (None = current working directory)
        
    Returns:
        Selected file path if user clicked OK, None otherwise
    """
    from file_navigation import IMAGE_FILE_FILTER
    
    if start_dir is None:
        start_dir = str(Path.cwd())
    
    selected_file, _ = QtWidgets.QFileDialog.getOpenFileName(
        parent_widget,
        "Open Image",
        start_dir,
        IMAGE_FILE_FILTER,
    )
    
    return selected_file if selected_file else None
SI_PREFIXES = [
    (1e9, 'G'),
    (1e6, 'M'),
    (1e3, 'k'),
    (1.0, ''),
    (1e-3, 'm'),
    (1e-6, 'μ'),
    (1e-9, 'n'),
    (1e-12, 'p'),
    (1e-15, 'f'),
]


def format_si_scale(value: float, base_unit: str = '', precision: int = 3) -> Tuple[float, str]:
    """
    Format a scale value into a nice SI unit.
    
    Args:
        value: The value in base units
        base_unit: The base unit (e.g., 'm', 'Hz', 'Å')
        precision: Number of significant figures
        
    Returns:
        (scaled_value, formatted_unit_string)
    """
    if value == 0 or not np.isfinite(value):
        return value, base_unit
    
    abs_value = abs(value)
    
    # Find appropriate SI prefix
    for factor, prefix in SI_PREFIXES:
        if abs_value >= factor * 0.95:  # Use 0.95 threshold to avoid .999k instead of 1M
            scaled = value / factor
            unit_str = f"{prefix}{base_unit}" if prefix else base_unit
            return scaled, unit_str
    
    # If value is extremely small, use the smallest prefix
    factor, prefix = SI_PREFIXES[-1]
    scaled = value / factor
    unit_str = f"{prefix}{base_unit}"
    return scaled, unit_str


def format_reciprocal_scale(value: float, axis_unit: str = "m") -> Tuple[float, str]:
    """Format reciprocal-space values with denominator-style SI units.

    Examples:
    - values in 1/m are displayed as 1/nm when appropriate
    - values in 1/nm can be displayed as 1/Å when appropriate
    """

    normalized_unit = unit_utils.normalize_axis_unit(axis_unit, default="m")
    denom_unit = unit_utils.reciprocal_denominator(normalized_unit) or normalized_unit

    to_meter = unit_utils.convert_distance_value(1.0, denom_unit, "m")
    if to_meter is None or not np.isfinite(to_meter) or to_meter <= 0:
        to_meter = 1.0

    value_in_inv_m = float(value) / float(to_meter)

    if value == 0 or not np.isfinite(value):
        return value, "1/m"

    abs_value = abs(value)
    best = None

    for factor, prefix in SI_PREFIXES:
        scaled = value_in_inv_m * factor
        abs_scaled = abs(scaled)

        if abs_scaled < 0.95 or abs_scaled >= 1000:
            continue

        score = abs(np.log10(abs_scaled))
        if best is None or score < best[0]:
            best = (score, scaled, prefix)

    if best is None:
        # Fall back to the closest scale in log space.
        for factor, prefix in SI_PREFIXES:
            scaled = value_in_inv_m * factor
            abs_scaled = abs(scaled)
            if abs_scaled <= 0 or not np.isfinite(abs_scaled):
                continue
            score = abs(np.log10(abs_scaled))
            if best is None or score < best[0]:
                best = (score, scaled, prefix)

    if best is None:
        return value_in_inv_m, "1/m"

    _score, scaled_value, chosen_prefix = best
    return float(scaled_value), f"1/{chosen_prefix}m"


def _get_hanning_window(shape: Tuple[int, int]) -> np.ndarray:
    """Get or create a cached Hanning window of specified shape."""
    if shape not in _window_cache:
        window = np.hanning(shape[0])[:, None] * np.hanning(shape[1])[None, :]
        _window_cache[shape] = window
    return _window_cache[shape]


def compute_fft(region: np.ndarray, scale_x: float, scale_y: float, apply_window: bool = True) -> Tuple[np.ndarray, float, float]:
    """
    Compute FFT of a 2D region and return magnitude spectrum with Nyquist frequencies.
    
    Optimizations:
    - Caches Hanning windows to avoid recomputation
    - Pre-calculates Nyquist frequencies
    - Uses vectorized NumPy operations
    
    Args:
        region: 2D numpy array
        scale_x: Physical scale along x-axis
        scale_y: Physical scale along y-axis
        apply_window: Whether to apply Hanning window before FFT
        
    Returns:
        magnitude_spectrum: Log-scaled FFT magnitude
        nyq_x: Nyquist frequency in x
        nyq_y: Nyquist frequency in y
    """
    if region is None or region.shape[0] < 2 or region.shape[1] < 2:
        return None, None, None
    
    # Apply window if requested (use cached window)
    if apply_window:
        window = _get_hanning_window(region.shape)
        region = region * window
    
    # Compute FFT with fftshift
    f = np.fft.fft2(region)
    fshift = np.fft.fftshift(f)
    
    # Compute magnitude spectrum with log scaling
    magnitude_spectrum = 20 * np.log10(np.abs(fshift) + 1e-8)
    
    # Pre-calculate Nyquist frequencies
    nyq_x = 0.5 / scale_x
    nyq_y = 0.5 / scale_y
    
    return magnitude_spectrum, nyq_x, nyq_y


def compute_inverse_fft(fft_data: np.ndarray) -> np.ndarray:
    """
    Compute inverse FFT efficiently.
    
    Args:
        fft_data: Complex FFT data (from fftshift)
        
    Returns:
        Real-space image (absolute values)
    """
    f_unshifted = np.fft.ifftshift(fft_data)
    real_image = np.fft.ifft2(f_unshifted)
    return np.abs(real_image)


def calculate_d_spacing(frequency: float, wavelength: float = 0.00251) -> float:
    """
    Calculate d-spacing from reciprocal space frequency.
    
    Args:
        frequency: Frequency in reciprocal space
        wavelength: Electron wavelength in angstroms (default: ~100 keV)
        
    Returns:
        d-spacing in angstroms
    """
    if frequency == 0:
        return float('inf')
    return 1.0 / frequency


def measure_line_distance(p1: Tuple[float, float], p2: Tuple[float, float], 
                         scale_x: float, scale_y: Optional[float] = None, 
                         is_reciprocal: bool = False) -> dict:
    """
    Measure distance between two points with optional d-spacing calculation.
    
    Uses vectorized NumPy for efficiency.
    
    Args:
        p1: First point (x, y)
        p2: Second point (x, y)
        scale_x: Physical scale along x-axis (units per pixel)
        scale_y: Physical scale along y-axis (units per pixel). If None, uses scale_x.
        is_reciprocal: Whether this is reciprocal space
        
    Returns:
        Dictionary with distance, d-spacing (if reciprocal), and scales
    """
    if scale_y is None:
        scale_y = scale_x
    
    # Calculate distance in pixels using vectorized operations
    diff = np.array([p2[0] - p1[0], p2[1] - p1[1]])
    dist_pixels = np.linalg.norm(diff)
    
    # Calculate distance in physical units with anisotropic scaling
    physical_scales = np.array([scale_x, scale_y])
    physical_diff = diff * physical_scales
    dist_physical = np.linalg.norm(physical_diff)
    
    result = {
        'distance_pixels': dist_pixels,
        'distance_physical': dist_physical,
        'scale_x': scale_x,
        'scale_y': scale_y
    }
    
    # Calculate d-spacing for reciprocal space
    if is_reciprocal and dist_physical != 0:
        frequency = 1.0 / dist_physical
        result['d_spacing'] = calculate_d_spacing(frequency)
    
    return result


def is_diffraction_pattern(image_data: np.ndarray, center_ratio: float = 2.0) -> bool:
    """
    Detect if image is a diffraction pattern using heuristics.
    
    Checks for symmetric, bright center with radiating features.
    Optimized to avoid unnecessary computations.
    
    Args:
        image_data: 2D image array
        center_ratio: Minimum ratio of center brightness to edge brightness
        
    Returns:
        True if likely a diffraction pattern
    """
    if image_data is None or len(image_data.shape) != 2:
        return False
    
    # Extract center and edge regions
    h, w = image_data.shape
    quarter_h, quarter_w = h // 4, w // 4
    
    center_region = image_data[quarter_h:3*quarter_h, quarter_w:3*quarter_w]
    edge_region = image_data[:quarter_h, :]
    
    # Compute means efficiently
    center_mean = np.mean(center_region)
    edge_mean = np.mean(edge_region)
    
    # Diffraction patterns typically have bright centers
    return center_mean > edge_mean * center_ratio


def apply_intensity_transform(
    image: np.ndarray,
    min_val: Optional[float],
    max_val: Optional[float],
    gamma: Optional[float] = 1.0,
) -> Optional[np.ndarray]:
    """Apply min/max window and gamma correction to an image.

    The transformation is defined as:

    - First clamp to the window ``[min_val, max_val]``.
    - Normalize to ``[0, 1]``.
    - Apply gamma mapping ``out = norm ** (1 / gamma)``.

    Args:
        image: Input image array (any numeric type).
        min_val: Input value mapped to black (0). If ``None`` uses ``image`` min.
        max_val: Input value mapped to white (1). If ``None`` uses ``image`` max.
        gamma: Gamma exponent (>0). ``1.0`` is linear, ``>1`` darkens mid-tones.

    Returns:
        New float32 array in the range [0, 1], or ``None`` if ``image`` is invalid.
    """

    if image is None:
        return None

    arr = np.asarray(image, dtype=np.float32)
    if arr.size == 0:
        return None

    finite_mask = np.isfinite(arr)
    if not np.any(finite_mask):
        # No finite values; just return zeros of the same shape
        return np.zeros_like(arr, dtype=np.float32)

    finite_vals = arr[finite_mask]

    if min_val is None or not np.isfinite(min_val):
        min_val = float(np.min(finite_vals))
    if max_val is None or not np.isfinite(max_val):
        max_val = float(np.max(finite_vals))

    # Ensure a non-degenerate window
    if max_val <= min_val:
        eps = np.finfo(np.float32).eps
        max_val = min_val + eps

    norm = (arr - float(min_val)) / float(max_val - min_val)
    norm = np.clip(norm, 0.0, 1.0, out=norm)

    if gamma is None or gamma <= 0 or not np.isfinite(gamma):
        gamma = 1.0

    inv_gamma = 1.0 / float(gamma)
    corrected = np.power(norm, inv_gamma, dtype=np.float32)

    return corrected
