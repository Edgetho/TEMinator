# SPDX-License-Identifier: GPL-2.0-only
# Copyright (C) 2026 Cooper Stuntz
# See LICENSE for full license terms.

"""Utility functions for FFT, measurements, and image analysis."""
import numpy as np
from typing import Tuple, Optional


# Cache for window functions to avoid recomputation
_window_cache = {}

# SI prefix conversions
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
