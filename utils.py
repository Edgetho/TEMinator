"""Utility functions for FFT, measurements, and image analysis."""
import numpy as np
from typing import Tuple, Optional


def compute_fft(region: np.ndarray, scale_x: float, scale_y: float, apply_window: bool = True) -> Tuple[np.ndarray, float, float]:
    """
    Compute FFT of a 2D region and return magnitude spectrum with Nyquist frequencies.
    
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
    
    if apply_window:
        window = np.hanning(region.shape[0])[:, None] * np.hanning(region.shape[1])[None, :]
        region = region * window
    
    f = np.fft.fft2(region)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log10(np.abs(fshift) + 1e-8)
    
    nyq_x = 1.0 / (2.0 * scale_x)
    nyq_y = 1.0 / (2.0 * scale_y)
    
    return magnitude_spectrum, nyq_x, nyq_y


def compute_inverse_fft(fft_data: np.ndarray) -> np.ndarray:
    """
    Compute inverse FFT.
    
    Args:
        fft_data: Complex FFT data (from fftshift)
        
    Returns:
        Real-space image
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
                         scale_x: float, scale_y: float = None, is_reciprocal: bool = False) -> dict:
    """
    Measure distance between two points.
    
    Args:
        p1: First point (x, y)
        p2: Second point (x, y)
        scale_x: Physical scale along x-axis (units per pixel)
        scale_y: Physical scale along y-axis (units per pixel). If None, uses scale_x for both.
        is_reciprocal: Whether this is reciprocal space
        
    Returns:
        Dictionary with distance and d-spacing (if reciprocal)
    """
    if scale_y is None:
        scale_y = scale_x
    
    # Calculate distance in pixels, accounting for anisotropic scaling
    dx_pixels = p2[0] - p1[0]
    dy_pixels = p2[1] - p1[1]
    dist_pixels = np.sqrt(dx_pixels**2 + dy_pixels**2)
    
    # Calculate distance in physical units
    dx_physical = dx_pixels * scale_x
    dy_physical = dy_pixels * scale_y
    dist_physical = np.sqrt(dx_physical**2 + dy_physical**2)
    
    result = {
        'distance_pixels': dist_pixels,
        'distance_physical': dist_physical,
        'scale_x': scale_x,
        'scale_y': scale_y
    }
    
    if is_reciprocal and dist_physical != 0:
        # For reciprocal space, frequency is the inverse of distance
        # d-spacing = 1 / frequency
        frequency = 1.0 / dist_physical
        result['d_spacing'] = calculate_d_spacing(frequency)
    
    return result


def is_diffraction_pattern(image_data: np.ndarray) -> bool:
    """
    Basic heuristic to detect if image is a diffraction pattern.
    Checks for symmetric, bright center with radiating features.
    
    Args:
        image_data: 2D image array
        
    Returns:
        True if likely a diffraction pattern
    """
    if image_data is None or len(image_data.shape) != 2:
        return False
    
    center_region = image_data[
        image_data.shape[0]//4:3*image_data.shape[0]//4,
        image_data.shape[1]//4:3*image_data.shape[1]//4
    ]
    
    center_mean = np.mean(center_region)
    edge_mean = np.mean(image_data[:image_data.shape[0]//8, :])
    
    # Diffraction patterns typically have bright centers
    return center_mean > edge_mean * 2
