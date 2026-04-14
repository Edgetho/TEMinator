# SPDX-License-Identifier: GPL-2.0-only
# Copyright (C) 2026 Cooper Stuntz
# See LICENSE for full license terms.

"""Application settings for rendering quality and performance."""

from __future__ import annotations

from typing import TypedDict

from PyQt5 import QtCore, QtGui


class RenderSettings(TypedDict):
    use_hardware_acceleration: bool
    image_resampling_quality: str


class PeakProfileSettings(TypedDict):
    integration_width_px: float
    radial_length_px: float
    azimuthal_span_deg: float


DEFAULT_RENDER_SETTINGS: RenderSettings = {
    "use_hardware_acceleration": True,
    "image_resampling_quality": "high",
}

RESAMPLING_FAST = "fast"
RESAMPLING_BALANCED = "balanced"
RESAMPLING_HIGH = "high"
RESAMPLING_CHOICES = {RESAMPLING_FAST, RESAMPLING_BALANCED, RESAMPLING_HIGH}
_HW_AVAILABLE_CACHE: bool | None = None
_EFFECTIVE_RENDER_SETTINGS: RenderSettings | None = None

DEFAULT_PEAK_PROFILE_SETTINGS: PeakProfileSettings = {
    "integration_width_px": 0.0,
    "radial_length_px": 100.0,
    "azimuthal_span_deg": 5.0,
}


def _settings_store() -> QtCore.QSettings:
    """Get the application's QSettings store for persisting render preferences.

    Returns:
        Detailed parameter description.

    """
    return QtCore.QSettings("TEMinator", "TEMinator")


def load_render_settings() -> RenderSettings:
    """Load render settings from persistent storage (QSettings).

    Returns:
        RenderSettings dict with hardware acceleration and resampling quality settings.
    """
    store = _settings_store()

    raw_hw = store.value(
        "render/use_hardware_acceleration",
        DEFAULT_RENDER_SETTINGS["use_hardware_acceleration"],
    )
    if isinstance(raw_hw, str):
        use_hardware = raw_hw.strip().lower() in {"1", "true", "yes", "on"}
    else:
        use_hardware = bool(raw_hw)

    quality = (
        str(
            store.value(
                "render/image_resampling_quality",
                DEFAULT_RENDER_SETTINGS["image_resampling_quality"],
            )
            or DEFAULT_RENDER_SETTINGS["image_resampling_quality"]
        )
        .strip()
        .lower()
    )
    if quality not in RESAMPLING_CHOICES:
        quality = DEFAULT_RENDER_SETTINGS["image_resampling_quality"]

    return {
        "use_hardware_acceleration": use_hardware,
        "image_resampling_quality": quality,
    }


def save_render_settings(settings: RenderSettings) -> None:
    """Save render settings to persistent storage (QSettings).

    Args:
        settings: RenderSettings dict to persist with validated values.
    """
    quality = str(settings.get("image_resampling_quality", "")).strip().lower()
    if quality not in RESAMPLING_CHOICES:
        quality = DEFAULT_RENDER_SETTINGS["image_resampling_quality"]

    use_hardware = bool(settings.get("use_hardware_acceleration", True))

    store = _settings_store()
    store.setValue("render/use_hardware_acceleration", use_hardware)
    store.setValue("render/image_resampling_quality", quality)
    store.sync()


def hardware_acceleration_available(*, force_refresh: bool = False) -> bool:
    """Check if OpenGL hardware acceleration is available on this system.

    Args:
        force_refresh: If True, re-test availability instead of using cached result.

    Returns:
        True if OpenGL context can be created, False if unavailable or error occurs.
    """
    global _HW_AVAILABLE_CACHE

    if not force_refresh and _HW_AVAILABLE_CACHE is not None:
        return _HW_AVAILABLE_CACHE

    app = QtCore.QCoreApplication.instance()
    if app is None:
        _HW_AVAILABLE_CACHE = False
        return False

    try:
        ctx = QtGui.QOpenGLContext()
        if not ctx.create():
            _HW_AVAILABLE_CACHE = False
            return False

        surface = QtGui.QOffscreenSurface()
        surface.setFormat(ctx.format())
        surface.create()

        ok = bool(ctx.makeCurrent(surface))
        if ok:
            ctx.doneCurrent()
        surface.destroy()
        _HW_AVAILABLE_CACHE = ok
        return ok
    except Exception:
        _HW_AVAILABLE_CACHE = False
        return False


def global_render_config_options(
    settings: RenderSettings,
    *,
    hardware_available: bool | None = None,
) -> dict[str, object]:
    """Generate pyqtgraph configuration options from render settings.

    Args:
        settings: RenderSettings with user preferences.
        hardware_available: If provided, use instead of auto-detecting.

    Returns:
        Dict of options to pass to pyqtgraph.setConfigOptions().
    """
    requested_hardware = bool(settings.get("use_hardware_acceleration", True))
    if hardware_available is None:
        hardware_available = hardware_acceleration_available()

    use_hardware = bool(requested_hardware and hardware_available)
    return {
        "useOpenGL": use_hardware,
        "imageAxisOrder": "row-major",
        "antialias": True,
    }


def set_effective_render_settings(settings: RenderSettings) -> None:
    """Cache the effective render settings for the current session.

    This is called from app.py after applying CLI overrides to ensure that
    render diagnostics and ImageViewer windows show the correct settings.

    Args:
        settings: Render settings mapping to persist or apply.

    """
    global _EFFECTIVE_RENDER_SETTINGS
    _EFFECTIVE_RENDER_SETTINGS = {
        "use_hardware_acceleration": settings.get("use_hardware_acceleration", True),
        "image_resampling_quality": settings.get("image_resampling_quality", "high"),
    }


def get_effective_render_settings() -> RenderSettings:
    """Get the effective render settings for the current session.

    Returns cached effective settings if set by app.py, otherwise loads from storage.

    Returns:
        Detailed parameter description.

    """
    global _EFFECTIVE_RENDER_SETTINGS
    if _EFFECTIVE_RENDER_SETTINGS is not None:
        return _EFFECTIVE_RENDER_SETTINGS
    return load_render_settings()


def load_peak_profile_settings() -> PeakProfileSettings:
    """Load persisted defaults for peak-profile collection parameters."""
    store = _settings_store()

    integration_width_px = float(
        store.value(
            "peak_profiles/integration_width_px",
            DEFAULT_PEAK_PROFILE_SETTINGS["integration_width_px"],
        )
    )
    radial_length_px = float(
        store.value(
            "peak_profiles/radial_length_px",
            DEFAULT_PEAK_PROFILE_SETTINGS["radial_length_px"],
        )
    )
    azimuthal_span_deg = float(
        store.value(
            "peak_profiles/azimuthal_span_deg",
            DEFAULT_PEAK_PROFILE_SETTINGS["azimuthal_span_deg"],
        )
    )

    return {
        "integration_width_px": max(0.0, integration_width_px),
        "radial_length_px": max(1.0, radial_length_px),
        "azimuthal_span_deg": max(0.1, azimuthal_span_deg),
    }


def save_peak_profile_settings(settings: PeakProfileSettings) -> None:
    """Persist peak-profile collection defaults to QSettings."""
    integration_width_px = max(0.0, float(settings.get("integration_width_px", 0.0)))
    radial_length_px = max(1.0, float(settings.get("radial_length_px", 100.0)))
    azimuthal_span_deg = max(0.1, float(settings.get("azimuthal_span_deg", 5.0)))

    store = _settings_store()
    store.setValue("peak_profiles/integration_width_px", integration_width_px)
    store.setValue("peak_profiles/radial_length_px", radial_length_px)
    store.setValue("peak_profiles/azimuthal_span_deg", azimuthal_span_deg)
    store.sync()
