# SPDX-License-Identifier: GPL-2.0-only
# Copyright (C) 2026 Cooper Stuntz
# See LICENSE for full license terms.

"""Image loading entrypoints for launching viewer windows."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import hyperspy.api as hs
import numpy as np
from pyqtgraph.Qt import QtWidgets

logger = logging.getLogger(__name__)


def _metadata_to_dict(metadata_obj: Any) -> Dict[str, Any]:
    """Convert a metadata object to a plain dictionary when possible.

    Args:
        metadata_obj: Metadata object or dictionary.

    Returns:
        Metadata as a dictionary, or empty dictionary when conversion fails.
    """
    if metadata_obj is None:
        return {}
    if isinstance(metadata_obj, dict):
        return metadata_obj
    if hasattr(metadata_obj, "as_dictionary"):
        try:
            converted = metadata_obj.as_dictionary()
            return converted if isinstance(converted, dict) else {}
        except Exception:
            return {}
    try:
        converted = dict(metadata_obj)
        return converted if isinstance(converted, dict) else {}
    except Exception:
        return {}


def _get_signal_metadata_dict(signal: Any) -> Dict[str, Any]:
    """Return mapped metadata dictionary from a signal.

    Args:
        signal: HyperSpy signal.

    Returns:
        Mapped metadata dictionary, or empty dictionary if unavailable.
    """
    return _metadata_to_dict(getattr(signal, "metadata", None))


def _get_signal_original_metadata_dict(signal: Any) -> Dict[str, Any]:
    """Return original metadata dictionary from a signal.

    Args:
        signal: HyperSpy signal.

    Returns:
        Original metadata dictionary, or empty dictionary if unavailable.
    """
    return _metadata_to_dict(getattr(signal, "original_metadata", None))


def _extract_colormix_selection(meta: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract ordered colormixSelection entries from metadata.

    Args:
        meta: Original metadata dictionary.

    Returns:
        Ordered list of colormix entries, each containing at least name and stem.
    """
    operations = meta.get("Operations", {})
    if not isinstance(operations, dict):
        return []

    iq_ops = operations.get("ImageQuantificationOperation", {})
    if not isinstance(iq_ops, dict):
        return []

    for op_data in iq_ops.values():
        if not isinstance(op_data, dict):
            continue
        colormix = op_data.get("colormixSelection")
        if not isinstance(colormix, list):
            continue

        parsed_entries: List[Dict[str, Any]] = []
        for entry in colormix:
            if not isinstance(entry, dict):
                continue
            name = str(entry.get("name", "")).strip()
            if not name:
                continue
            parsed_entries.append(
                {
                    "name": name,
                    "stem": bool(entry.get("stem", False)),
                    "selected": bool(entry.get("selected", False)),
                    "color": entry.get("color", "#ffffff"),
                }
            )

        if parsed_entries:
            return parsed_entries

    return []


def _find_colormix_selection(signals: List[Any]) -> List[Dict[str, Any]]:
    """Find colormixSelection metadata from the loaded signal set.

    Args:
        signals: Signals returned by HyperSpy for a file.

    Returns:
        Ordered colormix entries, or empty list when unavailable.
    """
    for signal in signals:
        original_meta = _get_signal_original_metadata_dict(signal)
        if not original_meta:
            continue
        colormix = _extract_colormix_selection(original_meta)
        if colormix:
            return colormix
    return []


def _extract_signal_title(signal: Any) -> Optional[str]:
    """Extract a cleaned signal title from mapped metadata.

    Args:
        signal: HyperSpy signal.

    Returns:
        Title text when available, otherwise None.
    """
    meta = _get_signal_metadata_dict(signal)
    general_meta = meta.get("General", {}) if isinstance(meta, dict) else {}
    if not isinstance(general_meta, dict):
        return None
    title = str(general_meta.get("title", "")).strip()
    return title or None


def _is_edx_elemental_map(signal: Any) -> bool:
    """Check if a signal is an EDS elemental or spectral intensity map.

    Args:
        signal: HyperSpy signal to check.

    Returns:
        True if signal appears to be an EDX elemental map.
    """
    try:
        # Check signal dimension (2D map with no navigation)
        if (
            signal.axes_manager.signal_dimension != 2
            or signal.axes_manager.navigation_dimension != 0
        ):
            return False

        # Safely access metadata
        meta = _get_signal_metadata_dict(signal)
        original_meta = _get_signal_original_metadata_dict(signal)
        
        # Only proceed if we have dict-like objects
        if not isinstance(meta, dict) or not isinstance(original_meta, dict):
            return False

        # Check if named with element symbols
        general_meta = meta.get("General", {})
        if not isinstance(general_meta, dict):
            general_meta = {}
        signal_name = general_meta.get("title", "").upper()
        
        # Common element symbols (periodic table)
        elements = {
            "H", "C", "N", "O", "F", "P", "S", "CL", "BR", "I",
            "NA", "K", "CA", "MG", "AL", "SI", "FE", "NI", "CU",
            "ZN", "AG", "AU", "PT", "PD", "TE", "TA", "W", "MO",
            "V", "CR", "MN", "CO", "TI", "SC", "Y", "ZR", "HF",
        }
        
        for elem in elements:
            if elem in signal_name:
                return True

        # Check if part of Features/SIFeature structure
        if "Features" in original_meta:
            return True

        # Check if signal has "map" or "intensity" in description
        if any(
            keyword in signal_name.lower()
            for keyword in ["map", "intensity", "counts", "signal"]
        ):
            return True

        return False

    except Exception as e:
        logger.debug(f"Error checking if signal is EDX map: {e}")
        return False


def _is_edx_spectrum_signal(signal: Any) -> bool:
    """Check if a signal likely carries EDS spectrum data.

    Args:
        signal: HyperSpy signal to check.

    Returns:
        True for 1D signal-space datasets (single spectra or spectrum images).
    """
    try:
        return (
            hasattr(signal, "axes_manager")
            and signal.axes_manager.signal_dimension == 1
        )
    except Exception:
        return False


def _extract_spectrum_name(signal: Any, index: int) -> str:
    """Generate stable display names for EDS spectrum sources."""
    title = _extract_signal_title(signal)
    if title:
        return title
    return f"Spectrum_{index}"


def _detect_edx_dataset(
    signals: List[Any],
) -> Tuple[Optional[Any], List[Tuple[str, Any]]]:
    """Detect if loaded signals form an EDX dataset and group them.

    Args:
        signals: List of HyperSpy signals.

    Returns:
        Tuple of (primary_signal, list_of_(element_name, signal_pairs)) 
        or (None, []) if not EDX data.
    """
    if len(signals) < 2:
        return None, []

    # Identify which signals are elemental maps
    elemental_maps = []
    primary_signals = []

    for sig in signals:
        if _is_edx_elemental_map(sig):
            # Try to extract element name from metadata or signal name
            element_name = _extract_element_name(sig)
            if element_name:
                elemental_maps.append((element_name, sig))
        else:
            primary_signals.append(sig)

    # If we found elemental maps + primary signal, it's EDX data
    if elemental_maps and primary_signals:
        logger.debug(
            f"Detected EDX dataset: {len(primary_signals)} primary signal(s) + "
            f"{len(elemental_maps)} elemental map(s): {[e[0] for e in elemental_maps]}"
        )
        return primary_signals[0], elemental_maps

    return None, []


def _extract_element_name(signal: Any) -> Optional[str]:
    """Extract element name from signal metadata or title.

    Args:
        signal: HyperSpy signal.

    Returns:
        Element symbol (e.g., "C", "Ni") or None.
    """
    try:
        # Try to get from metadata
        meta = _get_signal_metadata_dict(signal)
        original_meta = _get_signal_original_metadata_dict(signal)
        
        # Only proceed if we have dict-like objects
        if not isinstance(meta, dict) or not isinstance(original_meta, dict):
            return None

        # Check Sample.elements in mapped metadata (only reliable for single-entry lists)
        sample_data = meta.get("Sample", {})
        if isinstance(sample_data, dict) and "elements" in sample_data:
            elements = sample_data["elements"]
            if isinstance(elements, list) and len(elements) == 1:
                return elements[0]

        # Extract from signal title
        signal_name = _extract_signal_title(signal) or ""
        if signal_name:
            # Clean up name and return generic label
            cleaned_name = signal_name
            # Remove common suffixes
            for suffix in [" MAP", " INTENSITY", " SIGNAL", " COUNTS"]:
                cleaned_name = cleaned_name.replace(suffix, "")
            cleaned_name = cleaned_name.strip()
            if cleaned_name:
                return cleaned_name

        return None

    except Exception as e:
        logger.debug(f"Error extracting element name: {e}")
        return None


def open_image_file(file_path: str) -> None:
    """Open an image file; if it contains multiple images, open one window per image.

    For EDX datasets (.emd files), open a single window with all elemental maps available for selection.

    Args:
        file_path: Path to the target file on disk.

    """

    try:
        from image_viewer import ImageViewerWindow

        logger.debug("Opening image file: %s", file_path)
        loaded = hs.load(file_path)

        signals = loaded if isinstance(loaded, list) else [loaded]
        logger.debug("Loaded %d signal(s) from %s", len(signals), file_path)

        # For .emd files, treat all 2D signals as elemental maps (EDX dataset)
        is_emd_file = file_path.lower().endswith('.emd')
        
        if is_emd_file:
            two_d_signals: List[Any] = []
            spectrum_signals: List[Tuple[str, Any]] = []
            for sig in signals:
                try:
                    if (
                        hasattr(sig, "axes_manager")
                        and sig.axes_manager.signal_dimension == 2
                        and sig.axes_manager.navigation_dimension == 0
                    ):
                        two_d_signals.append(sig)
                    elif _is_edx_spectrum_signal(sig):
                        spectrum_signals.append(
                            (_extract_spectrum_name(sig, len(spectrum_signals)), sig)
                        )
                except Exception as e:
                    logger.debug(f"Error processing 2D signal for EDX: {e}")

            if two_d_signals:
                colormix_entries = _find_colormix_selection(signals)

                # Default: first 2D signal is primary image.
                primary_signal_index = 0

                # If colormix is aligned to 2D signals and has a STEM entry (e.g. HAADF),
                # use that index as primary image.
                if colormix_entries and len(colormix_entries) == len(two_d_signals):
                    stem_indices = [
                        idx
                        for idx, entry in enumerate(colormix_entries)
                        if bool(entry.get("stem", False))
                    ]
                    if stem_indices:
                        primary_signal_index = stem_indices[0]

                primary_signal = two_d_signals[primary_signal_index]
                map_signals = list(two_d_signals)

                # Build metadata-driven names by index alignment when possible.
                metadata_names: List[str] = []
                if colormix_entries:
                    if len(colormix_entries) == len(two_d_signals):
                        metadata_names = [
                            str(entry.get("name", "")).strip()
                            for entry in colormix_entries
                        ]
                    else:
                        metadata_names = [
                            str(entry.get("name", "")).strip()
                            for entry in colormix_entries
                            if str(entry.get("name", "")).strip()
                        ]

                elemental_maps = []
                used_names: Dict[str, int] = {}
                for idx, sig in enumerate(map_signals):
                    base_name: Optional[str] = None

                    if idx < len(metadata_names) and metadata_names[idx]:
                        base_name = metadata_names[idx]
                    else:
                        base_name = _extract_element_name(sig)

                    if not base_name:
                        base_name = f"Map_{idx}"

                    # Ensure unique labels so map dictionary keys do not collide.
                    count = used_names.get(base_name, 0) + 1
                    used_names[base_name] = count
                    label = base_name if count == 1 else f"{base_name} ({count})"

                    elemental_maps.append((label, sig))

                logger.debug(
                    "Resolved EDX labels for %d map(s): %s",
                    len(elemental_maps),
                    [name for name, _ in elemental_maps],
                )
            else:
                primary_signal, elemental_maps = None, []
                spectrum_signals = []
        else:
            # For non-.emd files, use original detection logic
            primary_signal, elemental_maps = _detect_edx_dataset(signals)
            spectrum_signals = [
                (_extract_spectrum_name(sig, idx), sig)
                for idx, sig in enumerate(signals)
                if _is_edx_spectrum_signal(sig)
            ]

        if primary_signal is not None:
            # EDX dataset: open single window with elemental maps
            try:
                logger.debug(f"Opening EDX dataset with {len(elemental_maps)} elemental maps")
                window = ImageViewerWindow(
                    file_path,
                    signal=primary_signal,
                    elemental_map_signals=elemental_maps,
                    eds_spectrum_signals=spectrum_signals,
                )
                window.show()
                logger.debug("Opened EDX viewer window")
            except Exception as e:
                logger.exception(
                    f"Could not open EDX dataset: {type(e).__name__}: {e}"
                )
                QtWidgets.QMessageBox.critical(
                    None, "Error", f"Could not open EDX dataset: {str(e)}"
                )
        else:
            # Regular dataset: open one window per signal
            for sig_index, signal in enumerate(signals):
                try:
                    if signal.axes_manager.navigation_dimension == 0:
                        suffix = f"[{sig_index}]" if len(signals) > 1 else None
                        window = ImageViewerWindow(
                            file_path, signal=signal, window_suffix=suffix
                        )
                        window.show()
                        logger.debug(
                            "Opened viewer for signal index %s suffix=%s", sig_index, suffix
                        )
                    else:
                        nav_shape = signal.axes_manager.navigation_shape
                        for nav_index in np.ndindex(nav_shape):
                            sub_signal = signal.inav[nav_index]
                            idx_str = ",".join(str(i) for i in nav_index)
                            if len(signals) > 1:
                                suffix = f"[{sig_index}; {idx_str}]"
                            else:
                                suffix = f"[{idx_str}]"
                            window = ImageViewerWindow(
                                file_path,
                                signal=sub_signal,
                                window_suffix=suffix,
                            )
                            window.show()
                            logger.debug(
                                "Opened viewer for signal index %s navigation index %s suffix=%s",
                                sig_index,
                                nav_index,
                                suffix,
                            )
                except Exception as e:
                    logger.exception(
                        f"Could not open signal {sig_index} from {file_path}; {type(e).__name__}."
                    )

    except Exception as exc:
        logger.exception("Could not open file: %s", file_path)
        QtWidgets.QMessageBox.critical(
            None, "Error", f"Could not open file: {str(exc)}"
        )
