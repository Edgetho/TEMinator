# SPDX-License-Identifier: GPL-2.0-only
# Copyright (C) 2026 Cooper Stuntz
# See LICENSE for full license terms.

"""Image loading entrypoints for launching viewer windows."""

from __future__ import annotations

import logging
from typing import Any, List, Optional, Tuple

import hyperspy.api as hs
import numpy as np
from pyqtgraph.Qt import QtWidgets

logger = logging.getLogger(__name__)


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
        meta = getattr(signal, "metadata", None) or {}
        original_meta = getattr(signal, "original_metadata", None) or {}
        
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
        meta = getattr(signal, "metadata", None) or {}
        original_meta = getattr(signal, "original_metadata", None) or {}
        
        # Only proceed if we have dict-like objects
        if not isinstance(meta, dict) or not isinstance(original_meta, dict):
            return None

        # Check Sample.elements in mapped metadata
        sample_data = meta.get("Sample", {})
        if isinstance(sample_data, dict) and "elements" in sample_data:
            elements = sample_data["elements"]
            if isinstance(elements, list) and elements:
                return elements[0]

        # Extract from signal title
        general_meta = meta.get("General", {})
        if not isinstance(general_meta, dict):
            general_meta = {}
        signal_name = general_meta.get("title", "")
        if signal_name:
            # Clean up name and try to match element
            name_upper = signal_name.upper()
            # Remove common suffixes
            for suffix in [" MAP", " INTENSITY", " SIGNAL", " COUNTS"]:
                name_upper = name_upper.replace(suffix, "")
            name_upper = name_upper.strip()
            
            if name_upper in {
                "H", "C", "N", "O", "F", "P", "S", "CL", "BR", "I",
                "NA", "K", "CA", "MG", "AL", "SI", "FE", "NI", "CU",
                "ZN", "AG", "AU", "PT", "PD", "TE", "TA", "W", "MO",
                "V", "CR", "MN", "CO", "TI", "SC", "Y", "ZR", "HF",
            }:
                return name_upper

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
            # Group 2D signals as elemental maps
            primary_signal = None
            elemental_maps = []
            
            for sig in signals:
                try:
                    # Check if 2D signal with no navigation
                    if (hasattr(sig, 'axes_manager') and 
                        sig.axes_manager.signal_dimension == 2 and
                        sig.axes_manager.navigation_dimension == 0):
                        # Extract element name if available
                        element_name = _extract_element_name(sig)
                        if element_name is None:
                            # Fall back to last part of signal title
                            meta = getattr(sig, "metadata", None) or {}
                            if isinstance(meta, dict):
                                general_meta = meta.get("General", {})
                                if isinstance(general_meta, dict):
                                    title = general_meta.get("title", "")
                                    element_name = title if title else f"Element_{len(elemental_maps)}"
                            else:
                                element_name = f"Element_{len(elemental_maps)}"
                        
                        if primary_signal is None:
                            primary_signal = sig
                        else:
                            elemental_maps.append((element_name, sig))
                except Exception as e:
                    logger.debug(f"Error processing 2D signal for EDX: {e}")
            
            primary_signal, elemental_maps = (primary_signal, elemental_maps) if primary_signal else (None, [])
        else:
            # For non-.emd files, use original detection logic
            primary_signal, elemental_maps = _detect_edx_dataset(signals)

        if primary_signal is not None:
            # EDX dataset: open single window with elemental maps
            try:
                logger.debug(f"Opening EDX dataset with {len(elemental_maps)} elemental maps")
                window = ImageViewerWindow(
                    file_path,
                    signal=primary_signal,
                    elemental_map_signals=elemental_maps,
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
