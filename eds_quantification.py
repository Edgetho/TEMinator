# SPDX-License-Identifier: GPL-2.0-only
# Copyright (C) 2026 Cooper Stuntz
# See LICENSE for full license terms.

"""Quantification service for EDS integration results."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from eds_models import EDSQuantResultRow


@dataclass(frozen=True)
class QuantificationRequest:
    """Input contract for region-level quantification."""

    region_id: int
    element_counts: Dict[str, float]
    method: str
    factor_text: str = ""


class EDSQuantificationService:
    """Compute CL/custom quantification rows from integrated element counts."""

    def __init__(self) -> None:
        self._atomic_weights: Dict[str, float] = self._load_atomic_weights()

    @staticmethod
    def _load_atomic_weights() -> Dict[str, float]:
        """Load atomic-weight table from exspy when available."""
        table: Dict[str, float] = {}
        try:
            import exspy  # type: ignore

            for symbol in dir(exspy.material.elements):
                if not symbol or not symbol[0].isupper():
                    continue
                try:
                    node = getattr(exspy.material.elements, symbol)
                    weight = float(node.General_properties.atomic_weight)
                    table[symbol] = weight
                except Exception:
                    continue
        except Exception:
            table = {}
        return table

    @staticmethod
    def _parse_factor_text(text: str) -> Dict[str, float]:
        """Parse factor string format: `Fe=1.1, Pt=0.9`."""
        out: Dict[str, float] = {}
        for raw_part in (text or "").split(","):
            part = raw_part.strip()
            if not part:
                continue
            if "=" not in part:
                continue
            key, value = part.split("=", 1)
            element = key.strip()
            if not element:
                continue
            try:
                out[element] = float(value.strip())
            except Exception:
                continue
        return out

    @staticmethod
    def _normalize(values: Dict[str, float]) -> Dict[str, float]:
        """Normalize positive values so they sum to 1."""
        cleaned = {k: max(0.0, float(v)) for k, v in values.items()}
        denom = sum(cleaned.values())
        if denom <= 0:
            return {k: 0.0 for k in cleaned}
        return {k: v / denom for k, v in cleaned.items()}

    def _atomic_to_weight(
        self,
        atomic_fraction: Dict[str, float],
    ) -> Tuple[Dict[str, Optional[float]], List[str]]:
        """Convert atomic fractions to weight fractions where possible."""
        warnings: List[str] = []
        weighted_terms: Dict[str, float] = {}
        missing: List[str] = []

        for element, frac in atomic_fraction.items():
            weight = self._atomic_weights.get(element)
            if weight is None:
                missing.append(element)
                continue
            weighted_terms[element] = frac * weight

        if missing:
            warnings.append(
                "Missing atomic weights for: " + ", ".join(sorted(missing))
            )

        denom = sum(weighted_terms.values())
        if denom <= 0:
            return ({k: None for k in atomic_fraction.keys()}, warnings)

        out: Dict[str, Optional[float]] = {}
        for element in atomic_fraction.keys():
            if element not in weighted_terms:
                out[element] = None
            else:
                out[element] = weighted_terms[element] / denom
        return out, warnings

    def quantify(self, request: QuantificationRequest) -> Sequence[EDSQuantResultRow]:
        """Quantify one region using CL or custom factors."""
        warnings: List[str] = []
        counts = {
            element: float(value)
            for element, value in sorted(request.element_counts.items(), key=lambda kv: kv[0])
            if float(value) >= 0
        }

        if not counts:
            return []

        method = (request.method or "CL").strip().upper()
        factors = self._parse_factor_text(request.factor_text)
        corrected: Dict[str, float] = {}

        if method == "CUSTOM":
            for element, count in counts.items():
                factor = factors.get(element, 1.0)
                corrected[element] = max(0.0, count * factor)
                if element not in factors:
                    warnings.append(f"Custom factor missing for {element}; defaulted to 1.0")
        else:
            method = "CL"
            for element, count in counts.items():
                factor = factors.get(element, 1.0)
                if abs(factor) < 1e-12:
                    factor = 1.0
                    warnings.append(f"k-factor for {element} was 0; defaulted to 1.0")
                if element not in factors:
                    warnings.append(f"k-factor missing for {element}; defaulted to 1.0")
                corrected[element] = max(0.0, count / factor)

        atomic_fraction = self._normalize(corrected)
        weight_fraction, weight_warnings = self._atomic_to_weight(atomic_fraction)
        warnings.extend(weight_warnings)

        rows: List[EDSQuantResultRow] = []
        for element in sorted(counts.keys()):
            at_val = atomic_fraction.get(element, 0.0) * 100.0
            wt_raw = weight_fraction.get(element)
            wt_val = None if wt_raw is None else wt_raw * 100.0
            rows.append(
                EDSQuantResultRow(
                    region_id=request.region_id,
                    element=element,
                    counts=counts[element],
                    weight_percent=wt_val,
                    atomic_percent=at_val,
                    method=method,
                    warnings=tuple(sorted(set(warnings))),
                )
            )
        return rows
