# Phase 5 Validation Checklist

This checklist captures Phase 5 hardening and verification coverage for EDS workflows.

## Automated Unit Coverage

1. Quantification adapter behavior:
- CL mode defaults/warnings for missing k-factors
- Custom mode multiplicative factors and normalization

2. Export schema formatting:
- Stable CSV header and deterministic record serialization

3. Metadata fallback behavior:
- Detector-path parsing for dispersion/offset/timing
- CustomProperties fallback for missing detector fields

Run tests:

```bash
python -m unittest discover -s tests -p "test_eds_*.py"
```

## Manual Integration Checks

1. Menu state transitions:
- Open non-EDS file: EDS menu disabled
- Open EDS map: EDS menu enabled with capability-gated actions
- Create region: region-dependent menu actions enable

2. ROI lifecycle:
- Create, move, resize, remove, clear ROI regions
- Confirm integration table and quant values refresh live

3. Hover spectra:
- Verify source label indicates true spectra vs pseudo-spectrum fallback
- Verify hover toggle menu action enables/disables updates

4. Persistence prompts:
- On new region creation, verify prompt-save appears
- Confirm CSV/JSON/PNG artifacts are written as expected
- Use Save View and verify optional EDS artifact prompt

## Current Risk Notes

1. Full high-resolution hover requires true per-pixel spectra cube loading in all supported EMD variants.
2. Absorption correction toggle is capability-gated and currently placeholder-level in quant backend.
