## Plan: EDS Quantification and Interactive Integration v3

This plan delivers a production-oriented EDS feature expansion centered on interactive ROI-driven spectra integration, Cliff-Lorimer plus custom quantification, dynamic menu surfacing of EDS capabilities, and prompted persistence of all derived outputs.

**Objectives**
1. Provide a mouse-over spectra mode for spatially resolved inspection.
2. Enable custom-area spectrum integration from user-defined ROIs.
3. Add robust quantification controls and outputs (CL + custom methods).
4. Make EDS workflows dynamically available through menus based on capability state.
5. Prompt-save every newly created derived view and data result.

**Phase 0: Foundations and Contracts**
1. Define canonical data models:
1.1 Spectrum sample model: channel index, energy axis, counts, source metadata.
1.2 ROI model: id, geometry type, coordinates, calibration units, timestamp.
1.3 Integration model: windows, baseline/background mode, included lines.
1.4 Quant result model: region id, element, counts, wt%, at%, method, warnings.
2. Define metadata fallback hierarchy for beam energy, detector dispersion/offset, live/real time, and line metadata.
3. Define capability-state object used by menu gating and button enablement.

**Phase 1: Interactive Spectra and ROI Integration**
1. Add hover spectra inspector in EDS-capable image windows:
1.1 Cursor position maps to spectrum index.
1.2 Debounced updates to avoid UI jitter.
1.3 Clear no-data/error state messaging when spectrum is unavailable.
2. Implement ROI integration workflow:
2.1 Rectangle ROI creation first for reliability.
2.2 Optional polygon ROI extension once baseline is stable.
2.3 ROI selection/edit/remove lifecycle with persistent table linkage.
3. Integrate ROI outputs in Integration tab:
3.1 Live updates of counts per selected lines/windows.
3.2 Region summary fields (area, pixel count, calibrated size).

**Phase 2: Quantification Engine (CL + Custom)**
1. Add quantification service module (separate from UI state).
2. Implement Cliff-Lorimer path:
2.1 Validate required inputs (line intensities, k-factors, metadata).
2.2 Compute atomic and weight outputs.
2.3 Emit warnings for incomplete detector/acquisition metadata.
3. Implement custom quantification path:
3.1 User-defined factor sets.
3.2 Method options persisted by profile/preset.
3.3 Deterministic ordering and reproducible outputs.
4. Add quant options UI:
4.1 Method selector.
4.2 Units selector (at%/wt%).
4.3 Factor editor and preset chooser.
4.4 Optional absorption correction toggle gated by metadata validity.

**Phase 3: Dynamic EDS Menu Surface**
1. Extend EDS menu availability from static EDX-present checks to capability-driven checks.
2. Add menu actions for major EDS workflows:
2.1 Spectra hover toggle.
2.2 ROI integration controls.
2.3 Integration window/background options.
2.4 Quantification options and method switching.
2.5 Export actions for images and tabular outputs.
3. Keep unavailable actions visible but disabled with explicit reason tooltips/messages.

**Phase 4: Prompted Persistence of Derived Outputs**
1. Define output artifact registry:
1.1 Derived images: overlays, ROI-marked snapshots, spectra plots.
1.2 Data outputs: region tables, quant summaries, metadata sidecars.
2. Trigger save prompt on creation of each derived artifact.
3. Extend existing save orchestration so prompted persistence is consistent across:
3.1 Main view and transform windows.
3.2 EDS integration/quant outputs.
3.3 Any new EDS result views opened after quant operations.
4. Use stable naming conventions and source-local default paths.

**Phase 5: Validation and Release Hardening**
1. Unit tests:
1.1 Quantification adapters and factor validation.
1.2 Metadata fallback behavior.
1.3 Export schema correctness.
2. Integration tests:
2.1 Menu gating state transitions.
2.2 ROI lifecycle and table synchronization.
2.3 Save-prompt behavior for newly created artifacts.
3. Manual workflow verification:
3.1 Open EMD dataset.
3.2 Inspect hover spectra.
3.3 Create ROI and integrate lines.
3.4 Quantify with CL and custom modes.
3.5 Save prompted outputs and verify re-open/readability.

**Parallelization Strategy**
1. Phase 1 is the primary dependency for all downstream user-visible EDS workflows.
2. Phase 2 and Phase 3 can proceed in parallel after data contracts stabilize.
3. Phase 4 depends on final artifact definitions from Phases 1-3.
4. Phase 5 runs continuously and as a final gate.

**Primary Implementation Targets**
- /Users/kcs/Source/TEMinator/viewer_edx.py — EDS manager, spectra/maps panel, integration table, export path.
- /Users/kcs/Source/TEMinator/image_viewer.py — menu callbacks, interaction mode arbitration, save orchestration.
- /Users/kcs/Source/TEMinator/menu_manager.py — capability-state-driven EDS menu availability.
- /Users/kcs/Source/TEMinator/measurement_tools.py — robust interaction patterns reused for ROI mode.
- /Users/kcs/Source/TEMinator/image_loader.py — EDS signal ingestion, calibration/metadata extraction.
- /Users/kcs/Source/TEMinator/metadata_export.txt — metadata-ground-truth validation reference.
- /Users/kcs/Source/TEMinator/README.md — user docs for EDS hover, ROI integration, quant options, and prompted saves.

**Accepted Decisions**
1. Delivery model: phased roadmap.
2. Quantification in first release: Cliff-Lorimer + custom method options.
3. Persistence policy: prompt user each time a new derived output is created.
4. Dependency policy: exspy/hyperspy integration allowed.

**Initial Exclusions**
1. Full parity with all advanced eXSpy model-fitting/calibration utilities in first release.
2. Broad multi-detector absorption-correction variants without validated metadata support.

**Completion Criteria**
1. Mouse-over spectra and ROI integration are stable across representative datasets.
2. Quantification tables are reproducible and exportable with validated schema.
3. Menu actions reflect runtime capability accurately.
4. All newly created derived outputs trigger save prompts and persist correctly.

## Patch Notes

### 2026-04-01

Implemented work completed through Phase 4.

1. Phase 0 completed:
1.1 Added canonical EDS contracts for spectrum samples, ROI regions, integration settings, quant rows, metadata context, and capability state.
1.2 Added metadata fallback parsing for beam energy, dispersion, offset, timing fields, and X-ray lines.
1.3 Introduced capability-state API for dynamic menu/feature gating.

2. Phase 1 completed:
2.1 Added hover spectra updates from image-view mouse position.
2.2 Added rectangle ROI integration workflow with interactive creation, edit, remove, and clear behavior.
2.3 Added live Integration table refresh from region-derived element counts.
2.4 Added source labeling for pseudo-spectrum fallback mode when full spectra are unavailable.

3. Phase 2 completed:
3.1 Added quantification service module with CL and custom methods.
3.2 Added factors parsing and warning propagation.
3.3 Added Integration tab quant controls (method, units preference, factor input, absorption toggle).
3.4 Added method/warning-aware quantified rows in results table.

4. Phase 3 completed:
4.1 Added capability-aware EDS menu gating beyond simple EDX-present checks.
4.2 Added dynamic EDS menu actions for tab navigation, hover toggle, region clear, method switching, and absorption toggle.
4.3 Added runtime menu refresh hooks tied to capability changes.

5. Phase 4 completed:
5.1 Added output artifact registry for EDS-derived files.
5.2 Added prompt-save flow when new region-derived outputs are created.
5.3 Implemented EDS export persistence to CSV and JSON sidecar files, plus optional snapshot PNG for region saves.
5.4 Hooked Save View flow to optionally persist EDS quantification artifacts.

6. Current known limitation:
6.1 Full high-resolution detector-spectrum hover depends on loading true spectral cube data; elemental-map fallback remains lower-resolution by design.