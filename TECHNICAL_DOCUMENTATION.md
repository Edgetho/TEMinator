TECHNICAL DOCUMENTATION
========================

This document describes the internal structure of the Fast FFT Image Analyzer project, focusing on what each module, class, and function does and how they interact.

Modules
-------

- `app.py` – main GUI implementation: windows, tools, and high‑level application logic.
- `utils.py` – numerical helpers for FFTs, unit scaling, measurements, and diffraction detection.
- `main.py` – lightweight entry point that runs the Qt application.


app.py
------

Top‑level
~~~~~~~~~

- **Module docstring** – states that this module provides an individual image viewer window with FFT analysis.
- **Imports** – pulls in standard modules (`sys`, `json`, `pathlib.Path`), NumPy, HyperSpy (`hyperspy.api as hs`), pyqtgraph, Qt bindings (`QtWidgets`, `QtCore`, `QtGui`), typing utilities, and the local `utils` module.
- **`Signal` alias** – resolves to `QtCore.pyqtSignal` or `QtCore.Signal` depending on which Qt binding is available; used to declare custom signals on pyqtgraph items.
- **Constants**:
  - `FFT_COLORS` – list of color codes used to assign distinct colors to FFT ROI boxes.
  - `PREVIEW_LINE_PEN` – yellow dashed `QPen` used for the temporary measurement line preview.
  - `DRAWN_LINE_PEN` – white `QPen` used for finalized measurement lines.
  - `LABEL_BRUSH_COLOR` – semi‑transparent yellow `QBrush` used as the background of measurement labels.
  - `DEFAULT_FFT_WINDOW_SIZE` – default size of FFT viewer windows.
  - `DEFAULT_IMAGE_WINDOW_SIZE` – default size of image viewer windows.
  - `DEFAULT_MAIN_WINDOW_SIZE` – default size of the main drag‑and‑drop window.


Class: `ScaleBarItem`
~~~~~~~~~~~~~~~~~~~~~~

Subclass of `pg.GraphicsObject` that draws a simple, non‑dynamic microscopy‑style scale bar anchored in data coordinates.

- **Purpose** – represent a physical distance on screen based on a known scale per pixel and express it using SI prefixes via `utils.format_si_scale`.
- **`__init__(scale_per_pixel: float, units: str = "px")`**
  - Stores the physical size represented by one pixel and the base units string.
  - Initializes internal fields (`bar_length_physical`, `display_unit`) and positions the item at `(0, 0)` in scene coordinates.
  - Calls `update_length()` to select an aesthetically reasonable bar length and label.
- **`update_length()`**
  - Chooses a “nice” physical bar length (1, 2, 5, or 10 × 10^n of the base unit) that should correspond to roughly 100 pixels on screen.
  - Converts that physical length into an SI‑prefixed value and unit string using `utils.format_si_scale`.
  - Stores both the exact physical length (used for geometry) and the formatted value/unit pair (used for text).
- **`set_scale(scale_per_pixel: float)`**
  - Updates the stored `scale_per_pixel`, recomputes the bar length and label via `update_length()`, and triggers a redraw.
- **`boundingRect()`**
  - Returns the rectangular area the item occupies, sized according to the bar length in pixels plus margins for text and end caps.
- **`paint(p, *args)`**
  - Draws the horizontal bar, vertical end caps, and the text label showing the formatted physical length and units.

Note: in the current UI, dynamic scaling is handled by `DynamicScaleBar`; `ScaleBarItem` remains available as a more traditional alternative.


Class: `DynamicScaleBar`
~~~~~~~~~~~~~~~~~~~~~~~~~

Subclass of `pg.GraphicsObject` implementing a scale bar that stays fixed in screen space but whose physical length and label adjust when the user zooms.

- **Purpose** – provide a zoom‑aware overlay scale bar:
  - Length in world units is a “nice” 2, 5, or 10 × 10^n in the underlying axis units (typically metres), formatted to nm‑scale labels.
  - On‑screen length is kept between configurable fractions of the current view width.
  - The bar ignores data transforms so its text size stays visually stable.
- **`__init__(viewbox: pg.ViewBox, units: str = "m", min_frac: float = 0.15, max_frac: float = 0.30, margin: int = 20)`**
  - Stores the associated `ViewBox`, base units, and fractions of the view width to use as minimum and maximum on‑screen bar length.
  - Keeps a rect cache (`_rect`) and the current length in pixels (`_length_px`) and label text (`_label_text`).
  - Sets `ItemIgnoresTransformations` to keep the item in screen coordinates and places it as a child of the viewbox so it overlays the data.
  - Connects to `sigRangeChanged` and `sigResized` on the viewbox to recompute geometry whenever the view or widget size changes.
- **`_choose_length(target_val: float, world_per_px: float, width_px: float) -> tuple`**
  - Internal helper that picks a physically “nice” length (2/5/10 × 10^n) near a target world‑space length.
  - Evaluates candidates over several orders of magnitude and filters them so their pixel length sits within `[min_frac, max_frac]` of the view width.
  - Returns `(length_val, length_px)` where `length_val` is in base units and `length_px` is the corresponding on‑screen length.
- **`_update_geometry(*args)`**
  - Reads the current x‑axis range and view width from the viewbox.
  - Converts this into world‑per‑pixel, chooses a suitable bar length via `_choose_length`, and formats the label using `utils.format_si_scale`.
  - Updates the cached bounding rect and positions the bar near the bottom‑left of the view (by `margin` pixels).
- **`boundingRect()`**
  - Returns the cached `_rect` used for Qt’s redraw calculations.
- **`paint(p, *args)`**
  - Draws a horizontal red bar with vertical end caps and the label text above it using the cached length and text.


Class: `MeasurementHistoryWindow`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Subclass of `QtWidgets.QMainWindow` that shows a history of measurement labels generated in an `ImageViewerWindow`.

- **Purpose** – provide a central place to review, copy, delete, or export all line‑measurement results taken in a given viewer.
- **`__init__(parent=None)`**
  - Initializes the window UI with:
    - A `QListWidget` listing each measurement as a row of text.
    - Buttons for “Clear All”, “Delete Selected”, “Copy Selected”, and “Export as CSV”.
  - Maintains an internal `measurements` list mirroring the list widget contents.
- **`add_measurement(measurement_text: str)`**
  - Appends a new measurement string to both the list widget and the internal list and scrolls the list to the bottom.
- **`clear_all()`**
  - Clears both the list widget and internal list and informs the user via an information message box.
- **`delete_selected()`**
  - Removes the currently selected entry from both the list widget and the internal list.
  - If the parent viewer implements `delete_measurement_by_label`, delegates to it so the corresponding annotation is removed from the image.
- **`copy_selected()`**
  - Copies the text of the currently selected measurement to the clipboard and shows a confirmation message box.
- **`export_as_csv()`**
  - Prompts the user for a file path and writes all measurements to a CSV file with a single `Measurement` column, reporting success or errors via message boxes.


Class: `MetadataWindow`
~~~~~~~~~~~~~~~~~~~~~~~~

Subclass of `QtWidgets.QMainWindow` that displays full HyperSpy metadata for the current image.

- **Purpose** – let users inspect the original metadata (including calibration information) as a formatted JSON‑style text block.
- **`__init__(parent=None, title: str = "Image Metadata", metadata: Optional[dict] = None)`**
  - Sets the window title and creates a central read‑only `QPlainTextEdit` widget.
  - If an initial metadata dictionary is provided, immediately calls `update_metadata()`.
- **`update_metadata(metadata: dict)`**
  - Serializes the metadata dictionary as pretty‑printed JSON (with `default=str` to handle non‑serializable objects) or falls back to `str(metadata)` if JSON serialization fails.
  - Sets the resulting string into the text edit.


Class: `FFTViewerWindow`
~~~~~~~~~~~~~~~~~~~~~~~~~

Subclass of `QtWidgets.QMainWindow` that displays the FFT (and optionally inverse FFT) of a rectangular region from an image.

- **Purpose** – provide a dedicated window for analyzing the frequency content of a selected ROI, reusing cached FFT results when the region is unchanged, with the same measurement tools and scale‑bar behaviour as the main image viewer but in reciprocal space.
- **`__init__(parent, region: np.ndarray, scale_x: float, scale_y: float, ax_x_name: str, ax_x_units: str, ax_y_name: str, ax_y_units: str, fft_name: str, parent_name: str = "")`**
  - Accepts the raw 2D `region` array and physical scaling and labeling information from the parent `ImageViewerWindow` axes.
  - Sets a descriptive window title via `_update_title()` and fixes the size to `DEFAULT_FFT_WINDOW_SIZE`.
  - Initializes caches for:
    - `_magnitude_spectrum` – log‑scaled magnitude of the FFT.
    - `_fft_complex` – complex FFT values used to compute inverse transforms.
    - `_inverse_fft_cache` – cached magnitude of the inverse FFT to avoid recomputation when toggling views.
    - `_nyq_x`, `_nyq_y` – Nyquist frequencies for axis labelling.
    - `_last_region_id` – `id(region)` for detecting when the ROI has changed.
  - Marks the FFT view as living in reciprocal space (`is_reciprocal_space = True`) and derives reciprocal‑space unit strings (e.g. `1/m` or `1/px`) for axis labels and the scale bar.
  - Initializes measurement state (line‑drawing tool, measurement history window, counters, and label tracking) analogous to `ImageViewerWindow`.
  - Calls `setup_ui()`, then computes and displays the FFT and installs keyboard shortcuts.
- **`_update_title(fft_name: str, parent_name: str)`**
  - Internal helper that composes and sets the window caption, including the parent image name when available.
- **`setup_ui()`**
  - Builds the Qt layout:
    - A toolbar row with:
      - “Show Inverse FFT” checkbox wired to `update_display()`.
      - “Measure Distance” (checkable), “Clear Measurements”, and “Measurement History” buttons wired to the FFT‑specific measurement handlers.
    - A `GraphicsLayoutWidget` containing a single `PlotItem` with a `pg.ImageItem` for displaying the FFT or inverse FFT image.
    - Configures the colormap to `magma`, hides pyqtgraph buttons/menus, and inverts the y‑axis to match the main viewer.
    - Attaches a `DynamicScaleBar` to the plot’s `ViewBox`, using reciprocal‑space units derived from the original real‑space axis.
    - Instantiates a `LineDrawingTool` on the FFT plot, with callbacks into the FFT viewer’s measurement logic.
- **`_compute_fft()`**
  - Computes the FFT only if the ROI has changed (`id(self.region)` differs from `_last_region_id` or no cached spectrum exists).
  - Uses `utils.compute_fft()` to compute the magnitude spectrum and Nyquist frequencies.
  - Independently computes a windowed complex FFT (with 2D Hanning window) and stores it in `_fft_complex` for inverse FFT operations.
  - Resets the inverse FFT cache when the region changes.
- **`update_display()`**
  - Decides which image to show based on the state of the “Show Inverse FFT” checkbox:
    - If unchecked, shows the magnitude spectrum.
    - If checked, computes and caches the inverse FFT via `utils.compute_inverse_fft()` and shows that instead.
  - Calls `setImage()` and `setRect()` on the `ImageItem` so the image is correctly mapped in frequency units from `-nyq` to `+nyq` along each axis.
  - Sets axis labels using reciprocal units derived from the axis units (e.g. `1/m` or `1/px`), stored as `fft_unit_x` / `fft_unit_y` and reused by the scale bar.

- **Keyboard shortcuts**
  - `setup_keyboard_shortcuts()` installs `Delete` / `Backspace` bindings that delete the currently selected measurement label/line and `Esc` to exit measurement mode.

- **Measurement tools in FFT view**
  - `_exit_measure_mode()` / `_toggle_line_measurement()` – enable or disable the `LineDrawingTool` and update the state of the “Measure Distance” button.
  - `_on_line_drawn(p1, p2)` – receives start and end points in FFT axis coordinates:
    - Computes the reciprocal‑space distance between the points.
    - Converts that to a pixel distance on the FFT grid using Nyquist frequencies and the ROI shape.
    - Computes a d‑spacing value (`1 / distance`) via `utils.calculate_d_spacing()`.
    - Draws a permanent line and creates a clickable `MeasurementLabel` at the midpoint, appending the pair to `measurement_items` and recording the text in a `MeasurementHistoryWindow`.
  - `_on_measurement_label_clicked(label)` – selects and highlights the clicked measurement label.
  - `_clear_measurements()` – removes all measurement graphics from the FFT plot and clears the history window if present.
  - `_delete_selected_measurement()` – removes the selected measurement and clears selection/highlighting.
  - `_set_label_fill()` – compatibility helper for setting the background of a `TextItem` across pyqtgraph versions.
  - `delete_measurement_by_label(text)` – deletes the first FFT measurement whose label text matches the given string.
  - `_format_measurement_label(result, measurement_id)` – formats FFT measurements as multi‑line labels showing d‑spacing in Å and the reciprocal‑space distance with an SI‑scaled base unit (e.g. `nm⁻¹`) plus the pixel distance.
  - `_show_measurement_history()` / `_add_to_measurement_history(text)` – lazily create and manage a `MeasurementHistoryWindow` owned by the FFT viewer, mirroring the behaviour of the main image viewer but scoped to this FFT view.


Class: `LineDrawingTool`
~~~~~~~~~~~~~~~~~~~~~~~~~

A non‑Qt class that wraps a `pg.PlotItem`’s `ViewBox` mouse events to implement interactive line drawing.

- **Purpose** – provide the line‑measurement interaction without coupling measurement logic directly into pyqtgraph’s event handlers.
- **`__init__(plot: pg.PlotItem, on_line_drawn_callback)`**
  - Stores the target `PlotItem` and callback to be invoked when a line is completed.
  - Captures the viewbox’s original `mousePressEvent`, `mouseMoveEvent`, and `mouseReleaseEvent` so they can be restored later.
  - Initializes state flags: `drawing`, `start_point`, `line_item`, and `is_enabled`.
- **`enable()`**
  - Marks the tool as enabled and installs its own mouse event handlers on the viewbox.
  - Resets drawing state and the start point.
- **`disable()`**
  - Marks the tool as disabled, restores the original viewbox mouse handlers, clears any preview line, and resets state.
- **`_clear_preview_line()`**
  - Removes the temporary preview `PlotDataItem` from the plot if one exists.
- **`_on_mouse_press(event)`**
  - When enabled and the click occurs inside the plot, starts a new line by recording the mouse position in view (data) coordinates and clearing any previous preview line.
  - Otherwise, delegates back to the original handler.
- **`_on_mouse_move(event)`**
  - When enabled and drawing is in progress, updates the preview line between the recorded start point and the current mouse position.
  - Outside of drawing mode, forwards events to the original handler.
- **`_on_mouse_release(event)`**
  - When enabled and a line is in progress, finalizes the line end point, clears the preview, and calls `on_line_drawn_callback(start, end)` with points in view (axis) coordinates.
  - If the release occurs outside the plot or no line is in progress, clears state as needed and passes the event back to the original handler.


Class: `FFTBoxROI`
~~~~~~~~~~~~~~~~~~~

Subclass of `pg.RectROI` that emits signals when clicked or double‑clicked.

- **Purpose** – add interactive selection and double‑click handling on top of a standard rectangular ROI used to define FFT regions.
- **Signals**:
  - `sigBoxClicked(object)` – emitted on single left‑click, with the ROI instance.
  - `sigBoxDoubleClicked(object)` – emitted on left double‑click, with the ROI instance.
- **`mouseClickEvent(ev)`**
  - Calls the base implementation (if present) to preserve default behaviour.
  - Emits `sigBoxClicked` or `sigBoxDoubleClicked` based on the mouse button and double‑click flag.


Class: `MeasurementLabel`
~~~~~~~~~~~~~~~~~~~~~~~~~~

Subclass of `pg.TextItem` that emits a signal when clicked.

- **Purpose** – represent a clickable label for a measurement, enabling selection and deletion via the toolbar.
- **Signals**:
  - `sigLabelClicked(object)` – emitted when the label is left‑clicked.
- **`mouseClickEvent(ev)`**
  - Calls the base implementation, then emits `sigLabelClicked` if the event is a left‑click.


Class: `ImageViewerWindow`
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Main per‑image window, subclassing `QtWidgets.QMainWindow`. It is responsible for displaying a single 2D image, managing FFT boxes and their windows, handling distance measurements, and exposing metadata.

Construction and state
^^^^^^^^^^^^^^^^^^^^^^^

- **`__init__(file_path: str, signal=None, window_suffix: Optional[str] = None)`**
  - Accepts the original file path and optionally a pre‑loaded HyperSpy signal and a suffix for the window title.
  - Initializes state fields:
    - `signal`, `data`, `ax_x`, `ax_y` – HyperSpy signal and axes; `data` is the 2D NumPy array.
    - `fft_boxes` – list of active `FFTBoxROI` instances.
    - `fft_count` – running counter used to generate FFT box IDs and labels.
    - `is_reciprocal_space` – boolean flag indicating whether the image is interpreted as a diffraction pattern.
    - `line_tool` – instance of `LineDrawingTool` used for distance measurements.
    - `fft_windows` – list of open `FFTViewerWindow` instances.
    - `fft_to_fft_window` – mapping from each FFT ROI to its associated `FFTViewerWindow`.
    - `fft_box_meta` – mapping from ROI to metadata including its numeric ID and label `TextItem`.
    - `selected_fft_box` – currently selected FFT ROI (for deletion).
    - `scale_bar` – dynamic scale bar overlay (for real‑space images).
    - `measurement_history_window` – optional `MeasurementHistoryWindow` instance.
    - `metadata_window` – optional `MetadataWindow` instance.
    - `measurement_count` – sequential counter for labeling measurements.
    - `btn_measure` – reference to the toolbar’s “Measure Distance” button.
    - `measurement_items` – list of `(line_item, text_item)` pairs for all drawn measurements.
    - `selected_measurement_index` – index of the currently selected measurement, if any.
  - If a `signal` is provided, calls `_setup_from_signal(signal, window_suffix)`; otherwise calls `_load_and_setup()` to load from `file_path` via HyperSpy.

Drag and drop
^^^^^^^^^^^^^

- **`dragEnterEvent(event: QtGui.QDragEnterEvent)`**
  - Accepts drag events that contain URLs so that files can be dropped onto individual viewer windows (not just the main window).
- **`dropEvent(event: QtGui.QDropEvent)`**
  - For each dropped file URL, resolves the local file path and, if it is a regular file, calls the module‑level `open_image_file()` helper to open it in a new viewer window.

Signal / data setup
^^^^^^^^^^^^^^^^^^^

- **`_load_and_setup()`**
  - Loads the image from `self.file_path` using `hs.load()`.
  - If the resulting signal has navigation dimensions, selects the first navigation slice (`inav[0, 0]`).
  - Delegates to `_setup_from_signal()` with the resulting 2D signal.
- **`_setup_from_signal(signal, window_suffix: Optional[str] = None)`**
  - Stores the HyperSpy signal and its data and axis objects.
  - Attempts to apply pixel calibration from the original metadata via `_apply_calibration_from_original_metadata()`; if this fails, shows a warning dialog and falls back to default pixel scaling.
  - Calls `utils.is_diffraction_pattern(self.data)` to set `is_reciprocal_space`.
  - Composes and sets the window title from the file name plus an optional suffix.
  - Calls `setup_ui()` and resizes the window to `DEFAULT_IMAGE_WINDOW_SIZE`.

Metadata helpers
^^^^^^^^^^^^^^^^

- **`_get_original_metadata_dict() -> Optional[dict]`**
  - Attempts to retrieve the “original metadata” from the HyperSpy signal, either via `signal.original_metadata` or `signal.metadata.original_metadata`.
  - Normalizes the metadata into a plain Python dictionary, using `.as_dictionary()` when available.
- **`_apply_calibration_from_original_metadata() -> bool`**
  - Retrieves the normalized original metadata and looks for a `ser_header_parameters` dictionary, handling case variations.
  - Extracts `CalibrationDeltaX` and `CalibrationDeltaY` (expected to be in metres for SER data) and uses them to set the `scale` and `units` attributes on the x and y axes.
  - Optionally reads `CalibrationOffsetX` and `CalibrationOffsetY` and applies them as axis offsets.
  - Returns `True` if calibration succeeds and `False` otherwise.

Geometry and UI
^^^^^^^^^^^^^^^

- **`image_bounds -> Tuple[float, float, float, float]` (property)**
  - Computes `(x_offset, y_offset, width, height)` using axis `offset`, `size`, and `scale`, falling back to simple defaults if axes are missing.
- **`setup_ui()`**
  - Builds the main viewer layout:
    - A horizontal toolbar row via `_create_toolbar()`.
    - A `GraphicsLayoutWidget` with a single `PlotItem` (`self.p1`) showing the image as a `pg.ImageItem`.
  - Lays out the image using `image_bounds`, locks aspect ratio to 1:1, inverts the y‑axis for conventional image orientation, and removes padding so the image fills the view.
  - Adds a `DynamicScaleBar` to `self.p1.vb` for real‑space images using the axis units.
  - Instantiates a `LineDrawingTool` bound to `self.p1` and wires up keyboard shortcuts for delete/backspace (deleting selected measurement or FFT box) and escape (exiting measurement mode).
- **`setup_keyboard_shortcuts()`**
  - Installs `QShortcuts` for `Delete`, `Backspace`, and `Esc` to trigger `_delete_selected_roi()` and `_exit_measure_mode()`.

Toolbar and FFT management
^^^^^^^^^^^^^^^^^^^^^^^^^^

- **`_delete_selected_roi()`**
  - If a measurement is currently selected, delegates to `_delete_selected_measurement()` and returns.
  - Otherwise, deletes the currently selected FFT ROI (if any) and its associated label and FFT window, updating all tracking collections and informing the user.
- **`_create_toolbar() -> QtWidgets.QHBoxLayout`**
  - Creates and wires toolbar buttons:
    - “Add New FFT Box” → `_add_new_fft()`.
    - “Measure Distance” (checkable) → `_toggle_line_measurement()`.
    - “Clear Measurements” → `_clear_measurements()`.
    - “Delete Selected” → `_delete_selected_roi()`.
    - “Image Metadata” → `_show_metadata_window()`.
    - “Measurement History” → `_show_measurement_history()`.
- **`_add_new_fft(x_offset=None, y_offset=None, w=None, h=None)`**
  - Computes a default FFT ROI rectangle based on the current `ViewBox` range (or full image bounds as a fallback), using half the visible width and height so the box is always reasonably sized.
  - Chooses a color from `FFT_COLORS`, constructs an `FFTBoxROI` at the computed position and size, and adds two scale handles for interactive resizing.
  - Creates a label `TextItem` (e.g. “FFT 0”), adds it to the plot, and wires signals:
    - `sigRegionChangeFinished` → `_on_fft_finished()`.
    - `sigBoxClicked` → `_on_fft_box_clicked()`.
    - `sigBoxDoubleClicked` → `_on_fft_box_double_clicked()`.
  - Stores bookkeeping metadata for the ROI in `fft_boxes` and `fft_box_meta` and increments the global FFT counter.
- **`_on_fft_finished(fft_box, fft_id, text_item)`**
  - Extracts the current ROI region as a NumPy array using `fft_box.getArrayRegion(self.data, self.img_orig)`.
  - If the region is large enough, updates the label position and calls `_open_or_update_fft_window()`.
- **`_open_or_update_fft_window(fft_box, fft_id, text_item, region)`**
  - If an FFT window already exists for the given ROI, updates its `region`, recomputes the FFT, and raises the window.
  - Otherwise, constructs a new `FFTViewerWindow` with the region and axis information, shows it, and records it in `fft_windows` and `fft_to_fft_window`.
- **`_on_fft_box_clicked(fft_box)`**
  - Marks the clicked ROI as the currently selected FFT box (for deletion via toolbar or shortcut).
- **`_on_fft_box_double_clicked(fft_box, fft_id, text_item)`**
  - Recomputes the ROI region and label position and (re)opens the associated FFT window via `_open_or_update_fft_window()`.

Measurement controls
^^^^^^^^^^^^^^^^^^^^

- **`_exit_measure_mode()`**
  - Disables the `LineDrawingTool`, unchecks the “Measure Distance” button if needed, and resets its appearance.
- **`_toggle_line_measurement()`**
  - When the button is checked, enables line drawing and visually highlights the button.
  - When unchecked, disables line drawing and restores the default style.
- **`_on_line_drawn(p1, p2)`**
  - Receives start and end points from `LineDrawingTool` in axis (physical) coordinates.
  - Computes physical distances using `ax_x.scale` and `ax_y.scale`, back‑computes pixel distances, and builds a result dictionary that may include `d_spacing` via `utils.calculate_d_spacing()` when `is_reciprocal_space` is `True`.
  - Draws the final measurement line (`DRAWN_LINE_PEN`), creates a `MeasurementLabel` at its midpoint using `_format_measurement_label()`, and tracks the `(line, label)` pair in `measurement_items`.
  - Adds the formatted label text to the measurement history via `_add_to_measurement_history()`.
- **`_on_measurement_label_clicked(label)`**
  - Locates the clicked label in `measurement_items`, updates `selected_measurement_index`, and visually highlights the selected label using `_set_label_fill()`.
- **`_clear_measurements()`**
  - Removes all measurement graphics from the plot and clears the history window if present.
- **`_delete_selected_measurement()`**
  - Removes the selected measurement line and label from both the plot and `measurement_items` and clears the selection.
- **`_set_label_fill(text_item, brush)`**
  - Compatibility helper that sets the background brush of a `TextItem` using either `setFill()` or `setBrush()`, depending on the pyqtgraph version.
- **`delete_measurement_by_label(label_text: str)`**
  - Deletes the first measurement whose label text matches the given string, used when the history window requests deletion by label.
- **`_format_measurement_label(result: dict, measurement_id: Optional[int] = None) -> str`**
  - Formats distances using `utils.format_si_scale()` and builds a multi‑line label string including the measurement index.
  - For diffraction patterns, reports d‑spacing in Å and the reciprocal distance; for real‑space images, reports physical distance and pixel distance.
- **`_show_measurement_history()` / `_add_to_measurement_history(text)`**
  - Lazily creates a `MeasurementHistoryWindow` as needed and forwards new measurement strings to it.

Metadata display
^^^^^^^^^^^^^^^^

- **`_show_metadata_window()`**
  - Assembles a metadata dictionary using `_get_original_metadata_dict()` or `signal.metadata.as_dictionary()` as a fallback.
  - Lazily creates or updates a `MetadataWindow` with the current file name in the title and the latest metadata content.


Function: `open_image_file(file_path: str)`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Module‑level helper for loading images and opening appropriate `ImageViewerWindow` instances. (Note: defined twice in `app.py` with the same body; functionally this is a single behaviour.)

- **Purpose** – centralize file loading so both the main window and individual viewers can support drag‑and‑drop using the same logic.
- **Behaviour**:
  - Calls `hs.load(file_path)` to obtain either a single signal or a list of signals.
  - Normalizes the result into a list `signals`.
  - For each signal:
    - If `navigation_dimension == 0`, treats it as a single 2D image and opens one `ImageViewerWindow` (with a suffix when there are multiple signals in the file).
    - Otherwise, iterates over the navigation indices via `np.ndindex(nav_shape)`, extracts each `sub_signal` via `inav[...]`, and opens one `ImageViewerWindow` per navigation position, including the signal index and navigation indices in the window title suffix.
  - On error, shows a critical message box describing the problem.


Class: `MainWindow`
~~~~~~~~~~~~~~~~~~~~

Subclass of `QtWidgets.QMainWindow` acting as the application’s initial drag‑and‑drop landing window.

- **`__init__()`**
  - Sets the window title to “Fast FFT Image Analyzer”, resizes to `DEFAULT_MAIN_WINDOW_SIZE`, enables drag‑and‑drop, and calls `_setup_ui()`.
- **`_setup_ui()`**
  - Creates a central widget with a vertically centered label explaining that users should drag and drop supported images.
- **`dragEnterEvent(event)` / `dropEvent(event)`**
  - Accept file URL drags and, for each dropped file, call `_open_image(file_path)`.
- **`_open_image(file_path: str)`**
  - Thin wrapper that delegates to the module‑level `open_image_file()`.


Function: `main()`
~~~~~~~~~~~~~~~~~~~

Application entry point that builds and runs the Qt event loop.

- Creates a `QApplication`, instantiates `MainWindow`, shows it, and executes the app via `app.exec()`.
- This function is exposed to `main.py` and also invoked when `app.py` is run directly as `__main__`.


utils.py
--------

Top‑level
~~~~~~~~~

- **Module docstring** – indicates this file contains utility functions for FFT, measurements, and general image analysis.
- **Imports** – brings in NumPy and typing helpers (`Tuple`, `Optional`).
- **`_window_cache`** – module‑level dictionary caching 2D Hanning windows keyed by array shape to avoid recomputing them for repeated FFTs on same‑sized ROIs.
- **`SI_PREFIXES`** – ordered list of `(factor, prefix)` pairs used to map values into common SI prefixed units (G, M, k, base, m, µ, n, p, f).


Function: `format_si_scale(value: float, base_unit: str = '', precision: int = 3) -> Tuple[float, str]`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **Purpose** – convert a raw value in base units into a value and unit string using a suitable SI prefix (e.g. `1e-9 m` → `(1, "nm")`).
- **Behaviour**:
  - If `value` is zero or not finite, returns `(value, base_unit)` unchanged.
  - Determines the absolute value and walks `SI_PREFIXES` from largest to smallest factor, choosing the first factor such that `abs_value >= factor * 0.95`.
  - Returns the scaled value and concatenated prefix + base unit.
  - For extremely small values, falls back to the smallest defined prefix.


Function: `_get_hanning_window(shape: Tuple[int, int]) -> np.ndarray`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **Purpose** – memoized creation of a 2D Hanning window of the requested shape.
- **Behaviour**:
  - If `shape` is not in `_window_cache`, computes `np.hanning(shape[0])[:, None] * np.hanning(shape[1])[None, :]`, stores it in the cache, and returns it.
  - Otherwise, returns the cached array.


Function: `compute_fft(region: np.ndarray, scale_x: float, scale_y: float, apply_window: bool = True) -> Tuple[np.ndarray, float, float]`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **Purpose** – compute the log‑magnitude FFT of a 2D region along with Nyquist frequencies used for axis labelling in FFT viewer windows.
- **Behaviour**:
  - Returns `(None, None, None)` if the region is `None` or has either dimension smaller than 2 pixels.
  - Optionally multiplies the region by a cached 2D Hanning window (`_get_hanning_window`) to reduce edge artefacts.
  - Computes the 2D FFT with `np.fft.fft2`, shifts it to center zero frequency with `np.fft.fftshift`, and computes a log‑scaled magnitude (`20 * log10(|F| + 1e-8)`).
  - Computes Nyquist frequencies `nyq_x = 0.5 / scale_x` and `nyq_y = 0.5 / scale_y` based on physical pixel scales.
  - Returns `(magnitude_spectrum, nyq_x, nyq_y)`.


Function: `compute_inverse_fft(fft_data: np.ndarray) -> np.ndarray`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **Purpose** – compute the magnitude of the inverse FFT from complex, shift‑centered FFT data.
- **Behaviour**:
  - Unshifts the spectrum with `np.fft.ifftshift`, performs `np.fft.ifft2`, and returns `np.abs(real_image)`.
  - Used by `FFTViewerWindow` when toggling “Show Inverse FFT”.


Function: `calculate_d_spacing(frequency: float, wavelength: float = 0.00251) -> float`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **Purpose** – translate a reciprocal‑space distance into d‑spacing (in Å), under a simple `d = 1 / frequency` relationship.
- **Behaviour**:
  - Returns `inf` if frequency is zero; otherwise returns `1.0 / frequency`.
  - The `wavelength` parameter is currently unused in the computation but documents the intended physical context (electron wavelength).


Function: `measure_line_distance(p1, p2, scale_x, scale_y=None, is_reciprocal: bool = False) -> dict`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **Purpose** – vectorized helper for computing distances in pixels and physical units, with optional d‑spacing for reciprocal‑space images.
- **Behaviour**:
  - If `scale_y` is `None`, uses `scale_x` (isotropic scaling).
  - Forms a difference vector between `p2` and `p1`, computes its Euclidean norm in pixel units, and then scales each axis by its physical scale to obtain a physical‑distance norm.
  - Returns a dictionary with keys `distance_pixels`, `distance_physical`, `scale_x`, `scale_y`.
  - If `is_reciprocal` is `True` and `distance_physical` is nonzero, computes a reciprocal frequency and adds `d_spacing` using `calculate_d_spacing()`.
  - Used conceptually for measurement logic; the viewer’s `_on_line_drawn` currently performs a similar computation in‑line for fine‑grained control.


Function: `is_diffraction_pattern(image_data: np.ndarray, center_ratio: float = 2.0) -> bool`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **Purpose** – heuristic for deciding whether an image is likely a diffraction pattern.
- **Behaviour**:
  - Returns `False` if `image_data` is not a 2D array.
  - Splits the image into a central region and an edge region; computes their mean intensities.
  - Returns `True` if the center mean exceeds `edge_mean * center_ratio`, i.e. if the image has a significantly brighter centre typical of many diffraction patterns.
  - The result is used by `ImageViewerWindow` to determine whether to display d‑spacings for measurements.


main.py
-------

- **Shebang and module docstring** – declares this file as the main entry point of the Fast FFT Image Analyzer.
- **`from app import main`** – imports the `main()` function defined in `app.py`.
- **`if __name__ == "__main__": main()`** – when `main.py` is executed directly, it instantiates the Qt application and shows the `MainWindow` via `app.main()`.

This document should give you enough detail to navigate and extend the project. The README focuses on usage; this file focuses on structure and behaviour of each program element.
