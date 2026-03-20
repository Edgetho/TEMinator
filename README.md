TEMcompanion
=======================

Desktop viewer for electron microscopy and related scientific images with fast, interactive FFT analysis, distance measurements, and metadata-aware scaling.

The application is built with PyQt / pyqtgraph and HyperSpy and is intended for quick inspection of both real‑space images and diffraction patterns.

Warning: This application is under very active development and will break. If you have ideas to fix things, please submit a PR.

---

Getting started
---------------

### 1. Install dependencies

From the project root, create the conda/mamba environment defined in `environment.yml`:

```bash
mamba env create -f environment.yml
```

or:

```bash
conda env create -f environment.yml
```

This creates the `teminator` environment with Python 3.14, NumPy, pyqtgraph, PyQt 5.15, and HyperSpy.

If you prefer to use either`pip` or `venv` instead, install the packages listed in `environment.yml` manually.

### 2. Run the application

From the project root, run:

```bash
./teminator
```

The launcher will:
- detect the environment name from `environment.yml`
- use `mamba run` (or `conda run`) to execute `app.py`
- forward any additional arguments to the app

Examples:

```bash
./teminator --help
./teminator /path/to/image.dm4
```

On startup you will see a small main window with drag‑and‑drop instructions.

If you already have a compatible Python environment active, you can also run directly:

```bash
python app.py
```


### Linux Wayland setup (optional)

If you want to run on Wayland with OpenGL acceleration, ensure the environment includes the Qt Wayland plugin:

```bash
mamba env update -n teminator -f environment.yml --prune
```

Then launch with:

```bash
QT_QPA_PLATFORM=wayland-egl QT_OPENGL=desktop mamba run -n teminator python app.py
```

Notes:
- If Wayland is unavailable or plugin loading fails, use `QT_QPA_PLATFORM=xcb`.
- Messages like `qt.qpa.wayland: Wayland does not support QWindow::requestActivate()` are normal on Wayland and can usually be ignored.



---

Basic usage
-----------

### Opening images

- Drag one or more supported files (DM3/DM4, TIFF, and any format HyperSpy can read) onto the main window.
- Each image (or navigation position in a HyperSpy signal) opens in its own **Image Viewer** window.
- If a file contains a navigation stack, one viewer is created per 2D slice.

### Image Viewer window

Each viewer window shows a single 2D image with tools for FFTs, measurements, and metadata.

Toolbar controls:

- **Add New FFT Box** – adds a colored rectangular ROI on the current view. The box size is chosen relative to the visible region so it is always a useful starting size.
- **Measure Distance** (toggle) – enables an interactive line‑drawing tool for distance and (for diffraction patterns) d‑spacing measurement.
- **Clear Measurements** – removes all measurement lines/labels from the image and clears the history window.
- **Delete Selected** – deletes the currently selected measurement or FFT box.
- **Image Metadata** – opens a window showing the full HyperSpy metadata dictionary for the current image.
- **Measurement History** – opens a separate window listing all measurements taken in this viewer.

Other behaviour:

- Images are shown in physical coordinates derived from HyperSpy metadata when available (axis scales and units).
- A dynamic overlay scale bar is drawn for real‑space images; it automatically updates size and label with zoom.
- Diffraction patterns are detected heuristically so measurements can be reported as d‑spacings.

### Working with FFT boxes

- Click **Add New FFT Box** to create a new ROI.
- Drag inside the box to move it; drag its corner handles to resize.
- When you release the mouse after moving or resizing, the FFT for that region is (re)computed.
- Each FFT box has:
  - A colored rectangle on the image.
  - A text label like “FFT 0”, “FFT 1”, … anchored near the box.

Interaction:

- **Single‑click** an FFT box to select it (for deletion via **Delete Selected**).
- **Double‑click** an FFT box to open or bring its FFT window to the front.
- Moving a box updates its existing FFT window; a new FFT window is only created the first time the box is finalized.

### FFT windows

- Each FFT box has at most one associated **FFT Viewer** window.
- The window title has the form `FFT – <image name> – FFT <id>`.
- The FFT is computed with a 2D Hanning window and displayed with a `magma` colormap.
- Axes are labelled in reciprocal units using the pixel scale (e.g. 1/m or 1/nm, depending on calibration).
- A dynamic overlay scale bar, analogous to the real‑space view, shows a convenient reciprocal‑space distance in screen coordinates.

Controls in an FFT window:

- **Show Inverse FFT** – when checked, displays the magnitude of the inverse FFT of the windowed ROI instead of the magnitude spectrum.
- **Measure Distance / Clear Measurements / Measurement History** – the same measurement tools available in the main image viewer, but operating directly in reciprocal‑space coordinates; d‑spacings are computed from measured reciprocal distances.

### Measuring distances

1. Click **Measure Distance** to enter measurement mode (the button is highlighted).
2. Click once on the image to set the start point of the line.
3. Move the mouse to preview the line (yellow dashed preview).
4. Click again to set the end point; a solid white line and label are added.
5. Press `Esc` or toggle **Measure Distance** off to leave measurement mode.

Reported values:

- Distances are computed in the physical units of the axes when calibration is available, and in pixels otherwise.
- For real‑space images, labels show physical distance plus the equivalent in pixels.
- For diffraction patterns, labels show d‑spacing (in Å) plus the reciprocal‑space distance.

Measurement management:

- Clicking a label selects that measurement; **Delete Selected** removes it from the image.
- **Clear Measurements** removes all measurement graphics and clears the history.
- The **Measurement History** window lists every measurement taken and supports clearing, deleting, copying to clipboard, and CSV export.

---

Architecture overview
---------------------

Source layout:

- `app.py` – app entrypoint and main drag-and-drop window.
- `image_viewer.py` – `ImageViewerWindow` and image loading/opening helpers.
- `fft_viewer.py` – `FFTViewerWindow` for ROI FFT views.
- `dialogs.py` – metadata/history/tone-curve dialogs.
- `measurement_tools.py` – line drawing tools, measurement labels, FFT ROI item classes.
- `scale_bars.py` – dynamic and static scale bar graphics.
- `utils.py` – numerical helpers for FFTs, line measurements, SI‑scaled units, and diffraction‑pattern detection.
- `teminator` – launcher script for conda/mamba-based startup.
- `environment.yml` – conda environment definition.

At runtime:

- The **MainWindow** accepts drag‑and‑drop and dispatches each dropped file to a helper that loads it with HyperSpy and opens one or more **ImageViewerWindow** instances.
- Each **ImageViewerWindow** owns its FFT boxes, measurement graphics, optional measurement history window, and metadata window.
- Each FFT box is a `pyqtgraph.RectROI` subclass wired to an **FFTViewerWindow** that displays and caches FFT results.
- Low‑level NumPy/FFT utilities live in `utils.py` and are shared between viewers.

---

Troubleshooting
---------------

- **No image opens** – check that the file format is supported by HyperSpy and that the path is readable.
- **FFT window is blank** – ensure the FFT box covers at least a 2×2 region of the image.
- **Measurements show “inf” d‑spacing** – this occurs when the measured reciprocal‑space distance is numerically zero; try measuring a larger feature.
- **Calibration warning** – if metadata does not contain usable pixel size information, the viewer falls back to unit‑per‑pixel scaling and shows a warning dialog.
- **Linux OpenGL warning at startup** – some systems without GLX/EGL support may print a Qt warning and then continue in software mode.

---

Development notes
-----------------

- The project targets Python 3.14 and PyQt via `pyqtgraph.Qt` for flexibility across Qt bindings.
- HyperSpy is used only for I/O and metadata; image display and interaction is handled entirely by pyqtgraph.
- GUI classes are kept in a single module for ease of navigation; numerical work is confined to `utils.py`.

License and citation
--------------------

Copyright (C) 2026 Cooper Stuntz, email: [cstuntz@princeton.edu](mailto:cstuntz@princeton.edu).

Code included in this repository is licensed under the GNU General Public License, version 2. The full license text is provided in `LICENSE`. Licenses for dependencies may differ; consult each dependency's license as needed.

If you use this tool in scientific work, consider acknowledging it as TEMinator and the authors in your methods or acknowledgements section.
