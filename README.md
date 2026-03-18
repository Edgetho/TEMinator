# Fast FFT Image Analyzer

A high-performance application for analyzing electron microscopy (EM) images with fast Fourier transform (FFT) analysis. Designed for clear interpretation of scientific EM images with support for both real-space and diffraction pattern analysis.

## Features

- **Drag & Drop Interface**: Simply drag image files into the application to open them
- **Multiple ROIs**: Create multiple regions of interest (ROIs), each with independent FFT analysis in separate windows
- **Real-time FFT**: FFT calculations occur only when ROI positioning is complete (optimized for performance)
- **Inverse FFT**: Toggle between FFT magnitude and inverse FFT display for each ROI
- **Distance Measurement**: Draw lines to measure distances with automatic d-spacing calculation for diffraction patterns
- **Automatic Image Type Detection**: Detects whether images are diffraction patterns or real-space images
- **Proper Unit Handling**: Preserves and displays physical units from image metadata throughout analysis
- **Multi-Window Analysis**: Analyze multiple images simultaneously with independent FFT windows for each ROI

## Installation

### Prerequisites

- **Anaconda** or **Miniconda** (recommended)
- Python 3.9 or higher

### Quick Setup

1. **Clone or download this repository**

2. **Create the conda environment**:
   ```bash
   conda env create -f environment.yml
   ```

3. **Activate the environment**:
   ```bash
   conda activate fft-image-analyzer
   ```

4. **Run the application**:
   ```bash
   python main.py
   ```

### Manual Installation (if not using conda)

If you prefer pip, install the required packages:

```bash
pip install numpy pyqtgraph pyqt5 hyperspy
```

Then run:
```bash
python main.py
```

## Usage Guide

### Opening Images

1. **Launch the application**: Run `python main.py`
2. **Drag and drop**: Simply drag an image file onto the main window (it will be blank/gray initially)
3. **Supported formats**: DM3, DM4, TIFF, and other formats supported by HyperSpy

The application automatically:
- Detects the image type (real-space or diffraction pattern)
- Sets appropriate axis labels and units from image metadata
- Creates an initial ROI and shows its FFT in a new window

### Analyzing Images

#### Creating and Managing ROIs

1. **Add a new ROI**: Click the "Add New ROI" button in the toolbar
   - Each ROI is displayed with a unique color and number label
   - ROI numbers appear on the image for easy identification

2. **Position an ROI**:
   - Click and drag to move it
   - Drag the corner handles to resize it
   - When you release the mouse, the FFT automatically updates

3. **View FFT**: Each completed ROI opens (or updates) its own FFT window
   - Window title shows: `FFT - [Image Name] - ROI [Number]`
   - FFT is calculated only when you finish moving/resizing (for performance)

#### FFT Display Options

In each FFT window:

1. **Show Inverse FFT**: Check this box to toggle between:
   - **Unchecked**: FFT magnitude spectrum (default)
   - **Checked**: Inverse FFT (reconstructed real-space image)

2. **Axis Labels**: 
   - Automatically shows reciprocal space units (e.g., 1/nm)
   - Scaling is computed from image metadata for accuracy

#### Measuring Distances

1. **Enable measurement tool**: Click "Measure Distance" in the toolbar
   - A message will appear with instructions
   - Click the button again to toggle the tool on/off

2. **Drawing a measurement line**:
   - Click once on the image to set the start point
   - Move the mouse to preview the line (yellow dashed)
   - Click again to set the end point and complete the measurement

3. **Results**:
   - A dialog shows the measured distance in physical units
   - Also displays the distance in pixels
   - **For diffraction patterns**: Shows d-spacing (in Ångströms)
   - **For real-space images**: Shows physical distance in image units

4. **Drawn lines**: Measurement lines remain visible as white lines on the image

### Example Workflow

1. Launch the app: `python main.py`
2. Drag a diffraction pattern image onto the window
3. The first ROI opens automatically with its FFT window
4. Click "Measure Distance" and measure a bright ring to get d-spacing
5. Click "Add New ROI" to select another region
6. Adjust the new ROI by dragging; its FFT window updates automatically
7. Check "Show Inverse FFT" in any FFT window to see the reconstructed real-space
8. Compare FFTs from different regions by having multiple FFT windows open

## Window Management

### Image Viewer Window
- **Title**: `Image Viewer - [Image Filename]`
- **Shows**: Original image with colored ROI boxes and labels
- **Toolbar**: "Add New ROI", "Measure Distance" buttons

### FFT Windows
- **Title**: `FFT - [Image Name] - ROI [Number]`
- **One window per ROI**: Each ROI has its own FFT window
- **Reuses windows**: Moving an ROI updates its existing FFT window instead of creating a new one

## Supported Image Formats

The application uses HyperSpy for image loading, supporting:

- **Electron microscopy formats**: DM3, DM4 (Gatan formats)
- **Standard formats**: TIFF, PNG, JPG
- **Scientific formats**: HDF5, NetCDF
- **Other formats**: See HyperSpy documentation

## Technical Details

### Performance Optimization

FFT calculations are expensive operations. The application is optimized by:

- **Calculation timing**: FFT only computes when you finish moving an ROI (using `sigRegionChangeFinished` signal)
- **Window reuse**: Moving an existing ROI updates its FFT window instead of creating a new one
- **Efficient algorithms**: Uses NumPy's optimized FFT library with Hanning windowing

### Image Metadata Preservation

Physical scaling, axis labels, and units are preserved from the image metadata:

- Displayed in the image viewer window labels
- Used to calculate accurate Nyquist frequencies for FFT display
- Maintained in distance measurements and d-spacing calculations

### Diffraction Pattern Detection

The app includes automatic detection of diffraction patterns by analyzing:

- Brightness ratio between image center and edges
- Assumes bright centers are diffraction patterns
- Used to automatically enable d-spacing calculations

If detection is incorrect, you can still manually measure distances (d-spacing won't calculate for real-space images).

## Troubleshooting

### "Could not determine pixel size" warning

This is a non-critical HyperSpy warning. It appears when image metadata doesn't include pixel size information. The app will still function normally.

### Image won't load

- Ensure the file format is supported by HyperSpy
- Check that the file path doesn't contain special characters
- Try a different image file to verify the app works

### FFT window doesn't update when moving ROI

- Ensure you released the mouse button completely
- FFT updates on `sigRegionChangeFinished`, not during dragging

### d-spacing showing as "inf" or very large values

- This occurs when measuring very small distances in reciprocal space
- The mathematical relationship is: d-spacing = 1 / distance
- Try measuring larger features (ring diameter instead of a point)

## Architecture

### File Structure

```
image_app/
├── main.py              # Entry point
├── app.py               # Main application classes and UI
├── utils.py             # FFT and measurement utilities
├── environment.yml      # Anaconda environment definition
├── README.md            # This file
└── REFACTOR_NOTES.md    # Technical refactoring notes
```

### Main Classes

- **`MainWindow`**: Drag-and-drop entry point for opening images
- **`ImageViewerWindow`**: Image display with ROI management
- **`FFTViewerWindow`**: FFT display window
- **`LineDrawingTool`**: Interactive line measurement tool

### Utility Functions

- `compute_fft()`: FFT calculation with windowing and scaling
- `compute_inverse_fft()`: Inverse FFT reconstruction
- `measure_line_distance()`: Distance and d-spacing calculations
- `is_diffraction_pattern()`: Image type detection

## Performance Notes

- Typical FFT computation time: 10-100ms depending on ROI size
- ~50MB RAM for typical EM images
- Multiple windows open simultaneously have minimal performance impact
- Line drawing adds minimal overhead

## Future Enhancements

- ROI presets and templates
- Automated diffraction ring detection
- Batch processing for multiple images
- Export analysis results to files
- Customizable electron wavelength for d-spacing
- Radial averaging of FFT data
- Measurement history tracking

## Contributing

For bug reports or feature requests, please document:
- Steps to reproduce
- Expected behavior
- Actual behavior
- Image file type (if applicable)

## License

Check the project repository for license information.

## Citation

If you use this tool in published research, please cite:

```
Fast FFT Image Analyzer
https://github.com/[repository-path]
```

## Support

For issues with:
- **Installation**: Check you have conda installed and internet connection
- **Image loading**: Verify the file format is supported by HyperSpy
- **FFT calculations**: Ensure ROI is large enough (minimum 2x2 pixels)
- **HyperSpy integration**: See https://hyperspy.org for documentation
