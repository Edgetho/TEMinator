## Known issues/ Desired features

---
Severe
-------

- inverse FFT option should work exactly the same way as for the FFT option, except that once the bounding box is selected, it should display the inverse FFT instead of the FFT. the options to draw bounding boxes for new (inverse) FFTs should be the same for FFT view windows as for parent image windows. 
- calibration errors do not fail loudly enough. When there is a failure of calibration, the scale bar should display UNCALIBRATED on it. there also needs to be a manual calibration option, which will display a tag like (manually calibrated) on it

- distance measurement is not properly context aware for whether this is a real-space or reciprocal-space image (perhaps need to use metadata to extract this)

- NOT FIXED: clearing measurements from the history pane does not remove annotations on the image, as it should. also there should not be a pop up confirmation dialogue after measurements are deleted.

- on the inital opening screen, all options except for parameters and open file should be greyed out.

- on opening an uncalibrated image, the request to calibrate should only open after the image has been opened

- measurement mode no longer works after an FFT has been enabled.

- There is no way to turn off distance measruement once it it turned on. There should be a status bar displayed at the bottom of the screen with the text "Measurement mode (esc to exit): {distance}" where distance stands for the current  mesurement values based on the instantaneous mouse position.

---
Cosmetic
-------

- scale bars for reciprocal-space images should display in 1/nm etc instead of G1/m etc for conformation with convention

- change the black bounding boxes to zero pixels wide on the image view screen



---
Desired features
-------

- option to save/load all open files as they stand so that work can be portably continued

- map of locations of all TEM files pulled from metadata /in a folder

- timeline of all exposures open/in a folder

- publication style figure generator

---
Desired UI/UX
------

Have consistent controls for all image viewers (i.e. for FFTs and direct images -- perhaps refactor code for this to make addition of image manipulation extensible in the future)

| Menu  | Option | Default Shortcut |
|------ |--------|------------------|
| File  | Open | Ctrl+O |
|       | Save View | Ctrl+S |
|       | Build Figure | Ctrl+B |
|       | Metadata | M |
|       | Calibrate | |
|       | Parameters | Ctrl+, |
| Manipulate | FFT | Ctrl+F |
|       | Inverse FFT | Ctrl+Shift+F |
|       | Adjust Display | A|
| Measure | Distance | D |
|       | History | H |
|       | Intensity | I |
|       | Profile | P |



New startup flow:

open a window that looks like the others and has the same menu bar across the top, but instead of an image it displays the welcome text as follows:

```html
"<div style='text-align:center;'>"
"<p style='margin:0;'>Drag and drop an image file here to open it.</p>"
"<p style='margin:6px 0 12px 0;'>(Supports DM3, DM4, TIFF, and other HyperSpy formats)</p>"
"<p style='margin:6px;'> Provided without warranty under the GNU GPLv2.<.</p>"
"<p style='margin:6px 0 0 0;'>Source: "
"<a href='https://github.com/Edgetho/TEMinator'>github.com/Edgetho/TEMinator</a></p>"
"<p style='margin:6px 0 0 0;'>&copy; 2026 Cooper Stuntz</p>"
"</div>"
```
All options except for File Open and File Parameters should be greyed out upon startup.