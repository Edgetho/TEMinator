## Known issues/ Desired features

---
Severe
-------

- The scaling of windows is limited by the width of the icons on the top bar
- FFT views can change their pixel aspect ratio. All image views should be locked at 1:1
- inverse FFT option not present for images symmetrically as for FFT displays
- calibration errors do not fail loudly enough. When there is a failure of calibration, the scale bar should display UNCALIBRATED on it. there also needs to be a manual calibration option, which will display a tag like (manually calibrated) on it

- distance measurement is not properly context aware for whether this is a real-space or reciprocal-space image (perhaps need to use metadata to extract this)


---
Cosmetic
-------
- it's hard to navigate and windows have inconstent options even though the fundamental controls should be the same for all image data views
- scale bars for reciprocal-space images should display in 1/nm etc instead of G1/m etc for conformation with convention




---
Desired features
-------

- option to save/load all open files as they stand so that work can be portably continued

- map of locations of all TEM files pulled from metadata /in a folder

- timeline of all exposures open/in a folder

- publication style figure generator

---
Desired UI/UX improvements
------

Have consistent controls for all image viewers (i.e. for FFTs and direct images -- perhaps refactor code for this to make addition of image manipulation extensible in the future)

| Menu  | Option | Default Shortcut |
|------ |--------|------------------|
| File  | Open | Ctrl+O |
|       | Save View | Ctrl+S |
|       | Build Figure | Ctrl+B |
|       | Parameters | Ctrl+, |
| Manipulate | FFT | Ctrl+F |
|       | Inverse FFT | Ctrl+Shift+F |
| Measure | Distance | D |
|       | History | H |
|       | Intensity | I |
|       | Profile | P |
| Display | Adjust | A |
|       | Metadata | M |


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
