## Known issues/ Desired features

---
Severe
-------

- when not running with --force-software sometimes selecting the fft option results in a teal screen all over. Not clear why this is.

- distance measurement is not properly context aware for whether this is a real-space or reciprocal-space image (perhaps need to use metadata to extract this)




- measurement mode no longer works after an FFT has been enabled in the main image window.

- scaling of diffraction images is currently not handled welll

- you cannot take the ft/ ift of a ft/ift

---
Cosmetic
-------


- change the black bounding boxes to zero pixels wide on the image view screen

- need a way to change the color scheme

- need to implement keyboard shortcuts

-boundary of images at black bounding box is not clear. Perhaps change it to something clearer.



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