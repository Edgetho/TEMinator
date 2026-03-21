# -*- mode: python ; coding: utf-8 -*-

from PyInstaller.building.build_main import Analysis, PYZ, EXE, COLLECT
from PyInstaller.building.osx import BUNDLE
from PyInstaller.utils.hooks import collect_all, copy_metadata
import importlib.util
import sys

print(f"[spec] Python: {sys.executable}")

for mod in ("pyqtgraph", "hyperspy", "PyQt5"):
    if importlib.util.find_spec(mod) is None:
        raise SystemExit(f"[spec] Missing module in build env: {mod}")

datas = [("app_icon.png", ".")]
binaries = []
hiddenimports = []

for pkg in [
    "pyqtgraph",
    "hyperspy",
    "rsciio",
    "PyQt5",
    "numpy",
    "scipy",
    "matplotlib",
    "skimage",
    "h5py",
    "tifffile",
    "imageio",
    "pandas",
    "numexpr",
    "natsort",
    "pint",
    "traits",
    "dask",
    "zarr",
]:
    d, b, h = collect_all(pkg)
    datas += d
    binaries += b
    hiddenimports += h

# ensure entry-point metadata is bundled
datas += copy_metadata("hyperspy")
datas += copy_metadata("rosettasciio")  # dist name for rsciio

hiddenimports += [
    "rsciio",
    "rsciio.tia",  
    "PyQt5.QtCore",
    "PyQt5.QtGui",
    "PyQt5.QtWidgets",
    "PyQt5.QtSvg",
    "PyQt5.QtOpenGL",
    "pyqtgraph.opengl",
]

def _dedupe(seq):
    seen, out = set(), []
    for x in seq:
        k = repr(x)
        if k not in seen:
            seen.add(k)
            out.append(x)
    return out

datas = _dedupe(datas)
binaries = _dedupe(binaries)
hiddenimports = _dedupe(hiddenimports)

a = Analysis(
    ["app.py"],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="TEMinator",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="TEMinator",
)

app = BUNDLE(
    coll,
    name="TEMinator.app",
    icon=None,
    bundle_identifier="org.teminator.app",
)