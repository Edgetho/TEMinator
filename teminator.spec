# -*- mode: python ; coding: utf-8 -*-

from PyInstaller.building.build_main import Analysis, PYZ, EXE, COLLECT
from PyInstaller.building.osx import BUNDLE


a = Analysis(
    ['app.py'],
    pathex=[],
    binaries=[],
    datas=[('app_icon.png', '.')],
    hiddenimports=[],
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
    name='TEMinator',
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
    name='TEMinator',
)

app = BUNDLE(
    coll,
    name='TEMinator.app',
    icon=None,
    bundle_identifier='org.teminator.app',
    info_plist={
        'CFBundleName': 'TEMinator',
        'CFBundleDisplayName': 'TEMinator',
        'CFBundleExecutable': 'TEMinator',
    },
)
