# FaceAttendance.spec
# PyInstaller spec for Face Attendance System
# Build:  pyinstaller FaceAttendance.spec --noconfirm
#
# Prerequisites:
#   pip install pyinstaller
#   python build/bundle_models.py   ← run first to pull in the AI models

import sys
import os
from PyInstaller.utils.hooks import collect_all, collect_data_files

block_cipher = None

# ── Collect dynamic packages (plugins, providers, submodules) ─────────────────
insight_datas, insight_bins, insight_hidden = collect_all("insightface")
ort_datas,     ort_bins,     ort_hidden     = collect_all("onnxruntime")
skimage_datas, skimage_bins, skimage_hidden = collect_all("skimage")
cv2_datas,     cv2_bins,     cv2_hidden     = collect_all("cv2")

# ── Build Analysis ────────────────────────────────────────────────────────────
a = Analysis(
    ["app.py"],
    pathex=["."],
    binaries=(
        insight_bins + ort_bins + skimage_bins + cv2_bins
    ),
    datas=(
        insight_datas + ort_datas + skimage_datas + cv2_datas
        + [
            # ── Bundled AI models (downloaded by build/bundle_models.py) ──
            ("insightface_models", "insightface_models"),
        ]
    ),
    hiddenimports=(
        insight_hidden + ort_hidden + skimage_hidden + cv2_hidden
        + [
            # Tkinter
            "tkinter", "tkinter.ttk", "tkinter.messagebox",
            "tkinter.simpledialog", "_tkinter",
            # Image
            "PIL", "PIL.Image", "PIL.ImageTk", "PIL.ImageDraw",
            # Numerics
            "numpy", "numpy.core", "numpy.core._multiarray_umath",
            "scipy", "scipy.spatial", "pandas",
            # DB
            "sqlite3", "_sqlite3",
            # ONNX providers
            "onnxruntime.capi._pybind_state",
            "onnxruntime.capi.onnxruntime_pybind11_state",
            # InsightFace internals
            "insightface.app", "insightface.model_zoo",
            "insightface.model_zoo.model_zoo",
            "insightface.utils", "insightface.utils.face_align",
            "insightface.utils.storage",
            # scikit-image (used by insightface.utils.face_align)
            "skimage.transform",
            "skimage._shared",
            # Other
            "packaging", "packaging.version",
        ]
    ),
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # Exclude large packages we don't use to keep binary size down
        "matplotlib", "IPython", "jupyter", "notebook",
        "tensorflow", "torch", "torchvision",
        "PyQt5", "PyQt6", "wx",
    ],
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

# ── Executable ────────────────────────────────────────────────────────────────
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="FaceAttendance",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,           # No black console window — GUI only
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon="build/app.ico",    # Windows taskbar / file icon
)

# ── Collect into one folder (onedir mode — faster startup than onefile) ───────
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="FaceAttendance",
)
