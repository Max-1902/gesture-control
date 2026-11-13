# PyInstaller spec for Gesture Control.app. Built by packaging/build_app.sh, not run directly.
import os

VERSION = os.environ["GC_VERSION"]   # set by build_app.sh from src/version.py

ROOT = os.path.abspath(os.path.join(SPECPATH, ".."))
SRC = os.path.join(ROOT, "src")

a = Analysis(
    [os.path.join(SRC, "run.py")],
    pathex=[SRC],
    datas=[(os.path.join(ROOT, "config", "default.yaml"), "config"),
           (os.path.join(SRC, "ui", "settings.html"), "ui")],
    hiddenimports=["overlay", "router", "av_source", "window_ctrl", "capture", "vision_tracker",
                   "AVFoundation", "CoreMedia", "Vision", "Quartz", "ApplicationServices",
                   "libdispatch", "WebKit", "ServiceManagement", "AppKit", "Foundation"],
    excludes=["cv2", "mediapipe", "debug_view", "tracker", "tkinter", "matplotlib", "PIL",
              "scipy", "pytest", "IPython", "setuptools", "pip"],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name="Gesture Control",
          console=False, argv_emulation=False, upx=False)
coll = COLLECT(exe, a.binaries, a.datas, upx=False, name="Gesture Control")
app = BUNDLE(
    coll,
    name="Gesture Control.app",
    icon=os.path.join(SPECPATH, "AppIcon.icns"),
    bundle_identifier="io.github.max-1902.gesture-control",
    version=VERSION,
    info_plist={
        "CFBundleDisplayName": "Gesture Control",
        "CFBundleShortVersionString": VERSION,
        "CFBundleVersion": VERSION,
        "LSUIElement": True,   # menu bar app: no Dock icon
        "LSMinimumSystemVersion": "12.0",
        "NSHighResolutionCapable": True,
        "NSCameraUsageDescription": "Gesture Control watches your hand to move the pointer and windows. "
                                    "The video stays on your Mac and is never saved or sent anywhere.",
        "NSHumanReadableCopyright": "MIT License",
    },
)
