"""Where things live, for both a source checkout and the packaged Gesture Control.app.

Inside the .app the bundle is read-only (writing into it would also break its code
signature), so logs and settings go to the standard per-user folders instead.
"""
from __future__ import annotations

import sys
from pathlib import Path

APP_NAME = "Gesture Control"
FROZEN = bool(getattr(sys, "frozen", False))

# Read-only resources (config/default.yaml, ui/settings.html)
if FROZEN:
    RESOURCES = Path(getattr(sys, "_MEIPASS", Path(sys.executable).resolve().parent))
    REPO = None
else:
    RESOURCES = Path(__file__).resolve().parent.parent
    REPO = RESOURCES

DEFAULT_CONFIG = RESOURCES / "config" / "default.yaml"
SETTINGS_HTML = (RESOURCES / "ui" / "settings.html") if FROZEN else (Path(__file__).resolve().parent / "ui" / "settings.html")

# Per-user, writable
SUPPORT_DIR = Path.home() / "Library" / "Application Support" / APP_NAME
USER_SETTINGS = SUPPORT_DIR / "settings.yaml"
LOG_DIR = (Path.home() / "Library" / "Logs" / APP_NAME) if FROZEN else REPO / "logs"
RECORDINGS_DIR = (SUPPORT_DIR / "recordings") if FROZEN else REPO / "recordings"
LOG_MAX_BYTES = 2_000_000


def open_log():
    """Append-mode run log, rotated to run.log.1 once it passes LOG_MAX_BYTES."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    path = LOG_DIR / "run.log"
    try:
        if path.exists() and path.stat().st_size > LOG_MAX_BYTES:
            path.replace(LOG_DIR / "run.log.1")
    except OSError:
        pass
    return path.open("a", buffering=1, encoding="utf-8")
