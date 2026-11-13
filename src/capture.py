"""Camera discovery, plus CameraSource: an OpenCV reader for the dev-only MediaPipe fallback.

The app itself captures through av_source.AVHandSource and never needs OpenCV.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass

import numpy as np


@dataclass
class CameraInfo:
    cv_index: int       # index to pass to cv2.VideoCapture
    name: str
    device_type: str
    unique_id: str

    @property
    def builtin(self) -> bool:
        return "BuiltIn" in self.device_type

    @property
    def is_phone(self) -> bool:
        return "continuity" in self.device_type.lower() or "iphone" in self.name.lower()


def list_cameras() -> list[CameraInfo]:
    """Cameras in the order OpenCV's AVFoundation backend numbers them.

    OpenCV takes [AVCaptureDevice devicesWithMediaType:] and sorts by uniqueID, so
    index 0 is NOT necessarily the Mac's own camera: with Continuity Camera, an
    iPhone whose ID sorts first becomes 0. We mirror that sort to map names → indices.
    """
    import AVFoundation as AV
    devs = list(AV.AVCaptureDevice.devicesWithMediaType_(AV.AVMediaTypeVideo) or [])
    devs += list(AV.AVCaptureDevice.devicesWithMediaType_(AV.AVMediaTypeMuxed) or [])
    devs.sort(key=lambda d: str(d.uniqueID()))
    return [CameraInfo(i, str(d.localizedName()), str(d.deviceType()), str(d.uniqueID()))
            for i, d in enumerate(devs)]


def resolve_camera(device: str | int | None = "builtin") -> CameraInfo:
    """device: "builtin" (the Mac's own camera), a name substring, or an OpenCV index.

    Never fails while any camera exists: desktop Macs have no built-in camera (→ first
    non-iPhone camera), and a camera picked in Settings may since have been unplugged.
    """
    cams = list_cameras()
    if not cams:
        raise RuntimeError("No camera found. Connect a camera and start Gesture Control again.")
    if device in (None, "", "builtin"):
        match = [c for c in cams if c.builtin] or [c for c in cams if not c.is_phone] or cams
    elif isinstance(device, int) or str(device).isdigit():
        match = [c for c in cams if c.cv_index == int(device)]
    else:
        match = [c for c in cams if str(device).lower() in c.name.lower()]
    return match[0] if match else resolve_camera("builtin")


class CameraSource:
    def __init__(self, device: str | int = "builtin", width: int = 640, height: int = 480,
                 fps: int = 30, mirror: bool = True):
        import cv2   # optional: only the MediaPipe fallback reads the camera through OpenCV
        self._cv2 = cv2
        self.mirror = mirror
        self.info = resolve_camera(device)
        self._cap = cv2.VideoCapture(self.info.cv_index, cv2.CAP_AVFOUNDATION)
        if not self._cap.isOpened():
            raise RuntimeError(
                "Could not open camera. On macOS, grant Camera access to your terminal/IDE "
                "in System Settings → Privacy & Security → Camera, then restart it."
            )
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self._cap.set(cv2.CAP_PROP_FPS, fps)
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self._cond = threading.Condition()
        self._frame: np.ndarray | None = None
        self._t: float = 0.0
        self.dropped = 0
        self._running = True
        self._thread = threading.Thread(target=self._loop, name="capture", daemon=True)
        self._thread.start()

    @property
    def resolution(self) -> tuple[int, int]:
        cv2 = self._cv2
        return (int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    def _loop(self) -> None:
        while self._running:
            ok, frame = self._cap.read()
            t = time.monotonic()
            if not ok:
                time.sleep(0.005)
                continue
            if self.mirror:
                frame = self._cv2.flip(frame, 1)
            with self._cond:
                if self._frame is not None:
                    self.dropped += 1   # previous frame was never consumed
                self._frame, self._t = frame, t
                self._cond.notify()

    def read(self, timeout: float = 1.0) -> tuple[np.ndarray, float] | None:
        """Block until a frame newer than the last one read is available. Returns (bgr, t)."""
        with self._cond:
            if self._frame is None and not self._cond.wait_for(
                    lambda: self._frame is not None or not self._running, timeout):
                return None
            if self._frame is None:
                return None
            frame, t, self._frame = self._frame, self._t, None
            return frame, t

    def close(self) -> None:
        self._running = False
        with self._cond:
            self._cond.notify_all()
        self._thread.join(timeout=1.0)
        self._cap.release()
