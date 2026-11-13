"""AVHandSource: native AVFoundation capture → Apple Vision hand pose, zero copies.

Why: with OpenCV's camera reader in the same process, Vision took 28 ms/frame (vs 4.5 ms
alone) and the app ran at ~23 fps. Here the camera hands CVPixelBuffers (native YUV)
straight to Vision on AVFoundation's own serial queue: no BGR conversion, no numpy copy,
no CGImage, and the main loop just picks up finished results.

Mirroring is done by Vision's orientation (UpMirrored), so coordinates come back already
mirrored like a selfie view, matching the rest of the pipeline.

read() → (hands, t, bgr_or_None). The BGR frame is only produced when want_image=True
(debug preview), because converting costs CPU.
"""
from __future__ import annotations

import threading
import time

import AVFoundation as AV
import CoreMedia
import libdispatch
import numpy as np
import objc
import Quartz as Q
import Vision as V
from Foundation import NSObject

from capture import resolve_camera
from vision_tracker import observation_to_hand

ORIENT_MIRRORED = 2   # kCGImagePropertyOrientationUpMirrored


def camera_permission() -> str:
    """"granted", "denied" (or restricted) or "unknown" (never asked)."""
    st = AV.AVCaptureDevice.authorizationStatusForMediaType_(AV.AVMediaTypeVideo)
    return {AV.AVAuthorizationStatusAuthorized: "granted",
            AV.AVAuthorizationStatusNotDetermined: "unknown"}.get(st, "denied")


def request_camera_permission(timeout: float = 120.0) -> bool:
    """Shows the system Camera prompt if it was never answered; blocks until answered."""
    if camera_permission() != "unknown":
        return camera_permission() == "granted"
    done = threading.Event()
    AV.AVCaptureDevice.requestAccessForMediaType_completionHandler_(
        AV.AVMediaTypeVideo, lambda granted: done.set())
    done.wait(timeout)
    return camera_permission() == "granted"


def _bgr_from_pixelbuffer(pb) -> np.ndarray | None:
    """Only for the debug preview: NV12 (420f) → BGR."""
    import cv2
    Q.CVPixelBufferLockBaseAddress(pb, 1)
    try:
        w, h = Q.CVPixelBufferGetWidth(pb), Q.CVPixelBufferGetHeight(pb)
        planes = []
        for i in (0, 1):
            base = Q.CVPixelBufferGetBaseAddressOfPlane(pb, i)
            stride = Q.CVPixelBufferGetBytesPerRowOfPlane(pb, i)
            ph = Q.CVPixelBufferGetHeightOfPlane(pb, i)
            buf = base.as_buffer(stride * ph)
            planes.append(np.frombuffer(buf, np.uint8).reshape(ph, stride).copy())
        y = planes[0][:, :w]
        uv = planes[1][:, :w]
        nv12 = np.vstack([y, uv])
        bgr = cv2.cvtColor(nv12, cv2.COLOR_YUV2BGR_NV12)
        return cv2.flip(bgr, 1)
    finally:
        Q.CVPixelBufferUnlockBaseAddress(pb, 1)


class _Delegate(NSObject):
    def initWithSource_(self, src):
        self = objc.super(_Delegate, self).init()
        if self is None:
            return None
        self.src = src
        return self

    def captureOutput_didOutputSampleBuffer_fromConnection_(self, output, sbuf, conn):
        # an exception escaping into AVFoundation's queue would abort the whole process
        try:
            self.src._on_frame(sbuf)
        except Exception as e:
            self.src.errors += 1
            self.src.last_error = repr(e)

    def captureOutput_didDropSampleBuffer_fromConnection_(self, output, sbuf, conn):
        self.src.dropped += 1


class AVHandSource:
    def __init__(self, device: str | int = "builtin", width: int = 640, height: int = 480,
                 fps: int = 30, max_hands: int = 2, min_confidence: float = 0.5):
        self.info = resolve_camera(device)
        dev = next((d for d in AV.AVCaptureDevice.devicesWithMediaType_(AV.AVMediaTypeVideo)
                    if str(d.uniqueID()) == self.info.unique_id), None)
        if dev is None:
            raise RuntimeError(f"camera {self.info.name} disappeared")
        self.min_confidence = min_confidence
        self.want_image = False
        self.dropped = 0
        self.errors = 0
        self.last_error: str | None = None
        self.running = False
        self.track_ms: list[float] = []
        self.convert_ms: list[float] = []   # part of track_ms spent turning results into Python

        self._req = V.VNDetectHumanHandPoseRequest.alloc().init()
        self._req.setMaximumHandCount_(max_hands)
        self._cond = threading.Condition()
        self._latest = None
        self._t0 = time.monotonic()

        self._session = AV.AVCaptureSession.alloc().init()
        preset = AV.AVCaptureSessionPreset640x480 if width <= 640 else AV.AVCaptureSessionPreset1280x720
        if self._session.canSetSessionPreset_(preset):
            self._session.setSessionPreset_(preset)
        inp, err = AV.AVCaptureDeviceInput.deviceInputWithDevice_error_(dev, None)
        if inp is None:
            raise RuntimeError(f"Could not open the camera {self.info.name!r} ({err}). "
                               "Check System Settings → Privacy & Security → Camera.")
        self._session.addInput_(inp)
        out = AV.AVCaptureVideoDataOutput.alloc().init()
        out.setAlwaysDiscardsLateVideoFrames_(True)   # never queue stale frames
        out.setVideoSettings_({Q.kCVPixelBufferPixelFormatTypeKey: Q.kCVPixelFormatType_420YpCbCr8BiPlanarFullRange})
        self._queue = libdispatch.dispatch_queue_create(b"gesture.capture", None)
        self._delegate = _Delegate.alloc().initWithSource_(self)
        out.setSampleBufferDelegate_queue_(self._delegate, self._queue)
        self._session.addOutput_(out)
        try:   # ask for the configured frame rate
            dev.lockForConfiguration_(None)
            dev.setActiveVideoMinFrameDuration_(CoreMedia.CMTimeMake(1, fps))
            dev.setActiveVideoMaxFrameDuration_(CoreMedia.CMTimeMake(1, fps))
            dev.unlockForConfiguration()
        except Exception:
            pass
        self.resume()

    @property
    def resolution(self) -> tuple[int, int]:
        return (640, 480)

    # ----- capture queue ----------------------------------------------------------------------

    def _on_frame(self, sbuf) -> None:
        t = time.monotonic()
        pb = CoreMedia.CMSampleBufferGetImageBuffer(sbuf)
        if pb is None:
            return
        a = time.perf_counter()
        handler = V.VNImageRequestHandler.alloc().initWithCVPixelBuffer_orientation_options_(
            pb, ORIENT_MIRRORED, None)
        ok, err = handler.performRequests_error_([self._req], None)
        b = time.perf_counter()
        hands = []
        if ok:
            for obs in self._req.results() or []:
                if obs.confidence() < self.min_confidence:
                    continue
                pts, err = obs.recognizedPointsForJointsGroupName_error_(
                    V.VNHumanHandPoseObservationJointsGroupNameAll, None)
                if pts is not None:
                    h = observation_to_hand(dict(pts), obs.chirality(), obs.confidence())
                    if h is not None:
                        hands.append(h)
        end = time.perf_counter()
        self.track_ms.append((end - a) * 1000)
        self.convert_ms.append((end - b) * 1000)
        if len(self.track_ms) > 120:
            del self.track_ms[:60]
            del self.convert_ms[:60]
        img = _bgr_from_pixelbuffer(pb) if self.want_image else None
        with self._cond:
            self._latest = (hands, t, img)
            self._cond.notify()

    # ----- consumer ---------------------------------------------------------------------------

    def read(self, timeout: float = 1.0):
        with self._cond:
            if self._latest is None:
                self._cond.wait(timeout)
            got, self._latest = self._latest, None
            return got

    def pause(self) -> None:
        """Stop the camera (its light goes off, no CPU used). resume() restarts it."""
        if self.running:
            self._session.stopRunning()
            self.running = False
            with self._cond:
                self._latest = None

    def resume(self) -> None:
        if not self.running:
            self._session.startRunning()
            self.running = True

    def close(self) -> None:
        self.pause()
