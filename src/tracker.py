"""Dev-only fallback tracker (--tracker mediapipe): MediaPipe Tasks HandLandmarker → list[Hand].

Uses VIDEO running mode (tracks between frames, much cheaper than re-detecting).
Input must already be mirrored: MediaPipe's handedness labels assume a selfie
(mirrored) image, so after the flip in CameraSource "Right" means your right hand.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

from hands import Hand

DEFAULT_MODEL = Path(__file__).resolve().parent.parent / "models" / "hand_landmarker.task"
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
             "hand_landmarker/float16/latest/hand_landmarker.task")

os.environ.setdefault("GLOG_minloglevel", "2")  # silence MediaPipe INFO spam


class HandTracker:
    def __init__(self, model_path: Path = DEFAULT_MODEL, max_hands: int = 2,
                 min_confidence: float = 0.6):
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Missing model {model_path}. Download it with:\n"
                                    f"  curl -L -o {model_path} {MODEL_URL}")
        opts = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=max_hands,
            min_hand_detection_confidence=min_confidence,
            min_hand_presence_confidence=min_confidence,
            min_tracking_confidence=0.5,
        )
        self._lm = vision.HandLandmarker.create_from_options(opts)
        self.min_confidence = min_confidence
        self._last_ms = -1

    def process(self, bgr: np.ndarray, t: float) -> list[Hand]:
        # VIDEO mode requires strictly increasing integer-ms timestamps
        ms = max(int(t * 1000), self._last_ms + 1)
        self._last_ms = ms
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        result = self._lm.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ms)

        hands: list[Hand] = []
        for lms, handed in zip(result.hand_landmarks, result.handedness):
            cat = handed[0]
            # NOTE: cat.score is LEFT/RIGHT certainty, not presence; it drops whenever the label
            # flickers. Presence is already gated by min_hand_presence_confidence above.
            pts = np.array([[p.x, p.y, p.z] for p in lms], dtype=np.float64)
            hands.append(Hand(cat.category_name, pts, cat.score))

        # Labels may flicker or collide; identity.HandIdentity makes them stable downstream.
        return hands

    def close(self) -> None:
        self._lm.close()


class CvSource:
    """OpenCV camera + MediaPipe, same read() interface as av_source.AVHandSource (dev fallback)."""

    def __init__(self, cam_cfg: dict, tr_cfg: dict):
        from capture import CameraSource
        self.cam = CameraSource(cam_cfg["device"], cam_cfg["width"], cam_cfg["height"],
                                cam_cfg["fps"], cam_cfg["mirror"])
        self.tracker = HandTracker(max_hands=tr_cfg["max_hands"], min_confidence=tr_cfg["min_confidence"])
        self.info, self.want_image, self.track_ms = self.cam.info, True, []
        self.running = True

    @property
    def resolution(self):
        return self.cam.resolution

    @property
    def dropped(self):
        return self.cam.dropped

    def read(self, timeout: float = 1.0):
        got = self.cam.read(timeout)
        if got is None:
            return None
        img, t = got
        a = time.perf_counter()
        hands = self.tracker.process(img, t)
        self.track_ms = (self.track_ms + [(time.perf_counter() - a) * 1000])[-120:]
        return hands, t, img

    def pause(self) -> None:
        pass

    def resume(self) -> None:
        pass

    def close(self):
        self.cam.close()
        self.tracker.close()
