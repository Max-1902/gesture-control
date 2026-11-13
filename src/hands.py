"""Core data types shared by every pipeline stage.

Coordinate convention (normalized once, here, at the boundary):
    x, y in [0, 1], origin top-left of the *mirrored* camera image, y grows down.
    z is MediaPipe's relative depth (smaller = closer to camera), roughly x-scaled.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# MediaPipe hand landmark indices
WRIST = 0
THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP = 1, 2, 3, 4
INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP = 5, 6, 7, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP = 9, 10, 11, 12
RING_MCP, RING_PIP, RING_DIP, RING_TIP = 13, 14, 15, 16
PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP = 17, 18, 19, 20

FINGERTIPS = (THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP)
PALM_POINTS = (WRIST, INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP)

# Skeleton edges, for drawing
CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),
)


@dataclass
class Hand:
    handedness: str              # "Left" | "Right" (from the user's point of view)
    landmarks: np.ndarray        # shape (21, 3), float64
    score: float                 # handedness/presence confidence, 0..1

    def to_json(self) -> dict:
        return {
            "handedness": self.handedness,
            "score": round(float(self.score), 4),
            "lm": np.round(self.landmarks, 5).tolist(),
        }

    @staticmethod
    def from_json(d: dict) -> "Hand":
        return Hand(d["handedness"], np.asarray(d["lm"], dtype=np.float64), float(d["score"]))


@dataclass
class Frame:
    t: float                     # seconds, monotonic, relative to session start
    hands: list[Hand] = field(default_factory=list)

    def hand(self, handedness: str) -> Hand | None:
        return next((h for h in self.hands if h.handedness == handedness), None)

    def to_json(self) -> dict:
        return {"t": round(self.t, 5), "hands": [h.to_json() for h in self.hands]}

    @staticmethod
    def from_json(d: dict) -> "Frame":
        return Frame(float(d["t"]), [Hand.from_json(h) for h in d["hands"]])
