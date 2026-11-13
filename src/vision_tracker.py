"""Apple Vision hand-pose observations → Hand (the 21-point layout from hands.py).

Vision's origin is bottom-left; it is flipped to top-left here, once. z is unavailable (0).
The capture + Vision request itself lives in av_source.py.
"""
from __future__ import annotations

import numpy as np
import Vision as V

from hands import Hand

_J = "VNHumanHandPoseObservationJointName"
# MediaPipe index order: wrist, thumb CMC..tip, index MCP..tip, middle, ring, little
JOINTS = [getattr(V, _J + n) for n in (
    "Wrist",
    "ThumbCMC", "ThumbMP", "ThumbIP", "ThumbTip",
    "IndexMCP", "IndexPIP", "IndexDIP", "IndexTip",
    "MiddleMCP", "MiddlePIP", "MiddleDIP", "MiddleTip",
    "RingMCP", "RingPIP", "RingDIP", "RingTip",
    "LittleMCP", "LittlePIP", "LittleDIP", "LittleTip",
)]


def observation_to_hand(points: dict, chirality: int, confidence: float,
                        min_point_conf: float = 0.2) -> Hand | None:
    """points: joint name → object with .location() (x, y bottom-left) and .confidence()."""
    lm = np.zeros((21, 3))
    weak = 0
    for i, name in enumerate(JOINTS):
        p = points.get(name)
        if p is None:
            return None
        loc = p.location()
        lm[i, 0], lm[i, 1] = loc.x, 1.0 - loc.y   # flip to top-left origin
        weak += p.confidence() < min_point_conf
    if weak > 10:         # half the hand is guesswork: drop it (pinched fingers occlude joints)
        return None
    # Vision reports chirality as seen in the (mirrored) image, which is the opposite of the
    # user's own hand: a real right hand came back as "left" in every live frame.
    side = {1: "Left", -1: "Right"}.get(int(chirality), "Right")
    return Hand(side, lm, float(confidence))
