import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hands import Hand  # noqa: E402


def make_hand(pose: str = "open", side: str = "Right", center=(0.5, 0.5), scale: float = 1.0) -> Hand:
    """Synthetic 21-point hand, upright, palm facing camera. Unit hand_size = 0.1 * scale."""
    s = 0.1 * scale
    wrist = np.array([0.0, 0.0])
    mcps = {5: (-0.35, -0.95), 9: (0.0, -1.0), 13: (0.3, -0.95), 17: (0.55, -0.85)}
    pts = np.zeros((21, 2))
    pts[0] = wrist
    for mcp, (x, y) in mcps.items():
        base = np.array([x, y])
        pts[mcp] = base
        if pose == "fist":
            tip = base * 0.35 + np.array([0.1, -0.6]) * 0.65              # folded into palm
            seg = [base + (0, 0.25), (base + tip) / 2, tip]
        else:
            seg = [base + (0, -0.4 * k) for k in (1, 2, 3)]                  # straight up
        if pose == "point" and mcp != 5:
            seg = [base + (0, 0.25), base + (0, 0.45), base + (0, 0.55)]
        pts[mcp + 1: mcp + 4] = seg
    # thumb
    if pose == "pinch":
        pts[1:5] = [(-0.3, -0.3), (-0.55, -0.6), (-0.6, -1.0), (-0.35, -2.1)]  # touches index tip
    elif pose == "loose":   # fingers just parting at the end of a slow tap (between pinch_on/off)
        pts[1:5] = [(-0.3, -0.3), (-0.55, -0.6), (-0.6, -1.0), (-0.35, -1.91)]
    elif pose == "fist":
        pts[1:5] = [(-0.3, -0.3), (-0.45, -0.55), (0.0, -0.75), (0.35, -0.7)]   # thumb across the fingers
    else:
        pts[1:5] = [(-0.3, -0.3), (-0.6, -0.5), (-0.9, -0.7), (-1.1, -0.9)]
    lm = np.zeros((21, 3))
    lm[:, :2] = pts * s + np.asarray(center)
    return Hand(side, lm, 0.95)


@pytest.fixture
def hand_factory():
    return make_hand
