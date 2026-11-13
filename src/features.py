"""FeatureExtractor: smoothed landmarks → scale-invariant scalars.

Everything is divided by hand_size (wrist → middle MCP), so values don't change
with distance from the camera. Uses x,y only: MediaPipe's z is noisy and its
scale differs from x,y, which would make the ratios distance-dependent again.

Thresholds are NOT decided here. This module only measures; Phase 2 picks the
thresholds from recorded data (see replay.py --csv).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from hands import FINGERTIPS, INDEX_MCP, INDEX_TIP, MIDDLE_MCP, PALM_POINTS, THUMB_TIP, WRIST, Frame, Hand


@dataclass
class HandFeatures:
    handedness: str
    hand_size: float          # normalized image units; also a proxy for distance to camera
    pinch: float              # thumb tip ↔ index tip / hand_size (small = pinched)
    openness: float           # mean fingertip ↔ palm center / hand_size (big = open palm)
    curl: float               # max fingertip ↔ palm center / hand_size (small = fist)
    index_extension: float    # index tip ↔ palm / hand_size
    others_extension: float   # mean(middle, ring, pinky) tip ↔ palm / hand_size
    min_extension: float      # LEAST extended of the 4 fingers (open palm needs all of them out)
    upright: float            # (wrist_y - mean fingertip_y) / hand_size; >0 = fingers point up
    pointer_x: float          # index fingertip (precise but jittery, moves when pinching)
    pointer_y: float
    knuckle_x: float          # index MCP: steady, doesn't move on pinch/fist → cursor anchor
    knuckle_y: float
    center_x: float
    center_y: float
    edge_margin: float = 1.0  # distance of the closest landmark to the frame edge (<0.015 = cut off)
    edge_side: str = ""       # which edge is closest: "bottom", "top", "left", "right"


@dataclass
class FrameFeatures:
    t: float
    hands: dict[str, HandFeatures]
    two_hand_span: float | None   # center ↔ center distance, in units of mean hand size

    def flat(self) -> dict:
        """One flat row for CSV export / plotting."""
        row: dict = {"t": self.t, "n_hands": len(self.hands), "two_hand_span": self.two_hand_span}
        for side in ("Left", "Right"):
            hf = self.hands.get(side)
            for k, v in (asdict(hf).items() if hf else
                         ((k, None) for k in HandFeatures.__dataclass_fields__)):
                if k != "handedness":
                    row[f"{side[0]}_{k}"] = v
        return row


def _d(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a[:2] - b[:2]))


def hand_features(hand: Hand) -> HandFeatures:
    p = hand.landmarks
    size = max(_d(p[WRIST], p[MIDDLE_MCP]), 1e-6)
    center = p[list(PALM_POINTS), :2].mean(axis=0)

    tip_dist = np.array([np.linalg.norm(p[i, :2] - center) for i in FINGERTIPS]) / size
    return HandFeatures(
        handedness=hand.handedness,
        hand_size=size,
        pinch=_d(p[THUMB_TIP], p[INDEX_TIP]) / size,
        openness=float(tip_dist.mean()),
        curl=float(tip_dist[1:].max()),   # ignore thumb: it sits near the palm even when open
        index_extension=float(tip_dist[1]),
        others_extension=float(tip_dist[2:].mean()),
        min_extension=float(tip_dist[1:].min()),
        upright=float((p[WRIST, 1] - p[list(FINGERTIPS[1:]), 1].mean()) / size),
        pointer_x=float(p[INDEX_TIP, 0]),
        pointer_y=float(p[INDEX_TIP, 1]),
        knuckle_x=float(p[INDEX_MCP, 0]),
        knuckle_y=float(p[INDEX_MCP, 1]),
        center_x=float(center[0]),
        center_y=float(center[1]),
        **_edge(p),
    )


def _edge(p: np.ndarray) -> dict:
    # gesture points only: a raised palm's wrist (0) and thumb base (1) sit at the bottom edge
    # even in good runs, and they don't matter for pinch/fist
    xs, ys = p[2:, 0], p[2:, 1]
    margins = {"left": xs.min(), "right": 1 - xs.max(), "top": ys.min(), "bottom": 1 - ys.max()}
    side = min(margins, key=margins.get)
    return {"edge_margin": float(margins[side]), "edge_side": side}


def extract(frame: Frame) -> FrameFeatures:
    hands = {h.handedness: hand_features(h) for h in frame.hands}
    span = None
    if "Left" in hands and "Right" in hands:
        l, r = hands["Left"], hands["Right"]
        mean_size = (l.hand_size + r.hand_size) / 2
        span = float(np.hypot(l.center_x - r.center_x, l.center_y - r.center_y) / mean_size)
    return FrameFeatures(frame.t, hands, span)
