"""HandIdentity: stable hand labels across frames.

MediaPipe's Left/Right classification flickers frame to frame (seen in logs/run.log: the
primary hand "vanished" 25× in 2 s). Every flicker reset the smoothing filter and made
the FSM think the hand was gone, so the cursor jumped and drags were dropped.

Here a hand keeps the label of the nearest existing track; MediaPipe's label is only
used to name a genuinely new hand.
"""
from __future__ import annotations

from itertools import permutations

import numpy as np

from hands import MIDDLE_MCP, PALM_POINTS, WRIST, Frame, Hand

SIDES = ("Left", "Right")


def _center(h: Hand) -> np.ndarray:
    return h.landmarks[list(PALM_POINTS), :2].mean(axis=0)


def _size(h: Hand) -> float:
    return max(float(np.linalg.norm(h.landmarks[WRIST, :2] - h.landmarks[MIDDLE_MCP, :2])), 1e-3)


class HandIdentity:
    def __init__(self, max_age_s: float = 0.5, max_jump_hand_sizes: float = 2.5):
        self.max_age = max_age_s
        self.max_jump = max_jump_hand_sizes
        self._tracks: dict[str, tuple[np.ndarray, float, float]] = {}   # label → (center, size, t)

    def __call__(self, frame: Frame) -> Frame:
        t = frame.t
        self._tracks = {k: v for k, v in self._tracks.items() if t - v[2] <= self.max_age}
        hands = frame.hands[:2]
        labels: list[str | None] = [None] * len(hands)

        # match hands to live tracks, minimizing total distance (≤ 2×2, brute force is fine)
        live = list(self._tracks)
        best, best_cost = None, float("inf")
        for k in range(min(len(hands), len(live)), 0, -1):
            for hi in permutations(range(len(hands)), k):
                for tl in permutations(live, k):
                    cost, ok = 0.0, True
                    for i, lab in zip(hi, tl):
                        c, s, _ = self._tracks[lab]
                        d = float(np.linalg.norm(_center(hands[i]) - c)) / s
                        if d > self.max_jump:
                            ok = False
                            break
                        cost += d
                    if ok and cost < best_cost:
                        best, best_cost = list(zip(hi, tl)), cost
            if best:
                break
        for i, lab in best or []:
            labels[i] = lab

        # new hands: MediaPipe's label if free, otherwise the other one
        for i, h in enumerate(hands):
            if labels[i] is None:
                want = h.handedness if h.handedness in SIDES else "Right"
                labels[i] = want if want not in labels else next(s for s in SIDES if s not in labels)

        out = []
        for h, lab in zip(hands, labels):
            self._tracks[lab] = (_center(h), _size(h), t)
            out.append(Hand(lab, h.landmarks, h.score))
        return Frame(t, out)
