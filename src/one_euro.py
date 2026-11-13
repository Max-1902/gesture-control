"""One Euro filter (Casiez et al., 2012): adaptive low-pass.

Slow motion → low cutoff → heavy smoothing (kills jitter at rest).
Fast motion → cutoff rises with speed → little lag.
Works elementwise on numpy arrays, so one instance filters all 21x3 coordinates.
"""
from __future__ import annotations

import math

import numpy as np

from hands import Frame, Hand


class OneEuro:
    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.02, d_cutoff: float = 1.0):
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self.reset()

    def reset(self) -> None:
        self.x_prev = self.dx_prev = self.t_prev = None

    @staticmethod
    def _alpha(cutoff, dt: float):
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x, t: float):
        x = np.asarray(x, dtype=np.float64)
        if self.t_prev is None:
            self.x_prev, self.dx_prev, self.t_prev = x, np.zeros_like(x), t
            return x
        dt = max(t - self.t_prev, 1e-6)
        dx = (x - self.x_prev) / dt
        a_d = self._alpha(self.d_cutoff, dt)
        dx_hat = a_d * dx + (1.0 - a_d) * self.dx_prev
        cutoff = self.min_cutoff + self.beta * np.abs(dx_hat)
        a = self._alpha(cutoff, dt)
        x_hat = a * x + (1.0 - a) * self.x_prev
        self.x_prev, self.dx_prev, self.t_prev = x_hat, dx_hat, t
        return x_hat


class LandmarkFilter:
    """One filter bank per hand side. A side's filter resets when that hand disappears,
    so a hand re-entering the frame doesn't get smeared from its old position."""

    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.02, d_cutoff: float = 1.0):
        self._params = (min_cutoff, beta, d_cutoff)
        self._filters: dict[str, OneEuro] = {}

    def __call__(self, frame: Frame) -> Frame:
        seen = set()
        out = []
        for h in frame.hands:
            seen.add(h.handedness)
            f = self._filters.setdefault(h.handedness, OneEuro(*self._params))
            out.append(Hand(h.handedness, f(h.landmarks, frame.t), h.score))
        for side in list(self._filters):
            if side not in seen:
                del self._filters[side]
        return Frame(frame.t, out)
