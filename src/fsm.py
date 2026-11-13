"""GestureFSM: features → state → intents. Pure and deterministic (time comes from frames).

    IDLE ──open palm, still, 0.5 s──▶ HOVER (armed; cursor follows index tip)
    HOVER ──pinch──▶ PRESS ──released quickly──▶ Click (2nd tap soon after → double click)
                     PRESS ──held / moved──▶ GRABBED ──release──▶ HOVER   (Grab / Move / Release)
    HOVER|GRABBED ──both hands pinch──▶ ZOOM ──either releases──▶ HOVER   (Resize*)
    HOVER ──fist held 1 s──▶ Minimize            (never while GRABBED: see fist_curl_on)
    armed, hand gone ──▶ ARMED ──back──▶ HOVER;  gone > 3 s ──▶ IDLE
    armed, no action for 8 s ──▶ IDLE   (you went back to typing)

Safety invariants (tested):
  * Nothing but the engage gesture leaves IDLE.
  * Every Grab is eventually followed by Release; every ResizeStart by ResizeEnd/Cancel,
    including when a hand vanishes or the kill switch fires. Never stuck mid-drag.

Arm rule (see config/default.yaml) was fitted on a real palm test vs two typing sessions:
open palm + raised toward the camera (hand_size) + still. Shape alone was not enough:
typing produced palm-shaped runs of >1 s.
"""
from __future__ import annotations

from collections import deque
from dataclasses import replace
from enum import Enum

import numpy as np

from features import FrameFeatures, HandFeatures
from intents import (Click, Grab, Intent, Minimize, Move, Release, Resize, ResizeCancel, ResizeEnd,
                     ResizeStart)


class State(Enum):
    IDLE = "idle"
    ARMED = "armed"        # armed, but the primary hand is out of view
    HOVER = "hover"
    PRESS = "press"        # fingers pinched, not yet decided: tap (click) or hold (drag)
    GRABBED = "grabbed"
    ZOOM = "zoom"


class Debounced:
    """Boolean that only flips after `n` consecutive frames disagreeing with it."""

    def __init__(self, n: int):
        self.n, self.value, self._count = n, False, 0

    def update(self, raw: bool) -> bool:
        if raw == self.value:
            self._count = 0
        else:
            self._count += 1
            if self._count >= self.n:
                self.value, self._count = raw, 0
        return self.value

    def reset(self) -> None:
        self.value, self._count = False, 0


class _Side:
    """Per-hand tracking state."""

    def __init__(self, debounce: int, pinch_debounce: int | None = None):
        self.pinch = Debounced(pinch_debounce or debounce)
        self.fist = Debounced(debounce)
        self.prev_center: tuple[float, float, float] | None = None  # x, y, t
        self.speed = float("inf")
        self.open_since: float | None = None
        self.last_seen = -1e9

    def reset(self) -> None:
        self.pinch.reset()
        self.fist.reset()
        self.prev_center = None
        self.speed = float("inf")
        self.open_since = None


class GestureFSM:
    def __init__(self, cfg: dict):
        g = cfg["gestures"]
        self.g = g
        self.debounce = int(g["debounce_frames"])
        self.state = State.IDLE
        self.enabled = True
        self.primary: str | None = None
        pd = int(g.get("pinch_debounce_frames", self.debounce))
        self.sides = {s: _Side(self.debounce, pd) for s in ("Left", "Right")}

        # HUD-facing outputs
        self.cursor: tuple[float, float] | None = None
        self.arm_progress = 0.0
        self.dwell_progress = 0.0

        self._history: deque[tuple[float, float]] = deque(maxlen=self.debounce + 4)
        self._last_seen = 0.0
        self._last_activity = 0.0
        self._grab_center: tuple[float, float] | None = None
        self._zoom_span0 = 1.0
        self._fist_since: float | None = None
        self._fist_point: tuple[float, float] | None = None
        self._need_release = False   # after zoom/minimize, wait for pinch to open before re-grab
        self._size_ref: float | None = None   # running primary hand size, to reject phantom hands
        self._last_edge = False   # primary hand was near the frame edge when last seen
        self.hint: str | None = None   # HUD hint: "low" / "edge" when the hand is leaving the camera view
        self._press_since: float | None = None
        self._press_center: tuple[float, float] | None = None
        self._last_click_t = -1e9
        self._last_click_count = 0

    # ----- public -----------------------------------------------------------------------------

    def set_enabled(self, on: bool) -> list[Intent]:
        """Kill switch. Turning off cleanly ends any drag/resize."""
        out = self._abort() if not on else []
        self.enabled = on
        if not on:
            self._disarm()
        return out

    def step(self, ff: FrameFeatures) -> list[Intent]:
        if not self.enabled:
            return []
        t = ff.t
        self._update_hint(ff)
        if self.state != State.IDLE and self.primary in ff.hands:
            h = ff.hands[self.primary]
            ref = self._size_ref
            if ref is not None and not (0.55 * ref <= h.hand_size <= 1.8 * ref):
                # phantom detection (live: a "hand" 1/4 the size pinched and grabbed a
                # full-screen app): treat as a dropout, never let it touch pinch state
                ff = replace(ff, hands={k: v for k, v in ff.hands.items() if k != self.primary})
            else:
                self._size_ref = h.hand_size if ref is None else 0.9 * ref + 0.1 * h.hand_size
        for side, s in self.sides.items():
            h = ff.hands.get(side)
            if h is None:
                if t - s.last_seen > self._grace():
                    s.reset()        # only forget pinch/fist state after a real loss
                s.prev_center = None
            else:
                s.last_seen = t
                self._update_side(s, h, t)

        if self.state == State.IDLE:
            self._step_idle(ff, t)
            return []
        return self._step_armed(ff, t)

    # ----- per-hand features ------------------------------------------------------------------

    def _update_side(self, s: _Side, h: HandFeatures, t: float) -> None:
        if s.prev_center is not None:
            px, py, pt = s.prev_center
            s.speed = np.hypot(h.center_x - px, h.center_y - py) / h.hand_size / max(t - pt, 1e-3)
        s.prev_center = (h.center_x, h.center_y, t)

        g = self.g
        if h.edge_margin < g.get("edge_clip_margin", 0.015):
            return   # hand cut off by the frame edge: its pinch/fist readings are garbage, freeze them
        # hysteresis: a curled pinch hand sits at curl ≈ 0.7-0.85, a real fist ≈ 0.3-0.6
        fist_raw = h.curl < (g["fist_curl_off"] if s.fist.value else g["fist_curl_on"])
        # a tight curled PINCH also has low curl (live: curl 0.24 with pinch 0.15 → false minimize);
        # a real fist has the thumb across the fingers, well away from the index tip (≥ 0.45)
        fist_raw = fist_raw and not s.pinch.value and h.pinch > g.get("fist_min_thumb_gap", 0.35)
        s.fist.update(fist_raw)
        if s.pinch.value:
            pinch_raw = h.pinch < g["pinch_off"]
        else:
            pinch_raw = h.pinch < g["pinch_on"] and not fist_raw  # a (tight) fist is not a pinch
        s.pinch.update(pinch_raw)

    def _is_open_palm(self, s: _Side, h: HandFeatures) -> bool:
        g = self.g
        return (h.edge_margin >= g.get("edge_warn_margin", 0.05)   # start with the whole hand in view
                and h.min_extension >= g["engage_min_extension"]
                and h.openness >= g["engage_min_openness"]
                and h.hand_size >= g["engage_min_hand_size"]
                and h.pinch >= g["engage_thumb_away"]
                and s.speed <= g["engage_max_speed"])

    def _grace(self) -> float:
        """A hand that slid out at the frame edge usually comes right back: wait longer."""
        g = self.g
        return g["lost_grace_s"] * (g.get("edge_grace_factor", 2.5) if self._last_edge else 1.0)

    def _update_hint(self, ff: FrameFeatures) -> None:
        h = ff.hands.get(self.primary) if self.primary else None
        if h is None and self.state == State.IDLE and ff.hands:
            h = max(ff.hands.values(), key=lambda x: x.hand_size)
        if h is None:
            if self.state in (State.IDLE,):
                self.hint = None
            return   # keep the last hint while the hand is briefly gone (it likely left at that edge)
        warn = self.g.get("edge_warn_margin", 0.05)
        self.hint = None if h.edge_margin >= warn else ("low" if h.edge_side == "bottom" else "edge")

    # ----- IDLE -------------------------------------------------------------------------------

    def _step_idle(self, ff: FrameFeatures, t: float) -> None:
        best = 0.0
        for side, s in self.sides.items():
            h = ff.hands.get(side)
            if h is None or not self._is_open_palm(s, h):
                s.open_since = None
                continue
            if s.open_since is None:
                s.open_since = t
            held = t - s.open_since
            best = max(best, held / self.g["engage_hold_s"])
            if held >= self.g["engage_hold_s"]:
                self._arm(side, t)
                return
        self.arm_progress = min(best, 1.0)

    def _arm(self, side: str, t: float) -> None:
        self.state, self.primary = State.HOVER, side
        self._last_seen = self._last_activity = t
        self.arm_progress = 0.0
        self._need_release = True   # the arming palm is open, but be explicit
        for s in self.sides.values():
            s.open_since = None

    def _disarm(self) -> None:
        self.state, self.primary, self.cursor = State.IDLE, None, None
        self._size_ref = None
        self._last_edge = False
        self.dwell_progress = 0.0
        self._fist_since = None
        self._history.clear()

    # ----- armed states -----------------------------------------------------------------------

    def _abort(self) -> list[Intent]:
        if self.state == State.GRABBED:
            return [Release()]
        if self.state == State.ZOOM:
            return [ResizeCancel()]
        return []

    def _step_armed(self, ff: FrameFeatures, t: float) -> list[Intent]:
        g = self.g
        h = ff.hands.get(self.primary)
        # a hand that slid out at the edge usually comes right back: wait longer before dropping a drag
        grace = self._grace()
        if h is not None:
            self._last_edge = h.edge_margin < g.get("edge_warn_margin", 0.05)
        if h is None and t - self._last_seen <= grace:
            return []   # brief tracking dropout: hold everything as-is, don't drop the window
        if h is None:
            out = self._abort()
            if self.state != State.ARMED:
                self.state = State.ARMED
                self._fist_since, self.dwell_progress = None, 0.0
                self._history.clear()
            if t - self._last_seen > g["disarm_timeout_s"]:
                self._disarm()
            return out

        self._last_seen = t
        if self.state == State.ARMED:
            self.state = State.HOVER
            self._last_activity = t
        s = self.sides[self.primary]
        if s.speed > g.get("activity_speed", 0.3) and s.speed != float("inf"):
            self._last_activity = t   # moving the pointer counts as using it
        self.cursor = (h.knuckle_x, h.knuckle_y)
        self._history.append(self.cursor)
        other = ff.hands.get("Left" if self.primary == "Right" else "Right")
        other_s = self.sides["Left" if self.primary == "Right" else "Right"]

        if (self._need_release and not s.pinch.value and not s.fist.value
                and h.curl > g["fist_curl_off"] and h.pinch > g["pinch_off"]):   # hand visibly open
            self._need_release = False

        out: list[Intent] = []

        # --- two-hand resize -----------------------------------------------------------------
        if self.state == State.ZOOM:
            self._last_activity = t
            if other is None:
                self.state, self._need_release = State.HOVER, True
                return [ResizeCancel()]
            if not (s.pinch.value and other_s.pinch.value):
                self.state, self._need_release = State.HOVER, True
                return [ResizeEnd()]
            scale = float(np.clip(ff.two_hand_span / self._zoom_span0,
                                  g["resize_min_scale"], g["resize_max_scale"]))
            return [Resize(scale)]

        if (other is not None and s.pinch.value and other_s.pinch.value
                and ff.two_hand_span and not self._need_release):
            if self.state == State.GRABBED:
                out.append(Release())
            self.state = State.ZOOM
            self._zoom_span0 = ff.two_hand_span
            self._last_activity = t
            x, y = self._pre_gesture_point()
            return out + [ResizeStart(x, y)]

        # --- fist dwell → minimize (only from HOVER: never yank a window mid-drag) -----------
        if s.fist.value and not self._need_release and self.state == State.HOVER:
            if self._fist_since is None:
                self._fist_since, self._fist_point = t, self._pre_gesture_point()
            self.dwell_progress = min((t - self._fist_since) / g["fist_dwell_s"], 1.0)
            self._last_activity = t
            if self.dwell_progress >= 1.0:
                x, y = self._fist_point
                out = self._abort() + [Minimize(x, y)]
                self.state, self._need_release = State.HOVER, True
                self._fist_since, self.dwell_progress = None, 0.0
                return out
            return []   # while making a fist, don't move the grabbed window
        self._fist_since, self.dwell_progress = None, 0.0

        # --- tap (click) vs hold (drag) ------------------------------------------------------
        if self.state == State.HOVER and s.pinch.value and not self._need_release:
            self.state = State.PRESS
            self._press_since, self._press_center = t, (h.center_x, h.center_y)
            self._last_activity = t
            return []

        if self.state == State.PRESS:
            self._last_activity = t
            if not s.pinch.value:     # released before it became a drag → click
                self.state = State.HOVER
                if t - self._press_since > g.get("tap_max_hold_s", 1.2):
                    return []         # a long still hold then release: not a tap, do nothing
                double = t - self._last_click_t <= g["double_click_s"] and self._last_click_count == 1
                count = 2 if double else 1
                self._last_click_t, self._last_click_count = t, count
                return [Click(count)]
            px, py = self._press_center
            moved = np.hypot(h.center_x - px, h.center_y - py) / h.hand_size
            firmly = h.pinch < g["pinch_on"]   # fingers opening = a slow tap ending, not a hold
            closed = h.pinch < g["pinch_off"]  # moving while the fingers part is a tap ending
            if (t - self._press_since >= g["tap_max_s"] and firmly) or (moved >= g["tap_max_move"] and closed):
                self.state = State.GRABBED
                self._grab_center = self._press_center   # drag counts from the pinch moment
                x, y = self._pre_gesture_point()
                gx, gy = self._grab_center
                dx, dy = (h.center_x - gx) * g["drag_gain"], (h.center_y - gy) * g["drag_gain"]
                self._grab_center = (h.center_x, h.center_y)
                return [Grab(x, y)] + ([Move(dx, dy)] if (dx or dy) else [])
            return []

        # --- drag / release -------------------------------------------------------------------
        if self.state == State.HOVER:
            if t - self._last_activity > g["armed_idle_timeout_s"]:
                self._disarm()
            return []

        if self.state == State.GRABBED:
            self._last_activity = t
            if not s.pinch.value:
                # open the hand before a fist counts: closing a drag into a fist must not minimize
                self.state, self._need_release = State.HOVER, True
                return [Release()]
            gx, gy = self._grab_center
            dx, dy = (h.center_x - gx) * g["drag_gain"], (h.center_y - gy) * g["drag_gain"]
            self._grab_center = (h.center_x, h.center_y)
            return [Move(dx, dy)] if (dx or dy) else []

        return out

    def _pre_gesture_point(self) -> tuple[float, float]:
        """Cursor from before the hand started changing shape (pinching/fisting moves the
        index tip). The debounce window is roughly how long the shape change has been going."""
        if not self._history:
            return self.cursor or (0.5, 0.5)
        back = min(len(self._history), self.debounce + 2)
        return self._history[-back]
