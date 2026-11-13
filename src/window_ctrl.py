"""WindowController: macOS Accessibility (AX) window operations + display geometry.

All coordinates here are AX/CoreGraphics GLOBAL coordinates: origin at the top-left of
the main display, y grows DOWN, other displays can have negative x/y. CGDisplayBounds
uses the same space, so no Cocoa (bottom-left) flip is needed anywhere in this module.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass

import ApplicationServices as AS
import Quartz as Q


@dataclass(frozen=True)
class Rect:
    x: float
    y: float
    w: float
    h: float

    def contains(self, px: float, py: float) -> bool:
        return self.x <= px < self.x + self.w and self.y <= py < self.y + self.h

    def clamp(self, px: float, py: float) -> tuple[float, float]:
        return (min(max(px, self.x), self.x + self.w - 1), min(max(py, self.y), self.y + self.h - 1))


class Displays:
    def __init__(self):
        err, ids, n = Q.CGGetActiveDisplayList(16, None, None)
        self.rects = []
        for d in ids[:n]:
            b = Q.CGDisplayBounds(d)
            self.rects.append(Rect(b.origin.x, b.origin.y, b.size.width, b.size.height))
        if not self.rects:   # e.g. mid display reconfiguration: fall back to the main display
            b = Q.CGDisplayBounds(Q.CGMainDisplayID())
            self.rects.append(Rect(b.origin.x, b.origin.y, max(b.size.width, 1), max(b.size.height, 1)))
        xs = [r.x for r in self.rects] + [r.x + r.w for r in self.rects]
        ys = [r.y for r in self.rects] + [r.y + r.h for r in self.rects]
        self.bbox = Rect(min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys))

    def clamp(self, px: float, py: float) -> tuple[float, float]:
        """Snap a point in the bounding box's dead zones onto the nearest real display."""
        if any(r.contains(px, py) for r in self.rects):
            return px, py
        best = min((r.clamp(px, py) for r in self.rects),
                   key=lambda q: (q[0] - px) ** 2 + (q[1] - py) ** 2)
        return best

    def __repr__(self) -> str:
        return f"Displays({len(self.rects)}: " + ", ".join(
            f"{r.w:.0f}x{r.h:.0f}@({r.x:.0f},{r.y:.0f})" for r in self.rects) + ")"


def is_trusted(prompt: bool = False) -> bool:
    if prompt:
        return bool(AS.AXIsProcessTrustedWithOptions({AS.kAXTrustedCheckOptionPrompt: True}))
    return bool(AS.AXIsProcessTrusted())


def _attr(el, name):
    err, val = AS.AXUIElementCopyAttributeValue(el, name, None)
    return val if err == 0 else None


class WindowController:
    def __init__(self):
        self._system = AS.AXUIElementCreateSystemWide()
        self._own_pid = os.getpid()

    def window_at(self, x: float, y: float):
        err, el = AS.AXUIElementCopyElementAtPosition(self._system, x, y, None)
        if err != 0 or el is None:
            return None
        err, pid = AS.AXUIElementGetPid(el, None)
        if err == 0 and pid == self._own_pid:
            return None   # never grab our own preview window
        win = _attr(el, AS.kAXWindowAttribute)
        if win is not None:
            return win
        for _ in range(30):   # walk up until role == AXWindow
            if _attr(el, AS.kAXRoleAttribute) == AS.kAXWindowRole:
                return el
            el = _attr(el, AS.kAXParentAttribute)
            if el is None:
                return None
        return None

    @staticmethod
    def title(win) -> str:
        return str(_attr(win, AS.kAXTitleAttribute) or "")

    @staticmethod
    def get_frame(win) -> Rect | None:
        pos, size = _attr(win, AS.kAXPositionAttribute), _attr(win, AS.kAXSizeAttribute)
        if pos is None or size is None:
            return None
        _, p = AS.AXValueGetValue(pos, AS.kAXValueCGPointType, None)
        _, s = AS.AXValueGetValue(size, AS.kAXValueCGSizeType, None)
        return Rect(p.x, p.y, s.width, s.height)

    @staticmethod
    def set_position(win, x: float, y: float) -> int:
        """Returns the AX error code (0 = success)."""
        v = AS.AXValueCreate(AS.kAXValueCGPointType, Q.CGPoint(x, y))
        return AS.AXUIElementSetAttributeValue(win, AS.kAXPositionAttribute, v)

    @staticmethod
    def set_size(win, w: float, h: float) -> bool:
        v = AS.AXValueCreate(AS.kAXValueCGSizeType, Q.CGSize(w, h))
        return AS.AXUIElementSetAttributeValue(win, AS.kAXSizeAttribute, v) == 0

    @staticmethod
    def is_fullscreen(win) -> bool:
        return bool(_attr(win, "AXFullScreen"))

    def exit_fullscreen(self, win, timeout: float = 2.5) -> bool:
        """Native full-screen windows refuse moves (-25200). Leave full screen and wait for
        the animation to finish, i.e. until the position becomes settable again."""
        AS.AXUIElementSetAttributeValue(win, "AXFullScreen", False)
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            time.sleep(0.1)
            err, ok = AS.AXUIElementIsAttributeSettable(win, AS.kAXPositionAttribute, None)
            if not self.is_fullscreen(win) and err == 0 and ok:
                time.sleep(0.15)   # let the frame settle
                return True
        return False

    def minimize(self, win) -> int:
        """Returns the AX error code (0 = success). Full-screen windows can't be minimized
        directly, so they leave full screen first."""
        if self.is_fullscreen(win) and not self.exit_fullscreen(win):
            return -25200
        # NOTE: with Stage Manager on, this works (window shrinks into the side strip) but
        # AXMinimized keeps reading False, so don't "verify" via that attribute.
        return AS.AXUIElementSetAttributeValue(win, AS.kAXMinimizedAttribute, True)


def warp_mouse(x: float, y: float) -> None:
    Q.CGWarpMouseCursorPosition(Q.CGPoint(x, y))
    Q.CGAssociateMouseAndMouseCursorPosition(True)  # keep the physical mouse usable
