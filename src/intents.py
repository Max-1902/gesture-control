"""Intents: what the FSM wants done. The ActionRouter turns these into window operations.

Positions/deltas are in normalized camera units (0..1, top-left origin, mirrored);
the router maps them to screen pixels. Keeping the FSM in camera units keeps it
display-agnostic and testable.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Grab:            # pick the window under (x, y) and start dragging it
    x: float
    y: float


@dataclass(frozen=True)
class Click:           # quick pinch tap at the pointer; count 2 = double click
    count: int = 1


@dataclass(frozen=True)
class Move:            # move the grabbed window by (dx, dy), gain already applied
    dx: float
    dy: float


@dataclass(frozen=True)
class Release:         # stop dragging; window stays where it is
    pass


@dataclass(frozen=True)
class ResizeStart:     # start two-hand resize of the grabbed window, or the one under (x, y)
    x: float
    y: float


@dataclass(frozen=True)
class Resize:          # scale relative to the size at ResizeStart
    scale: float


@dataclass(frozen=True)
class ResizeEnd:       # commit the current size
    pass


@dataclass(frozen=True)
class ResizeCancel:    # a hand dropped out mid-resize: restore the size from ResizeStart
    pass


@dataclass(frozen=True)
class Minimize:        # minimize the grabbed window, or the one under (x, y)
    x: float
    y: float


Intent = Grab | Click | Move | Release | ResizeStart | Resize | ResizeEnd | ResizeCancel | Minimize
