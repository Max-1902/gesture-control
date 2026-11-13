"""Developer-only camera window (run.py --preview / --record): landmarks, features, FSM state.

Needs OpenCV (requirements-dev.txt). The app itself never imports this module.
"""
from __future__ import annotations

import time

import cv2
import numpy as np

from features import hand_features
from fsm import State
from hands import CONNECTIONS, FINGERTIPS, Frame

COLORS = {"Left": (255, 160, 60), "Right": (60, 200, 255)}  # BGR
STATE_COLORS = {State.IDLE: (150, 150, 150), State.ARMED: (0, 200, 255), State.HOVER: (0, 220, 0),
                State.PRESS: (0, 255, 255), State.GRABBED: (255, 120, 0), State.ZOOM: (255, 0, 200)}


def draw_hand(img: np.ndarray, lm: np.ndarray, color) -> None:
    h, w = img.shape[:2]
    pts = [(int(x * w), int(y * h)) for x, y, _ in lm]
    for a, b in CONNECTIONS:
        cv2.line(img, pts[a], pts[b], color, 2, cv2.LINE_AA)
    for i, p in enumerate(pts):
        cv2.circle(img, p, 5 if i in FINGERTIPS else 3, color, -1, cv2.LINE_AA)


def ring(img, center, radius, progress, color) -> None:
    cv2.ellipse(img, center, (radius, radius), -90, 0, 360, (60, 60, 60), 3, cv2.LINE_AA)
    cv2.ellipse(img, center, (radius, radius), -90, 0, int(360 * progress), color, 4, cv2.LINE_AA)


def text(img, s: str, y: int, color=(255, 255, 255)) -> None:
    cv2.putText(img, s, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, s, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)


def draw_fsm(img, fsm, smoothed: Frame) -> None:
    """State badge, arm ring around the palm, cursor, minimize dwell ring."""
    h, w = img.shape[:2]
    color = STATE_COLORS[fsm.state]
    if fsm.state == State.IDLE:   # show why the palm does/doesn't arm
        g, y = fsm.g, h - 40
        for hand in smoothed.hands:
            hf = hand_features(hand)
            s = fsm.sides[hand.handedness]
            checks = [("fingers", hf.min_extension, g["engage_min_extension"], hf.min_extension >= g["engage_min_extension"]),
                      ("thumb", hf.pinch, g["engage_thumb_away"], hf.pinch >= g["engage_thumb_away"]),
                      ("still", s.speed, g["engage_max_speed"], s.speed <= g["engage_max_speed"])]
            msg = "  ".join(f"{n} {v:.2f}{'>=' if n != 'still' else '<='}{t} {'OK' if ok else 'NO'}"
                            for n, v, t, ok in checks)
            text(img, f"arm {hand.handedness}: {msg}", y, (0, 255, 0) if all(c[3] for c in checks) else (0, 0, 255))
            y -= 20
    label = ("OFF (k)" if not fsm.enabled else
             fsm.state.value.upper() + (f" ({fsm.primary})" if fsm.primary else ""))
    cv2.rectangle(img, (w - 190, 32), (w - 10, 62), color, -1)
    cv2.putText(img, label, (w - 182, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)
    if fsm.state == State.IDLE and fsm.arm_progress > 0:
        for hand in smoothed.hands:   # ring on the hand(s) currently holding an open palm
            if fsm.sides[hand.handedness].open_since is not None:
                c = hand.landmarks[9, :2]
                ring(img, (int(c[0] * w), int(c[1] * h)), 45, fsm.arm_progress, (0, 220, 0))
    if fsm.cursor and fsm.state != State.IDLE:
        cx, cy = int(fsm.cursor[0] * w), int(fsm.cursor[1] * h)
        cv2.drawMarker(img, (cx, cy), color, cv2.MARKER_CROSS, 24, 2, cv2.LINE_AA)
        if fsm.dwell_progress > 0:
            ring(img, (cx, cy), 30, fsm.dwell_progress, (0, 0, 255))


def draw_preview(img, raw, smoothed, feats, pipe, source, rec, label, auto_start, record_s, t_cap,
                 frame_times, infer_ms, cpu_pct, show_filtered) -> int:
    """Draws and shows the debug window. Returns the pressed key, or -1."""
    for hand in (smoothed if show_filtered else raw).hands:
        draw_hand(img, hand.landmarks, COLORS.get(hand.handedness, (200, 200, 200)))

    fps = (len(frame_times) - 1) / (frame_times[-1] - frame_times[0]) if len(frame_times) > 1 else 0
    latency = (time.monotonic() - t_cap) * 1000
    infer = np.mean(infer_ms) if infer_ms else 0.0
    text(img, f"{fps:4.1f} fps  infer {infer:4.1f} ms  "
              f"cap->draw {latency:4.0f} ms  CPU {cpu_pct:3.0f}%  drop {source.dropped}", 20)
    text(img, f"overlay: {'filtered' if show_filtered else 'raw'}   keys (click this window first):"
              f" r=rec f=filter k=kill q=quit", 40, (180, 180, 180))
    if auto_start is not None and rec is None:
        cv2.putText(img, f"{label}: starts in {auto_start - t_cap:.0f}", (img.shape[1] // 2 - 150,
                    img.shape[0] // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3, cv2.LINE_AA)
    y = 62
    for side, hf in sorted(feats.hands.items()):
        text(img, f"{side:5s} pinch {hf.pinch:4.2f}  open {hf.openness:4.2f}  "
                  f"curl {hf.curl:4.2f}  size {hf.hand_size:4.2f}", y, COLORS[side])
        y += 20
    if feats.two_hand_span is not None:
        text(img, f"span {feats.two_hand_span:4.2f}", y)
    draw_fsm(img, pipe.fsm, smoothed)
    if rec:
        cv2.circle(img, (img.shape[1] - 20, 20), 8, (0, 0, 255), -1)
        rem = f"  {auto_start + record_s - t_cap:.0f}s left" if auto_start else ""
        text(img, f"REC {label} {rec.frames}{rem}", img.shape[0] - 12, (80, 80, 255))

    cv2.imshow("gesture-control: debug", img)
    key = cv2.waitKey(1) & 0xFF
    return -1 if key == 255 else key


def close() -> None:
    cv2.destroyAllWindows()
