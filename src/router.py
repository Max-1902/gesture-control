"""ActionRouter: FSM state + intents → mouse pointer and window operations.

Runs on its own thread at `max_write_hz` (AX calls can block for tens of ms on slow apps,
which must never stall the camera/logic loop).

Pointer = RELATIVE with acceleration, like a mouse (architecture doc §2, decision 5):
  * PointerMapper: hand delta × gain(speed), with a deadband anchor that ignores resting
    jitter. Gains are high, so the hand barely needs to move: a quick ~10 cm flick
    crosses all three screens, slow motion stays precise.
  * It starts from wherever your real mouse is when you arm (clutch: re-arm to re-center).
  * The pointer glides toward its target at 60 Hz (`follow_tau_s`), so motion is fluid
    even though the camera only delivers ~25 samples/s.

"Holding it": a grabbed window moves by exactly the pointer's movement, so the spot you
pinched stays under the pointer. Pop on grab, tink on release.
"""
from __future__ import annotations

import math
import queue
import subprocess
import threading
import time

import Quartz as Q

from fsm import State
from intents import Click, Grab, Intent, Minimize, Release, Resize, ResizeCancel, ResizeEnd, ResizeStart
from window_ctrl import Displays, Rect, WindowController, warp_mouse

SOUNDS = {"grab": "/System/Library/Sounds/Pop.aiff", "release": "/System/Library/Sounds/Tink.aiff",
          "minimize": "/System/Library/Sounds/Bottle.aiff", "nothing": "/System/Library/Sounds/Funk.aiff",
          "click": "/System/Library/Sounds/Tink.aiff", "start": "/System/Library/Sounds/Hero.aiff"}
CAMERA_ASPECT = 480 / 640   # normalized y units are taller than x units; equalize hand motion


def accel_gain(speed: float, c: dict) -> float:
    """px per camera-width, as a smooth function of hand speed (camera-widths per second)."""
    lo, hi = c["pointer_speed_slow"], c["pointer_speed_fast"]
    u = min(max((speed - lo) / (hi - lo), 0.0), 1.0)
    u = u * u * (3 - 2 * u)   # smoothstep
    gain = c["pointer_gain_slow"] + (c["pointer_gain_fast"] - c["pointer_gain_slow"]) * u
    return gain * c.get("pointer_speed", 1.0)


class PointerMapper:
    """Hand position (normalized camera coords) → pointer delta in screen px.

    Deadband anchor instead of a per-frame dead zone: the pointer only moves once the hand
    leaves a small circle around an anchor, and the anchor is then dragged along. Resting
    jitter (inside the circle) is ignored entirely, yet slow deliberate motion still
    passes through 1:1 (a per-frame dead zone would eat it). The delta then gets an
    accelerated gain, so a small, quick hand flick crosses all displays while slow motion
    stays precise.
    """

    def __init__(self, c: dict):
        self.c = c
        self.reset()

    def reset(self) -> None:
        self._anchor: list[float] | None = None
        self._t = 0.0
        self._speed = 0.0

    def rebase(self, nx: float, ny: float, t: float) -> None:
        """Re-anchor without moving (pointer frozen while pinching)."""
        self._anchor, self._t = [nx, ny * CAMERA_ASPECT], t

    def step(self, nx: float, ny: float, t: float) -> tuple[float, float]:
        y = ny * CAMERA_ASPECT   # equalize x/y: both in camera-width units
        if self._anchor is None or t <= self._t:
            if self._anchor is None:
                self.rebase(nx, ny, t)
            return 0.0, 0.0
        dt, self._t = t - self._t, t
        if dt > self.c.get("pointer_max_gap_s", 0.15):
            self.rebase(nx, ny, t)   # hand was briefly lost: resume from here, never jump
            return 0.0, 0.0
        ex, ey = nx - self._anchor[0], y - self._anchor[1]
        d = math.hypot(ex, ey)
        r = self.c["pointer_deadband"]
        if d <= r:
            self._speed *= 0.5
            return 0.0, 0.0
        k = (d - r) / d
        mx, my = ex * k, ey * k
        self._anchor[0] += mx
        self._anchor[1] += my
        self._speed = 0.5 * self._speed + 0.5 * math.hypot(mx, my) / dt   # steadier gain
        g = accel_gain(self._speed, self.c)
        return mx * g, my * g * self.c.get("pointer_gain_y", 1.0)


def mouse_location() -> tuple[float, float]:
    p = Q.CGEventGetLocation(Q.CGEventCreate(None))
    return p.x, p.y


class Router:
    def __init__(self, cfg: dict, dry_run: bool = False, log=print, overlay=None):
        self.c = c = cfg["control"]
        self.dry_run, self.log = dry_run, log
        self.displays = Displays()
        self.ctrl = WindowController()
        self.max_hz = c["max_write_hz"]
        self.tau = c.get("follow_tau_s", 0.03)

        self._q: queue.Queue = queue.Queue()
        self._lock = threading.Lock()
        self._sample: tuple[float, float, float] | None = None   # latest anchor (nx, ny, t)
        self._active = False
        self._frozen = False
        self._state = State.IDLE
        self._arm = self._dwell = 0.0
        self._hint: str | None = None
        self.overlay = overlay
        self.enabled = True   # kill switch state, shown on the HUD

        # AX-thread state
        self._hover_win: tuple | None = None       # (x, y, w, h) of the window a pinch would grab
        self._hover_check_t = 0.0
        self._last_hud: dict | None = None
        self._mapper = PointerMapper(c)
        self._target: list[float] | None = None    # pointer target, screen px
        self._shown: list[float] | None = None     # pointer as displayed
        self._win = None
        self._win_start: Rect | None = None
        self._grab_ptr: tuple[float, float] | None = None
        self._resize_scale = 1.0
        self._move_err_logged = False
        self._fs_pending = False                   # grabbed a full-screen window: exit only once really dragged
        self._ax_moved_for: tuple | None = None
        self._displays_t = time.monotonic()
        self._running = True
        # Pointer thread: glide + warp + HUD only, never touches other apps → never stalls.
        # AX thread: everything that talks to other apps (can block 20-50 ms, or seconds).
        self._thread = threading.Thread(target=self._loop, name="pointer", daemon=True)
        self._ax_thread = threading.Thread(target=self._ax_loop, name="ax", daemon=True)
        self._thread.start()
        self._ax_thread.start()
        log(f"router: {self.displays}{' [DRY RUN]' if dry_run else ''}")

    # ----- called from the logic loop (cheap, non-blocking) -----------------------------------

    def submit(self, intents: list[Intent], cursor: tuple[float, float] | None, state: State,
               t: float | None = None, arm: float = 0.0, dwell: float = 0.0,
               hint: str | None = None) -> None:
        for it in intents:
            self._q.put(it)
        with self._lock:
            self._state, self._arm, self._dwell, self._hint = state, arm, dwell, hint
            self._active = state in (State.HOVER, State.PRESS, State.GRABBED, State.ZOOM)
            self._frozen = state == State.PRESS   # fingers together: hold the pointer still
            if cursor and self._active:
                self._sample = (cursor[0], cursor[1], time.monotonic() if t is None else t)
            elif not self._active:
                self._sample = None

    def close(self) -> None:
        self._q.put(Release())   # never leave a window mid-drag
        self._running = False
        self._ax_thread.join(timeout=3.0)
        self._thread.join(timeout=1.0)

    # ----- AX thread --------------------------------------------------------------------------

    def play(self, name: str) -> None:
        if self.c.get("sounds", True):
            subprocess.Popen(["afplay", "-v", "0.4", SOUNDS[name]],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def _update_target(self) -> None:
        with self._lock:
            sample, active, frozen = self._sample, self._active, self._frozen
        if not active or sample is None:
            self._mapper.reset()
            if not active:
                self._target = self._shown = None   # clutch: next arm starts at the real mouse
            return
        if self._target is None:
            self._target = list(mouse_location())
            self._shown = list(self._target)
        if frozen:
            self._mapper.rebase(*sample)   # unfreezing must not jump
            return
        dx, dy = self._mapper.step(*sample)
        if dx or dy:
            self._target = list(self.displays.clamp(self._target[0] + dx, self._target[1] + dy))

    def _loop(self) -> None:
        """Pointer thread: 60 Hz glide/warp/HUD. No calls into other apps, so it never stalls."""
        period = 1.0 / self.max_hz
        last_t = time.monotonic()
        last_mouse = None
        errors = 0
        while self._running:
            t0 = time.monotonic()
            dt, last_t = t0 - last_t, t0
            try:
                last_mouse = self._tick(dt, t0, last_mouse)
                errors = 0
            except Exception as e:
                errors += 1
                if errors <= 3:
                    import traceback
                    self.log(f"  !! pointer error (recovered): {e!r}\n{traceback.format_exc(limit=3)}")
                self._target = self._shown = None
                last_mouse = None
            time.sleep(max(0.0, period - (time.monotonic() - t0)))

    def _tick(self, dt: float, t0: float, last_mouse):
        self._update_target()
        if (self._target is not None and last_mouse is not None and self._grab_ptr is None
                and self.c["move_mouse"] and not self.dry_run):
            real = mouse_location()
            if math.hypot(real[0] - last_mouse[0], real[1] - last_mouse[1]) > 4:
                # you moved the real mouse/trackpad: follow it instead of yanking it back
                self._target, self._shown = list(real), list(real)
                last_mouse = real
        if self._target is not None and self._shown is not None:
            a = 1.0 - math.exp(-dt / self.tau) if self.tau > 0 else 1.0
            self._shown[0] += (self._target[0] - self._shown[0]) * a
            self._shown[1] += (self._target[1] - self._shown[1]) * a
            mouse = (round(self._shown[0], 1), round(self._shown[1], 1))
            if mouse != last_mouse:
                if self.c["move_mouse"] and not self.dry_run:
                    warp_mouse(*mouse)
                last_mouse = mouse
        if self.overlay is not None:
            self._send_hud(t0)
        return last_mouse

    def _ax_loop(self) -> None:
        """AX thread: intents, window moves (latest pointer only), hover-window lookups."""
        period = 1.0 / self.max_hz
        errors = 0
        while self._running or not self._q.empty():
            try:
                try:
                    it = self._q.get(timeout=period)
                except queue.Empty:
                    it = None
                dirty_size = False
                while it is not None:
                    if isinstance(it, Resize):
                        self._resize_scale, dirty_size = it.scale, True
                    elif isinstance(it, (Grab, Click, Release, ResizeStart, ResizeEnd, ResizeCancel, Minimize)):
                        if dirty_size:
                            self._apply_size()
                            dirty_size = False
                        self._handle(it)
                    # Move intents are ignored: the window follows the pointer instead
                    try:
                        it = self._q.get_nowait()
                    except queue.Empty:
                        it = None
                if dirty_size:
                    self._apply_size()
                if self._grab_ptr is not None:
                    p = self._pointer()
                    if p != self._ax_moved_for:
                        self._ax_moved_for = p
                        self._apply_move()
                now = time.monotonic()
                self._update_hover(now)
                if now - self._displays_t > 2.0:   # monitors plugged in / removed / rearranged
                    self._displays_t = now
                    d = Displays()
                    if d.rects and d.rects != self.displays.rects:
                        self.displays = d
                        self.log(f"displays changed: {d}")
                errors = 0
            except Exception as e:   # a window vanished mid-drag, an app hung, ...: recover
                errors += 1
                if errors <= 3:
                    import traceback
                    self.log(f"  !! window error (recovered): {e!r}\n{traceback.format_exc(limit=3)}")
                self._win = self._win_start = self._grab_ptr = None
                self._fs_pending = False
                time.sleep(period)

    def _update_hover(self, now: float) -> None:
        with self._lock:
            state = self._state
        if state not in (State.HOVER, State.PRESS) or self._shown is None or self._grab_ptr is not None:
            self._hover_win = None
            return
        if now - self._hover_check_t > 0.12:
            self._hover_check_t = now
            win = self.ctrl.window_at(*self._shown)
            f = self.ctrl.get_frame(win) if win is not None else None
            self._hover_win = (round(f.x), round(f.y), round(f.w), round(f.h)) if f else None

    def _send_hud(self, now: float) -> None:
        with self._lock:
            state, arm, dwell, hint = self._state, self._arm, self._dwell, self._hint
        msg: dict = {"s": state.value}
        if hint:
            msg["hint"] = hint
        if not self.enabled:
            msg["off"] = True
        if self._shown is not None:
            msg["p"] = (round(self._shown[0]), round(self._shown[1]))
        elif arm > 0:
            msg["p"] = tuple(round(v) for v in mouse_location())   # arming: ring around the mouse
        if arm > 0:
            msg["arm"] = round(arm, 2)
        if dwell > 0:
            msg["dwell"] = round(dwell, 2)
        # target window outline: live while dragging, looked up ~8x/s while hovering
        if self._win is not None and self._win_start is not None and self._grab_ptr is not None and self._shown:
            s = self._win_start
            msg["win"] = (round(s.x + self._shown[0] - self._grab_ptr[0]),
                          round(s.y + self._shown[1] - self._grab_ptr[1]), round(s.w), round(s.h))
        elif state in (State.HOVER, State.PRESS) and self._shown is not None:
            hw = self._hover_win   # looked up by the AX thread
            if hw:
                msg["win"] = hw
        if msg != self._last_hud:
            self.overlay.send(msg)
            self._last_hud = msg

    def _pointer(self) -> tuple[float, float]:
        s = self._shown   # read once: the pointer thread may swap it
        return (s[0], s[1]) if s is not None else mouse_location()

    def _pick(self):
        x, y = self._pointer()
        win = self.ctrl.window_at(x, y)
        frame = self.ctrl.get_frame(win) if win is not None else None
        if (win is not None and frame is not None and not self.ctrl.title(win)
                and any(abs(frame.w - d.w) < 2 and abs(frame.h - d.h) < 2 for d in self.displays.rects)):
            self.log(f"  skipped untitled display-sized window ({frame.w:.0f}x{frame.h:.0f}) as desktop")
            win = None   # the Finder desktop: not a window
        if win is None or frame is None:
            self.log(f"  no window at ({x:.0f}, {y:.0f})")
            self.play("nothing")   # audible "nothing there" instead of silent failure
            return None, None
        self.log(f"  → {self.ctrl.title(win)!r} at ({frame.x:.0f}, {frame.y:.0f}) {frame.w:.0f}x{frame.h:.0f}")
        return win, frame

    def _end_drag(self) -> None:
        if self._win is not None and self._grab_ptr is not None:
            self._apply_move()
        self._win = self._win_start = self._grab_ptr = None
        self._fs_pending = False

    def _click(self, count: int) -> None:
        x, y = self._pointer()
        self.log(f"  click ×{count} at ({x:.0f}, {y:.0f})")
        if self.dry_run:
            return
        for kind in (Q.kCGEventLeftMouseDown, Q.kCGEventLeftMouseUp):
            ev = Q.CGEventCreateMouseEvent(None, kind, Q.CGPoint(x, y), Q.kCGMouseButtonLeft)
            Q.CGEventSetIntegerValueField(ev, Q.kCGMouseEventClickState, count)   # 2 = double click
            Q.CGEventPost(Q.kCGHIDEventTap, ev)

    def _handle(self, it: Intent) -> None:
        self.log(f"{type(it).__name__}")
        if isinstance(it, Click):
            self._click(it.count)
            self.play("click")
        elif isinstance(it, Grab):
            self._win, self._win_start = self._pick()
            self._move_err_logged = False
            if self._win is None:
                return
            self._grab_ptr = self._pointer()
            self.play("grab")
            self._fs_pending = self.ctrl.is_fullscreen(self._win) and not self.dry_run
            if self._fs_pending:
                self.log("  window is native full screen: will leave full screen once you drag it")
        elif isinstance(it, Release):
            if self._win is not None:
                self.play("release")
            self._end_drag()
        elif isinstance(it, ResizeStart):
            if self._win is None:
                self._win, self._win_start = self._pick()
            else:
                win = self._win
                self._end_drag()
                self._win = win
                self._win_start = self.ctrl.get_frame(win)
            self._resize_scale = 1.0
            if self._win is not None:
                self.play("grab")
        elif isinstance(it, ResizeEnd):
            if self._win is not None:
                self.play("release")
            self._win = self._win_start = None
        elif isinstance(it, ResizeCancel):
            if self._win is not None and self._win_start is not None and not self.dry_run:
                s = self._win_start
                self.ctrl.set_size(self._win, s.w, s.h)
                self.ctrl.set_position(self._win, s.x, s.y)
            self._win = self._win_start = None
        elif isinstance(it, Minimize):
            win = self._win
            if win is None:
                win, _ = self._pick()
            if win is not None and not self.dry_run:
                err = self.ctrl.minimize(win)
                if err:
                    self.log(f"  !! minimize refused (AX error {err})")
                else:
                    self.play("minimize")
            self._win = self._win_start = self._grab_ptr = None

    def _apply_move(self) -> None:
        if self._win is None or self._win_start is None or self._grab_ptr is None or self.dry_run:
            return
        px, py = self._pointer()
        if self._fs_pending:
            gx, gy = self._grab_ptr
            if math.hypot(px - gx, py - gy) < self.c.get("fullscreen_drag_px", 80):
                return   # a quick pinch on a full-screen app must not kick it out of full screen
            self._fs_pending = False
            self.log("  dragging a full-screen window → exiting full screen")
            if not self.ctrl.exit_fullscreen(self._win):
                self.log("  !! could not exit full screen; can't move this window")
                self._win = self._win_start = self._grab_ptr = None
                return
            f = self.ctrl.get_frame(self._win)
            if f is None:
                self._win = self._win_start = self._grab_ptr = None
                return
            px, py = self._pointer()
            self._grab_ptr = (px, py)
            self._win_start = Rect(px - f.w / 2, py - 20, f.w, f.h)   # title bar under pointer
        s = self._win_start
        err = self.ctrl.set_position(self._win, s.x + px - self._grab_ptr[0], s.y + py - self._grab_ptr[1])
        if err and not self._move_err_logged:
            self.log(f"  !! move rejected by app (AX error {err})")
            self._move_err_logged = True

    def _apply_size(self) -> None:
        if self._win is None or self._win_start is None or self.dry_run:
            return
        s = self._win_start
        w, h = s.w * self._resize_scale, s.h * self._resize_scale
        self.ctrl.set_size(self._win, w, h)
        self.ctrl.set_position(self._win, s.x + (s.w - w) / 2, s.y + (s.h - h) / 2)  # keep centered
