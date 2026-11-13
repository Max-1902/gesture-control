"""Gesture Control: camera → hand landmarks → gestures → pointer and window control.

    python run.py                  # the app (menu bar icon, HUD, settings window)
    python run.py --dry-run        # track and log, but never touch the mouse or windows
    python run.py --preview        # developer camera window (needs OpenCV)
    python run.py --label pinch --record 60    # record 60 s of landmarks after a countdown

Preview keys (click the camera window first): r = start/stop recording, f = raw/filtered,
k = on/off, q / Esc = quit.
"""
from __future__ import annotations

import argparse
import multiprocessing
import resource
import signal
import sys
import threading
import time
import traceback
from collections import deque
from pathlib import Path

import numpy as np

import config
import paths
from hands import Frame
from intents import Move
from pipeline import Pipeline
from recorder import Recorder
from version import __version__


def cpu_seconds() -> float:
    r = resource.getrusage(resource.RUSAGE_SELF)
    return r.ru_utime + r.ru_stime


class Camera:
    """Owns the hand source and copes with everything that can be wrong with the camera:
    permission not yet answered or denied, no camera, camera unplugged, camera changed in
    Settings. state: starting | asking | denied | missing | error | running | paused."""

    def __init__(self, cfg: dict, backend: str, log):
        self.cfg, self.backend, self.log = cfg, backend, log
        self.source = None
        self.state, self.error = "starting", ""
        self._asked = False
        self._next_try = 0.0
        self.want_image = False

    def ensure(self, enabled: bool) -> None:
        """Called ~10x/s: (re)open the camera when possible, pause it while disabled."""
        if self.source is not None:
            if enabled and not self.source.running:
                self.source.resume()
            elif not enabled and self.source.running:
                self.source.pause()
            self.state = "running" if self.source.running else "paused"
            return
        now = time.monotonic()
        if now < self._next_try:
            return
        self._next_try = now + 1.0
        cam, tr = self.cfg["camera"], self.cfg["tracker"]
        try:
            if self.backend == "vision":
                from av_source import AVHandSource, camera_permission, request_camera_permission
                perm = camera_permission()
                if perm == "unknown":
                    if not self._asked:   # the system prompt; answered asynchronously
                        self._asked = True
                        threading.Thread(target=request_camera_permission, daemon=True).start()
                    self.state = "asking"
                    return
                if perm == "denied":
                    if self.state != "denied":
                        self.log("camera access denied: System Settings → Privacy & Security → Camera")
                    self.state = "denied"
                    return
                self.source = AVHandSource(cam["device"], cam["width"], cam["height"], cam["fps"],
                                           tr["max_hands"], tr["vision_min_confidence"])
            else:
                from tracker import CvSource
                self.source = CvSource(cam, tr)
        except RuntimeError as e:
            state = "missing" if "No camera" in str(e) else "error"
            if (state, str(e)) != (self.state, self.error):
                self.log(f"camera: {e}")
            self.state, self.error = state, str(e)
            self._next_try = now + 3.0
            return
        self.source.want_image = self.want_image
        self.state, self.error = "running", ""
        self.log(f"camera: {self.source.info.name}, tracker: {self.backend}")
        if not enabled:
            self.source.pause()
            self.state = "paused"

    def reopen(self) -> None:
        self.close()
        self._next_try = 0.0

    def read(self):
        if self.source is None or not self.source.running:
            return None
        return self.source.read(0.25)

    def close(self) -> None:
        if self.source is not None:
            self.source.close()
            self.source = None
            self.state = "starting"


def list_camera_names() -> list[str]:
    from capture import list_cameras
    try:
        return [c.name for c in list_cameras()]
    except Exception:
        return []


def main(argv: list[str] | None = None) -> None:
    # Vision results are read via ~100 tiny pyobjc calls per frame on the capture thread; each
    # call releases and re-takes the GIL. With the default 5 ms switch interval every re-take
    # can wait 5 ms behind the logic/router threads: measured 2 ms → 27-57 ms per frame.
    sys.setswitchinterval(0.0005)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", help="extra YAML merged over the defaults and your settings")
    ap.add_argument("--camera", help='override camera: "builtin", name substring, or index')
    ap.add_argument("--list-cameras", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="log window actions, don't perform them")
    ap.add_argument("--no-control", action="store_true", help="no window control at all (viewer only)")
    ap.add_argument("--preview", action="store_true", help="show the developer camera window")
    ap.add_argument("--no-hud", action="store_true", help="no on-screen HUD / menu bar icon")
    ap.add_argument("--tracker", choices=["vision", "mediapipe"], help="override tracker backend")
    ap.add_argument("--label", default="session", help="label for recordings made with 'r' / --record")
    ap.add_argument("--record", type=float, metavar="SECONDS",
                    help="auto-record for SECONDS after a 3 s countdown, then quit")
    ap.add_argument("--version", action="version", version=f"Gesture Control {__version__}")
    ap.add_argument("--self-test", action="store_true", help=argparse.SUPPRESS)   # used by build_app.sh
    # parse_known_args: macOS may pass extra arguments (e.g. -psn_…) to an app it launches
    args, _ = ap.parse_known_args(argv)
    cfg = config.load(args.config)

    if args.self_test:   # import everything the app loads lazily, so a broken build fails here
        import AVFoundation, CoreMedia, ServiceManagement, Vision, WebKit  # noqa: F401
        import av_source, overlay, router, window_ctrl  # noqa: F401
        assert paths.SETTINGS_HTML.exists() and paths.DEFAULT_CONFIG.exists()
        print(f"Gesture Control {__version__}: self-test OK")
        return

    if args.list_cameras:
        from capture import list_cameras
        for c in list_cameras():
            print(f"{c.cv_index}: {c.name}  [{c.device_type}]{'  ← builtin' if c.builtin else ''}")
        return

    if args.camera:
        cfg["camera"]["device"] = args.camera
    backend = args.tracker or cfg["tracker"].get("backend", "vision")
    control = not args.no_control and not args.record
    preview = args.preview or bool(args.record) or args.no_control
    if preview:
        try:
            import debug_view
        except ImportError:
            raise SystemExit("--preview / --record need OpenCV:  pip install -r requirements-dev.txt")

    t_start = time.monotonic()
    log_file = paths.open_log()
    log_file.write(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')}  Gesture Control {__version__} =====\n")

    def log(msg: str) -> None:
        line = f"{time.monotonic() - t_start:7.2f}s  {msg}"
        print(line, flush=True)
        try:
            log_file.write(line + "\n")
        except (OSError, ValueError):
            pass

    overlay = router = None
    camera = Camera(cfg, backend, log)
    camera.want_image = preview
    pipe = Pipeline(cfg)
    stop = {"now": False}

    def _stop(*_):
        stop["now"] = True

    signal.signal(signal.SIGINT, _stop)    # Ctrl+C / logout: finish the frame, then clean teardown
    signal.signal(signal.SIGTERM, _stop)

    live_rec = rec = None
    try:
        if control:
            from window_ctrl import is_trusted
            if not args.no_hud:
                from overlay import Overlay
                try:
                    html = paths.SETTINGS_HTML.read_text(encoding="utf-8")
                except OSError:
                    html = ""
                overlay = Overlay(html)
            from router import Router
            trusted = args.dry_run or is_trusted(prompt=True)   # prompt: the system "allow" dialog
            if not trusted:
                log("Accessibility permission missing: tracking only, until it is granted "
                    "(System Settings → Privacy & Security → Accessibility)")
            router = Router(cfg, dry_run=args.dry_run or not trusted, log=log, overlay=overlay)
            router.play("start")   # audible "I'm on"
            if overlay and (not config.user_settings_exist() or not trusted):
                config.mark_onboarded()
                overlay.send(("show_settings",))   # first launch: welcome + permissions
            print("running. Raise an open palm to start.  ctrl+opt+cmd+G = on/off,  Ctrl+C here = quit")

        record_live = cfg.get("debug", {}).get("record_live")
        if record_live is None:
            record_live = not paths.FROZEN
        if record_live and not args.record:
            live_dir = paths.RECORDINGS_DIR / "live"
            old = sorted(live_dir.glob("*.jsonl")) if live_dir.exists() else []
            for f in old[:-9]:          # keep the last 10 live sessions
                f.unlink(missing_ok=True)
            live_rec = Recorder("live", meta={"camera": cfg["camera"], "tracker": cfg["tracker"]},
                                directory=live_dir)

        run_loop(args, cfg, camera, pipe, router, overlay, log, stop, t_start, preview, live_rec)
    except Exception as e:   # anything unexpected: log it and tell the user instead of vanishing
        log(f"!! fatal: {e!r}\n{traceback.format_exc()}")
        if overlay and overlay.alive:
            overlay.send(("fatal", f"{e}\n\nDetails are in the log folder. Opening the app again usually helps."))
            overlay.wait(600)
        else:
            raise
    finally:
        if live_rec:
            live_rec.close()
        if router:
            router.close()
        if overlay:
            overlay.close()
        camera.close()
        if preview:
            debug_view.close()
        if stop.get("restart") and paths.FROZEN:
            # reopen the .app once this process is gone (e.g. to pick up a fresh permission)
            import subprocess
            bundle = str(Path(sys.executable).resolve().parents[2])
            subprocess.Popen(["/bin/sh", "-c", 'sleep 1; open "$0"', bundle], start_new_session=True)
            log("restarting")
        log("quit")
        log_file.close()


def run_loop(args, cfg, camera: Camera, pipe: Pipeline, router, overlay, log, stop, t_start,
             preview: bool, live_rec) -> None:
    rec: Recorder | None = None
    auto_start = time.monotonic() + 3.0 if args.record else None
    show_filtered = True
    frame_times: deque[float] = deque(maxlen=60)
    infer_ms: deque[float] = deque(maxlen=60)
    cpu_t0, wall_t0, cpu_pct = cpu_seconds(), time.monotonic(), 0.0
    last_state = pipe.fsm.state
    perf_t = time.monotonic()
    slow_t = 0.0   # first iteration sends the settings right away
    last_frame_t = time.monotonic()
    stall_logged = False
    errors = 0
    sent_settings = None
    camera_names = list_camera_names()

    def set_enabled(on: bool) -> None:
        released = pipe.fsm.set_enabled(on)
        if router:
            router.enabled = on
            router.submit(released, None, pipe.fsm.state)
        log(f"gesture control {'ON' if on else 'OFF'}")

    def settings_payload() -> dict:
        return {"pointer_speed": cfg["control"].get("pointer_speed", 1.0),
                "sounds": bool(cfg["control"].get("sounds", True)),
                "show_status": bool(cfg.get("ui", {}).get("show_status", True)),
                "camera": str(cfg["camera"].get("device", "builtin")),
                "cameras": camera_names, "camera_state": camera.state, "camera_error": camera.error}

    while not stop["now"]:
        # ----- UI events (handled every iteration, even without camera frames) -------------
        key = -1
        if overlay:
            for ev in overlay.poll_events():
                if ev == "quit":
                    return
                if ev == "restart":
                    stop["restart"] = True
                    return
                if ev == "toggle":
                    set_enabled(not pipe.fsm.enabled)
                elif isinstance(ev, tuple) and ev and ev[0] == "set":
                    applied = config.apply_user(cfg, ev[1])
                    log(f"settings: {applied}")
                    if "camera.device" in applied:
                        camera.reopen()

        now = time.monotonic()
        if now - slow_t >= 1.0:   # once a second: permissions, camera list, settings window
            slow_t = now
            if router and router.dry_run and not args.dry_run:
                from window_ctrl import is_trusted
                if is_trusted():
                    router.dry_run = False
                    log("Accessibility permission granted: control enabled")
            if overlay:
                camera_names = list_camera_names()
                payload = settings_payload()
                if payload != sent_settings:
                    overlay.send(("settings", payload))
                    sent_settings = payload
        cam_state = camera.state
        camera.ensure(pipe.fsm.enabled)
        if camera.state != cam_state:   # opening the camera can take seconds: not a stall
            last_frame_t = time.monotonic()

        got = camera.read()
        if got is None:
            if camera.state == "running" and now - last_frame_t > 3.0 and not stall_logged:
                stall_logged = True
                log("!! no camera frames for 3 s (camera unplugged or used by another app?)")
            if camera.state != "running":
                time.sleep(0.1)
                last_frame_t = time.monotonic()
            continue
        last_frame_t, stall_logged = time.monotonic(), False
        hands, t_cap, img = got

        if auto_start is not None:
            if rec is None and t_cap >= auto_start:
                rec = Recorder(args.label, meta={"camera": cfg["camera"], "tracker": cfg["tracker"]})
                print(f"recording {args.record:.0f}s → {rec.path}")
            elif rec is not None and t_cap >= auto_start + args.record:
                break
        raw = Frame(t_cap - t_start, hands)
        src = camera.source
        if src is not None:
            infer_ms.extend(src.track_ms[-1:])
        try:
            smoothed, feats, intents = pipe.step(raw)
        except Exception as e:   # one bad frame must never take the whole app down
            errors += 1
            if errors <= 3:
                log(f"!! frame error (skipped): {e!r}\n{traceback.format_exc(limit=4)}")
            continue
        if router:
            router.submit(intents, pipe.fsm.cursor, pipe.fsm.state, t_cap,
                          arm=pipe.fsm.arm_progress, dwell=pipe.fsm.dwell_progress, hint=pipe.fsm.hint)
        else:
            for it in intents:
                if not isinstance(it, Move):
                    print(f"{raw.t:7.2f}s  {it}")
        if live_rec:
            live_rec.write(raw)
        if rec:
            rec.write(raw)  # always record RAW, so filter params can be re-tuned on replay

        frame_times.append(t_cap)
        now = time.monotonic()
        if now - wall_t0 >= 1.0:
            cpu_pct = 100 * (cpu_seconds() - cpu_t0) / (now - wall_t0)
            cpu_t0, wall_t0 = cpu_seconds(), now
        if router and now - perf_t >= 10.0 and len(frame_times) > 2 and infer_ms:
            perf_t = now
            gaps = np.diff(np.array(frame_times)) * 1000
            convert = getattr(src, "convert_ms", None)
            log(f"perf: {len(frame_times) / (frame_times[-1] - frame_times[0]):.1f} fps, "
                f"track {np.median(infer_ms):.1f}/{np.percentile(infer_ms, 95):.1f} ms (p50/p95), "
                f"frame gap p95 {np.percentile(gaps, 95):.0f} ms, CPU {cpu_pct:.0f}%"
                + (f", convert p95 {np.percentile(convert, 95):.1f} ms" if convert else "")
                + (f", {src.errors} capture errors (last {src.last_error})" if getattr(src, "errors", 0) else ""))

        if preview and img is not None:
            import debug_view
            key = debug_view.draw_preview(img, raw, smoothed, feats, pipe, src, rec, args.label, auto_start,
                                          args.record, t_cap, frame_times, infer_ms, cpu_pct, show_filtered)
        if key in (ord("q"), 27):
            break
        if key == ord("f"):
            show_filtered = not show_filtered
        if key == ord("k"):
            set_enabled(not pipe.fsm.enabled)
        if key == ord("r"):
            if rec:
                print(f"saved {rec.frames} frames → {rec.close()}")
                rec = None
            else:
                rec = Recorder(args.label, meta={"camera": cfg["camera"], "tracker": cfg["tracker"]})
                print(f"recording → {rec.path}")
        if router and pipe.fsm.state != last_state:
            log(f"state {last_state.value} → {pipe.fsm.state.value}")
        last_state = pipe.fsm.state
    if rec:
        print(f"saved {rec.frames} frames → {rec.close()}")


if __name__ == "__main__":
    multiprocessing.freeze_support()   # the packaged app re-runs this file for the UI process
    main()
