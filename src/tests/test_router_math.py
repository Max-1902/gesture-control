import config
import math

from router import PointerMapper, accel_gain


def test_accel_gain_monotonic_and_bounded():
    c = config.load()["control"]
    gains = [accel_gain(v, c) for v in (0, 0.05, 0.1, 0.3, 0.6, 1.0, 3.0)]
    assert gains == sorted(gains)
    assert gains[0] == c["pointer_gain_slow"] and gains[-1] == c["pointer_gain_fast"]


def test_resting_jitter_does_not_move_pointer():
    m = PointerMapper(config.load()["control"])
    rng = __import__("numpy").random.default_rng(0)
    moved = 0.0
    for i in range(300):   # ±1 camera px of noise at rest
        dx, dy = m.step(0.5 + rng.uniform(-1, 1) / 640, 0.5 + rng.uniform(-1, 1) / 480, i / 30)
        moved += math.hypot(dx, dy)
    assert moved == 0.0


def test_slow_motion_still_moves():
    m = PointerMapper(config.load()["control"])
    total = sum(m.step(0.5 + 0.0005 * i, 0.5, i / 30)[0] for i in range(60))   # 0.015 cw/s
    assert total > 30


def test_small_quick_flick_crosses_all_displays():
    m = PointerMapper(config.load()["control"])
    total = sum(m.step(0.5 + 0.2 * min(i, 10) / 10, 0.5, i / 30)[0] for i in range(15))  # 20% of frame in 1/3 s
    assert total > 2500


def test_rebase_does_not_jump():
    m = PointerMapper(config.load()["control"])
    m.step(0.5, 0.5, 0.0)
    m.rebase(0.6, 0.5, 0.1)
    assert m.step(0.6, 0.5, 0.133) == (0.0, 0.0)


def test_router_survives_errors_in_window_calls():
    """One failing AX call (window closed mid-drag, hung app) must not kill the pointer thread."""
    import time
    from fsm import State
    from intents import Click, Grab
    from router import Router
    logs = []
    r = Router(config.load(), dry_run=True, log=logs.append)
    try:
        def boom(*a, **k):
            raise RuntimeError("AX went away")
        r.ctrl.window_at = boom
        r.submit([Grab(0.5, 0.5)], (0.5, 0.5), State.GRABBED)
        time.sleep(0.2)
        assert r._thread.is_alive() and r._ax_thread.is_alive()
        assert any("error (recovered)" in str(l) for l in logs)
        r._click = lambda count: logs.append(f"clicked {count}")
        r.submit([Click(1)], (0.5, 0.5), State.HOVER)
        time.sleep(0.2)
        assert "clicked 1" in logs   # still processing intents after the error
    finally:
        r.close()
