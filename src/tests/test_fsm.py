import glob
from pathlib import Path

import numpy as np
import pytest

import config
from conftest import make_hand
from features import extract
from fsm import GestureFSM, State
from hands import Frame
from intents import (Click, Grab, Minimize, Move, Release, Resize, ResizeCancel, ResizeEnd,
                     ResizeStart)
from pipeline import Pipeline
from recorder import read_session

FPS = 30


class Sim:
    """Drives the FSM with synthetic hands at 30 fps."""

    def __init__(self, **overrides):
        cfg = config.load()
        cfg["gestures"].update(overrides)
        self.fsm = GestureFSM(cfg)
        self.t = 0.0
        self.log: list = []

    def run(self, seconds: float, *hands_fn):
        """hands_fn: callables (frame_index) -> Hand | None."""
        out = []
        for i in range(round(seconds * FPS)):
            hands = [h for h in (fn(i) for fn in hands_fn) if h is not None]
            out += self.fsm.step(extract(Frame(self.t, hands)))
            self.t += 1 / FPS
        self.log += out
        return out

    def arm(self, side="Right"):
        # arm with the same-size hand the test continues with (a sudden 2.5x size change
        # is rejected as a phantom detection); the size gate is covered by its own tests
        g = self.fsm.g
        saved, g["engage_min_hand_size"] = g["engage_min_hand_size"], 0.05
        self.run(1.0, lambda i: make_hand("open", side))
        g["engage_min_hand_size"] = saved
        assert self.fsm.state == State.HOVER, self.fsm.state


def still(pose, side="Right", center=(0.5, 0.5)):
    return lambda i: make_hand(pose, side, center)


def types(intents):
    return [type(i) for i in intents]


# --- engage ------------------------------------------------------------------------------------

def test_open_palm_held_arms():
    sim = Sim()
    sim.arm()
    assert sim.fsm.primary == "Right"


def test_short_palm_does_not_arm():
    sim = Sim()
    sim.run(0.4, lambda i: make_hand("open", center=(0.5, 0.8), scale=2.5))
    assert sim.fsm.state == State.IDLE
    assert 0.3 < sim.fsm.arm_progress < 1.0


def test_moving_palm_does_not_arm():
    sim = Sim()
    sim.run(2.0, lambda i: make_hand("open", center=(0.3 + 0.01 * i, 0.8), scale=2.5))  # 1.2 hand sizes/s
    assert sim.fsm.state == State.IDLE


@pytest.mark.parametrize("pose", ["pinch", "fist", "point"])
def test_nothing_but_palm_leaves_idle(pose):
    """Including when held close to the camera."""
    sim = Sim()
    assert sim.run(3.0, lambda i: make_hand(pose, center=(0.5, 0.8), scale=2.5)) == []
    assert sim.fsm.state == State.IDLE


# --- drag --------------------------------------------------------------------------------------

def test_small_far_palm_does_not_arm():
    sim = Sim()
    sim.run(2.0, still("open"))     # hand_size 0.1: resting at keyboard distance
    assert sim.fsm.state == State.IDLE


def test_grab_drag_release():
    sim = Sim()
    sim.arm()
    sim.run(0.3, still("open"))
    sim.run(0.8, still("pinch"))
    drag = sim.run(1.0, lambda i: make_hand("pinch", center=(0.5 + 0.003 * i, 0.5)))
    rel = sim.run(0.3, still("open", center=(0.59, 0.5)))
    assert types(sim.log)[0] is Grab
    total_dx = sum(m.dx for m in drag if isinstance(m, Move))
    assert total_dx == pytest.approx(0.087 * sim.fsm.g["drag_gain"], rel=0.05)   # 29 steps * 0.003
    assert Release in types(rel)


def test_hand_lost_while_grabbed_releases_then_disarms():
    sim = Sim()
    sim.arm()
    sim.run(0.3, still("open"))
    sim.run(0.8, still("pinch"))
    out = sim.run(0.4, lambda i: None)          # longer than lost_grace_s
    assert types(out) == [Release]
    assert sim.fsm.state == State.ARMED
    sim.run(3.5, lambda i: None)
    assert sim.fsm.state == State.IDLE


def test_brief_dropout_does_not_drop_window():
    sim = Sim()
    sim.arm()
    sim.run(0.3, still("open"))
    sim.run(0.8, still("pinch"))
    out = sim.run(0.15, lambda i: None)         # tracker loses the hand for ~4 frames
    out += sim.run(0.8, still("pinch"))
    assert Release not in types(out) and Grab not in types(out)
    assert sim.fsm.state == State.GRABBED


def test_no_regrab_from_arming_pinch_carryover():
    """Arming then immediately pinching must still need the pinch to register fresh."""
    sim = Sim()
    sim.arm()
    out = sim.run(0.9, still("pinch"))
    assert types(out) == [Grab]


def test_quick_pinch_tap_is_click():
    sim = Sim()
    sim.arm()
    sim.run(0.3, still("open"))
    out = sim.run(0.2, still("pinch")) + sim.run(0.3, still("open"))
    assert out == [Click(1)]
    assert sim.fsm.state == State.HOVER


def test_double_tap_is_double_click():
    sim = Sim()
    sim.arm()
    sim.run(0.3, still("open"))
    out = []
    for _ in range(2):
        out += sim.run(0.2, still("pinch")) + sim.run(0.2, still("open"))
    assert out == [Click(1), Click(2)]


def test_slow_taps_are_two_single_clicks():
    sim = Sim()
    sim.arm()
    sim.run(0.3, still("open"))
    out = sim.run(0.2, still("pinch")) + sim.run(1.0, still("open"))
    out += sim.run(0.2, still("pinch")) + sim.run(0.3, still("open"))
    assert out == [Click(1), Click(1)]


def test_slow_tap_opening_is_click_not_flash_grab():
    """Live bug: a 0.6 s+ tap whose fingers were already parting became Grab+Release in 30 ms."""
    sim = Sim()
    sim.arm()
    sim.run(0.3, still("open"))
    out = sim.run(0.4, still("pinch")) + sim.run(0.4, still("loose")) + sim.run(0.3, still("open"))
    assert out == [Click(1)]


def test_long_still_hold_released_does_nothing_or_drags():
    """A firm still pinch past tap_max_s is a hold (grab), never a click."""
    sim = Sim()
    sim.arm()
    sim.run(0.3, still("open"))
    out = sim.run(1.0, still("pinch")) + sim.run(0.3, still("open"))
    assert Click not in types(out) and types(out) == [Grab, Release]


def test_phantom_small_hand_cannot_grab():
    """Live bug: a false detection 1/4 the size of the real hand pinched and grabbed."""
    sim = Sim()
    sim.arm()
    sim.run(0.5, still("open"))
    out = sim.run(1.0, lambda i: make_hand("pinch", scale=0.25))
    assert Grab not in types(out) and Click not in types(out)


def test_tight_curled_pinch_is_not_a_fist():
    """Live bug: curl 0.24 + pinch 0.15 minimized the window instead of pinching."""
    sim = Sim()
    sim.arm()
    sim.run(0.3, still("open"))
    sim.fsm.g.update(fist_curl_on=2.0, fist_curl_off=2.1)   # now every pose is "curled enough"
    out = sim.run(1.5, still("pinch"))
    assert Minimize not in types(out) and Grab in types(out)


# --- hand at the camera's frame edge (live: hand resting low, 47-68% of frames cut off) ---

def test_palm_cut_off_by_frame_edge_does_not_arm():
    sim = Sim()
    sim.run(2.0, lambda i: make_hand("open", center=(0.5, 0.45), scale=2.5))   # fingertips above the frame
    assert sim.fsm.state == State.IDLE and sim.fsm.hint == "edge"


def test_cut_off_hand_cannot_click_or_grab():
    sim = Sim()
    sim.arm()
    sim.run(0.3, still("open"))
    out = sim.run(1.0, still("pinch", center=(0.5, 1.08)))   # wrist below the frame
    assert out == [] and sim.fsm.hint == "low"


def test_grab_survives_sliding_out_at_the_edge():
    sim = Sim()
    sim.arm()
    sim.run(0.3, still("open"))
    sim.run(0.8, still("pinch", center=(0.5, 1.04)))
    assert sim.fsm.state == State.GRABBED
    out = sim.run(0.4, lambda i: None)          # longer than lost_grace_s, shorter than 2.5x
    out += sim.run(0.3, still("pinch", center=(0.5, 1.04)))
    assert Release not in types(out) and sim.fsm.state == State.GRABBED


def test_pinch_and_move_drags_immediately():
    sim = Sim()
    sim.arm()
    sim.run(0.3, still("open"))
    out = sim.run(0.3, lambda i: make_hand("pinch", center=(0.5 + 0.01 * i, 0.5)))
    assert Grab in types(out) and Click not in types(out)


def test_armed_idle_timeout():
    sim = Sim()
    sim.arm()
    sim.run(9.0, still("point"))
    assert sim.fsm.state == State.IDLE


# --- minimize ----------------------------------------------------------------------------------

def test_fist_dwell_minimizes():
    sim = Sim()
    sim.arm()
    sim.run(0.3, still("open"))
    out = sim.run(1.3, still("fist"))
    assert types(out) == [Minimize]


def test_fist_released_early_cancels():
    sim = Sim()
    sim.arm()
    sim.run(0.3, still("open"))
    sim.run(0.6, still("fist"))
    assert 0.3 < sim.fsm.dwell_progress < 1.0
    out = sim.run(0.5, still("open"))
    assert out == [] and sim.fsm.dwell_progress == 0.0


def test_fist_while_grabbed_never_minimizes():
    sim = Sim()
    sim.arm()
    sim.run(0.3, still("open"))
    sim.run(0.8, still("pinch"))
    out = sim.run(1.5, still("fist"))
    assert Minimize not in types(out)


# --- two-hand resize ---------------------------------------------------------------------------

def test_two_hand_resize_and_end():
    sim = Sim()
    sim.arm()
    sim.run(0.3, still("open"), still("open", "Left", (0.3, 0.5)))
    sim.run(0.3, still("pinch"), still("pinch", "Left", (0.3, 0.5)))
    spread = sim.run(1.0, lambda i: make_hand("pinch", "Right", (0.5 + 0.003 * i, 0.5)),
                     lambda i: make_hand("pinch", "Left", (0.3 - 0.003 * i, 0.5)))
    end = sim.run(0.3, still("open", center=(0.59, 0.5)), still("open", "Left", (0.21, 0.5)))
    kinds = types(sim.log)
    assert ResizeStart in kinds and Grab not in kinds[: kinds.index(ResizeStart)] or Release in kinds
    scales = [r.scale for r in spread if isinstance(r, Resize)]
    assert scales[-1] == pytest.approx(0.38 / 0.2, rel=0.1)
    assert ResizeEnd in types(end)


def test_hand_drop_mid_resize_cancels():
    sim = Sim()
    sim.arm()
    sim.run(0.3, still("open"), still("open", "Left", (0.3, 0.5)))
    sim.run(0.5, still("pinch"), still("pinch", "Left", (0.3, 0.5)))
    out = sim.run(0.8, still("pinch"))           # left hand vanished
    assert ResizeCancel in types(out)
    assert sim.fsm.state == State.HOVER
    assert Grab not in types(out)                # still-pinched right hand must not re-grab


# --- kill switch & invariants ------------------------------------------------------------------

def test_kill_switch_releases_and_silences():
    sim = Sim()
    sim.arm()
    sim.run(0.3, still("open"))
    sim.run(0.8, still("pinch"))
    assert types(sim.fsm.set_enabled(False)) == [Release]
    assert sim.fsm.state == State.IDLE
    assert sim.run(2.0, still("open")) == []


def test_random_sequences_never_leave_drag_or_resize_open():
    rng = np.random.default_rng(7)
    poses = ["open", "pinch", "fist", "point", None]
    for trial in range(30):
        sim = Sim()
        seq_r = rng.choice(len(poses), 400, p=[0.35, 0.3, 0.1, 0.1, 0.15])
        seq_l = rng.choice(len(poses), 400, p=[0.2, 0.3, 0.05, 0.05, 0.4])
        rpose = [poses[k] for k in np.repeat(seq_r, 6)]
        lpose = [poses[k] for k in np.repeat(seq_l, 6)]
        sim.run(len(rpose) / FPS,
                lambda i: rpose[i] and make_hand(rpose[i], "Right"),
                lambda i: lpose[i] and make_hand(lpose[i], "Left", (0.3, 0.5)))
        sim.log += sim.fsm.set_enabled(False)
        dragging = resizing = False
        for it in sim.log:
            if isinstance(it, Grab):
                assert not dragging and not resizing; dragging = True
            elif isinstance(it, Release):
                assert dragging; dragging = False
            elif isinstance(it, ResizeStart):
                assert not dragging and not resizing; resizing = True
            elif isinstance(it, (ResizeEnd, ResizeCancel)):
                assert resizing; resizing = False
            elif isinstance(it, Move):
                assert dragging
            elif isinstance(it, Resize):
                assert resizing
        assert not dragging and not resizing, f"trial {trial} left an operation open"


# --- real recordings ---------------------------------------------------------------------------

IDLE_SESSIONS = sorted(glob.glob(str(Path(__file__).resolve().parents[2] / "recordings" / "*idle-typing*.jsonl"))
                       + glob.glob(str(Path(__file__).resolve().parents[2] / "recordings" / "*talking*.jsonl")))


@pytest.mark.skipif(not IDLE_SESSIONS, reason="no idle recordings")
@pytest.mark.parametrize("path", IDLE_SESSIONS, ids=lambda p: Path(p).name)
def test_zero_false_arms_on_idle_recordings(path):
    """Phase 3 exit criterion: typing/talking never arms the system."""
    _, frames = read_session(path)
    pipe = Pipeline(config.load())
    for fr in frames:
        _, _, intents = pipe.step(fr)
        assert pipe.fsm.state == State.IDLE, f"false arm at t={fr.t:.2f}s"
        assert intents == []
