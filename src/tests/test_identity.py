import numpy as np

from conftest import make_hand
from hands import Frame, Hand
from identity import HandIdentity
from pipeline import Pipeline
import config


def relabel(h: Hand, side: str) -> Hand:
    return Hand(side, h.landmarks.copy(), h.score)


def test_label_flicker_is_ignored():
    ident = HandIdentity()
    h = make_hand("open", "Right", (0.5, 0.5))
    labels = [ident(Frame(i / 30, [relabel(h, "Left" if i % 2 else "Right")])).hands[0].handedness
              for i in range(30)]
    assert set(labels) == {"Right"}


def test_two_hands_keep_identity_when_labels_swap():
    ident = HandIdentity()
    l, r = make_hand("open", "Left", (0.3, 0.5)), make_hand("open", "Right", (0.7, 0.5))
    ident(Frame(0, [l, r]))
    out = ident(Frame(1 / 30, [relabel(l, "Right"), relabel(r, "Left")]))   # MediaPipe swaps
    by_x = sorted(out.hands, key=lambda h: h.landmarks[0, 0])
    assert [h.handedness for h in by_x] == ["Left", "Right"]


def test_duplicate_labels_resolved():
    ident = HandIdentity()
    out = ident(Frame(0, [make_hand("open", "Right", (0.3, 0.5)), make_hand("open", "Right", (0.7, 0.5))]))
    assert sorted(h.handedness for h in out.hands) == ["Left", "Right"]


def test_far_jump_is_a_new_hand():
    ident = HandIdentity()
    ident(Frame(0, [make_hand("open", "Right", (0.2, 0.5))]))
    out = ident(Frame(1 / 30, [make_hand("open", "Left", (0.8, 0.5))]))  # 6 hand sizes away
    assert out.hands[0].handedness == "Left"


def test_flicker_does_not_disarm_fsm():
    pipe = Pipeline(config.load())
    for i in range(30):
        pipe.step(Frame(i / 30, [make_hand("open", "Right", center=(0.5, 0.8), scale=2.5)]))
    from fsm import State
    assert pipe.fsm.state == State.HOVER
    states = set()
    for i in range(30, 90):
        side = "Left" if i % 3 == 0 else "Right"
        pipe.step(Frame(i / 30, [make_hand("open", side, center=(0.5, 0.8), scale=2.5)]))
        states.add(pipe.fsm.state)
    assert states == {State.HOVER}
