from types import SimpleNamespace as NS

import numpy as np

from vision_tracker import JOINTS, observation_to_hand


def fake_points(conf=0.9):
    return {name: NS(location=lambda i=i: NS(x=0.1 + 0.01 * i, y=0.8 - 0.01 * i), confidence=lambda: conf)
            for i, name in enumerate(JOINTS)}


def test_y_is_flipped_to_top_left_origin_and_order_matches_mediapipe():
    h = observation_to_hand(fake_points(), chirality=-1, confidence=0.95)
    assert h.handedness == "Right"
    np.testing.assert_allclose(h.landmarks[0, :2], [0.1, 0.2])        # wrist: y 0.8 → 0.2
    np.testing.assert_allclose(h.landmarks[8, :2], [0.18, 0.28])      # index tip = MediaPipe index 8


def test_low_confidence_hand_dropped():
    assert observation_to_hand(fake_points(conf=0.1), chirality=-1, confidence=0.9) is None


def test_missing_joint_dropped():
    pts = fake_points()
    pts.pop(JOINTS[4])
    assert observation_to_hand(pts, chirality=-1, confidence=0.9) is None
