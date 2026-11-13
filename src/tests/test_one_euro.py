import numpy as np

from hands import Frame, Hand
from one_euro import LandmarkFilter, OneEuro


def test_first_sample_passes_through():
    f = OneEuro()
    assert f(3.0, 0.0) == 3.0


def test_reduces_jitter_at_rest():
    rng = np.random.default_rng(0)
    f = OneEuro(min_cutoff=1.0, beta=0.02)
    noisy = 0.5 + rng.normal(0, 0.005, 300)
    out = np.array([f(x, i / 30) for i, x in enumerate(noisy)])
    assert out[30:].std() < noisy[30:].std() / 3


def test_fast_motion_has_less_lag_with_higher_beta():
    def lag(beta):
        f = OneEuro(min_cutoff=1.0, beta=beta)
        ts = np.arange(60) / 30
        xs = ts * 1.0  # 1 unit/s ramp, fast for normalized coords
        out = [f(x, t) for x, t in zip(xs, ts)]
        return xs[-1] - out[-1]

    assert lag(5.0) < lag(0.0) / 2


def test_vectorized_matches_scalar():
    rng = np.random.default_rng(1)
    xs = rng.random((50, 3))
    vec = OneEuro()
    scal = [OneEuro() for _ in range(3)]
    for i, x in enumerate(xs):
        v = vec(x, i / 30)
        s = [scal[j](x[j], i / 30) for j in range(3)]
        np.testing.assert_allclose(v, s)


def test_landmark_filter_resets_when_hand_disappears():
    lf = LandmarkFilter()
    a = Hand("Right", np.full((21, 3), 0.2), 1.0)
    b = Hand("Right", np.full((21, 3), 0.8), 1.0)
    lf(Frame(0.0, [a]))
    lf(Frame(1 / 30, []))                  # hand lost
    out = lf(Frame(2 / 30, [b]))           # reappears elsewhere
    np.testing.assert_allclose(out.hands[0].landmarks, b.landmarks)  # no smear from old position
