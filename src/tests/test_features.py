import pytest

from features import extract, hand_features
from hands import Frame


@pytest.mark.parametrize("scale", [0.5, 1.0, 2.0])
def test_scale_invariant(hand_factory, scale):
    ref = hand_features(hand_factory("open"))
    hf = hand_features(hand_factory("open", scale=scale))
    assert hf.pinch == pytest.approx(ref.pinch)
    assert hf.openness == pytest.approx(ref.openness)
    assert hf.hand_size == pytest.approx(ref.hand_size * scale)


def test_pinch_separates(hand_factory):
    assert hand_features(hand_factory("pinch")).pinch < 0.15
    assert hand_features(hand_factory("open")).pinch > 0.5


def test_fist_vs_open(hand_factory):
    fist, open_ = hand_features(hand_factory("fist")), hand_features(hand_factory("open"))
    assert fist.curl < 0.9 < 1.2 < open_.curl
    assert fist.openness < open_.openness


def test_pointing(hand_factory):
    hf = hand_features(hand_factory("point"))
    assert hf.index_extension > 1.2
    assert hf.others_extension < 0.9


def test_two_hand_span(hand_factory):
    near = extract(Frame(0, [hand_factory(side="Left", center=(0.4, 0.5)),
                             hand_factory(side="Right", center=(0.6, 0.5))]))
    far = extract(Frame(0, [hand_factory(side="Left", center=(0.2, 0.5)),
                            hand_factory(side="Right", center=(0.8, 0.5))]))
    assert near.two_hand_span == pytest.approx(2.0)   # 0.2 apart / hand_size 0.1
    assert far.two_hand_span == pytest.approx(3 * near.two_hand_span)
    assert extract(Frame(0, [hand_factory()])).two_hand_span is None
