"""Phase 1 exit criterion: a recorded session replays to identical output twice."""
import numpy as np

import config
import replay
from hands import Frame
from recorder import Recorder, read_session


def _record(tmp_path, hand_factory, n=120):
    rng = np.random.default_rng(42)
    rec = Recorder("synthetic", directory=tmp_path)
    for i in range(n):
        hands = []
        if 10 <= i < 100:
            pose = "pinch" if 40 <= i < 70 else "open"
            h = hand_factory(pose, center=(0.4 + i * 0.002, 0.5))
            h.landmarks += rng.normal(0, 0.002, h.landmarks.shape)  # tracker jitter
            hands.append(h)
        rec.write(Frame(100.0 + i / 30, hands))  # absolute t; recorder should rebase to 0
    return rec.close()


def test_roundtrip(tmp_path, hand_factory):
    path = _record(tmp_path, hand_factory)
    header, frames = read_session(path)
    frames = list(frames)
    assert header["label"] == "synthetic"
    assert len(frames) == 120
    assert frames[0].t == 0.0
    assert sum(1 for f in frames if f.hands) == 90


def test_deterministic(tmp_path, hand_factory):
    path = _record(tmp_path, hand_factory)
    cfg = config.load()
    assert replay.digest(replay.run(path, cfg)) == replay.digest(replay.run(path, cfg))
    assert replay.main([str(path), "--check-determinism"]) == 0


def test_pinch_visible_in_replay(tmp_path, hand_factory):
    # Pinch starts at frame 40; with default filter params it must register within 5 frames
    rows = replay.run(_record(tmp_path, hand_factory), config.load())
    pinched = [r["R_pinch"] for r in rows[45:65]]
    released = [r["R_pinch"] for r in rows[80:95]]
    assert max(pinched) < 0.25 < 0.35 < min(released)


def test_csv_export(tmp_path, hand_factory):
    out = tmp_path / "f.csv"
    assert replay.main([str(_record(tmp_path, hand_factory)), "--csv", str(out)]) == 0
    assert out.read_text().splitlines()[0].startswith("t,n_hands,two_hand_span")
