"""The pure, hardware-free part of the pipeline: Frame → identity → filter → features → FSM → intents.

Live mode and replay both go through this, so a recorded session reproduces
exactly what the live app computed.
"""
from __future__ import annotations

from features import FrameFeatures, extract
from fsm import GestureFSM
from hands import Frame
from identity import HandIdentity
from intents import Intent
from one_euro import LandmarkFilter


class Pipeline:
    def __init__(self, cfg: dict):
        f = cfg["filter"]
        self.filter = LandmarkFilter(f["min_cutoff"], f["beta"], f.get("d_cutoff", 1.0))
        self.identity = HandIdentity()
        self.fsm = GestureFSM(cfg)

    def step(self, frame: Frame) -> tuple[Frame, FrameFeatures, list[Intent]]:
        smoothed = self.filter(self.identity(frame))
        feats = extract(smoothed)
        return smoothed, feats, self.fsm.step(feats)
