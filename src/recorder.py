"""Recorder / reader for landmark sessions (JSONL).

Line 1 is a header: {"type": "header", "version": 1, "label": ..., "meta": {...}}
Every following line is one Frame.to_json() (raw, unfiltered landmarks).
No images are ever stored, only 21 points per hand.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Iterator

from hands import Frame
from paths import RECORDINGS_DIR

FORMAT_VERSION = 1


class Recorder:
    def __init__(self, label: str, meta: dict | None = None, directory: Path = RECORDINGS_DIR):
        directory.mkdir(parents=True, exist_ok=True)
        safe = "".join(c if c.isalnum() or c in "-_" else "-" for c in label) or "session"
        self.path = directory / f"{time.strftime('%Y%m%d-%H%M%S')}_{safe}.jsonl"
        self._f = self.path.open("w", buffering=1, encoding="utf-8")  # line-buffered: a crash loses ≤ 1 frame
        self._t0: float | None = None
        self.frames = 0
        self._write({"type": "header", "version": FORMAT_VERSION, "label": label,
                     "created": time.strftime("%Y-%m-%dT%H:%M:%S"), "meta": meta or {}})

    def _write(self, obj: dict) -> None:
        self._f.write(json.dumps(obj, separators=(",", ":")) + "\n")

    def write(self, frame: Frame) -> None:
        """Timestamps are stored relative to the first recorded frame."""
        if self._t0 is None:
            self._t0 = frame.t
        self._write(Frame(frame.t - self._t0, frame.hands).to_json())
        self.frames += 1

    def close(self) -> Path:
        self._f.close()
        return self.path


def read_session(path: Path | str) -> tuple[dict, Iterator[Frame]]:
    path = Path(path)
    with path.open() as f:
        header = json.loads(f.readline())
    if header.get("type") != "header":
        raise ValueError(f"{path}: missing header line")
    if header.get("version") != FORMAT_VERSION:
        raise ValueError(f"{path}: unsupported version {header.get('version')}")

    def frames() -> Iterator[Frame]:
        with path.open() as f:
            next(f)
            for line in f:
                if line.strip():
                    yield Frame.from_json(json.loads(line))

    return header, frames()
