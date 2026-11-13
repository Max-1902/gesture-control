"""Replay recorded sessions through the pipeline, headless.

    python replay.py ../recordings/*.jsonl              # summary per session
    python replay.py session.jsonl --csv out.csv        # per-frame features, for plotting
    python replay.py session.jsonl --check-determinism  # Phase 1 exit criterion
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

import config
from pipeline import Pipeline
from recorder import read_session


def run(path: Path, cfg: dict) -> list[dict]:
    _, frames = read_session(path)
    pipe = Pipeline(cfg)
    rows = []
    for fr in frames:
        _, feats, intents = pipe.step(fr)
        rows.append({**feats.flat(), "state": pipe.fsm.state.value,
                     "intents": ";".join(type(i).__name__ for i in intents)})
    return rows


def digest(rows: list[dict]) -> str:
    return hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()[:16]


def summarize(path: Path, rows: list[dict]) -> str:
    if not rows:
        return f"{path.name}: empty"
    dur = rows[-1]["t"] - rows[0]["t"]
    fps = (len(rows) - 1) / dur if dur > 0 else float("nan")
    present = sum(r["n_hands"] > 0 for r in rows) / len(rows)

    def stat(key: str) -> str:
        v = np.array([r[key] for r in rows if r[key] is not None])
        if not len(v):
            return f"{key}: -"
        p5, p50, p95 = np.percentile(v, [5, 50, 95])
        return f"{key}: p5={p5:.2f} p50={p50:.2f} p95={p95:.2f}"

    arms = sum(1 for a, b in zip(rows, rows[1:]) if a["state"] == "idle" and b["state"] != "idle")
    counts: dict[str, int] = {}
    for r in rows:
        for name in filter(None, r["intents"].split(";")):
            counts[name] = counts.get(name, 0) + 1
    lines = [f"{path.name}: {len(rows)} frames, {dur:.1f}s, {fps:.1f} fps, hand present {present:.0%}",
             f"   arms: {arms}   intents: {counts or 'none'}"]
    for side in "LR":
        lines.append("   " + "  ".join(stat(f"{side}_{k}") for k in ("pinch", "openness", "curl")))
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sessions", nargs="+", type=Path)
    ap.add_argument("--config", type=Path)
    ap.add_argument("--csv", type=Path, help="write per-frame features (single session only)")
    ap.add_argument("--check-determinism", action="store_true")
    args = ap.parse_args(argv)
    cfg = config.load(args.config)

    if args.csv and len(args.sessions) != 1:
        ap.error("--csv takes exactly one session")

    ok = True
    for path in args.sessions:
        rows = run(path, cfg)
        print(summarize(path, rows))
        if args.check_determinism:
            a, b = digest(rows), digest(run(path, cfg))
            same = a == b
            ok &= same
            print(f"   determinism: {'OK' if same else 'MISMATCH'} ({a} vs {b})")
        if args.csv:
            with args.csv.open("w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader()
                w.writerows(rows)
            print(f"   wrote {args.csv}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
