"""Calibration opponent: wraps a real Stockfish binary behind the same
get_move(fen, time_left_ms) -> str contract as agent.py, so the harness
can run it exactly like any other agent.

Third-party engines are banned from the SUBMISSION ZIP, not from local
testing — this module never ships (it's not in tools/build_zip.py's
whitelist) and only exists for calibrating our own rating scale against a
known one via UCI_LimitStrength/UCI_Elo (with Skill Level as a
cross-check).

Configured via environment variables (read once at import, since the
wire protocol's stdio_runner.py always launches a fixed module path — the
harness passes these via AgentProcess(..., env=...) per gauntlet leg):

    STOCKFISH_PATH   path to the engine binary (required)
    STOCKFISH_ELO    if set: setoption UCI_LimitStrength=true, UCI_Elo=<N>
    STOCKFISH_SKILL  if set (and STOCKFISH_ELO isn't): setoption Skill Level=<N>
    STOCKFISH_THREADS  default "1", matching the competition's 1-core constraint

If neither STOCKFISH_ELO nor STOCKFISH_SKILL is set, Stockfish runs at
full strength (Skill Level 20, UCI_LimitStrength false) — not useful for
calibration, but a legitimate "how far are we from full-strength SF" gut
check.
"""
from __future__ import annotations

import atexit
import os
import queue
import subprocess
import sys
import threading

_STOCKFISH_PATH = os.environ.get("STOCKFISH_PATH")
if not _STOCKFISH_PATH:
    raise RuntimeError("STOCKFISH_PATH environment variable must point at a Stockfish binary")

_ELO = os.environ.get("STOCKFISH_ELO")
_SKILL = os.environ.get("STOCKFISH_SKILL")
_THREADS = os.environ.get("STOCKFISH_THREADS", "1")


class _StockfishUCI:
    def __init__(self, path: str):
        self.proc = subprocess.Popen(
            [path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )
        self._queue: "queue.Queue[str]" = queue.Queue()
        self._reader = threading.Thread(target=self._pump, daemon=True)
        self._reader.start()
        self._send("uci")
        self._wait_for("uciok")
        self._configure()
        self._send("isready")
        self._wait_for("readyok")

    def _pump(self) -> None:
        try:
            for line in self.proc.stdout:
                self._queue.put(line.rstrip("\n"))
        except Exception:
            pass

    def _send(self, command: str) -> None:
        self.proc.stdin.write(command + "\n")
        self.proc.stdin.flush()

    def _wait_for(self, token: str, timeout_s: float = 30.0) -> None:
        while True:
            line = self._queue.get(timeout=timeout_s)
            if line.strip() == token or line.strip().startswith(token):
                return

    def _configure(self) -> None:
        self._send(f"setoption name Threads value {_THREADS}")
        if _ELO:
            self._send("setoption name UCI_LimitStrength value true")
            self._send(f"setoption name UCI_Elo value {_ELO}")
        elif _SKILL:
            self._send(f"setoption name Skill Level value {_SKILL}")

    def best_move(self, fen: str, time_left_ms: int) -> str:
        self._send(f"position fen {fen}")
        # We only know our own remaining time, not the opponent's;
        # approximating both sides as equal lets Stockfish's own time
        # manager make a sensible call rather than reimplementing one.
        t = max(1, int(time_left_ms))
        self._send(f"go wtime {t} btime {t} winc 500 binc 500")
        while True:
            line = self._queue.get(timeout=max(5.0, time_left_ms / 1000 + 10))
            if line.startswith("bestmove"):
                return line.split()[1]

    def quit(self) -> None:
        try:
            self._send("quit")
            self.proc.wait(timeout=3)
        except Exception:
            try:
                self.proc.kill()
            except Exception:
                pass


_engine = _StockfishUCI(_STOCKFISH_PATH)
atexit.register(_engine.quit)


def get_move(fen: str, time_left_ms: int) -> str:
    return _engine.best_move(fen, time_left_ms)
