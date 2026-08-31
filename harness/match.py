"""Runs one game between two agent processes over the stdio wire protocol,
enforcing the real clock and detecting illegal moves / crashes / flags.
"""
from __future__ import annotations

import dataclasses
import json
import os
import queue
import subprocess
import sys
import threading
import time

import chess

from cb_time import INIT_BUDGET_MS
from harness.clock import SideClock

DEFAULT_TIME_MS = 120_000
DEFAULT_INCREMENT_MS = 500
MAX_OUTPUT_BYTES = 4096
MAX_FULLMOVES = 300  # adjudicate as a draw beyond this, rather than loop forever

_STDIO_RUNNER = os.path.join(os.path.dirname(__file__), "stdio_runner.py")


class AgentProcess:
    """A running agent subprocess, speaking the stdio wire protocol.

    stdout is drained by a background thread into a queue so reads can be
    timed out cross-platform (no reliance on POSIX-only select() on pipes).
    """

    def __init__(self, agent_path: str, name: str):
        self.name = name
        self.agent_path = agent_path
        self.proc = subprocess.Popen(
            [sys.executable, _STDIO_RUNNER, agent_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        self.init_ms: float | None = None
        self._queue: "queue.Queue[str]" = queue.Queue()
        self._reader = threading.Thread(target=self._pump_stdout, daemon=True)
        self._reader.start()

    def _pump_stdout(self) -> None:
        try:
            for line in self.proc.stdout:
                self._queue.put(line.rstrip("\n"))
        except Exception:
            pass

    def _readline(self, timeout_s: float) -> str | None:
        try:
            return self._queue.get(timeout=max(0.0, timeout_s))
        except queue.Empty:
            return None

    def wait_ready(self, timeout_s: float = INIT_BUDGET_MS / 1000 + 10) -> None:
        start = time.perf_counter()
        line = self._readline(timeout_s)
        self.init_ms = (time.perf_counter() - start) * 1000
        if line is None:
            stderr = self._drain_stderr()
            raise RuntimeError(
                f"{self.name}: no ready signal within {timeout_s:.1f}s "
                f"(crashed or hung during import/init). stderr:\n{stderr}"
            )
        msg = json.loads(line)
        if msg.get("status") != "ready":
            raise RuntimeError(f"{self.name}: unexpected message before ready: {msg}")

    def request_move(
        self, fen: str, time_left_ms: float, timeout_s: float
    ) -> tuple[str | None, float, str | None]:
        """Returns (move_uci_or_None, elapsed_ms, error_reason_or_None)."""
        request = json.dumps({"fen": fen, "time_left_ms": int(time_left_ms)}) + "\n"
        start = time.perf_counter()
        try:
            self.proc.stdin.write(request)
            self.proc.stdin.flush()
        except Exception as exc:
            return None, 0.0, f"stdin_write_failed:{exc}"

        line = self._readline(timeout_s)
        elapsed_ms = (time.perf_counter() - start) * 1000
        if line is None:
            return None, elapsed_ms, "timeout_or_crash"

        payload_bytes = len(line.encode("utf-8"))
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            return None, elapsed_ms, "malformed_output"

        if payload_bytes > MAX_OUTPUT_BYTES:
            return None, elapsed_ms, f"output_too_large:{payload_bytes}"
        if "error" in msg:
            return None, elapsed_ms, f"engine_error:{msg['error']}"

        move = msg.get("move")
        if not isinstance(move, str):
            return None, elapsed_ms, "malformed_output"
        return move, elapsed_ms, None

    def _drain_stderr(self, max_chars: int = 2000) -> str:
        try:
            self.proc.stderr.flush()
        except Exception:
            pass
        try:
            data = self.proc.stderr.read(max_chars) or ""
        except Exception:
            data = ""
        return data

    def close(self) -> None:
        try:
            self.proc.stdin.close()
        except Exception:
            pass
        try:
            self.proc.terminate()
            self.proc.wait(timeout=5)
        except Exception:
            try:
                self.proc.kill()
            except Exception:
                pass


@dataclasses.dataclass
class GameResult:
    winner: str | None  # "white", "black", or None for a draw
    reason: str
    plies: int
    white_times_ms: list
    black_times_ms: list


def _opponent(side: str) -> str:
    return "black" if side == "white" else "white"


def play_game(
    white_agent_path: str,
    black_agent_path: str,
    time_ms: float = DEFAULT_TIME_MS,
    increment_ms: float = DEFAULT_INCREMENT_MS,
    max_fullmoves: int = MAX_FULLMOVES,
) -> GameResult:
    white = AgentProcess(white_agent_path, "white")
    black = AgentProcess(black_agent_path, "black")
    procs = {"white": white, "black": black}
    try:
        white.wait_ready()
        black.wait_ready()
        for side, proc in procs.items():
            if proc.init_ms is not None and proc.init_ms > INIT_BUDGET_MS:
                return GameResult(_opponent(side), f"{side}_init_budget_exceeded", 0, [], [])

        board = chess.Board()
        clocks = {
            "white": SideClock(time_ms, increment_ms),
            "black": SideClock(time_ms, increment_ms),
        }
        times = {"white": [], "black": []}

        while not board.is_game_over(claim_draw=True):
            if board.fullmove_number > max_fullmoves:
                return GameResult(None, "adjudicated_move_limit", board.ply(), times["white"], times["black"])

            side = "white" if board.turn == chess.WHITE else "black"
            proc = procs[side]
            timeout_s = clocks[side].remaining_ms / 1000 + 2  # IPC/OS grace only

            move_str, elapsed_ms, err = proc.request_move(board.fen(), clocks[side].remaining_ms, timeout_s)
            times[side].append(elapsed_ms)

            if clocks[side].consume(elapsed_ms):
                return GameResult(_opponent(side), f"{side}_flag", board.ply(), times["white"], times["black"])

            if err is not None:
                return GameResult(_opponent(side), f"{side}_{err}", board.ply(), times["white"], times["black"])

            try:
                move = chess.Move.from_uci(move_str)
            except Exception:
                return GameResult(_opponent(side), f"{side}_illegal_move_format", board.ply(), times["white"], times["black"])

            if move not in board.legal_moves:
                return GameResult(_opponent(side), f"{side}_illegal_move", board.ply(), times["white"], times["black"])

            board.push(move)

        outcome = board.outcome(claim_draw=True)
        winner = None
        if outcome is not None:
            if outcome.winner is True:
                winner = "white"
            elif outcome.winner is False:
                winner = "black"
        reason = outcome.termination.name.lower() if outcome is not None else "unknown"
        return GameResult(winner, reason, board.ply(), times["white"], times["black"])
    finally:
        white.close()
        black.close()
