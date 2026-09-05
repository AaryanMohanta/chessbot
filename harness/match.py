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
MAX_PLIES = 300  # matches the competition's own 300-ply adjudication-on-material rule

# Score-based adjudication (self-play testing only -- see play_game's
# ``adjudicate`` param and AgentProcess.request_move's optional score).
# Long dead endgames are a large fraction of wall clock in dev SPRT and
# contribute almost nothing to the statistical signal, so cutting them
# short (only when both sides are actually reporting a score, i.e. our
# own agents playing each other -- Stockfish/third-party opponents don't
# report one over this wire protocol, so adjudication silently never
# triggers against them) is close to free throughput.
RESIGN_THRESHOLD_CP = 600
RESIGN_CONSECUTIVE_PLIES = 4
DRAW_SCORE_EPSILON_CP = 5  # "0cp" with a little slack for engine rounding
DRAW_CONSECUTIVE_PLIES = 60

_STDIO_RUNNER = os.path.join(os.path.dirname(__file__), "stdio_runner.py")


class AgentProcess:
    """A running agent subprocess, speaking the stdio wire protocol.

    stdout is drained by a background thread into a queue so reads can be
    timed out cross-platform (no reliance on POSIX-only select() on pipes).
    """

    def __init__(self, agent_path: str, name: str, env: dict | None = None, cpu_affinity: int | None = None):
        self.name = name
        self.agent_path = agent_path
        proc_env = {**os.environ, **env} if env else None
        # cpu_affinity: hard-pin this process to one specific core via
        # taskset (Linux only), rather than relying on a cgroup CPU quota
        # (e.g. docker --cpus=N) to keep concurrent games from stealing
        # cycles from each other. A soft quota shared between two
        # processes (this agent + its opponent) was observed to starve
        # the numba compile badly enough to cause spurious
        # init_budget_exceeded losses under concurrent load -- a
        # test-harness artifact, not real engine weakness.
        command = [sys.executable, _STDIO_RUNNER, agent_path]
        if cpu_affinity is not None and sys.platform.startswith("linux"):
            # taskset doesn't exist on Windows -- pinned separately below
            # via psutil instead, since Popen itself has no cross-platform
            # affinity knob.
            command = ["taskset", "-c", str(cpu_affinity)] + command
        self.proc = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            env=proc_env,
        )
        if cpu_affinity is not None and sys.platform.startswith("win"):
            # Windows has no taskset; psutil wraps SetProcessAffinityMask.
            # Best-effort only (2026-09, for approximating the
            # competition's "1 dedicated core" constraint on a Windows
            # dev box) -- this pins to one CORE, it does NOT reproduce the
            # reference hardware's actual clock speed (AMD EPYC 9V74 @
            # 2.60 GHz), which Windows has no supported per-process knob
            # for. Silently skipped if psutil isn't installed, same
            # "don't fail the launch over a missing benchmarking nicety"
            # policy as the Linux taskset path.
            try:
                import psutil
                psutil.Process(self.proc.pid).cpu_affinity([cpu_affinity])
            except Exception:
                pass
        self.init_ms: float | None = None
        self._queue: "queue.Queue[str]" = queue.Queue()
        self._reader = threading.Thread(target=self._pump_stdout, daemon=True)
        self._reader.start()

        # stderr must be drained continuously too, not just lazily on an
        # error path -- discovered via a real ~2-hour hang during a
        # concurrent Stockfish calibration run (2026-09): once a child's
        # stderr pipe (64KB on Windows) fills up because nobody is
        # reading it, the child's own next print(..., file=sys.stderr)
        # blocks forever inside the OS pipe write, which means it never
        # gets back to reading stdin or writing stdout -- and this
        # class's own request_move timeout, bounded on stdout alone,
        # can't detect or recover from that: the child is alive and
        # "responding" as far as the process table is concerned, just
        # permanently stuck on a write() syscall. Kept bounded (last
        # _STDERR_TAIL_CHARS) since a runaway warning loop could
        # otherwise grow this without limit for the lifetime of the game.
        self._stderr_lines: list[str] = []
        self._stderr_lock = threading.Lock()
        self._stderr_reader = threading.Thread(target=self._pump_stderr, daemon=True)
        self._stderr_reader.start()

    def _pump_stdout(self) -> None:
        try:
            for line in self.proc.stdout:
                self._queue.put(line.rstrip("\n"))
        except Exception:
            pass

    def _pump_stderr(self) -> None:
        try:
            for line in self.proc.stderr:
                with self._stderr_lock:
                    self._stderr_lines.append(line.rstrip("\n"))
                    del self._stderr_lines[:-200]  # bounded tail, see __init__ comment
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
    ) -> tuple[str | None, float, str | None, float | None]:
        """Returns (move_uci_or_None, elapsed_ms, error_reason_or_None,
        score_cp_or_None). ``score`` is the mover's own last-search score
        in centipawns from the mover's perspective (None if the agent
        doesn't report one -- see stdio_runner.py's optional
        get_last_score() hook, and agent.py's -- purely diagnostic, never
        required by the real get_move contract)."""
        request = json.dumps({"fen": fen, "time_left_ms": int(time_left_ms)}) + "\n"
        start = time.perf_counter()
        try:
            self.proc.stdin.write(request)
            self.proc.stdin.flush()
        except Exception as exc:
            return None, 0.0, f"stdin_write_failed:{exc}", None

        line = self._readline(timeout_s)
        elapsed_ms = (time.perf_counter() - start) * 1000
        if line is None:
            return None, elapsed_ms, "timeout_or_crash", None

        payload_bytes = len(line.encode("utf-8"))
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            return None, elapsed_ms, "malformed_output", None

        if payload_bytes > MAX_OUTPUT_BYTES:
            return None, elapsed_ms, f"output_too_large:{payload_bytes}", None
        if "error" in msg:
            return None, elapsed_ms, f"engine_error:{msg['error']}", None

        move = msg.get("move")
        if not isinstance(move, str):
            return None, elapsed_ms, "malformed_output", None
        score = msg.get("score")
        score = float(score) if isinstance(score, (int, float)) else None
        return move, elapsed_ms, None, score

    def _drain_stderr(self, max_chars: int = 2000) -> str:
        # Reads the tail buffer _pump_stderr has been continuously
        # filling in the background -- must not read self.proc.stderr
        # directly here, since that pipe now belongs exclusively to that
        # thread (a second concurrent reader would race it).
        with self._stderr_lock:
            data = "\n".join(self._stderr_lines)
        return data[-max_chars:]

    def close(self) -> None:
        """Close stdin (EOF) and give the process a chance to exit on its
        own before escalating to terminate/kill. This matters beyond tidy
        shutdown: a well-behaved agent (ours, or a wrapper around a nested
        subprocess like the Stockfish calibration opponent) runs its own
        cleanup — e.g. telling a child engine process to quit — when its
        stdin loop ends naturally. TerminateProcess on Windows (what
        subprocess.terminate() does) skips all of that and would orphan
        any subprocess the agent itself spawned."""
        try:
            self.proc.stdin.close()
        except Exception:
            pass
        try:
            self.proc.wait(timeout=3)
            return
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
    moves_uci: list = dataclasses.field(default_factory=list)


def _opponent(side: str) -> str:
    return "black" if side == "white" else "white"


def play_game(
    white_agent_path: str,
    black_agent_path: str,
    time_ms: float = DEFAULT_TIME_MS,
    increment_ms: float = DEFAULT_INCREMENT_MS,
    max_plies: int = MAX_PLIES,
    start_fen: str | None = None,
    white_env: dict | None = None,
    black_env: dict | None = None,
    white_cpu: int | None = None,
    black_cpu: int | None = None,
    adjudicate: bool = True,
) -> GameResult:
    """Play one game. ``start_fen`` defaults to the standard starting
    position; pass an opening-book FEN to start from there instead (the
    moves that produced it are not recorded here — the caller, e.g. the
    gauntlet runner, knows the opening_id and can record that separately).
    ``white_env``/``black_env`` are extra environment variables for each
    agent's subprocess (e.g. configuring baselines/stockfish_agent.py's
    strength for a calibration opponent). ``white_cpu``/``black_cpu``
    optionally hard-pin each side to one core via taskset — see
    AgentProcess's cpu_affinity docstring. ``adjudicate`` enables the
    resign/draw score-based cutoffs (see module constants) -- they only
    ever fire when both sides are reporting a score each move, which in
    practice means both sides are our own agent.py; against an opponent
    that doesn't report one, games always run to a real conclusion (or
    the ``max_plies`` cap) regardless of this flag."""
    white = AgentProcess(white_agent_path, "white", env=white_env, cpu_affinity=white_cpu)
    black = AgentProcess(black_agent_path, "black", env=black_env, cpu_affinity=black_cpu)
    procs = {"white": white, "black": black}
    moves_uci: list = []
    try:
        white.wait_ready()
        black.wait_ready()
        for side, proc in procs.items():
            if proc.init_ms is not None and proc.init_ms > INIT_BUDGET_MS:
                return GameResult(_opponent(side), f"{side}_init_budget_exceeded", 0, [], [], moves_uci)

        board = chess.Board(start_fen) if start_fen else chess.Board()
        clocks = {
            "white": SideClock(time_ms, increment_ms),
            "black": SideClock(time_ms, increment_ms),
        }
        times = {"white": [], "black": []}
        recent_white_pov_scores: list[float] = []  # every ply with a reported score, white's POV

        while not board.is_game_over(claim_draw=True):
            if board.ply() >= max_plies:
                return GameResult(None, "adjudicated_move_limit", board.ply(), times["white"], times["black"], moves_uci)

            side = "white" if board.turn == chess.WHITE else "black"
            proc = procs[side]
            timeout_s = clocks[side].remaining_ms / 1000 + 2  # IPC/OS grace only

            move_str, elapsed_ms, err, score = proc.request_move(board.fen(), clocks[side].remaining_ms, timeout_s)
            times[side].append(elapsed_ms)

            if clocks[side].consume(elapsed_ms):
                return GameResult(_opponent(side), f"{side}_flag", board.ply(), times["white"], times["black"], moves_uci)

            if err is not None:
                return GameResult(_opponent(side), f"{side}_{err}", board.ply(), times["white"], times["black"], moves_uci)

            try:
                move = chess.Move.from_uci(move_str)
            except Exception:
                return GameResult(_opponent(side), f"{side}_illegal_move_format", board.ply(), times["white"], times["black"], moves_uci)

            if move not in board.legal_moves:
                return GameResult(_opponent(side), f"{side}_illegal_move", board.ply(), times["white"], times["black"], moves_uci)

            board.push(move)
            moves_uci.append(move_str)

            if adjudicate:
                if score is None:
                    recent_white_pov_scores.clear()  # one side not reporting -- can't trust "both agree"
                else:
                    white_pov = score if side == "white" else -score
                    recent_white_pov_scores.append(white_pov)

                    tail = recent_white_pov_scores[-RESIGN_CONSECUTIVE_PLIES:]
                    if len(tail) == RESIGN_CONSECUTIVE_PLIES and all(s >= RESIGN_THRESHOLD_CP for s in tail):
                        return GameResult("white", "adjudicated_resign", board.ply(), times["white"], times["black"], moves_uci)
                    if len(tail) == RESIGN_CONSECUTIVE_PLIES and all(s <= -RESIGN_THRESHOLD_CP for s in tail):
                        return GameResult("black", "adjudicated_resign", board.ply(), times["white"], times["black"], moves_uci)

                    draw_tail = recent_white_pov_scores[-DRAW_CONSECUTIVE_PLIES:]
                    if len(draw_tail) == DRAW_CONSECUTIVE_PLIES and all(abs(s) <= DRAW_SCORE_EPSILON_CP for s in draw_tail):
                        return GameResult(None, "adjudicated_draw", board.ply(), times["white"], times["black"], moves_uci)

        outcome = board.outcome(claim_draw=True)
        winner = None
        if outcome is not None:
            if outcome.winner is True:
                winner = "white"
            elif outcome.winner is False:
                winner = "black"
        reason = outcome.termination.name.lower() if outcome is not None else "unknown"
        return GameResult(winner, reason, board.ply(), times["white"], times["black"], moves_uci)
    finally:
        white.close()
        black.close()
