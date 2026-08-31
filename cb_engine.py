"""Engine: owns per-game state and wires cb_time -> cb_search -> cb_eval.

``Engine.__init__`` is where one-time, expensive setup belongs (loading
weights, warming up a JIT, allocating tables) — it runs during the 60 s
init budget, before the game clock starts. Everything in ``get_move`` runs
under the per-move clock instead.
"""
from __future__ import annotations

import chess

import cb_eval
import cb_search
from cb_time import TimeManager


class Engine:
    def __init__(self) -> None:
        """One-time init, spent against the init budget (see
        ``cb_time.INIT_BUDGET_MS``), not the per-move clock.

        TODO: load weights (.onnx / .safetensors / .pt), warm up
        onnxruntime / numba JIT, allocate transposition tables, etc.
        """
        self.time_manager = TimeManager()
        self.ply = 0

    def evaluate(self, board: chess.Board) -> int:
        return cb_eval.evaluate(board)

    def search(self, board: chess.Board, budget_ms: int) -> chess.Move:
        """Run search under a soft time budget and return the chosen move."""
        move, _score, _info = cb_search.search(board, budget_ms, stop_check=lambda: False)
        return move

    def get_move(self, fen: str, time_left_ms: int) -> str:
        """Per-move entry point. Raises on any internal failure — the
        caller (``agent.py``) is responsible for catching that and falling
        back to a trivially-legal move."""
        board = chess.Board(fen)
        budget = self.time_manager.budget(time_left_ms, ply=self.ply)
        self.ply += 1

        move = self.search(board, budget.soft_ms)
        return move.uci()
