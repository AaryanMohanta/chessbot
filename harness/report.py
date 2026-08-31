"""Formats a GameResult (or a batch of them) into a human-readable summary."""
from __future__ import annotations

from harness.match import GameResult


def _avg(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def format_game(result: GameResult, white_name: str, black_name: str) -> str:
    if result.winner == "white":
        outcome = f"{white_name} (white) wins"
    elif result.winner == "black":
        outcome = f"{black_name} (black) wins"
    else:
        outcome = "draw"

    lines = [
        f"{outcome} - {result.reason} - {result.plies} plies",
        f"  {white_name} (white): avg {_avg(result.white_times_ms):.1f} ms, "
        f"max {max(result.white_times_ms, default=0):.1f} ms/move",
        f"  {black_name} (black): avg {_avg(result.black_times_ms):.1f} ms, "
        f"max {max(result.black_times_ms, default=0):.1f} ms/move",
    ]
    return "\n".join(lines)


def summarize_match(results: list[tuple[GameResult, str, str]]) -> str:
    """results: list of (GameResult, white_name, black_name)."""
    wins: dict[str, int] = {}
    draws = 0
    for result, white_name, black_name in results:
        if result.winner == "white":
            wins[white_name] = wins.get(white_name, 0) + 1
        elif result.winner == "black":
            wins[black_name] = wins.get(black_name, 0) + 1
        else:
            draws += 1

    lines = [format_game(r, w, b) for r, w, b in results]
    lines.append("")
    lines.append(f"Summary: {dict(wins)} draws={draws} / {len(results)} games")
    return "\n".join(lines)
