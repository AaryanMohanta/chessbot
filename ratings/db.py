"""The game database: every game ever played, append-only, source of truth
for the rating fit.

Format: CSV, not PGN. Reasoning:
  - Every field the rating fit and reporting need (versions, colours,
    opening, result, termination, plies) is scalar/tabular — there's no
    tree-structured or freeform content that PGN's format earns its
    complexity for.
  - Appends are one line each. That's safe under concurrent writers (the
    gauntlet runner uses a thread pool): a single `write()` of one line is
    effectively atomic at the OS level for lines well under the pipe/
    filesystem buffer size, whereas PGN's blank-line-delimited multi-line
    records are much easier for two concurrent writers to interleave into
    a corrupt file.
  - Trivial to load into anything (csv module, pandas, a spreadsheet) for
    the rating fit or ad hoc analysis, with no PGN parser dependency.
  - The one thing PGN gives you for free — the move list, for later
    replay/debugging — isn't lost: it's kept as one extra column
    (space-separated UCI moves) rather than as the file's whole structure.

Concurrency note: this module serializes writes within one process with a
lock. It does not take an OS-level file lock, so two *separate* processes
appending at the same instant aren't fully guarded against interleaving —
acceptable here because the gauntlet runner parallelizes with threads in a
single process, not with multiple processes.
"""
from __future__ import annotations

import csv
import dataclasses
import datetime
import threading
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent / "games.csv"

FIELDS = [
    "timestamp",
    "white_version",
    "black_version",
    "opening_id",
    "result",       # "1-0", "0-1", or "1/2-1/2"
    "termination",  # harness.match.GameResult.reason
    "plies",
    "moves_uci",    # space-separated UCI moves; may be empty for older rows
]

_write_lock = threading.Lock()


@dataclasses.dataclass(frozen=True)
class GameRecord:
    timestamp: str
    white_version: str
    black_version: str
    opening_id: str
    result: str
    termination: str
    plies: int
    moves_uci: str = ""

    @property
    def white_score(self) -> float:
        return {"1-0": 1.0, "0-1": 0.0, "1/2-1/2": 0.5}[self.result]


def result_string(winner: str | None) -> str:
    if winner == "white":
        return "1-0"
    if winner == "black":
        return "0-1"
    return "1/2-1/2"


def append_game(record: GameRecord, db_path: Path = DB_PATH) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not db_path.exists()
    with _write_lock:
        with open(db_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDS)
            if is_new:
                writer.writeheader()
            writer.writerow(dataclasses.asdict(record))


def load_games(db_path: Path = DB_PATH) -> list[GameRecord]:
    if not db_path.exists():
        return []
    games = []
    with open(db_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            games.append(
                GameRecord(
                    timestamp=row["timestamp"],
                    white_version=row["white_version"],
                    black_version=row["black_version"],
                    opening_id=row["opening_id"],
                    result=row["result"],
                    termination=row["termination"],
                    plies=int(row["plies"]),
                    moves_uci=row.get("moves_uci", "") or "",
                )
            )
    return games


def now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def played_pairings(db_path: Path = DB_PATH) -> set[tuple[str, str, str]]:
    """(white_version, black_version, opening_id) triples already recorded
    — used by the gauntlet runner for resumability. Load once per run
    rather than rescanning the whole file per candidate game."""
    return {(g.white_version, g.black_version, g.opening_id) for g in load_games(db_path)}
