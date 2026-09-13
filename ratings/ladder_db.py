"""The ladder game database: every real competition-ladder game we have a
log and/or PGN for, append-only, parallel to ratings/db.py's self-play
games.csv but for a different shape of data.

Why a separate file from db.py's GameRecord/games.csv rather than
reusing it: db.py's schema is white_version/black_version -- both sides
are one of *our own* tracked zoo versions, which is what the joint Elo
fit in ratings/elo.py needs. A ladder game is us against a real house bot
we don't control or necessarily know the exact strength of, tagged by
*our own* build/config (see cb_build_tag.py) rather than a zoo version --
a fundamentally different join key, so forcing it into the same table
would mean a lot of always-empty or overloaded columns. Real ladder games
also arrive from parsed log/PGN text (see ratings/parse_match_log.py and
ratings/ingest_ladder_logs.py), not from a game we played ourselves via
harness.match.play_game, so they carry telemetry fields (median depth,
clock spend, v1 usage) db.py's schema has no room for.

This is the table item 1 of the 600-ply-rule adaptation ("use the ladder
as our test harness") is built around: tag every real ladder game by
which build/config played it, accumulate enough of them under two
different build tags, then compare -- see ratings/ladder_report.py.
"""
from __future__ import annotations

import csv
import dataclasses
import threading
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent / "ladder_games.csv"

FIELDS = [
    "timestamp",
    "build_tag",
    "our_color",       # "white" or "black"
    "opponent",         # PGN's tag for the other side, "" if unknown
    "opponent_elo",     # PGN WhiteElo/BlackElo for the opponent, "" if absent
    "score_for_us",     # 1.0 / 0.5 / 0.0
    "termination",      # PGN Termination header, "" if absent
    "plies",
    "median_depth",     # "" if no searched moves were logged
    "total_clock_spend_s",
    "v1_ever_played",   # "1" or "0"
    "depth_collapse_count",
    "moves_uci",        # space-separated, "" if the PGN wasn't available
    "source_log",       # filename the telemetry came from, "" if none
    "source_pgn",       # filename the PGN came from, "" if none
]

_write_lock = threading.Lock()


@dataclasses.dataclass(frozen=True)
class LadderGameRecord:
    timestamp: str
    build_tag: str
    our_color: str
    opponent: str
    opponent_elo: str
    score_for_us: float
    termination: str
    plies: int
    median_depth: float | None
    total_clock_spend_s: float
    v1_ever_played: bool
    depth_collapse_count: int
    moves_uci: str = ""
    source_log: str = ""
    source_pgn: str = ""


def _to_row(record: LadderGameRecord) -> dict:
    d = dataclasses.asdict(record)
    d["median_depth"] = "" if record.median_depth is None else record.median_depth
    d["v1_ever_played"] = "1" if record.v1_ever_played else "0"
    return d


def _from_row(row: dict) -> LadderGameRecord:
    return LadderGameRecord(
        timestamp=row["timestamp"],
        build_tag=row["build_tag"],
        our_color=row["our_color"],
        opponent=row["opponent"],
        opponent_elo=row["opponent_elo"],
        score_for_us=float(row["score_for_us"]),
        termination=row["termination"],
        plies=int(row["plies"]),
        median_depth=None if row["median_depth"] == "" else float(row["median_depth"]),
        total_clock_spend_s=float(row["total_clock_spend_s"]),
        v1_ever_played=row["v1_ever_played"] == "1",
        depth_collapse_count=int(row["depth_collapse_count"]),
        moves_uci=row.get("moves_uci", "") or "",
        source_log=row.get("source_log", "") or "",
        source_pgn=row.get("source_pgn", "") or "",
    )


def append_game(record: LadderGameRecord, db_path: Path = DB_PATH) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not db_path.exists()
    with _write_lock:
        with open(db_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDS)
            if is_new:
                writer.writeheader()
            writer.writerow(_to_row(record))


def load_games(db_path: Path = DB_PATH) -> list[LadderGameRecord]:
    if not db_path.exists():
        return []
    with open(db_path, newline="", encoding="utf-8") as f:
        return [_from_row(row) for row in csv.DictReader(f)]


def already_ingested_sources(db_path: Path = DB_PATH) -> set[tuple[str, str]]:
    """(source_log, source_pgn) pairs already recorded -- lets the
    ingestion CLI be re-run over the same directory (e.g. after new files
    were added) without double-counting games it already has."""
    return {(g.source_log, g.source_pgn) for g in load_games(db_path)}
