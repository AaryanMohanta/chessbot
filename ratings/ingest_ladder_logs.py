"""CLI: pull real competition-ladder game log/PGN pairs into
ratings/ladder_games.csv (see ratings/ladder_db.py), tagged by which
build/config played each one (see cb_build_tag.py and agent.py's startup
log line) -- item 1 of the 600-ply-rule adaptation, "use the ladder as
our test harness."

Usage: python -m ratings.ingest_ladder_logs <path> [<path> ...] [--our-name NAME]
(--our-name defaults to "chessbot", our registered ladder name)

Each <path> is a .log/.txt file, a .pgn file, or a directory containing
either/both. Files are paired by filename stem (e.g. "round19.log" +
"round19.pgn") -- the dashboard's actual naming convention isn't assumed
beyond that, since a stem with only one of the two still gets ingested
with whatever telemetry/metadata is available from it alone. Already-
ingested (source_log, source_pgn) pairs are skipped, so re-running over a
directory you keep adding new files to is safe.

Determining which colour we played (needed to turn a PGN Result into a
score_for_us): preferred is the log's own first "mv ply=" line -- ply=0
means we answered before Black's first move, i.e. we're White; ply=1
means we're Black. When no log is present, falls back to matching
--our-name (case-insensitive) against the PGN's White/Black tags. A stem
where neither method resolves our colour is skipped with a warning
printed, not silently dropped -- a row with no score is not useful for
the whole point of this table, but silently losing a game other tooling
doesn't know about would be worse than a loud skip.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import chess
import chess.pgn

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from ratings.db import now_iso  # noqa: E402
from ratings.ladder_db import DB_PATH, LadderGameRecord, already_ingested_sources, append_game  # noqa: E402
from ratings.parse_match_log import parse_build_tag, parse_init, parse_moves, report_from_moves  # noqa: E402

LOG_SUFFIXES = {".log", ".txt", ".err"}
PGN_SUFFIXES = {".pgn"}


def _collect_files(paths: list[str]) -> list[Path]:
    files: list[Path] = []
    for raw in paths:
        p = Path(raw)
        if p.is_dir():
            for suffix in LOG_SUFFIXES | PGN_SUFFIXES:
                files.extend(sorted(p.glob(f"*{suffix}")))
        elif p.is_file():
            files.append(p)
    return files


def _group_by_stem(files: list[Path]) -> dict[str, dict[str, Path]]:
    groups: dict[str, dict[str, Path]] = {}
    for f in files:
        group = groups.setdefault(f.stem, {})
        if f.suffix.lower() in PGN_SUFFIXES:
            group["pgn"] = f
        elif f.suffix.lower() in LOG_SUFFIXES:
            group["log"] = f
    return groups


def _our_color_from_log(log_text: str) -> str | None:
    moves = parse_moves(log_text)
    if not moves:
        return None
    return "white" if moves[0].ply % 2 == 0 else "black"


def _our_color_from_pgn_headers(headers: dict, our_name: str | None) -> str | None:
    if our_name is None:
        return None
    our_name_lower = our_name.lower()
    if our_name_lower in headers.get("White", "").lower():
        return "white"
    if our_name_lower in headers.get("Black", "").lower():
        return "black"
    return None


def _score_for_us(result: str, our_color: str) -> float | None:
    if result not in ("1-0", "0-1", "1/2-1/2"):
        return None
    if result == "1/2-1/2":
        return 0.5
    winner = "white" if result == "1-0" else "black"
    return 1.0 if winner == our_color else 0.0


def ingest_one(stem: str, group: dict[str, Path], our_name: str | None) -> LadderGameRecord | None:
    log_path = group.get("log")
    pgn_path = group.get("pgn")

    log_text = log_path.read_text(encoding="utf-8", errors="replace") if log_path else ""
    build_tag = parse_build_tag(log_text) if log_text else None
    moves = parse_moves(log_text) if log_text else []
    init = parse_init(log_text) if log_text else None
    move_report = report_from_moves(moves, init)

    headers: dict = {}
    moves_uci = ""
    if pgn_path:
        with open(pgn_path, encoding="utf-8", errors="replace") as f:
            game = chess.pgn.read_game(f)
        if game is not None:
            headers = dict(game.headers)
            board = game.board()
            uci_moves = []
            for move in game.mainline_moves():
                uci_moves.append(move.uci())
                board.push(move)
            moves_uci = " ".join(uci_moves)

    our_color = _our_color_from_log(log_text) or _our_color_from_pgn_headers(headers, our_name)
    if our_color is None:
        print(f"[ingest] {stem}: could not determine our colour (no log moves, no --our-name match) -- skipped",
              file=sys.stderr)
        return None

    result = headers.get("Result", "")
    score = _score_for_us(result, our_color)
    if score is None:
        print(f"[ingest] {stem}: no usable PGN Result ({result!r}) -- skipped", file=sys.stderr)
        return None

    opponent_side = "Black" if our_color == "white" else "White"
    opponent = headers.get(opponent_side, "")
    opponent_elo = headers.get(f"{opponent_side}Elo", "")
    termination = headers.get("Termination", "")
    plies = int(headers.get("PlyCount", 0)) or len(moves_uci.split()) or move_report["moves_seen"]

    return LadderGameRecord(
        timestamp=now_iso(),
        build_tag=build_tag or "unknown",
        our_color=our_color,
        opponent=opponent,
        opponent_elo=str(opponent_elo),
        score_for_us=score,
        termination=termination,
        plies=plies,
        median_depth=move_report["median_depth"],
        total_clock_spend_s=move_report["total_clock_spend_s"],
        v1_ever_played=move_report["v1_ever_played"],
        depth_collapse_count=len(move_report["depth_collapses"]),
        moves_uci=moves_uci,
        source_log=log_path.name if log_path else "",
        source_pgn=pgn_path.name if pgn_path else "",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("paths", nargs="+", help="log/PGN files or directories containing them")
    parser.add_argument("--our-name", default="chessbot",
                         help="fallback for determining our colour when a stem has no log "
                              "(matched case-insensitively against the PGN's White/Black tags); "
                              "our registered ladder name, default 'chessbot'")
    args = parser.parse_args()

    files = _collect_files(args.paths)
    groups = _group_by_stem(files)
    already = already_ingested_sources(db_path=DB_PATH)

    ingested = 0
    skipped_seen = 0
    skipped_unresolved = 0
    for stem, group in sorted(groups.items()):
        source_key = (group.get("log").name if "log" in group else "", group.get("pgn").name if "pgn" in group else "")
        if source_key in already:
            skipped_seen += 1
            continue
        record = ingest_one(stem, group, args.our_name)
        if record is None:
            skipped_unresolved += 1
            continue
        append_game(record, db_path=DB_PATH)
        ingested += 1
        print(f"[ingest] {stem}: build_tag={record.build_tag} our_color={record.our_color} "
              f"score={record.score_for_us} vs {record.opponent or '?'}")

    print(f"\nOK: ingested {ingested} game(s) into {DB_PATH} "
          f"({skipped_seen} already ingested, {skipped_unresolved} skipped/unresolved)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
