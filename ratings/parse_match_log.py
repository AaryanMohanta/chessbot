r"""Parses one real AI Chessathon match .log file's OUTPUT/stderr section
into structured per-move telemetry (see agent.py's _log_move and
cb_nb_engine.py's init logging for the line formats being parsed).

Deliberately regex-scans for lines matching the known formats anywhere in
the text, rather than assuming a fixed line order or contiguous block --
the new 8KB-per-game cap (first 4KB + last 4KB, PGN shown separately on
the dashboard) means a long game's middle third of moves is simply
missing from what we get back, and the parser has to produce a sane
partial result from that rather than choke on it.

Every pattern below allows leading whitespace before its anchor (`^\s*`,
not bare `^`): the raw stderr agent.py writes has none, but the actual
downloaded dashboard .log file (MATCH/INIT/CLOCK/RESULT/MOVES sections,
then an "OUTPUT" section) re-indents that captured stderr by two spaces
for readability. A bare `^` anchor matches synthetic unindented fixtures
fine but silently matches nothing at all against every real downloaded
log -- caught 2026-09 when ingesting real round files returned zero
telemetry across the board.

Usage: python ratings/parse_match_log.py <log_file_or_dir> [...]
Prints one summary per file; see report_from_moves() for the fields.
"""
from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path

_MOVE_RE = re.compile(
    r"^\s*mv ply=(?P<ply>\d+) layer=(?P<layer>\S+) d=(?P<depth>\S+) sc=(?P<score>\S+) "
    r"n=(?P<nodes>\S+) t=(?P<t>[\d.]+)s left=(?P<left>[\d.]+)s",
    re.MULTILINE,
)
_INIT_RE = re.compile(
    r"^\s*\[cb_nb_engine\] init returned after (?P<ms>[\d.]+) ms \((?P<status>[^)]*)\) "
    r"ready_before_move1=(?P<ready>True|False)(?: stages: (?P<stages>.*))?$",
    re.MULTILINE,
)
_BG_WARMUP_RE = re.compile(
    r"^\s*\[cb_nb_engine\] background warmup finished after (?P<ms>[\d.]+) ms \(numba_failed=(?P<failed>True|False)\)$",
    re.MULTILINE,
)
_BUILD_TAG_RE = re.compile(r"^\s*\[agent\] build_tag=(?P<tag>\S+)$", re.MULTILINE)


@dataclass
class MoveRecord:
    ply: int
    layer: str
    depth: int | None
    score: float | None
    nodes: int | None
    elapsed_s: float
    time_left_s: float


@dataclass
class InitRecord:
    total_ms: float
    status: str
    ready_before_move1: bool
    stage_times_ms: dict[str, float]


def parse_moves(log_text: str) -> list[MoveRecord]:
    moves = []
    for m in _MOVE_RE.finditer(log_text):
        depth = None if m["depth"] == "-" else int(m["depth"])
        score = None if m["score"] == "-" else float(m["score"])
        nodes = None if m["nodes"] == "-" else int(m["nodes"])
        moves.append(MoveRecord(
            ply=int(m["ply"]), layer=m["layer"], depth=depth, score=score,
            nodes=nodes, elapsed_s=float(m["t"]), time_left_s=float(m["left"]),
        ))
    return moves


def parse_init(log_text: str) -> InitRecord | None:
    m = _INIT_RE.search(log_text)
    if m is None:
        return None
    stages = {}
    if m["stages"]:
        for part in m["stages"].split():
            name, _, ms = part.partition("=")
            try:
                stages[name] = float(ms.rstrip("ms"))
            except ValueError:
                pass
    return InitRecord(
        total_ms=float(m["ms"]), status=m["status"],
        ready_before_move1=(m["ready"] == "True"), stage_times_ms=stages,
    )


def parse_build_tag(log_text: str) -> str | None:
    """Which build/config produced this log (see agent.py's startup line
    and tools/build_zip.py's --tag), or None if the line isn't present --
    true for any log predating this instrumentation, and harmless: the
    caller just can't attribute that game to a build."""
    m = _BUILD_TAG_RE.search(log_text)
    return m["tag"] if m else None


def parse_background_warmup(log_text: str) -> tuple[float, bool] | None:
    """(finished_after_ms, numba_failed), if a background-warmup
    completion line is present -- only fires when init missed the
    blocking deadline, see cb_nb_engine.py's _warm_up_remaining."""
    m = _BG_WARMUP_RE.search(log_text)
    if m is None:
        return None
    return float(m["ms"]), m["failed"] == "True"


def report_from_moves(moves: list[MoveRecord], init: InitRecord | None) -> dict:
    """The four things asked for: median depth, clock spend, whether v1
    ever played, and any move where depth collapsed relative to its
    neighbours. Depths/nodes are only ever present for layer in
    {numba, v1} -- book and trivial moves carry no search info by
    construction (see agent.py's _log_move)."""
    searched = [m for m in moves if m.depth is not None]
    depths = [m.depth for m in searched]
    v1_moves = [m for m in moves if m.layer == "v1"]

    collapses = []
    for i in range(1, len(searched) - 1):
        prev_d, cur_d, next_d = searched[i - 1].depth, searched[i].depth, searched[i + 1].depth
        neighbour_avg = (prev_d + next_d) / 2
        if neighbour_avg - cur_d >= 3 and cur_d <= neighbour_avg * 0.6:
            collapses.append((searched[i].ply, cur_d, prev_d, next_d))

    total_clock_spend_s = sum(m.elapsed_s for m in moves)

    return {
        "moves_seen": len(moves),
        "moves_missing_note": "counts only reflect whichever moves survived the 8KB truncation",
        "median_depth": _median(depths) if depths else None,
        "min_depth": min(depths) if depths else None,
        "max_depth": max(depths) if depths else None,
        "total_clock_spend_s": round(total_clock_spend_s, 2),
        "v1_ever_played": len(v1_moves) > 0,
        "v1_move_count": len(v1_moves),
        "v1_first_ply": v1_moves[0].ply if v1_moves else None,
        "depth_collapses": collapses,  # [(ply, depth, prev_depth, next_depth), ...]
        "init": None if init is None else {
            "total_ms": init.total_ms,
            "status": init.status,
            "ready_before_move1": init.ready_before_move1,
            "stage_times_ms": init.stage_times_ms,
        },
    }


def _median(xs: list[float]) -> float:
    s = sorted(xs)
    n = len(s)
    mid = n // 2
    return s[mid] if n % 2 else (s[mid - 1] + s[mid]) / 2


def parse_log_file(path: Path) -> dict:
    text = path.read_text(encoding="utf-8", errors="replace")
    moves = parse_moves(text)
    init = parse_init(text)
    report = report_from_moves(moves, init)
    report["build_tag"] = parse_build_tag(text)
    report["file"] = str(path)
    return report


if __name__ == "__main__":
    paths: list[Path] = []
    for arg in sys.argv[1:]:
        p = Path(arg)
        if p.is_dir():
            paths.extend(sorted(p.glob("*.log")))
        else:
            paths.append(p)

    if not paths:
        print("Usage: python ratings/parse_match_log.py <log_file_or_dir> [...]")
        sys.exit(1)

    for path in paths:
        r = parse_log_file(path)
        print(f"\n=== {path.name} ===")
        print(f"  moves_seen={r['moves_seen']}  median_depth={r['median_depth']}  "
              f"depth_range={r['min_depth']}-{r['max_depth']}")
        print(f"  total_clock_spend={r['total_clock_spend_s']}s")
        print(f"  v1_ever_played={r['v1_ever_played']} (count={r['v1_move_count']}, first_ply={r['v1_first_ply']})")
        if r["depth_collapses"]:
            print(f"  DEPTH COLLAPSES: {r['depth_collapses']}")
        if r["init"]:
            print(f"  init: {r['init']}")
