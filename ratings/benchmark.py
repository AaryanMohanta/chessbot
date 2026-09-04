"""Search benchmarking: median depth and node rate at a fixed time budget,
over real middlegame positions specifically.

Averaging across a mixed bag of opening/middlegame/endgame positions (as
an earlier ad hoc check in this repo's history did) is misleading — a
simple K+R vs K endgame reaches much greater depth than a contested
middlegame at the same time budget, since its branching factor is tiny.
That inflates the "average" without saying anything about strength where
strength is actually decided. Report the middlegame median only.

Benchmarks the engine currently importable as ``cb_search``/``cb_tt`` —
i.e. whatever's first on sys.path, normally this repo's live root. Not
built to benchmark an arbitrary zoo snapshot in the same process: Python
caches modules by name, so a second import of a differently-pathed
cb_search wouldn't actually reload it. Fine for "before/after a change to
the live tree", which is what this exists for.
"""
from __future__ import annotations

import statistics
import time

import chess

import cb_search
from cb_tt import TranspositionTable

# Real, varied middlegame positions — contested, roughly balanced,
# meaningful piece counts still on the board. Deliberately excludes
# openings (too few pieces developed, artificially narrow) and endgames
# (branching factor collapses, depth is not comparable).
MIDDLEGAME_POSITIONS = [
    # Italian, both sides developed, tension on e5/d4
    "r1bqk2r/ppp2ppp/2n2n2/2bpp3/2B1P3/3P1N2/PPP2PPP/RNBQ1RK1 w kq - 0 7",
    # Ruy Lopez middlegame, closed centre
    "r1bq1rk1/2p1bppp/p1n2n2/1p1pp3/4P3/1B3N2/PPPP1PPP/RNBQR1K1 w - - 0 9",
    # Sicilian Najdorf-ish, opposite-side-castling texture
    "r1bq1rk1/1p2bppp/p1nppn2/6B1/3NP3/2N5/PPP1BPPP/R2Q1RK1 w - - 0 10",
    # Queen's Gambit Declined middlegame, IQP structure
    "r1bq1rk1/pp1n1ppp/2p1pn2/3p4/2PP4/2N1PN2/PP3PPP/R1BQ1RK1 w - - 0 8",
    # King's Indian, closed centre, opposite wing play brewing
    "r1bq1rk1/ppp1npbp/3p1np1/3Pp3/2P1P3/2N1BP2/PP2N1PP/R2QKB1R w KQ - 0 9",
    # Nimzo-Indian, doubled pawns structure
    "rnbq1rk1/pp3ppp/4pn2/2pp4/1bPP4/2NBPN2/PP3PPP/R1BQK2R w KQ - 0 8",
    # Caro-Kann, minor piece activity fight
    "r2q1rk1/pp1nbppp/2p1pn2/3p4/2PP4/1PN1PN2/PB3PPP/R2QKB1R w KQ - 0 9",
    # English/Reti, symmetrical-ish, fight for the centre
    "r1bq1rk1/pp2ppbp/2np1np1/2p5/2P5/2NP1NP1/PP2PPBP/R1BQ1RK1 w - - 0 8",
]


def bench_position(fen: str, soft_ms: float, hard_ms: float) -> dict:
    board = chess.Board(fen)
    tt = TranspositionTable()
    start = time.perf_counter()
    move, score, info = cb_search.search(board, soft_ms, hard_ms, tt, generation=1)
    elapsed = time.perf_counter() - start
    nps = info["nodes"] / elapsed if elapsed > 0 else float("inf")
    return {"fen": fen, "depth": info["depth"], "nodes": info["nodes"], "elapsed": elapsed, "nps": nps, "move": move.uci(), "score": score}


def median_middlegame_depth_and_nps(soft_ms: float = 1500, hard_ms: float = 4500) -> dict:
    results = [bench_position(fen, soft_ms, hard_ms) for fen in MIDDLEGAME_POSITIONS]
    depths = [r["depth"] for r in results]
    npss = [r["nps"] for r in results]
    return {
        "results": results,
        "median_depth": statistics.median(depths),
        "median_nps": statistics.median(npss),
        "min_depth": min(depths),
        "max_depth": max(depths),
    }


if __name__ == "__main__":
    summary = median_middlegame_depth_and_nps()
    for r in summary["results"]:
        print(f"depth={r['depth']:2d} nodes={r['nodes']:7d} nps={r['nps']:>10,.0f} move={r['move']:6s} score={r['score']:+6d}  {r['fen']}")
    print(f"\nmedian depth (middlegame only): {summary['median_depth']}")
    print(f"median nps   (middlegame only): {summary['median_nps']:,.0f}")
    print(f"depth range: {summary['min_depth']}-{summary['max_depth']}")
