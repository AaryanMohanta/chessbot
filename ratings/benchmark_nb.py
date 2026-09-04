"""Same median-depth/nps benchmark as ratings/benchmark.py, but for the
numba bitboard engine (cb_nb_search) instead of v1 -- numba is the
primary shipped search now, so this is the number that actually matters
for tracking real engine performance over time. Reuses v1's benchmark's
exact position set (MIDDLEGAME_POSITIONS) so the two numbers stay
comparable to each other.

Note the first call in a fresh process pays numba's JIT compile cost
(see cb_nb_engine.py) -- callers that want a clean nps number should
either warm up first (a throwaway search()) or expect the first
position's numbers to look artificially slow/shallow.
"""
from __future__ import annotations

import statistics
import time

import cb_nb_search as S
from ratings.benchmark import MIDDLEGAME_POSITIONS


def bench_position_nb(fen: str, soft_ms: float, hard_ms: float) -> dict:
    arrays = S.SearchArrays()
    start = time.perf_counter()
    move, score, info = S.search(fen, soft_ms, hard_ms, arrays, generation=1)
    elapsed = time.perf_counter() - start
    nps = info["nodes"] / elapsed if elapsed > 0 else float("inf")
    return {"fen": fen, "depth": info["depth"], "nodes": info["nodes"], "elapsed": elapsed, "nps": nps, "move": move, "score": score}


def median_middlegame_depth_and_nps_nb(soft_ms: float = 1500, hard_ms: float = 4500, warm_up: bool = True) -> dict:
    if warm_up:
        bench_position_nb(MIDDLEGAME_POSITIONS[0], soft_ms=1, hard_ms=50)

    results = [bench_position_nb(fen, soft_ms, hard_ms) for fen in MIDDLEGAME_POSITIONS]
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
    import cb_nb_fast as F

    summary = median_middlegame_depth_and_nps_nb()
    for r in summary["results"]:
        print(f"depth={r['depth']:2d} nodes={r['nodes']:7d} nps={r['nps']:>10,.0f} move={F.move_to_uci(r['move']):6s} score={r['score']:+6d}  {r['fen']}")
    print(f"\nmedian depth (middlegame only): {summary['median_depth']}")
    print(f"median nps   (middlegame only): {summary['median_nps']:,.0f}")
    print(f"depth range: {summary['min_depth']}-{summary['max_depth']}")
