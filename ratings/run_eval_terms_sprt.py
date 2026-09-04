"""Batch SPRT: all six new eval terms (passed pawns, isolated/doubled
pawns, rook on open/semi-open file, bishop pair, mobility, king safety --
plus lazy eval, which only affects speed, not the score these terms
produce) together against the material+PST-only baseline.

Deliberately one run, not six: a per-term SPRT would need six separate
attribution runs when only the batched accept/reject decision is
actually going to be acted on. If this fails or is inconclusive, bisect
starting with mobility off -- it's the known expensive term (real nps
cost measured directly, see cb_nb_fast.py's evaluate_from_state
docstring) and the prime suspect for costing more search depth than the
eval richness returns.

Same agent script (cb_nb_agent.py) on both sides, toggled via
CB_NB_ENABLE_EXTENDED_EVAL -- see cb_nb_fast.py. 10+0.1, [0, 10] bounds
(a novel change, not a non-regression check), 4000-pair cap, concurrency
7, meant to run overnight.
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from ratings.sprt import run_sprt_pentanomial  # noqa: E402

AGENT = str(REPO_ROOT / "cb_nb_agent.py")

TIME_MS = 5_000
INCREMENT_MS = 50
ELO0, ELO1 = 0.0, 15.0
MAX_PAIRS = 4000
CONCURRENCY = 7

if __name__ == "__main__":
    print("##### SPRT: all six eval terms + lazy eval vs material+PST baseline #####", flush=True)
    result = run_sprt_pentanomial(
        AGENT, AGENT,
        elo0=ELO0, elo1=ELO1, max_pairs=MAX_PAIRS,
        time_ms=TIME_MS, increment_ms=INCREMENT_MS, concurrency=CONCURRENCY,
        candidate_env={"CB_NB_ENABLE_EXTENDED_EVAL": "1"},
        baseline_env={"CB_NB_ENABLE_EXTENDED_EVAL": "0"},
    )
    games = result["games"]
    flags = result["flag_counts"]
    errored = result.get("errored_pairs", 0)
    print(
        f"\nDECISION: {result['decision']} - {result['reason']} "
        f"after {games} games ({result['pairs']} pairs, {errored} pairs errored/skipped), "
        f"mean_pair_score={result['mean_pair_score']:.3f}/2, LLR={result['llr']:+.3f}"
    )
    if games:
        print(
            f"Flag rate -- candidate (extended eval): {flags['candidate']}/{games} ({100*flags['candidate']/games:.1f}%), "
            f"baseline (material+PST): {flags['baseline']}/{games} ({100*flags['baseline']/games:.1f}%)"
        )
        if flags["candidate"] or flags["baseline"]:
            print("WARNING: nonzero flag rate -- should always be 0%, this needs investigating before trusting the result above.")
    if errored:
        print(f"NOTE: {errored} pair(s) errored (see per-pair WARNING lines above) and were excluded, not counted as losses.")

    from ratings.benchmark_nb import median_middlegame_depth_and_nps_nb

    bench = median_middlegame_depth_and_nps_nb()
    print(f"Median depth (middlegame only, extended eval as currently configured): {bench['median_depth']} "
          f"(range {bench['min_depth']}-{bench['max_depth']})")
    print(f"Median nps:                                                            {bench['median_nps']:,.0f}\n")
