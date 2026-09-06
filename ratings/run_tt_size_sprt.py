"""SPRT: TT_SIZE_BITS=23 (~403MB, 6 int64/uint64 arrays x 8M entries)
vs the current 20 (~50MB, 1M entries) -- see cb_nb_search.py's own
comment on TT_SIZE_BITS for the reasoning (at ~1M nps and a multi-second
move budget, a single move alone visits several million nodes against a
1M-entry table, before even counting that the table persists and
accumulates across the whole game).

Same agent script (cb_nb_agent.py) on both sides, matching every other
same-day SPRT's pattern. Time-boxed max_pairs via CLI override.
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
MAX_PAIRS = int(sys.argv[1]) if len(sys.argv) > 1 else 4000
CONCURRENCY = 7

CANDIDATE_BIG_TT = {"CB_NB_TT_SIZE_BITS": "23"}
BASELINE_SMALL_TT = {"CB_NB_TT_SIZE_BITS": "20"}

if __name__ == "__main__":
    print("##### SPRT: TT_SIZE_BITS=23 (~403MB) vs 20 (~50MB, current) #####", flush=True)
    result = run_sprt_pentanomial(
        AGENT, AGENT,
        elo0=ELO0, elo1=ELO1, max_pairs=MAX_PAIRS,
        time_ms=TIME_MS, increment_ms=INCREMENT_MS, concurrency=CONCURRENCY,
        candidate_env=CANDIDATE_BIG_TT,
        baseline_env=BASELINE_SMALL_TT,
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
            f"Flag rate -- candidate (2^23 TT): {flags['candidate']}/{games} ({100*flags['candidate']/games:.1f}%), "
            f"baseline (2^20 TT): {flags['baseline']}/{games} ({100*flags['baseline']/games:.1f}%)"
        )
        if flags["candidate"] or flags["baseline"]:
            print("WARNING: nonzero flag rate -- should always be 0%, this needs investigating before trusting the result above.")
    if errored:
        print(f"NOTE: {errored} pair(s) errored (see per-pair WARNING lines above) and were excluded, not counted as losses.")
