"""SPRT: retuned king safety (raised shield/file/attacker weights plus a
new castling incentive) against the original small-weights baseline.

Built after a real ladder loss (round 4) where the engine never castled,
wrecked its own kingside pawn shield, and pushed queenside pawns while
exposed -- the old weights (max realistic total ~15-70cp) were too small
to ever outweigh other eval terms. One batch run, not per-term: the raised
weights and the castling incentive are evaluated together since only the
combined accept/reject decision will be acted on (same reasoning as the
eval-terms and pruning-family SPRTs).

Same agent script (cb_nb_agent.py) on both sides, toggled via
CB_NB_KING_SAFETY_V2 (see cb_nb_fast.py). 5+0.05, [0, 15] bounds, 4000-pair
cap, concurrency 7 -- this machine's real ceiling (7 physical cores).
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

KING_SAFETY_V2 = {"CB_NB_KING_SAFETY_V2": "1"}
KING_SAFETY_V1 = {"CB_NB_KING_SAFETY_V2": "0"}

if __name__ == "__main__":
    print("##### SPRT: retuned king safety (raised weights + castling incentive) vs original weights #####", flush=True)
    result = run_sprt_pentanomial(
        AGENT, AGENT,
        elo0=ELO0, elo1=ELO1, max_pairs=MAX_PAIRS,
        time_ms=TIME_MS, increment_ms=INCREMENT_MS, concurrency=CONCURRENCY,
        candidate_env=KING_SAFETY_V2,
        baseline_env=KING_SAFETY_V1,
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
            f"Flag rate -- candidate (retuned king safety): {flags['candidate']}/{games} ({100*flags['candidate']/games:.1f}%), "
            f"baseline (original weights): {flags['baseline']}/{games} ({100*flags['baseline']/games:.1f}%)"
        )
        if flags["candidate"] or flags["baseline"]:
            print("WARNING: nonzero flag rate -- should always be 0%, this needs investigating before trusting the result above.")
    if errored:
        print(f"NOTE: {errored} pair(s) errored (see per-pair WARNING lines above) and were excluded, not counted as losses.")
