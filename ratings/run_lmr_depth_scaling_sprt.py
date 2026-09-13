"""SPRT: LMR depth-scaled reduction (CB_NB_ENABLE_LMR_DEPTH_SCALING) vs
the existing flat LMR_REDUCTION -- see cb_nb_search.py's comment on the
flag for the reasoning (deeper remaining search affords a bigger
reduction, most engines scale by depth as well as move index; this
change only adds the depth dimension).

Same agent script (cb_nb_agent.py) on both sides, matching the pruning
batch SPRT's pattern. Time-boxed max_pairs (see ratings/
run_pruning_batch_sprt.py's own CLI-override comment) since a full
overnight run isn't available before the deadline.
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

CANDIDATE_ON = {"CB_NB_ENABLE_LMR_DEPTH_SCALING": "1"}
BASELINE_OFF = {"CB_NB_ENABLE_LMR_DEPTH_SCALING": "0"}

if __name__ == "__main__":
    print("##### SPRT: LMR depth-scaled reduction vs flat reduction #####", flush=True)
    result = run_sprt_pentanomial(
        AGENT, AGENT,
        elo0=ELO0, elo1=ELO1, max_pairs=MAX_PAIRS,
        time_ms=TIME_MS, increment_ms=INCREMENT_MS, concurrency=CONCURRENCY,
        candidate_env=CANDIDATE_ON,
        baseline_env=BASELINE_OFF,
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
            f"Flag rate -- candidate (depth-scaling on): {flags['candidate']}/{games} ({100*flags['candidate']/games:.1f}%), "
            f"baseline (flat): {flags['baseline']}/{games} ({100*flags['baseline']/games:.1f}%)"
        )
        if flags["candidate"] or flags["baseline"]:
            print("WARNING: nonzero flag rate -- should always be 0%, this needs investigating before trusting the result above.")
    if errored:
        print(f"NOTE: {errored} pair(s) errored (see per-pair WARNING lines above) and were excluded, not counted as losses.")
