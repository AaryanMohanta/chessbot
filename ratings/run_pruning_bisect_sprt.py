"""Bisection for the rejected pruning batch (reverse futility + futility
+ LMP together scored 0.752/2 per pair against no pruning -- H0,
LLR=-3.123, see ratings/run_pruning_batch_sprt.py's log). Tests each
technique alone against the same no-pruning baseline to find which one(s)
are actually responsible for the regression, rather than guessing.

Runs the three one at a time (not concurrently -- avoids stacking SPRT
runs the way calibration + SPRT contention taught us not to). Pass
--technique rfp|futility|lmp to run just one.
"""
import argparse
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

ALL_OFF = {"CB_NB_ENABLE_RFP": "0", "CB_NB_ENABLE_FUTILITY": "0", "CB_NB_ENABLE_LMP": "0"}

TECHNIQUES = {
    "rfp": {**ALL_OFF, "CB_NB_ENABLE_RFP": "1"},
    "futility": {**ALL_OFF, "CB_NB_ENABLE_FUTILITY": "1"},
    "lmp": {**ALL_OFF, "CB_NB_ENABLE_LMP": "1"},
}


def run_one(name: str, candidate_env: dict) -> None:
    print(f"\n##### SPRT: {name} alone vs no shallow pruning #####", flush=True)
    result = run_sprt_pentanomial(
        AGENT, AGENT,
        elo0=ELO0, elo1=ELO1, max_pairs=MAX_PAIRS,
        time_ms=TIME_MS, increment_ms=INCREMENT_MS, concurrency=CONCURRENCY,
        candidate_env=candidate_env,
        baseline_env=ALL_OFF,
    )
    games = result["games"]
    flags = result["flag_counts"]
    errored = result.get("errored_pairs", 0)
    print(
        f"\n{name.upper()} DECISION: {result['decision']} - {result['reason']} "
        f"after {games} games ({result['pairs']} pairs, {errored} errored/skipped), "
        f"mean_pair_score={result['mean_pair_score']:.3f}/2, LLR={result['llr']:+.3f}"
    )
    if games:
        print(f"Flag rate -- {name}: {flags['candidate']}/{games} ({100*flags['candidate']/games:.1f}%), "
              f"baseline: {flags['baseline']}/{games} ({100*flags['baseline']/games:.1f}%)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--technique", choices=list(TECHNIQUES), default=None,
                         help="run just one technique; default runs all three sequentially")
    args = parser.parse_args()

    if args.technique:
        run_one(args.technique, TECHNIQUES[args.technique])
    else:
        for name, env in TECHNIQUES.items():
            run_one(name, env)
