"""Batch SPRT: reverse futility pruning + futility pruning + late move
pruning together against the same engine with all three off. One run,
not three -- matches the eval-terms SPRT's reasoning: only the batched
accept/reject decision is going to be acted on, so per-technique
attribution runs would just be three extra overnight runs for
information we won't use.

If this fails or is inconclusive, bisect by disabling one flag at a time
(CB_NB_ENABLE_RFP / CB_NB_ENABLE_FUTILITY / CB_NB_ENABLE_LMP, see
cb_nb_search.py) rather than guessing which one is the problem -- they
were built independently toggleable from the start for exactly this.

Same agent script (cb_nb_agent.py) on both sides. 10+0.1, [0, 10] bounds,
4000-pair cap, concurrency 7, meant to run overnight.

Note: the earlier eval-terms SPRT crashed once at this concurrency
(both sides being numba-based doubles simultaneous JIT-compile load per
pair vs. a numba-vs-v1 comparison) -- ratings/sprt.py's
run_sprt_pentanomial now catches a failed pair and skips it rather than
taking down the whole run, so a repeat of that here won't lose the batch.
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
# 4000 is the "run it overnight" cap. Optional CLI override for a
# time-boxed run before a deadline, e.g. `python run_pruning_batch_sprt.py
# 250` -- SPRT still resolves early on a clear effect either way; a lower
# cap only risks an "inconclusive" outcome if the true effect sits near
# the elo0/elo1 boundary, it doesn't bias the result.
MAX_PAIRS = int(sys.argv[1]) if len(sys.argv) > 1 else 4000
CONCURRENCY = 7

PRUNING_ON = {"CB_NB_ENABLE_RFP": "1", "CB_NB_ENABLE_FUTILITY": "1", "CB_NB_ENABLE_LMP": "1"}
PRUNING_OFF = {"CB_NB_ENABLE_RFP": "0", "CB_NB_ENABLE_FUTILITY": "0", "CB_NB_ENABLE_LMP": "0"}

if __name__ == "__main__":
    print("##### SPRT: reverse futility + futility + LMP vs no shallow pruning #####", flush=True)
    result = run_sprt_pentanomial(
        AGENT, AGENT,
        elo0=ELO0, elo1=ELO1, max_pairs=MAX_PAIRS,
        time_ms=TIME_MS, increment_ms=INCREMENT_MS, concurrency=CONCURRENCY,
        candidate_env=PRUNING_ON,
        baseline_env=PRUNING_OFF,
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
            f"Flag rate -- candidate (pruning on): {flags['candidate']}/{games} ({100*flags['candidate']/games:.1f}%), "
            f"baseline (pruning off): {flags['baseline']}/{games} ({100*flags['baseline']/games:.1f}%)"
        )
        if flags["candidate"] or flags["baseline"]:
            print("WARNING: nonzero flag rate -- should always be 0%, this needs investigating before trusting the result above.")
    if errored:
        print(f"NOTE: {errored} pair(s) errored (see per-pair WARNING lines above) and were excluded, not counted as losses.")

    from ratings.benchmark_nb import median_middlegame_depth_and_nps_nb

    bench = median_middlegame_depth_and_nps_nb()
    print(f"Median depth (middlegame only, pruning on as currently configured): {bench['median_depth']} "
          f"(range {bench['min_depth']}-{bench['max_depth']})")
    print(f"Median nps:                                                         {bench['median_nps']:,.0f}\n")
