"""Replicates the validation harness's smoke checks:

  1. Fresh import + init well under the 60 s init budget.
  2. One full game as each colour against a baseline opponent.
  3. Every move returned was legal (a crash or illegal move ends the game
     as a loss for that side — this test asserts that never happens to us,
     which is exactly what agent.py's fallback path exists to guarantee).
  4. No move's wire-protocol output exceeded the 4096 byte cap.

Run: python tools/smoke_test.py
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from harness.match import AgentProcess, play_game  # noqa: E402
from harness.report import format_game  # noqa: E402
from cb_time import INIT_BUDGET_MS  # noqa: E402

AGENT_PATH = str(REPO_ROOT / "agent.py")
BASELINE_PATH = str(REPO_ROOT / "baselines" / "random_mover.py")

OUR_LOSS_REASONS_ARE_BUGS = (
    "illegal_move",
    "illegal_move_format",
    "malformed_output",
    "engine_error",
    "flag",
    "timeout_or_crash",
    "init_budget_exceeded",
    "output_too_large",
)


def check_init_time() -> bool:
    print("== init time ==")
    proc = AgentProcess(AGENT_PATH, "agent")
    try:
        proc.wait_ready()
    finally:
        proc.close()

    ok = proc.init_ms is not None and proc.init_ms < INIT_BUDGET_MS
    print(f"  init took {proc.init_ms:.1f} ms (budget {INIT_BUDGET_MS} ms) -> {'PASS' if ok else 'FAIL'}")
    if proc.init_ms is not None and proc.init_ms > INIT_BUDGET_MS * 0.8:
        print("  WARNING: close to the init budget")
    return ok


def check_game(agent_is_white: bool) -> bool:
    label = "white" if agent_is_white else "black"
    print(f"== game: agent plays {label} vs random_mover ==")
    if agent_is_white:
        result = play_game(AGENT_PATH, BASELINE_PATH)
        white_name, black_name = "agent", "random_mover"
        our_side = "white"
    else:
        result = play_game(BASELINE_PATH, AGENT_PATH)
        white_name, black_name = "random_mover", "agent"
        our_side = "black"

    print(format_game(result, white_name, black_name))

    our_reason_is_our_fault = result.reason.startswith(our_side) and any(
        bug in result.reason for bug in OUR_LOSS_REASONS_ARE_BUGS
    )
    times = result.white_times_ms if our_side == "white" else result.black_times_ms
    ok = not our_reason_is_our_fault
    print(f"  -> {'PASS' if ok else 'FAIL'} ({len(times)} moves by agent)")
    return ok


def main() -> int:
    results = [
        check_init_time(),
        check_game(agent_is_white=True),
        check_game(agent_is_white=False),
    ]
    passed = all(results)
    print()
    print("SMOKE TEST: " + ("PASS" if passed else "FAIL"))
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
