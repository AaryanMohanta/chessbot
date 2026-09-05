# AI Chessathon Agent

## Hard environment constraints

These come from the competition runtime and are non-negotiable — the
tooling in this repo (`tools/build_zip.py` especially) enforces them so a
violation is caught locally instead of at validation.

- Submission is a zip with `agent.py` **at the root** (no wrapping folder), 50 MB unzipped.
- `agent.py` exposes exactly: `def get_move(fen: str, time_left_ms: int) -> str` returning a UCI move (e.g. `e2e4`, `e7e8q`).
- Python 3.12, full stdlib, and only: `torch==2.13.0+cpu`, `numpy==2.5.2`, `python-chess==1.11.2` (imported as `chess`), `onnxruntime==1.29.0`, `numba==0.67.0`. Nothing else installs at validation; `requirements.txt` in the zip is ignored. Any other import crashes the agent.
- No native binaries in the zip (no Cython/compiled extensions). `.onnx` / `.safetensors` / `.pt` weight files are fine.
- The zip goes first on `sys.path` — no shipped file may be named after a module we import (`chess.py`, `random.py`, `types.py`, ...). Hence the `cb_` prefix on every module we add.
- Runtime: 1 core of an AMD EPYC 9V74 @ 2.60 GHz (specified 2026-09; previously just "1 core"), 2 GB RAM, no network, no GPU, read-only FS except 256 MB at `/tmp`.
- Clock: 120 s per side + 0.5 s increment, with a 90 s init budget spent *before* the clock starts (raised from 60 s, 2026-09 rules update).
- Output cap: 4096 bytes/move. Malformed output = illegal move = loss.
- One process per game, alive between moves, but suspended while the opponent is thinking (2026-09 rules update: pondering is explicitly disabled, not just unimplemented).

See `chessathon-engine-design.md` for the full architecture and the day-by-day
plan. This repo currently implements days 1-2 of that plan: a correct,
never-crashing, never-flagging v1 engine on python-chess. No TT, no null
move / LMR, no numba, no NNUE, no opening book yet — those are days 3+.

## Repo layout

```
agent.py            entry point, fallback guard — ships at the zip root
cb_engine.py         per-game state, orchestration            \
cb_search.py         iterative deepening, negamax, quiescence   } ship at
cb_eval.py           tapered material + PST evaluation          } the zip
cb_order.py          MVV-LVA move ordering                      } root
cb_time.py           §7 time budgeting                         /
cb_tables.py         piece-square tables (plain lists, tuner-replaceable)

harness/              dev-only: subprocess wire-protocol test harness
  stdio_runner.py      subprocess entry point (loads an agent, speaks stdio JSON)
  match.py             plays one game between two agent processes, enforces the clock
  clock.py             per-side clock bookkeeping
  report.py            formats results

baselines/            dev-only opponents to test against
  random_mover.py
  greedy_capture.py

tools/
  build_zip.py         builds & validates the submission zip
  smoke_test.py        replicates the validator's smoke checks

tests/
  positions.py          six standard perft positions
  test_perft.py          depth<=3 fast (default run); depth 4-5 marked `slow`
  test_tactics.py        verified mate-in-1/mate-in-2 fixtures + stubs for eval-dependent cases
  test_time_management.py  real invariant tests for cb_time (it's plumbing, not a stub)
  test_fallback_guard.py   proves the §3 guard by deliberately breaking the engine
```

Only `agent.py` and the `cb_*.py` files ship in the submission zip. Everything
else (`harness/`, `baselines/`, `tools/`, `tests/`) is local dev tooling.

Expensive tests (perft depth >= 4, running into millions of nodes) are
marked `slow` and skipped by default — run them explicitly:

```bash
pytest -m slow tests/test_perft.py
```

## Design choices and open questions

- **Safety net**: `agent.py` wraps engine construction and every
  `get_move` call in a guard, then independently re-verifies the returned
  move is legal against the FEN (not just "no exception was raised") —
  see `tests/test_fallback_guard.py`, which proves this by deliberately
  making the engine raise, lie about legality, and return garbage.
- **Check extensions have no cap.** The design doc specifies "+1 ply when
  in check" unconditionally. A long forced-check sequence could in theory
  drive Python's recursion depth up; the periodic hard-time-limit node
  check (every 2048 nodes) still fires regardless of depth, and the
  fallback guard catches a `RecursionError` like any other exception, so
  this can't crash a game — but it's an unbounded-recursion smell worth
  revisiting when killers/history land in days 3-4.
- **Quiescence delta pruning uses raw captured-piece value, not SEE.**
  The design doc's qsearch sketch prices the margin off `see(capture)`,
  but SEE is explicitly out of scope for v1 move ordering ("MVV-LVA, or
  SEE if you get there"). Using full SEE for pruning while not using it
  for ordering would be an odd asymmetry, so v1 uses the simpler
  MVV-LVA-consistent stand-in.
- **`cb_time`'s exact §7 formula can produce `soft > hard`** when usable
  time is small relative to the increment (e.g. 2.5s left -> soft=459ms,
  hard=150ms). Verified harmless (see `tests/test_time_management.py`):
  hard is enforced independently inside the search and its wall-clock
  deadline still lands first in that regime, so the hard cutoff always
  wins regardless of which number is nominally bigger.
- **Repetition tracking is partial by design.** `cb_engine` counts
  positions it's asked to move from, keyed on everything relevant to
  repetition *except* the move-counters — but the wire protocol only
  hands us a FEN on our own turns, with no move history, so this can only
  see positions where it was our move. It's bookkeeping for a future
  search heuristic (e.g. contempt near a draw), not authoritative
  threefold detection — the referee claims that itself.
- **Null-move pruning and LMR were each SPRT-tested against the version
  immediately before it, not against each other factorially** — null-move
  was tested with LMR off on both sides, LMR was tested with null-move on
  on both sides. Both scored positive (52% and 57% over 100 games each,
  both inconclusive against a 0/20 Elo bound — see
  `ratings/calibration_logs/`, gitignored) and neither regressed, so both
  are kept on. But because they weren't tested factorially, **their
  measured effects are not additive** — don't read "null-move is worth
  ~+14 Elo and LMR is worth ~+49 Elo" as "the pair is worth +63"; the two
  techniques interact (LMR's reductions change which lines null-move's
  verification search even reaches), and only a joint test would say by
  how much.
- **SPRT bounds and game budgets, standardized:** [0, 10] Elo for a novel
  change, [-10, 0] (non-regression) for a change with a strong literature
  prior, 4000-8000 games. Narrower bounds need *more* games, not fewer —
  expected sample size scales roughly as 1/(elo1-elo0)^2, so don't reach
  for [0, 5] hoping for a faster answer; it costs ~16x the games of
  [0, 20] for the same statistical power. `ratings/sprt.py`'s
  `run_sprt_pentanomial` pairs games from the same opening with reversed
  colours (a GSPRT on the paired-score distribution) for the same
  variance-reduction benefit cutechess-cli/fastchess get from pentanomial
  LLR — implemented directly rather than adopted from either tool, since
  our agent speaks a custom JSON wire protocol, not UCI, so a UCI-only
  driver can't run it regardless.
- **All testing (SPRT and Stockfish calibration) runs at a fast time
  control (10+0.1), not the tournament one (120+0.5).** The tournament
  clock is reserved for final validation runs and the wire-protocol smoke
  test — nobody SPRTs at their competition TC, it's needlessly slow for
  no statistical benefit.

## Running the harness

Install dev dependencies once (see `requirements-dev.txt`; this file is
never shipped and never read by the validator):

```bash
pip install -r requirements-dev.txt
```

Play one game, agent (white) vs. the random-mover baseline:

```bash
python -c "from harness.match import play_game; from harness.report import format_game; r = play_game('agent.py', 'baselines/random_mover.py'); print(format_game(r, 'agent', 'random_mover'))"
```

Swap in `baselines/greedy_capture.py` for a slightly stronger opponent, or
swap the FEN/side order to play as black.

Run the smoke test (fresh import, init-time check, one game as each colour
vs. a baseline, legality + output-size checks — this is what validation
will effectively do to your submission):

```bash
python tools/smoke_test.py
```

Run the test suite (perft + whatever tactical/time-management tests you've
filled in):

```bash
pytest
```

## Building the submission zip

```bash
python tools/build_zip.py
```

This AST-scans every shipped `.py` file's imports (no code execution),
checks for filename shadowing of stdlib/allowed-package names, rejects
native binaries and `__pycache__`, and checks total unzipped size against
the 50 MB cap. It refuses to write the zip if anything fails, and prints
the size and file list on success. Output goes to `dist/submission.zip`
(pass a different path as the first argument to override).

## Numba bitboard experiment (`cb_nb_*.py`) — parallel, unshipped

Not in `tools/build_zip.py`'s whitelist and doesn't touch `agent.py` or
any shipped module — v1 (`cb_search.py` on python-chess) stays the
fallback and the perft oracle throughout. Status by stage:

| Stage | What | Gate | Result |
|---|---|---|---|
| 1 | Flat bitboard representation, FEN parse/serialize | 10k-position round-trip vs python-chess | 0 errors / 11,859 |
| 2 | Movegen + make/unmake (magic bitboards, self-verified) | Perft exact, depth 5 (6 for startpos), all six standard positions | Exact, 1.1-1.6M nps |
| 3 | Incremental Zobrist through make/unmake | Matches from-scratch recompute at every node, full games | 0 errors / 31,880+ checks |
| 4 | Incremental material+PST (reuses v1's `cb_tables.py` data) | Same treatment as stage 3, plus cross-check vs v1's `cb_eval.py` | 0 errors; exact match on a real position |
| 5 | Search (negamax/PVS/quiescence/TT/MVV-LVA) ported into njit | Real search nps on middlegame positions; mate-in-1/2 fixtures; node-for-node match vs v1 | See below |

**Stage 5 result: median 1.1M search nps, median depth 7.0** on the same
middlegame benchmark v1 scores depth 5.0 / 19.6k nps on — roughly 56x, not
the 3-5x-below-perft (~250-500k) that was expected going in. That gap was
investigated rather than accepted: cross-checked node counts and best
moves against v1's own search (null-move/LMR disabled in v1, since this
port doesn't have them yet) at fixed depths 4-7 on the same position —
**identical scores and identical best moves at every depth**, which is
strong evidence the algorithm's behavior was preserved exactly rather than
numba silently over-pruning. The likely explanation: the "3-5x" heuristic
assumes eval/ordering/TT overhead comparable to many existing engines'
implementations; here that overhead is proportionally small relative to
movegen (simple material+PST eval, insertion-sort ordering, direct
array-indexed TT), so search nps lands much closer to the raw
movegen-only rate than the heuristic predicted.

**Scope not yet ported from v1**: killers, history heuristic, null-move
pruning, LMR. Core negamax/PVS/quiescence/TT/MVV-LVA only.

**Compile budget is tightening.** Cold-start (numba cache cleared, target
30s / hard budget 60s) by stage: movegen alone 12.85s -> +Zobrist 9.8s ->
+eval 7.6s (noise, not a real improvement) -> **+search 22.16s**. Still
under the 30s target but with less margin than is comfortable, and this
is *before* porting killers/history/null-move/LMR, which will add more
compiled code. Worth watching closely, possibly splitting kernels, before
adding more.
