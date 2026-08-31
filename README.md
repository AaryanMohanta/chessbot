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
- Runtime: 1 core, 2 GB RAM, no network, no GPU, read-only FS except 256 MB at `/tmp`.
- Clock: 120 s per side + 0.5 s increment, with a 60 s init budget spent *before* the clock starts.
- Output cap: 4096 bytes/move. Malformed output = illegal move = loss.
- One process per game, alive between moves, keeps its core after `get_move` returns.

## Repo layout

```
agent.py            entry point — ships at the zip root
cb_engine.py         \
cb_search.py          } stubs with real signatures — ship at the zip root
cb_eval.py            } (see "What's stubbed" below)
cb_time.py           / time budgeting — implemented for real, not a stub

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
  positions.py          known perft positions
  test_perft.py          wired up against python-chess's move generator
  test_tactics.py        empty structured stubs — fill in once search/eval exist
  test_time_management.py  invariant tests for cb_time + stubs for search-level behavior
```

Only `agent.py` and the `cb_*.py` files ship in the submission zip. Everything
else (`harness/`, `baselines/`, `tools/`, `tests/`) is local dev tooling.

## What's stubbed vs. implemented

- **Stubbed** (`NotImplementedError`-adjacent, no algorithm): `cb_search.search`
  returns a uniformly random legal move; `cb_eval.evaluate` returns raw
  material count. Both have real signatures and docstrings — fill in the
  actual logic.
- **Implemented for real**: `cb_time.TimeManager.budget` — soft/hard
  per-move budgets from remaining time, increment, and ply. This is
  plumbing, not chess strategy, so it's done.
- **Safety net**: `agent.py` wraps engine construction and every
  `get_move` call in a guard. Any exception anywhere in `cb_engine` /
  `cb_search` / `cb_eval` falls back to the first legal move python-chess
  enumerates for the given FEN — dependency-free of our own (possibly
  buggy) code, and trivially correct.

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
