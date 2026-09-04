# AI Chessathon — Engine Design Document

**Lock: 11 September 12:00 (uploads close 11:00). Written 31 August — 11 days.**

---

## 1. The box you're building in

Everything downstream follows from these, so they're worth internalising before any design decision.

| Constraint | Consequence |
|---|---|
| `agent.py` at zip root, 50 MB unzipped | Flat module layout; book + tablebases share the budget |
| Python 3.12 + torch / numpy / python-chess / onnxruntime / numba only | No Cython, no external chess libs, no compiler |
| Native binaries rejected, source must be judge-readable | numba is the only path to speed |
| 1 dedicated core, 2 GB RAM | Single-threaded search; threads past the first cost you time |
| No network, no GPU | Everything ships in the zip |
| 120 s + 0.5 s increment | See §7 — the increment dominates |
| 60 s init budget before the clock | JIT compilation must land here |
| 4096-byte output cap; malformed output = loss | Guard the output path |
| 300-ply adjudication on material | Games can run 150 moves per side |
| One process per game, alive between moves | State persists; pondering is legal |
| Third-party engines banned inside the zip | Training data annotated by an engine is fine |

### The clock, computed

At the 300-ply cap you get 150 moves per side. Total time available is 120 + 150 × 0.5 = 195 s, i.e. about **1.3 s per move** in the worst case. Realistically most games end sooner and you'll average 1.5–2 s.

That number sets your entire strength ceiling:

| Node rate | Nodes per move | Approx. depth |
|---|---|---|
| 30k nps (python-chess) | ~45k | 6–7 |
| 500k nps (numba bitboards) | ~750k | 9–10 |
| 5M nps (C, not permitted) | ~7.5M | 12–13 |

Three plies is roughly 150–200 Elo. This is why §4 is the highest-stakes decision in the document.

---

## 2. Target

GM strength is ~2500. A first engine, in Python, on one core, in eleven days, realistically lands **2000–2300**. That beats almost every club player and loses to a GM.

It is also a strong competition entry, because your opponents are other people's eleven-day bots — not Magnus.

Elo budget, roughly, from a 1400-ish baseline:

| Component | Gain |
|---|---|
| Alpha-beta + material + PST | baseline |
| Quiescence search | +200 |
| Move ordering (TT move, MVV-LVA, killers, history) | +200 |
| Transposition table | +100 |
| Texel-tuned eval | +150 |
| Extra eval terms (pawns, king safety, mobility) | +200 |
| Late move reductions | +80 |
| Null-move pruning | +70 |
| Faster movegen (numba) | +300–400 |
| Aspiration windows | +25 |
| Opening book | +20 and clock savings |
| Pondering | +50 |

The top five are cheap and non-negotiable. The numba line is the largest single item and the riskiest.

---

## 3. Architecture

```
agent.py              entry point, fallback guard, init hook
cb_engine.py          per-game state, orchestration
cb_search.py          iterative deepening, negamax, quiescence
cb_movegen.py         v1: python-chess wrapper / v2: numba bitboards
cb_eval.py            tapered evaluation
cb_tt.py              transposition table
cb_order.py           move ordering, killers, history
cb_time.py            clock budgeting
cb_book.py            polyglot book probe
cb_tables.py          generated constants (PSTs, zobrist, magics)
book.bin              opening book
```

`cb_` prefixes throughout. The zip goes first on `sys.path`, so a file called `chess.py` or `types.py` shadows the real module and kills your agent in validation.

### The fallback guard

`get_move` must never raise and never return an illegal move. Structure it as:

```python
def get_move(fen, time_left_ms):
    try:
        move = ENGINE.search(fen, time_left_ms)
        if move is not None and _is_legal(fen, move):
            return move
    except Exception:
        pass
    return _first_legal_move(fen)   # dependency-free, obviously correct
```

A crash is a loss. A bad move is a slightly worse position. The asymmetry is enormous and this guard is the cheapest Elo in the whole project.

---

## 4. Board representation — the decision that matters

**v1: python-chess.** Correct, pleasant, and roughly 100× too slow. ~20–50k nps in search.

**v2: numba bitboards.** Position as a small array of `uint64`. Magic bitboards for sliders, precomputed attack tables built at import. Search recursion jitted end-to-end. 500k–2M nps.

The rule: **build v1 first and get it legal, then attempt v2 behind a flag.**

Rewriting movegen is where eleven-day projects die. A subtly wrong en passant or castling-rights rule surfaces as an illegal move in a rated game, which is an instant loss. So:

- Validate against `python-chess` with perft on the six standard positions to depth 6.
- If perft does not match **exactly**, v2 does not ship. No exceptions, no "it's probably fine".
- Keep v1 working the whole time as the fallback path and the perft oracle.

numba specifics:
- `@njit(nopython=True)` throughout the hot path. No Python objects, no dicts, no lists inside jitted code.
- Preallocate move arrays; pass numpy buffers, never allocate per node.
- Every Python↔jit boundary crossing costs microseconds. Cross once per search, not once per node.
- Warm every jitted function with a dummy call at import so compilation lands in the 60 s init budget.
- **Measure the compile time.** A full jitted search kernel can take 30–60 s to compile. If you're near the budget, you fail with `init` and lose. Caching to `/tmp` doesn't help — the process is fresh every game.

---

## 5. Search

### Iterative deepening

Search depth 1, then 2, then 3... Counterintuitively faster than going straight to depth N, because each iteration fills the TT with move-ordering information for the next. It also guarantees a usable move whenever the clock stops.

### Negamax with alpha-beta and TT

```
def negamax(depth, alpha, beta, ply):
    alpha_orig = alpha

    entry = tt.probe(hash)
    if entry and entry.depth >= depth:
        if entry.flag == EXACT:  return entry.score
        if entry.flag == LOWER:  alpha = max(alpha, entry.score)
        if entry.flag == UPPER:  beta  = min(beta, entry.score)
        if alpha >= beta:        return entry.score

    if depth <= 0:
        return qsearch(alpha, beta, ply)

    # ... null move, move loop with LMR ...

    flag = EXACT
    if best <= alpha_orig: flag = UPPER
    elif best >= beta:     flag = LOWER
    tt.store(hash, depth, best, flag, best_move)
    return best
```

Two traps: **mate scores must be ply-adjusted** on store and probe (store `score + ply`, probe `score - ply`), or you'll get bogus mates from transposed positions. And a TT move must be **verified legal** before playing it — hash collisions happen and an unverified TT move is an illegal-move loss.

TT design: fixed-size array of packed entries, power-of-two count, index by `hash & mask`. 2 GB memory, so 2^22 entries (~64 MB) is comfortable. Replace by depth, preferring the current search generation. Zobrist keys generated at import from a fixed seed.

### Move ordering

The single highest-leverage part of the engine. Alpha-beta's pruning depends entirely on searching the best move first; with good ordering the effective branching factor drops from ~35 to ~3.

Priority:
1. TT move
2. Winning and equal captures, MVV-LVA (or SEE if you get there)
3. Killer moves — two quiet moves per ply that caused a cutoff
4. History heuristic — `history[piece][to_square] += depth²` on cutoff
5. Losing captures

### Quiescence search

At depth 0, don't evaluate immediately — search captures until the position is quiet. Without this your engine evaluates mid-capture-sequence and hallucinates constantly. About forty lines, worth ~200 Elo.

```
def qsearch(alpha, beta, ply):
    stand_pat = evaluate()
    if stand_pat >= beta: return beta
    alpha = max(alpha, stand_pat)
    for capture in ordered_captures():
        if stand_pat + see(capture) + 200 < alpha: continue   # delta pruning
        score = -qsearch(-beta, -alpha, ply+1)
        ...
```

Cap the depth (say 8 plies) so pathological positions can't explode.

### Pruning and reductions

- **Null-move pruning**: give the opponent a free move at reduced depth; if you still fail high, prune. Conditions: not in check, depth ≥ 3, side to move has non-pawn material (zugzwang), no null move at the previous ply. R = 2 or 3.
- **Late move reductions**: after the first ~4 moves, reduce quiet moves at depth ≥ 3. Re-search at full depth if the reduced search fails high.
- **Check extensions**: +1 ply when in check. Cheap, prevents horizon blunders.
- **Aspiration windows**: from depth 5, search a narrow window (±30 cp) around the previous score; widen and re-search on fail.

Add these one at a time, each behind a flag, each SPRT-tested. Several will not gain what the literature claims in your specific engine.

---

## 6. Evaluation

Tapered: compute a phase value from remaining material, evaluate middlegame and endgame separately, interpolate.

```
score = (mg_score * phase + eg_score * (256 - phase)) // 256
```

**v1 terms:** material, piece-square tables, both phase-tapered.

**v2 terms, in rough order of value:** passed pawns (by rank), isolated and doubled pawns, rook on open/semi-open file, bishop pair, mobility (legal move count by piece type), king safety (attacker count and weight on the king zone), king pawn shield.

Adding a term the eval currently cannot express at all is worth much more than refining one it already has. Resist the temptation to bucket PSTs by king square: it breaks incremental PST updates, since a king move invalidates every piece's contribution and forces a full recompute.

### Texel tuning

Fit the weights by logistic regression against game outcomes:

```
E = (1/N) Σ (result_i − σ(K · eval_i))²
```

where `σ` is the logistic function and `K` is fitted first by a 1-D scan. Minimise by coordinate descent or Adam over the weight vector.

Use **`quiet-labeled.epd`** from Zurichess: 725k positions, pre-filtered so quiescence found no winning capture, labelled with game results. Because the positions are quiet you can call `evaluate()` directly rather than going through qsearch, which makes tuning tractable in Python. ~30 MB. This is what most engines at your target strength tuned on.

Do **not** reach for the Lichess eval database this time — the positions aren't quiet, the evals come from mixed depths and Stockfish versions, and the distribution is human-analysis moments rather than positions your search visits. Filtering it properly needs a working qsearch and a day of pipeline you don't have.

Tuning data never ships. Only the fitted numbers go in the zip, as a Python file.

---

## 7. Time management

Own module, tested independently. It must never flag — a flag is a loss regardless of how well you were playing.

```
reserve      = 2000 ms                       # process overhead, safety
moves_to_go  = max(20, 55 - ply // 2)
usable       = max(0, time_left - reserve)
soft         = increment * 0.9 + usable / moves_to_go
hard         = min(soft * 4, usable * 0.3)
```

- **Soft limit**: checked between iterative-deepening iterations. If exceeded, don't start another depth.
- **Hard limit**: checked inside the search, every 2048 nodes (not every node — the syscall costs more than you think). On hit, unwind immediately and return the best move from the last *completed* iteration.

Never return a move from a partially completed iteration unless it's better than the previous iteration's move at the point you stopped. A half-searched depth can be worse than the depth below it.

The 0.5 s increment means that in steady state you can spend ~0.5 s per move forever. Lean on it. Bank time early, spend it in the middlegame.

---

## 8. Shipped data

| Item | Size | Value |
|---|---|---|
| Agent source | < 1 MB | — |
| Opening book (Polyglot) | 2–8 MB | ~20 Elo + clock savings |
| 3–4 man Syzygy WDL | ~3 MB | marginal, cheap |
| Headroom | rest | |

`chess.polyglot` and `chess.syzygy` are both in the base image.

Build the book yourself from Lichess monthly PGN dumps, filtered to high-rated games. Better than a downloaded generic book because you control depth and breadth against your size budget — and it removes any question about provenance. 3–5 man Syzygy WDL is 378 MiB and won't fit; 3–4 man alone will.

---

## 9. Pondering

The process keeps its dedicated core after `get_move` returns, and the rules explicitly permit thinking while the opponent moves. Almost nobody implements this under time pressure. Worth ~50 Elo.

Design: on returning a move, predict the opponent's reply (the second move of your PV), start a background thread searching that position, and store results in the TT. When `get_move` arrives, check whether the actual position matches your prediction — if so you inherit a warm TT, if not you discard.

**Do this last, and carefully.** A background thread still running when the next `get_move` arrives will contend for your single core and cost you the game. Signal it to stop and join it before starting your own search.

---

## 10. Testing

This is not optional infrastructure — most changes that feel like improvements are noise or regressions. Stockfish's dominance owes as much to Fishtest as to any single algorithm.

**Perft.** Six standard positions to depth 6, matched against python-chess. Gate on v2 movegen shipping.

**Tactical suites.** WAC (300 positions), Bratko-Kopec (24), STS (1500 across 15 themes). Also filter the Lichess puzzle CSV (6M rated, themed puzzles) by rating band to build custom suites. More diagnostic than Elo when debugging *why* the engine is bad.

**Version zoo.** Freeze every version. Ratings only mean anything if old versions stay playable.

**Gauntlets.** Alternate colours, draw start positions from a balanced book (8moves_v3.pgn or 2moves_v1.pgn — the biased UHO books are for engines much stronger than yours). Without varied openings, deterministic bots replay identical games and you learn nothing.

**Rating.** Store every game ever played, then fit all ratings jointly by logistic maximum likelihood with one version pinned at 0. Not incremental Elo updates.

**SPRT** for A/B decisions. `elo0=0, elo1=5, alpha=beta=0.05` for gainers; `elo0=-5, elo1=0` for non-regressions. Answers "is v5 better than v4" in 200–400 games instead of several thousand.

**Smoke test** replicating validation: fresh import, init well under 60 s, one game as each colour, every move legal, no output over 4096 bytes.

---

## 11. Build checks

The build script produces the zip and **refuses to write it** if any check fails:

- `agent.py` present at the root, not inside a folder
- Unzipped size under 50 MB
- AST-scan every shipped `.py` for imports outside the allowed set
- No native binaries, no `__pycache__`
- No filename shadowing a stdlib or allowed-package module
- Smoke test passes

Print size and file list every time. Six uploads per team per day — don't waste one on a zip that fails validation on a structural mistake.

---

## 12. Plan

| Days | Work |
|---|---|
| 1–2 | Negamax, alpha-beta, quiescence, MVV-LVA, material + PST, on python-chess. Legal, doesn't flag. **Submit it.** |
| 3–4 | TT, killers, history, iterative deepening, aspiration windows. Harness, zoo, SPRT working. |
| 5–7 | numba movegen attempt. Perft-validated. Abandoned without regret if not clean by end of day 7. |
| 8–9 | Null move, LMR, check extensions. Eval terms. Texel tuning. |
| 10 | Opening book. Tablebases. Pondering if time permits. |
| 11 | Gauntlets only. No new features. Final submission well before 11:00. |

**Submit something playable on day 2.** The ladder runs hourly from 08:00 to 22:00 and the house bots show public CCRL ratings — that's free calibration of your internal Elo scale onto a real one, and it's available from the moment you have a legal agent.

Day 11 is for testing, not building. Anything not SPRT-validated by day 10 doesn't ship.

---

## 13. Risk register

| Risk | Mitigation |
|---|---|
| numba movegen has a legality bug | Perft gate; v1 stays shippable throughout |
| JIT compile exceeds 60 s init | Measure early; split kernels; fall back to v1 |
| Time manager flags | Hard limit with 2 s reserve; explicit test asserting the budget is never exceeded |
| Crash in search | Fallback guard in `get_move` |
| Zip fails validation on structure | Automated build checks; never hand-assemble |
| Pondering thread contends for the core | Join before searching; ship it last or not at all |
| Tuning overfits | Hold out 20% of positions; validate by SPRT, not by training error |
| Feature added on day 11 loses Elo | Feature freeze on day 10 |

---

## 14. What not to do

**NNUE.** The environment ships torch and onnxruntime and the 50 MB budget invites it. Don't. The incremental accumulator requires your own make/unmake, so it's gated behind the numba rewrite rather than parallel to it. Neither torch nor onnxruntime is usable at search time — per-call overhead is tens of microseconds, orders of magnitude too slow at 100k+ nps — so you'd hand-write quantised integer matmuls in numba. Then data prep, training, quantisation-aware validation, and the near-certainty that your first three nets are worse than a tuned handcrafted eval. It's the natural v2, after the competition.

**MCTS.** Conquered Go, but plain MCTS is weak at chess: the game is sharp, a single buried refutation flips the assessment, and a sampling search under-explores exactly that move. It works in AlphaZero only because a trained policy network fixes that weakness — and that needs GPUs you don't have.

**C.** Native binaries are rejected, Cython is called out by name, there's no compiler in the image and no network to fetch one. The stated reason is that a judge must be able to read your source to clear a flagged game. numba is the sanctioned escape hatch and gets within 2–4× of C on tight bitboard loops.

**Multithreaded search.** One core. Threads past the first share it and cost you time.

---

## 15. The short version

Alpha-beta with a Texel-tuned evaluation, ruthless move ordering, and a real testing loop. That isn't a compromise version of the modern approach — it's the modern approach with the one component swapped out that needs a GPU and a month.

Correct and never flagging beats clever and crashing. Ship early, measure everything, freeze on day 10.
