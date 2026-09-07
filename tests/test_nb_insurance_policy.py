"""Insurance-policy gate for cb_nb_engine.Engine's deadline-based compile:
__init__ blocks through the three priority-ordered compile stages
(movegen/make-unmake, eval/Zobrist, search kernel) against a wall-clock
deadline; whatever doesn't finish in time is handed to a background
thread, with get_move serving from v1 until it catches up.

Stage functions (Engine._STAGES) are monkeypatched to fast/slow/raising
fakes so these tests run in a fraction of a second instead of the ~40s a
real compile takes, and so each of the three required outcomes can be
triggered deterministically rather than depending on real hardware
timing:

  1. test_finishes_inside_deadline_costs_nothing   -- all stages finish
     before the deadline; numba is used immediately, no background
     thread is ever needed.
  2. test_overruns_deadline_and_hands_off_to_background -- movegen and
     eval/Zobrist finish blocking, but the deadline lands before the
     (much larger, in reality) search stage is attempted, so it's
     deferred to the background thread; get_move serves legal moves from
     v1 in the meantime and switches over once the background compile
     finishes. This mirrors the real measured shape on the Linux
     reference container: movegen+eval/Zobrist together take ~6s, the
     search kernel alone takes ~30s -- by far the dominant cost, and the
     stage most likely to still be running when a deadline lands.
  3. test_falls_back_to_v1_permanently_if_a_stage_raises -- a stage
     raising is treated as a permanent numba failure, not retried.

Every test drains its engine's background thread (waits for
_numba_ready) before returning, so a leftover daemon thread doesn't keep
running into later tests -- competing for the GIL/CPU there previously
made one of these tests genuinely flaky (see git history).
"""
import time

import chess
import pytest

import cb_nb_engine
import cb_nb_fast as F

STARTPOS = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
MIDDLEGAME_FEN = "r1bqk2r/ppp2ppp/2n2n2/2bpp3/2B1P3/3P1N2/PPP2PPP/RNBQ1RK1 w kq - 0 7"

# Small enough that v1's own soft/hard time budget (see cb_time.py's §7
# formula) is a few hundred ms, not multiple seconds -- keeps tests fast
# and avoids racing v1's own thinking time against a fake stage's delay.
V1_FAST_TIME_LEFT_MS = 3_000

_PROMO_FROM_CHESS = {
    None: F.NO_PROMO,
    chess.KNIGHT: F.KNIGHT,
    chess.BISHOP: F.BISHOP,
    chess.ROOK: F.ROOK,
    chess.QUEEN: F.QUEEN,
}


def _pack_first_legal_move(fen: str) -> int:
    board = chess.Board(fen)
    move = next(iter(board.legal_moves))
    promo = _PROMO_FROM_CHESS[move.promotion]
    return F.pack_move(move.from_square, move.to_square, promo, F.FLAG_NORMAL)


def _assert_legal(fen: str, uci: str) -> None:
    board = chess.Board(fen)
    assert chess.Move.from_uci(uci) in board.legal_moves, f"{uci} illegal in {fen}"


def _fake_stage(name: str, delay: float = 0.0, raises: bool = False):
    """Builds a fake replacement for one of Engine._STAGES's compile
    stage methods: optionally sleeps (standing in for real compile time),
    optionally raises (standing in for a genuine compile failure)."""

    def _stage(self):
        if delay:
            time.sleep(delay)
        if raises:
            raise RuntimeError(f"simulated failure in stage {name!r}")

    _stage.__name__ = f"fake_stage_{name}"
    return _stage


class _FakeNumbaSearch:
    """Stands in for cb_nb_search.search once the (faked) stages report
    ready -- always returns a real legal move for the given FEN via
    python-chess, so legality assertions are meaningful without needing
    an actual compiled search."""

    def __init__(self):
        self.calls = 0

    def __call__(self, fen, soft_ms, hard_ms, arrays, generation=1, max_depth=64):
        self.calls += 1
        move = _pack_first_legal_move(fen)
        return move, 0, {"depth": 1, "nodes": 1}


@pytest.fixture
def fake_numba_search(monkeypatch):
    fake = _FakeNumbaSearch()
    monkeypatch.setattr(cb_nb_engine.S, "search", fake)
    return fake


def _drain(engine, timeout=10.0):
    """Wait for the background warmup thread (if any) to finish so it
    doesn't leak into later tests. Not an assertion -- callers that care
    whether warmup actually completed check the return value/flags
    themselves; this just enforces test-to-test isolation."""
    engine._numba_ready.wait(timeout=timeout)


def test_finishes_inside_deadline_costs_nothing(monkeypatch, fake_numba_search):
    monkeypatch.setattr(cb_nb_engine.Engine, "_STAGES", (
        _fake_stage("movegen", delay=0.02),
        _fake_stage("eval_zobrist", delay=0.02),
        _fake_stage("search", delay=0.02),
    ))
    start = time.monotonic()
    engine = cb_nb_engine.Engine(process_start=start)
    elapsed = time.monotonic() - start

    assert elapsed < 1.0, f"blocking compile took {elapsed:.2f}s for three 0.02s fake stages"
    assert engine._numba_ready.is_set()
    assert not engine._numba_failed

    move = engine.get_move(STARTPOS, time_left_ms=V1_FAST_TIME_LEFT_MS)
    _assert_legal(STARTPOS, move)
    assert fake_numba_search.calls == 1, "should have gone straight to numba, no v1 fallback needed"


def test_overruns_deadline_and_hands_off_to_background(monkeypatch, fake_numba_search):
    stage3_delay = 1.2  # comfortably longer than v1's real ~0.3-0.5s search under V1_FAST_TIME_LEFT_MS
    monkeypatch.setattr(cb_nb_engine, "COMPILE_DEADLINE_S", 0.03)
    monkeypatch.setattr(cb_nb_engine.Engine, "_STAGES", (
        _fake_stage("movegen", delay=0.02),
        _fake_stage("eval_zobrist", delay=0.02),
        _fake_stage("search", delay=stage3_delay),
    ))

    start = time.monotonic()
    engine = cb_nb_engine.Engine(process_start=start)
    elapsed = time.monotonic() - start

    # Stages 1+2 (0.04s total) run blocking -- the deadline (0.03s) is
    # already behind by the time stage 3 is considered, so it's deferred
    # to the background thread instead of blocking __init__ for another
    # 1.2s.
    assert elapsed < 0.5, f"__init__ blocked for {elapsed:.2f}s -- stage 3 should have been backgrounded"
    assert not engine._numba_ready.is_set()
    assert not engine._numba_failed

    move = engine.get_move(STARTPOS, time_left_ms=V1_FAST_TIME_LEFT_MS)
    _assert_legal(STARTPOS, move)
    assert not engine._numba_ready.is_set(), "stage 3 fake is still sleeping -- this answer must have come from v1"
    assert fake_numba_search.calls == 0

    ready = engine._numba_ready.wait(timeout=stage3_delay + 5)
    assert ready, "background stage 3 never completed"
    assert not engine._numba_failed

    move = engine.get_move(MIDDLEGAME_FEN, time_left_ms=V1_FAST_TIME_LEFT_MS)
    _assert_legal(MIDDLEGAME_FEN, move)
    assert fake_numba_search.calls == 1, "should now be routed through the (now-ready) numba path"


@pytest.mark.parametrize("failing_stage_index", [0, 2], ids=["movegen_raises", "search_raises"])
def test_falls_back_to_v1_permanently_if_a_stage_raises(monkeypatch, fake_numba_search, failing_stage_index):
    stages = [
        _fake_stage("movegen", delay=0.01),
        _fake_stage("eval_zobrist", delay=0.01),
        _fake_stage("search", delay=0.01),
    ]
    stages[failing_stage_index] = _fake_stage("failing", raises=True)
    monkeypatch.setattr(cb_nb_engine.Engine, "_STAGES", tuple(stages))

    engine = cb_nb_engine.Engine(process_start=time.monotonic())

    assert engine._numba_ready.is_set(), "ready must fire immediately -- no point backgrounding a known failure"
    assert engine._numba_failed

    for fen in [STARTPOS, MIDDLEGAME_FEN]:
        move = engine.get_move(fen, time_left_ms=V1_FAST_TIME_LEFT_MS)
        _assert_legal(fen, move)
    assert fake_numba_search.calls == 0, "a failed compile must never fall through to the numba search"


def test_stays_legal_across_a_short_game_spanning_the_deadline_handoff(monkeypatch, fake_numba_search):
    """Plays a handful of real plies through the exact window where the
    backgrounded stage 3 finishes mid-game, using real python-chess to
    advance the board -- the strongest available proof that the v1-to-
    numba handoff can't hand back an illegal or malformed move at any
    point in that transition."""
    stage3_delay = 1.2
    monkeypatch.setattr(cb_nb_engine, "COMPILE_DEADLINE_S", 0.03)
    monkeypatch.setattr(cb_nb_engine.Engine, "_STAGES", (
        _fake_stage("movegen", delay=0.02),
        _fake_stage("eval_zobrist", delay=0.02),
        _fake_stage("search", delay=stage3_delay),
    ))

    engine = cb_nb_engine.Engine(process_start=time.monotonic())
    board = chess.Board(STARTPOS)

    for _ in range(6):
        move_uci = engine.get_move(board.fen(), time_left_ms=V1_FAST_TIME_LEFT_MS)
        move = chess.Move.from_uci(move_uci)
        assert move in board.legal_moves, f"{move_uci} illegal in {board.fen()}"
        board.push(move)
        time.sleep(0.4)  # let the backgrounded stage 3 clock advance across the loop

    _drain(engine)
    assert engine._numba_ready.is_set()
    assert not engine._numba_failed


def test_real_unmocked_compile_succeeds_and_numba_actually_plays():
    """Every test above fakes Engine._STAGES, which is exactly why a real
    ladder loss (round 52) went undetected here: a plain argument-count
    mismatch in _stage_eval_zobrist's real body (introduced alongside an
    evaluate_from_state signature change, missed at this one call site)
    made numba fail to compile every single time, with every mocked test
    above still passing clean since none of them ever call the real
    stage functions. tools/smoke_test.py had the same blind spot --
    "no crash, legal moves" is exactly what a silent, permanent v1
    fallback also produces.

    This test pays the real ~30-70s compile cost specifically so a
    signature mismatch like that one fails LOUDLY in the default test
    suite instead of silently on the real ladder -- worth the added
    runtime given what the alternative already cost once."""
    engine = cb_nb_engine.Engine(process_start=time.monotonic())
    assert not engine._numba_failed, "real numba compile failed -- see stderr for the traceback"
    assert engine._numba_ready.is_set(), "real compile should finish well inside the test's own patience"

    move = engine.get_move(STARTPOS, time_left_ms=30_000)
    _assert_legal(STARTPOS, move)
    assert engine.last_layer == "numba", (
        f"get_move answered via {engine.last_layer!r}, not the real numba search -- "
        "a silent fallback would still return a legal move, which is exactly the gap this test closes"
    )
