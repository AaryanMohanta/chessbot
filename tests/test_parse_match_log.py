"""Hand-verified tests for ratings/parse_match_log.py -- both that it
correctly extracts telemetry from the real line formats agent.py and
cb_nb_engine.py emit, and that it degrades gracefully on a log from
before this instrumentation existed (no move lines at all) rather than
raising, since real ladder logs from before this change will still be
in the pipeline.
"""
from ratings.parse_match_log import parse_build_tag, parse_init, parse_moves, report_from_moves

_SAMPLE_LOG = """\
[agent] build_tag=abc1234
[cb_engine] init took 1.7 ms (budget 90000 ms)
[cb_engine] init took 1.7 ms (budget 90000 ms)
[cb_nb_engine] init returned after 40000.0 ms (fully warm) ready_before_move1=True stages: stage_movegen=5000ms stage_eval_zobrist=3500ms stage_search=28000ms stage_cpu_warmup=3500ms
mv ply=0 layer=book d=- sc=- n=- t=0.00s left=120.0s
mv ply=1 layer=book d=- sc=- n=- t=0.00s left=120.5s
mv ply=2 layer=numba d=9 sc=+20 n=294146 t=1.02s left=118.5s
mv ply=3 layer=numba d=2 sc=-18 n=1200 t=0.05s left=117.0s
mv ply=4 layer=numba d=9 sc=+28 n=63625 t=0.24s left=117.5s
mv ply=5 layer=v1 d=6 sc=-45 n=5000 t=0.90s left=100.0s
"""


def test_parse_moves_extracts_all_fields():
    moves = parse_moves(_SAMPLE_LOG)
    assert len(moves) == 6
    assert moves[0].layer == "book" and moves[0].depth is None
    assert moves[2].layer == "numba" and moves[2].depth == 9 and moves[2].nodes == 294146
    assert moves[2].score == 20.0
    assert moves[5].layer == "v1"


def test_parse_init_extracts_stages_and_ready_flag():
    init = parse_init(_SAMPLE_LOG)
    assert init is not None
    assert init.ready_before_move1 is True
    assert init.status == "fully warm"
    assert init.stage_times_ms["stage_search"] == 28000.0


def test_report_flags_v1_usage_and_median_depth():
    moves = parse_moves(_SAMPLE_LOG)
    init = parse_init(_SAMPLE_LOG)
    report = report_from_moves(moves, init)
    assert report["v1_ever_played"] is True
    assert report["v1_move_count"] == 1
    assert report["v1_first_ply"] == 5
    # searched depths: 9, 2, 9, 6 -> median (9+6)/2? sorted [2,6,9,9] -> median (6+9)/2=7.5
    assert report["median_depth"] == 7.5


def test_report_detects_a_depth_collapse():
    moves = parse_moves(_SAMPLE_LOG)
    report = report_from_moves(moves, None)
    # ply=3 (depth 2) sits between two depth-9 searches -- a clear collapse
    collapsed_plies = [c[0] for c in report["depth_collapses"]]
    assert 3 in collapsed_plies


def test_parse_build_tag_extracts_the_tag():
    assert parse_build_tag(_SAMPLE_LOG) == "abc1234"


def test_parse_build_tag_returns_none_when_absent():
    assert parse_build_tag("no build tag line here\nmv ply=0 layer=book d=- sc=- n=- t=0.00s left=120.0s") is None


_REAL_DASHBOARD_LOG = """\
AI CHESSATHON MATCH LOG
=======================

MATCH
  Round          Rated 49

OUTPUT
  3,785 bytes on stderr

  [agent] build_tag=712f2aa
  [cb_engine] init took 1.5 ms (budget 90000 ms)
  [cb_nb_engine] init returned after 56723.1 ms (fully warm) ready_before_move1=True stages: stage_movegen=5423ms
  mv ply=16 layer=numba d=10 sc=-86 n=593794 t=2.15s left=120.0s
  mv ply=18 layer=numba d=9 sc=-136 n=241604 t=0.87s left=118.3s
"""


def test_parse_moves_matches_the_real_indented_dashboard_format():
    """The downloaded dashboard .log file re-indents the captured stderr
    by two spaces under its OUTPUT section (see this module's docstring)
    -- a real bug caught 2026-09 when every real downloaded log parsed to
    zero moves despite the synthetic (unindented) fixture above working
    fine. Locks in the ^\\s* anchor fix."""
    moves = parse_moves(_REAL_DASHBOARD_LOG)
    assert len(moves) == 2
    assert moves[0].ply == 16 and moves[0].score == -86.0
    assert parse_build_tag(_REAL_DASHBOARD_LOG) == "712f2aa"
    init = parse_init(_REAL_DASHBOARD_LOG)
    assert init is not None and init.ready_before_move1 is True


def test_gracefully_handles_a_pre_instrumentation_log():
    old_log = "[cb_engine] init took 1.3 ms (budget 90000 ms)\nOUTPUT\n  153 bytes on stderr\n"
    moves = parse_moves(old_log)
    init = parse_init(old_log)
    assert moves == []
    assert init is None
    report = report_from_moves(moves, init)
    assert report["moves_seen"] == 0
    assert report["median_depth"] is None
    assert report["v1_ever_played"] is False
