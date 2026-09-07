"""Internal iterative reduction (IIR, 2026-09, item 3): no TT move means
no good first guess for move ordering at this node, so reduce depth by
one rather than searching it in full. The condition itself is the only
thing worth testing directly -- the reduction (depth -= 1) is a one-line
consequence applied at the call site in negamax.
"""
import cb_nb_search as s


def test_no_reduction_when_a_tt_move_exists():
    assert s._should_apply_iir(tt_move=42, depth=10) is False


def test_no_reduction_below_min_depth():
    assert s._should_apply_iir(tt_move=-1, depth=s.IIR_MIN_DEPTH - 1) is False


def test_reduces_at_exactly_min_depth_with_no_tt_move():
    assert s._should_apply_iir(tt_move=-1, depth=s.IIR_MIN_DEPTH) is True


def test_reduces_above_min_depth_with_no_tt_move():
    assert s._should_apply_iir(tt_move=-1, depth=s.IIR_MIN_DEPTH + 5) is True
