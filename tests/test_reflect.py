from kronos_integrated_bot.reflect import (
    PARAM_BOUNDS,
    bump_version,
    compute_metrics,
    validate_proposal,
)


def test_bump_version():
    assert bump_version("kronos-v14") == "kronos-v15"
    assert bump_version("kronos-v09") == "kronos-v10"


def test_validate_clamps_to_bounds():
    # min_confidence bounds (60, 92, step 3), current 82 -> 95 clamps to 85 (step)
    p = validate_proposal(
        {"parameter": "min_confidence", "new_value": 95, "hypothesis": "h", "analysis": "a"},
        {"min_confidence": 82},
    )
    assert p is not None and p["new_value"] == 85


def test_validate_rejects_locked_param():
    assert validate_proposal(
        {"parameter": "risk_per_trade_pct", "new_value": 5, "hypothesis": "h", "analysis": "a"},
        {"risk_per_trade_pct": 2.0},
    ) is None


def test_validate_rejects_unknown_param():
    assert validate_proposal(
        {"parameter": "made_up_param", "new_value": 1, "hypothesis": "h", "analysis": "a"},
        {},
    ) is None


def test_validate_rejects_noop_after_clamp():
    # already at lower bound, proposal below bound clamps back to current value
    assert validate_proposal(
        {"parameter": "min_confidence", "new_value": 40, "hypothesis": "h", "analysis": "a"},
        {"min_confidence": 60},
    ) is None


def test_validate_rejects_non_numeric():
    assert validate_proposal(
        {"parameter": "min_confidence", "new_value": "high", "hypothesis": "h", "analysis": "a"},
        {"min_confidence": 82},
    ) is None


def test_int_params_stay_int():
    p = validate_proposal(
        {"parameter": "min_confidence", "new_value": 84.0, "hypothesis": "h", "analysis": "a"},
        {"min_confidence": 82},
    )
    assert isinstance(p["new_value"], int) and p["new_value"] == 84


def test_compute_metrics_empty():
    m = compute_metrics([])
    assert m["closed_trades"] == 0 and m["win_rate"] == 0.0


def test_compute_metrics_basic():
    trades = [{"pnl": 10}, {"pnl": -5}, {"pnl": 15}, {"pnl": -2}]
    m = compute_metrics(trades)
    assert m["closed_trades"] == 4
    assert m["win_rate"] == 50.0
    assert m["total_pnl"] == 18
    assert m["max_drawdown"] == -5  # after the first win, the -5 dip


def test_all_bounds_are_sane():
    for param, (lo, hi, step) in PARAM_BOUNDS.items():
        assert lo < hi, param
        assert step >= 0, param
        assert step <= (hi - lo), param
