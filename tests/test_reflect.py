from kronos_integrated_bot.reflect import (
    PARAM_BOUNDS,
    build_proposal_prompts,
    bump_version,
    compute_metrics,
    is_oscillation,
    recently_adjusted_params,
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


def test_oscillation_guard_blocks_flip_flop():
    # Last applied change moved min_confidence 85 -> 82 (current). Proposing
    # 82 -> 85 reverts that change: oscillation.
    hyp = [{"parameter_changed": "min_confidence", "action": "change",
            "old_value": 85, "new_value": 82}]
    assert is_oscillation("min_confidence", 85, 82, hyp) is True


def test_oscillation_guard_allows_continuation():
    # Last change moved 85 -> 82; proposing 82 -> 80 continues the same
    # direction (further down), not a reversal.
    hyp = [{"parameter_changed": "min_confidence", "action": "change",
            "old_value": 85, "new_value": 82}]
    assert is_oscillation("min_confidence", 80, 82, hyp) is False


def test_oscillation_guard_ignores_other_params():
    hyp = [{"parameter_changed": "min_rr_ratio", "action": "change",
            "old_value": 2.2, "new_value": 1.9}]
    assert is_oscillation("min_confidence", 85, 82, hyp) is False


def test_oscillation_guard_no_history():
    assert is_oscillation("min_confidence", 85, 82, []) is False


def test_recently_adjusted_params_collects_recent():
    hyp = [
        {"action": "change", "parameter_changed": "min_rr_ratio"},
        {"action": "rejected_oscillation", "parameter_changed": "min_confidence"},
        {"action": "change", "parameter_changed": "min_adx_trending"},
    ]
    out = recently_adjusted_params(hyp, n=4)
    assert set(out) == {"min_rr_ratio", "min_confidence", "min_adx_trending"}


def test_prompt_flags_recent_and_blind_params():
    strategy = {"version": "kronos-v18", "params": {"min_confidence": 82, "min_adx_trending": 22}}
    metrics = {"closed_trades": 100, "win_rate": 45.0, "total_pnl": -100.0,
               "avg_pnl_per_trade": -1.0}
    hyp = [{"action": "rejected_oscillation", "parameter_changed": "min_confidence"},
           {"action": "change", "parameter_changed": "min_adx_trending"}]
    system_prompt, user_prompt = build_proposal_prompts(strategy, metrics, [], hyp)
    # Recently-adjusted params surface in the avoid list and are flagged inline.
    assert "RECENTLY ADJUSTED" in user_prompt
    assert "min_confidence" in user_prompt
    # min_confidence is a replay-blind param and should be flagged as such.
    assert "not replay-validatable" in user_prompt
    # System prompt instructs avoiding flagged params.
    assert "RECENTLY ADJUSTED" in system_prompt


def test_all_bounds_are_sane():
    for param, (lo, hi, step) in PARAM_BOUNDS.items():
        assert lo < hi, param
        assert step >= 0, param
        assert step <= (hi - lo), param
