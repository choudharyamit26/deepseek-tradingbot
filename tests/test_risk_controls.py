from risk_controls import normalize_stop_loss_percent, stop_loss_floor_percent


def test_stop_loss_floor_uses_atr_for_high_price_stock():
    # DMART example: Rs 7.28 3m ATR at Rs 4169.90 needs about 0.26% stop,
    # not a Rs 6-ish stop below one ATR.
    assert stop_loss_floor_percent(7.28, 4169.90) == 0.26
    assert normalize_stop_loss_percent(0.15, 7.28, 4169.90) == 0.26


def test_stop_loss_floor_uses_absolute_min_when_atr_is_smaller():
    assert stop_loss_floor_percent(0.5, 1000.0) == 0.25
    assert normalize_stop_loss_percent(None, 0.5, 1000.0) == 0.25
