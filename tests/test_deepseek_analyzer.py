from deepseek_analyzer import DeepSeekStockAnalyzer as A


def test_confidence_clamped():
    r = A._validate_signal_fields({"signal": "BUY", "confidence": 150}, "X")
    assert r["confidence"] == 100
    r = A._validate_signal_fields({"signal": "SELL", "confidence": -5}, "X")
    assert r["confidence"] == 0
    r = A._validate_signal_fields({"signal": "BUY", "confidence": "abc"}, "X")
    assert r["confidence"] == 0


def test_invalid_signal_becomes_hold():
    r = A._validate_signal_fields({"signal": "YOLO", "confidence": 80}, "X")
    assert r["signal"] == "HOLD"


def test_out_of_range_sl_tp_dropped():
    r = A._validate_signal_fields(
        {"signal": "BUY", "confidence": 80, "stop_loss_percent": -3,
         "target_percent": 99}, "X")
    assert "stop_loss_percent" not in r
    assert "target_percent" not in r


def test_in_range_sl_tp_kept():
    r = A._validate_signal_fields(
        {"signal": "BUY", "confidence": 80, "stop_loss_percent": "1.5",
         "target_percent": 3.0}, "X")
    assert r["stop_loss_percent"] == 1.5
    assert r["target_percent"] == 3.0


def test_repair_truncated_json():
    raw = '{"signal": "SELL", "confidence": 72, "reasoning": "price below vwap'
    r = A._repair_truncated_json(raw, "X")
    assert r["signal"] == "SELL"
    assert r["confidence"] == 72


def test_repair_validates_ranges():
    raw = '{"signal": "BUY", "confidence": 80, "stop_loss_percent": 88.0}'
    r = A._repair_truncated_json(raw, "X")
    assert "stop_loss_percent" not in r  # out of range, dropped by validation


def test_repair_empty_returns_hold():
    r = A._repair_truncated_json("", "X")
    assert r["signal"] == "HOLD" and r["confidence"] == 0
