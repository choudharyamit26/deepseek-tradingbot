"""Shared execution risk controls."""


def stop_loss_floor_percent(
    atr_value,
    entry_price,
    atr_multiplier: float = 1.5,
    min_percent: float = 0.25,
) -> float:
    """Minimum stop distance, expressed as percent of entry price."""
    try:
        atr = float(atr_value)
        price = float(entry_price)
    except (TypeError, ValueError):
        atr = 0.0
        price = 0.0

    atr_floor = (atr / price * 100 * atr_multiplier) if atr > 0 and price > 0 else 0.0
    return round(max(atr_floor, float(min_percent)), 2)


def normalize_stop_loss_percent(
    stop_loss_percent,
    atr_value,
    entry_price,
    atr_multiplier: float = 1.5,
    min_percent: float = 0.25,
) -> float:
    """Coerce a proposed stop-loss percent and enforce the execution floor."""
    floor = stop_loss_floor_percent(atr_value, entry_price, atr_multiplier, min_percent)
    try:
        proposed = float(stop_loss_percent)
    except (TypeError, ValueError):
        proposed = floor

    if proposed <= 0:
        proposed = floor
    return round(max(proposed, floor), 2)
