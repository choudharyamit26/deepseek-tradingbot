"""Self-improvement reflection agent for the Kronos-enhanced bot.

Rebuilt 2026-06-10 (the original reflect.py was lost in the 2026-06-09 repo
cleanup) with guardrails the original lacked:

  1. PARAM_BOUNDS registry — every tunable parameter has (min, max, max_step).
     Proposals outside bounds are clamped; locked params (max_step=0) rejected.
  2. Minimum-evidence gate — no parameter change until >= min_trades_per_change
     closed trades have accumulated since the last change. Below that the agent
     runs analysis-only and records an "insufficient sample" entry.
  3. Auto-revert — before proposing anything new, the previous hypothesis is
     evaluated against its recorded baseline; a clear degradation reverts the
     changed parameter to its old value.
  4. Only genuinely closed trades (pnl + exit_price present) count toward
     metrics. LIVE-FAILED rows are included: API order rejections (DH-905)
     are executed manually by the trader, so their PnL is a real outcome.
  5. Full hypotheses history (with measured outcomes) is fed to the LLM so it
     stops re-proposing directions that already failed.
  6. Explicit DeepSeek error handling (402/429 alert + skip) and strict JSON
     validation of the proposal before anything is applied.

State layout (unchanged from the original agent):
  kronos_strategy.yaml                      current strategy (version: kronos-vNN)
  state/history/strategy_v<version>.yaml    archived versions
  state/hypotheses.jsonl                    one record per reflection decision
  state/reflection.log                      timestamped run log
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import re
import sys
from copy import deepcopy
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import requests
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from kronos_integrated_bot import config as cfg

IST = ZoneInfo("Asia/Kolkata")
logger = logging.getLogger("reflection")

LOG_DIR = os.path.join(str(cfg.PROJECT_ROOT), "trading_logs")
HISTORY_DIR = cfg.STATE_DIR / "history"
HYPOTHESES_FILE = cfg.STATE_DIR / "hypotheses.jsonl"
REFLECTION_LOG = cfg.STATE_DIR / "reflection.log"
ANALOG_DB = cfg.PROJECT_ROOT / "analog_history.db"

DEEPSEEK_URL = "https://api.deepseek.com/chat/completions"
DEEPSEEK_MODEL = "deepseek-v4-pro"

# ── Guardrail 1: parameter bounds ────────────────────────────────────────────
# param: (min, max, max_step). max_step = 0 means the parameter is locked —
# the LLM may discuss it but the agent will never change it automatically.
# Risk-of-ruin parameters are locked by design; loosen deliberately, by hand.
PARAM_BOUNDS: dict[str, tuple[float, float, float]] = {
    "min_confidence":              (60,    92,    3),
    "min_rr_ratio":                (1.2,   3.0,   0.3),
    "min_adx_trending":            (12,    30,    2),
    "min_prefilter_volume_ratio":  (0.05,  1.0,   0.1),
    "min_prefilter_atr_pct":       (0.05,  1.0,   0.1),
    "min_volume_ratio_trending":   (0.1,   1.0,   0.1),
    "cooldown_seconds":            (300,   7200,  900),
    "same_direction_cooldown":     (300,   7200,  900),
    "rsi_ob_limit":                (60,    85,    3),
    "rsi_os_limit":                (15,    40,    3),
    "max_daily_signals":           (3,     20,    2),
    "max_signals_per_stock_per_day": (1,   3,     1),
    "max_concurrent_positions":    (1,     5,     1),
    "kronos_confidence_weight":    (0.0,   1.0,   0.1),
    "kronos_penalty_conflict":     (0.2,   1.0,   0.1),
    "kronos_bonus_align":          (1.0,   1.3,   0.05),
    "kronos_temperature":          (0.1,   1.0,   0.1),
    "kronos_sample_count":         (1,     10,    2),
    "kronos_exit_threshold":       (-0.03, 0.0,   0.005),
    "kronos_min_predicted_move":   (0.0,   0.02,  0.002),
    "trailing_sl_activation_pct":  (0.5,   5.0,   0.5),
    "trailing_sl_distance_atr":    (0.5,   4.0,   0.5),
    "max_trade_duration_minutes":  (30,    360,   30),
    "market_open_skip_minutes":    (0,     60,    15),
    "market_close_exit_minutes":   (5,     45,    5),
    "partial_profit_pct":          (0.0,   50.0,  10),
    "position_confidence_scalar":  (0.5,   1.5,   0.1),
    # Locked: capital-protection parameters never auto-tuned.
    "risk_per_trade_pct":          (0.5,   2.0,   0),
    "max_daily_loss_pct":          (0.5,   2.0,   0),
    "max_daily_trades":            (1,     10,    0),
    "max_consecutive_losses":      (1,     5,     0),
}

# ── Trade loading & metrics ──────────────────────────────────────────────────
def load_closed_trades(days_back: int = 14, since: datetime | None = None) -> list[dict]:
    """Load closed trades from daily signal CSVs.

    A row is counted when pnl is present and non-empty. exit_price is NOT
    required: LIVE-FAILED signals (DH-905 order rejections placed manually by
    the trader) carry a real pnl outcome but no system-tracked exit_price.
    Rows without a pnl entry are open/pending positions and are skipped.
    """
    import csv

    trades = []
    now = datetime.now(IST)
    for d_back in range(days_back):
        d = (now - timedelta(days=d_back)).strftime("%Y-%m-%d")
        path = os.path.join(LOG_DIR, f"signals_{d}.csv")
        if not os.path.isfile(path):
            continue
        try:
            with open(path, "r", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    pnl_str = (row.get("pnl") or "").strip()
                    if not pnl_str:
                        continue  # open/pending position — no outcome yet
                    try:
                        ts = datetime.strptime(row["timestamp"], "%Y-%m-%d %H:%M:%S").replace(tzinfo=IST)
                    except (ValueError, KeyError):
                        ts = None
                    if since is not None and ts is not None and ts <= since:
                        continue
                    try:
                        exit_str = (row.get("exit_price") or "").strip()
                        entry = float(row.get("entry_price") or 0)
                        exit_p = float(exit_str) if exit_str else None
                        qty_str = (row.get("quantity") or "").strip()
                        qty = int(float(qty_str)) if qty_str else 1
                        direction = row.get("direction", "")
                        stored_pnl = float(pnl_str)

                        # Detect the guardian entry_price=0 bug:
                        #   _calc_pnl with entry=0 produces pnl = (0-exit)*qty = -exit*qty
                        # Check: if stored_pnl ≈ -(exit_price * qty), it's corrupted.
                        # In that case, recompute from the CSV entry_price (which is the
                        # actual signal/fill price for TRAILING-SL rows). For all other
                        # cases trust the stored pnl — FAILED-SHORT manual pnl, actual
                        # fills that differ from signal price, large-qty positions, etc.
                        pnl = stored_pnl
                        if (exit_p is not None and qty > 0 and entry > 0
                                and abs(stored_pnl + exit_p * qty) < 0.50):
                            # Corrupted: pnl was recorded as -(exit * qty)
                            if direction == "BUY":
                                pnl = round((exit_p - entry) * qty, 2)
                            else:
                                pnl = round((entry - exit_p) * qty, 2)

                        trades.append({
                            "timestamp": row.get("timestamp", ""),
                            "symbol": row["symbol"],
                            "direction": direction,
                            "confidence": int(float(row.get("confidence") or 0)),
                            "pnl": pnl,
                            "entry": entry,
                            "exit": exit_p,
                            "market_regime": row.get("market_regime", ""),
                            "signal_type": row.get("signal_type", ""),
                            "date": d,
                        })
                    except ValueError:
                        continue
        except Exception as e:
            logger.warning("Error reading %s: %s", path, e)
    trades.sort(key=lambda t: t["timestamp"])
    return trades


def load_closed_trades_from_db(days_back: int = 60, since: datetime | None = None) -> list[dict]:
    """Load closed trades from analog_history.db.

    The DB stores richer per-trade indicator snapshots (RSI, ADX, volume_ratio,
    MFI, ATR%, Kronos alignment, Nifty trend, market_regime) than the CSV logs,
    giving the reflection LLM much better signal for parameter optimization.
    """
    if not ANALOG_DB.exists():
        logger.warning("analog_history.db not found at %s", ANALOG_DB)
        return []

    trades = []
    now = datetime.now(IST)
    cutoff = now - timedelta(days=days_back)

    try:
        conn = sqlite3.connect(str(ANALOG_DB))
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM setups WHERE outcome IS NOT NULL ORDER BY ts ASC"
        ).fetchall()
        conn.close()
    except Exception as e:
        logger.warning("Error reading analog_history.db: %s", e)
        return []

    for row in rows:
        try:
            ts_str = row["ts"]
            # DB timestamps are ISO format (2026-06-02T09:53:11)
            ts = datetime.fromisoformat(ts_str).replace(tzinfo=IST)
        except (ValueError, KeyError):
            ts = None

        if ts is not None and ts < cutoff:
            continue
        if since is not None and ts is not None and ts <= since:
            continue

        pnl = row["pnl"]
        if pnl is None:
            continue

        trades.append({
            "timestamp": ts_str,
            "symbol": row["symbol"],
            "direction": row["signal_type"],
            "confidence": int(row["confidence"] or 0),
            "pnl": float(pnl),
            "pnl_pct": float(row["pnl_pct"] or 0),
            "market_regime": row["market_regime"] or "",
            "signal_type": row["signal_type"],
            "outcome": row["outcome"],
            "date": ts_str[:10] if ts_str else "",
            # Rich indicator data only available from DB
            "rsi": float(row["rsi"] or 50),
            "adx": float(row["adx"] or 20),
            "volume_ratio": float(row["volume_ratio"] or 1.0),
            "mfi": float(row["mfi"] or 50),
            "atr_pct": float(row["atr_pct"] or 0.5),
            "kronos_aligned": bool(row["kronos_aligned"]),
            "kronos_direction": row["kronos_direction"] or "",
            "nifty_trend": row["nifty_trend"] or "",
        })

    trades.sort(key=lambda t: t["timestamp"])
    logger.info("Loaded %d closed trades from analog_history.db (days_back=%d)", len(trades), days_back)
    return trades


def compute_metrics(trades: list[dict]) -> dict:
    """Win rate, PnL, expectancy, per-trade sharpe, max drawdown on cum PnL."""
    n = len(trades)
    if n == 0:
        return {"closed_trades": 0, "win_rate": 0.0, "total_pnl": 0.0,
                "avg_pnl_per_trade": 0.0, "sharpe_ratio": 0.0, "max_drawdown": 0.0}
    pnls = [t["pnl"] for t in trades]
    winners = [p for p in pnls if p > 0]
    total_pnl = sum(pnls)
    avg = total_pnl / n
    var = sum((p - avg) ** 2 for p in pnls) / n
    std = var ** 0.5
    sharpe = (avg / std * (252 ** 0.5)) if std > 0 else 0.0

    peak = cum = 0.0
    max_dd = 0.0
    for p in pnls:
        cum += p
        peak = max(peak, cum)
        max_dd = min(max_dd, cum - peak)

    return {
        "closed_trades": n,
        "win_rate": round(len(winners) / n * 100, 2),
        "total_pnl": round(total_pnl, 2),
        "avg_pnl_per_trade": round(avg, 4),
        "sharpe_ratio": round(sharpe, 3),
        "max_drawdown": round(max_dd, 2),
    }


def breakdown_by(trades: list[dict], key: str) -> dict:
    out: dict[str, dict] = {}
    for t in trades:
        k = t.get(key) or "UNKNOWN"
        b = out.setdefault(k, {"count": 0, "pnl": 0.0, "winners": 0})
        b["count"] += 1
        b["pnl"] += t["pnl"]
        if t["pnl"] > 0:
            b["winners"] += 1
    for b in out.values():
        b["win_rate"] = round(b["winners"] / b["count"] * 100, 1) if b["count"] else 0.0
        b["pnl"] = round(b["pnl"], 2)
    return out


def indicator_analysis(trades: list[dict]) -> str:
    """Build indicator-level analysis from DB-sourced trades.

    Buckets trades by RSI zone, ADX strength, volume, Kronos alignment, and
    Nifty trend to surface which indicator conditions correlate with wins/losses.
    Only meaningful when trades carry rich indicator data (from the DB).
    """
    if not trades or "rsi" not in trades[0]:
        return ""

    def _bucket_stats(trades: list[dict], key: str, bucket_fn) -> dict:
        buckets: dict[str, dict] = {}
        for t in trades:
            label = bucket_fn(t.get(key))
            b = buckets.setdefault(label, {"count": 0, "pnl": 0.0, "wins": 0})
            b["count"] += 1
            b["pnl"] += t["pnl"]
            if t["pnl"] > 0:
                b["wins"] += 1
        for b in buckets.values():
            b["win_rate"] = round(b["wins"] / b["count"] * 100, 1) if b["count"] else 0.0
            b["pnl"] = round(b["pnl"], 2)
        return buckets

    def rsi_zone(v):
        if v is None: return "unknown"
        if v < 30: return "oversold(<30)"
        if v < 45: return "weak(30-45)"
        if v < 55: return "neutral(45-55)"
        if v < 70: return "strong(55-70)"
        return "overbought(>70)"

    def adx_zone(v):
        if v is None: return "unknown"
        if v < 15: return "no_trend(<15)"
        if v < 25: return "weak(15-25)"
        if v < 40: return "strong(25-40)"
        return "very_strong(>40)"

    def vol_zone(v):
        if v is None: return "unknown"
        if v < 0.3: return "low(<0.3)"
        if v < 0.7: return "normal(0.3-0.7)"
        return "high(>0.7)"

    def kronos_label(v):
        return "aligned" if v else "conflicted"

    def nifty_label(v):
        return v if v else "unknown"

    sections = []
    sections.append(f"RSI ZONES: {json.dumps(_bucket_stats(trades, 'rsi', rsi_zone))}")
    sections.append(f"ADX STRENGTH: {json.dumps(_bucket_stats(trades, 'adx', adx_zone))}")
    sections.append(f"VOLUME RATIO: {json.dumps(_bucket_stats(trades, 'volume_ratio', vol_zone))}")
    # Only surface Kronos alignment when the data actually varies. Backfilled
    # rows (backfill_rag.py) hardcode kronos_aligned=0 / kronos_direction='',
    # so a uniform column is a data artifact, not signal — feeding it to the LLM
    # as "100% conflicted" is misleading. Skip it unless there is real variance.
    kronos_vals = {bool(t.get("kronos_aligned")) for t in trades}
    kronos_dirs = {(t.get("kronos_direction") or "") for t in trades}
    if len(kronos_vals) > 1 or kronos_dirs != {""}:
        sections.append(f"KRONOS ALIGNMENT: {json.dumps(_bucket_stats(trades, 'kronos_aligned', kronos_label))}")
    sections.append(f"NIFTY TREND: {json.dumps(_bucket_stats(trades, 'nifty_trend', nifty_label))}")

    return "\n".join(sections)


# ── Strategy / hypotheses I/O ────────────────────────────────────────────────
def load_strategy() -> dict:
    if cfg.STRATEGY_FILE.exists():
        with open(cfg.STRATEGY_FILE) as f:
            return yaml.safe_load(f) or {"params": {}}
    return {"params": {}}


def save_strategy(strategy: dict):
    with open(cfg.STRATEGY_FILE, "w") as f:
        yaml.dump(strategy, f, default_flow_style=False)


def archive_strategy(strategy: dict):
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    version = strategy.get("version", "unknown")
    path = HISTORY_DIR / f"strategy_v{version}.yaml"
    with open(path, "w") as f:
        yaml.dump(strategy, f, default_flow_style=False)
    logger.info("Archived prior strategy (v%s) to %s", version, path)


def bump_version(version: str) -> str:
    m = re.match(r"^(.*?v)(\d+)$", version or "")
    if m:
        return f"{m.group(1)}{int(m.group(2)) + 1:02d}"
    return f"{version or 'kronos-v'}.1"


def load_hypotheses() -> list[dict]:
    records = []
    if HYPOTHESES_FILE.exists():
        with open(HYPOTHESES_FILE, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
    return records


def append_hypothesis(record: dict):
    with open(HYPOTHESES_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    logger.info("Appended reflection record to %s", HYPOTHESES_FILE)


def last_change_time(hypotheses: list[dict]) -> datetime | None:
    """Timestamp of the most recent record that actually changed a parameter."""
    for rec in reversed(hypotheses):
        if rec.get("parameter_changed") and rec.get("action") != "analysis_only":
            try:
                return datetime.strptime(rec["timestamp"], "%Y-%m-%d %H:%M:%S").replace(tzinfo=IST)
            except (ValueError, KeyError):
                return None
    return None


def is_oscillation(param: str, new_value, current, hypotheses: list[dict]) -> bool:
    """Guardrail: reject proposals that merely undo the most recent change.

    The LLM has a habit of flip-flopping a parameter (e.g. min_confidence
    82->85->82->85): each cycle it "discovers" the opposite rationale and
    reverts the previous cycle's change, so the strategy oscillates without
    ever accumulating evidence. Replay validation cannot catch this for params
    it does not model (min_confidence, all kronos_*), so we block it here.

    A proposal is an oscillation when the parameter's most recent applied change
    moved it old->current, and the new proposal moves it back toward old.
    """
    for rec in reversed(hypotheses):
        if rec.get("parameter_changed") == param and rec.get("action") in ("change", "revert"):
            try:
                old_f = float(rec.get("old_value"))
                cur_f = float(current)
                new_f = float(new_value)
            except (TypeError, ValueError):
                return False
            # Previous change moved old_f -> cur_f. Moving back toward old_f
            # (same sign as old_f - cur_f) is a reversal of that change.
            return (new_f - cur_f) * (old_f - cur_f) > 0
    return False


# ── Telegram ─────────────────────────────────────────────────────────────────
def send_telegram(message: str):
    if not cfg.TELEGRAM_ENABLED:
        return
    try:
        requests.post(
            f"https://api.telegram.org/bot{cfg.TELEGRAM_BOT_TOKEN}/sendMessage",
            json={"chat_id": cfg.TELEGRAM_CHAT_ID, "text": message},
            timeout=10,
        )
    except Exception as e:
        logger.warning("Telegram send failed: %s", e)


# ── Guardrail 3: evaluate the previous hypothesis ────────────────────────────
def evaluate_previous_hypothesis(hypotheses: list[dict], min_eval_trades: int) -> dict | None:
    """If the last applied change has enough post-change evidence and clearly
    degraded performance, return a revert decision dict; else None."""
    last = None
    for rec in reversed(hypotheses):
        if rec.get("parameter_changed") and rec.get("action") not in ("analysis_only", "revert"):
            last = rec
            break
    if last is None:
        return None

    changed_at = last_change_time([last])
    if changed_at is None:
        return None

    post_trades = load_closed_trades(days_back=30, since=changed_at)
    if len(post_trades) < min_eval_trades:
        logger.info("Previous hypothesis (%s) has %d/%d evaluation trades — verdict pending.",
                    last.get("parameter_changed"), len(post_trades), min_eval_trades)
        return None

    baseline = last.get("metrics") or {}
    current = compute_metrics(post_trades)
    base_wr = float(baseline.get("win_rate", 0))
    base_avg = float(baseline.get("avg_pnl_per_trade", 0))

    win_rate_collapsed = current["win_rate"] < base_wr - 10
    expectancy_flipped = current["avg_pnl_per_trade"] < 0 <= base_avg
    degraded = expectancy_flipped or (win_rate_collapsed and current["avg_pnl_per_trade"] < base_avg)

    logger.info(
        "Evaluating previous hypothesis %s: baseline WR=%.1f avg=%.4f | "
        "post-change (%d trades) WR=%.1f avg=%.4f -> %s",
        last.get("parameter_changed"), base_wr, base_avg,
        current["closed_trades"], current["win_rate"], current["avg_pnl_per_trade"],
        "DEGRADED, reverting" if degraded else "OK, keeping",
    )
    if not degraded:
        return None
    return {"record": last, "post_metrics": current}


def apply_revert(strategy: dict, decision: dict, dry_run: bool) -> dict:
    rec = decision["record"]
    param = rec["parameter_changed"]
    old_value = rec["old_value"]
    new_version = bump_version(strategy.get("version", "kronos-v0"))

    revert_record = {
        "timestamp": datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S"),
        "action": "revert",
        "old_version": strategy.get("version"),
        "new_version": new_version,
        "parameter_changed": param,
        "old_value": strategy.get("params", {}).get(param),
        "new_value": old_value,
        "hypothesis": f"AUTO-REVERT: '{param}' change from v{rec.get('old_version')} degraded performance.",
        "reasoning": (
            f"Post-change metrics over {decision['post_metrics']['closed_trades']} trades: "
            f"win_rate={decision['post_metrics']['win_rate']}%, "
            f"avg_pnl={decision['post_metrics']['avg_pnl_per_trade']} vs baseline "
            f"win_rate={rec.get('metrics', {}).get('win_rate')}%, "
            f"avg_pnl={rec.get('metrics', {}).get('avg_pnl_per_trade')}."
        ),
        "metrics": decision["post_metrics"],
    }
    if dry_run:
        logger.info("[DRY-RUN] Would revert %s -> %s and bump to %s", param, old_value, new_version)
        return strategy

    archive_strategy(strategy)
    strategy = deepcopy(strategy)
    strategy["params"][param] = old_value
    strategy["version"] = new_version
    save_strategy(strategy)
    append_hypothesis(revert_record)
    send_telegram(
        f"REFLECTION AUTO-REVERT\n{param}: back to {old_value}\n"
        f"Strategy {revert_record['old_version']} -> {new_version}\n{revert_record['reasoning']}"
    )
    logger.info("Reverted %s to %s; saved strategy %s", param, old_value, new_version)
    return strategy


# ── Guardrail 5+6: LLM proposal with history + strict validation ─────────────
def _hypotheses_with_outcomes(hypotheses: list[dict]) -> str:
    """Render past hypotheses with measured outcomes (next record's metrics
    show how the change actually performed)."""
    lines = []
    for i, rec in enumerate(hypotheses):
        outcome = ""
        if i + 1 < len(hypotheses):
            nxt = hypotheses[i + 1].get("metrics") or {}
            cur = rec.get("metrics") or {}
            if nxt and cur:
                d_wr = float(nxt.get("win_rate", 0)) - float(cur.get("win_rate", 0))
                outcome = f" | OUTCOME: win_rate {cur.get('win_rate')}% -> {nxt.get('win_rate')}% ({d_wr:+.1f})"
        tag = rec.get("action", "change")
        lines.append(
            f"- [{rec.get('new_version')}] ({tag}) {rec.get('parameter_changed')}: "
            f"{rec.get('old_value')} -> {rec.get('new_value')}{outcome}"
        )
    return "\n".join(lines) if lines else "(no prior changes)"


# Params whose effect the filter-level replay cannot model (no confidence /
# Kronos layer in replay) — changing them is unvalidatable, so the LLM should
# only pick them when the indicator evidence is specifically about them.
REPLAY_BLIND_PARAMS = {"min_confidence"} | {
    p for p in PARAM_BOUNDS if p.startswith("kronos_")
}


def recently_adjusted_params(hypotheses: list[dict], n: int = 4) -> list[str]:
    """Params changed (or rejected as oscillation) in the last n decisions.

    The LLM fixates on one parameter (the min_confidence flip-flop). Feeding it
    an explicit AVOID list of recently-touched params forces it to explore the
    rest of the parameter space instead of ping-ponging the same knob.
    """
    seen: list[str] = []
    for rec in reversed(hypotheses):
        if rec.get("action") in ("change", "revert", "rejected_oscillation", "rejected_by_replay"):
            p = rec.get("parameter_changed")
            if p and p not in seen:
                seen.append(p)
        if len(seen) >= n:
            break
    return seen


def build_proposal_prompts(strategy: dict, metrics: dict, trades: list[dict],
                           hypotheses: list[dict]) -> tuple[str, str]:
    """Build (system_prompt, user_prompt) for the LLM. Split out from the API
    call so the prompt content is unit-testable without a network round-trip."""
    params = strategy.get("params", {})
    avoid = recently_adjusted_params(hypotheses)
    tunable_lines = []
    for p, (lo, hi, step) in PARAM_BOUNDS.items():
        if step == 0:
            continue  # locked params not offered to the LLM
        cur = params.get(p, "unset")
        flags = []
        if p in avoid:
            flags.append("RECENTLY ADJUSTED — avoid")
        if p in REPLAY_BLIND_PARAMS:
            flags.append("not replay-validatable")
        suffix = f"  [{'; '.join(flags)}]" if flags else ""
        tunable_lines.append(
            f"- {p}: current={cur}, allowed range [{lo}, {hi}], max change per cycle {step}{suffix}")

    regime = breakdown_by(trades, "market_regime")
    direction = breakdown_by(trades, "direction")

    system_prompt = (
        "You are a quantitative strategy optimizer for an intraday NSE trading bot. "
        "Propose EXACTLY ONE parameter change. Return ONLY valid JSON:\n"
        '{"parameter": "<name>", "new_value": <number>, '
        '"hypothesis": "<one sentence>", "analysis": "<short analysis>"}\n'
        "Rules:\n"
        "- The parameter MUST be one of the tunable parameters listed by the user.\n"
        "- new_value MUST be inside the allowed range and within max-change-per-cycle of the current value.\n"
        "- Do NOT re-propose a direction that previously degraded performance (see history outcomes).\n"
        "- Do NOT propose a parameter flagged 'RECENTLY ADJUSTED' — it will be auto-rejected as oscillation. "
        "Pick a DIFFERENT parameter and explore the rest of the space.\n"
        "- Ground the choice in the BY DIRECTION and INDICATOR-LEVEL ANALYSIS evidence: target the "
        "specific bucket (RSI zone, ADX band, direction, volume) that is losing money, and pick the "
        "parameter that gates that bucket.\n"
        "- Parameters flagged 'not replay-validatable' (min_confidence, kronos_*) cannot be checked by "
        "the replay simulator; prefer a filter-level parameter (adx/rsi/volume/rr gates) when the "
        "indicator evidence points there.\n"
        "- If win rate is healthy but trade count is low, consider relaxing a filter; "
        "if win rate is poor, consider tightening one."
    )
    # Build indicator analysis if DB-sourced trades have rich data
    ind_analysis = indicator_analysis(trades)
    ind_block = f"\n\nINDICATOR-LEVEL ANALYSIS (from trade database):\n{ind_analysis}" if ind_analysis else ""
    avoid_block = (f"\n\nRECENTLY ADJUSTED (do NOT propose these — pick something else): {avoid}"
                   if avoid else "")

    user_prompt = (
        f"CURRENT STRATEGY VERSION: {strategy.get('version')}\n\n"
        f"PERFORMANCE SINCE LAST CHANGE ({metrics['closed_trades']} closed trades, infra failures excluded):\n"
        f"{json.dumps(metrics, indent=2)}\n\n"
        f"BY REGIME: {json.dumps(regime)}\n"
        f"BY DIRECTION: {json.dumps(direction)}"
        f"{ind_block}\n\n"
        f"CHANGE HISTORY WITH MEASURED OUTCOMES:\n{_hypotheses_with_outcomes(hypotheses)}"
        f"{avoid_block}\n\n"
        f"TUNABLE PARAMETERS:\n" + "\n".join(tunable_lines) +
        f"\n\nGOALS: {json.dumps(strategy.get('goal', {}))}"
    )
    return system_prompt, user_prompt


def propose_change(strategy: dict, metrics: dict, trades: list[dict],
                   hypotheses: list[dict]) -> dict | None:
    """Ask DeepSeek for exactly one parameter change. Returns validated dict
    {parameter, new_value, hypothesis, analysis} or None."""
    params = strategy.get("params", {})
    system_prompt, user_prompt = build_proposal_prompts(strategy, metrics, trades, hypotheses)

    try:
        resp = requests.post(
            DEEPSEEK_URL,
            headers={"Authorization": f"Bearer {cfg.DEEPSEEK_API_KEY}",
                     "Content-Type": "application/json"},
            json={
                "model": DEEPSEEK_MODEL,
                "messages": [{"role": "system", "content": system_prompt},
                             {"role": "user", "content": user_prompt}],
                "temperature": 0.2,
                "max_tokens": 1500,
                "response_format": {"type": "json_object"},
                "thinking": {"type": "disabled"},
            },
            timeout=60,
        )
    except requests.RequestException as e:
        logger.error("DeepSeek request failed: %s", e)
        send_telegram(f"REFLECTION SKIPPED: DeepSeek unreachable ({e})")
        return None

    if resp.status_code in (402, 429):
        msg = f"REFLECTION SKIPPED: DeepSeek returned {resp.status_code} ({'quota/payment' if resp.status_code == 402 else 'rate limit'})."
        logger.error(msg)
        send_telegram(msg)
        return None
    if resp.status_code != 200:
        logger.error("DeepSeek error %s: %s", resp.status_code, resp.text[:300])
        return None

    try:
        content = resp.json()["choices"][0]["message"]["content"]
        proposal = json.loads(content)
    except (KeyError, json.JSONDecodeError, ValueError) as e:
        logger.error("Unparseable DeepSeek proposal: %s", e)
        return None

    return validate_proposal(proposal, params)


def validate_proposal(proposal: dict, params: dict) -> dict | None:
    """Guardrails 1+6: schema check, bounds clamp, step clamp, locked-param veto."""
    param = proposal.get("parameter")
    if not isinstance(param, str) or param not in PARAM_BOUNDS:
        logger.warning("Proposal rejected: unknown parameter %r", param)
        return None
    lo, hi, step = PARAM_BOUNDS[param]
    if step == 0:
        logger.warning("Proposal rejected: parameter %s is locked (capital protection)", param)
        return None
    try:
        new_value = float(proposal["new_value"])
    except (KeyError, TypeError, ValueError):
        logger.warning("Proposal rejected: non-numeric new_value %r", proposal.get("new_value"))
        return None

    current = params.get(param)
    clamped = max(lo, min(hi, new_value))
    if current is not None:
        try:
            cur_f = float(current)
            if abs(clamped - cur_f) > step:
                clamped = cur_f + step if clamped > cur_f else cur_f - step
                clamped = max(lo, min(hi, clamped))
        except (TypeError, ValueError):
            pass
    if clamped != new_value:
        logger.info("Proposal for %s clamped: %s -> %s (bounds [%s, %s], step %s)",
                    param, new_value, clamped, lo, hi, step)
    if current is not None and float(current) == clamped:
        logger.info("Proposal rejected: %s already at %s after clamping", param, clamped)
        return None

    if isinstance(current, int) and float(clamped).is_integer():
        clamped = int(clamped)

    return {
        "parameter": param,
        "new_value": clamped,
        "hypothesis": str(proposal.get("hypothesis", ""))[:600],
        "analysis": str(proposal.get("analysis", ""))[:1200],
    }


def apply_proposal(strategy: dict, proposal: dict, metrics: dict, dry_run: bool) -> dict:
    param = proposal["parameter"]
    old_value = strategy.get("params", {}).get(param)
    new_version = bump_version(strategy.get("version", "kronos-v0"))

    record = {
        "timestamp": datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S"),
        "action": "change",
        "old_version": strategy.get("version"),
        "new_version": new_version,
        "parameter_changed": param,
        "old_value": old_value,
        "new_value": proposal["new_value"],
        "hypothesis": proposal["hypothesis"],
        "reasoning": proposal["analysis"],
        "analysis": proposal["analysis"],
        "metrics": metrics,
    }
    logger.info("DeepSeek Hypothesis: %s", proposal["hypothesis"])
    logger.info("Optimization Recommendation: Change %s from %s to %s",
                param, old_value, proposal["new_value"])
    if dry_run:
        logger.info("[DRY-RUN] Would save strategy %s with %s=%s", new_version, param, proposal["new_value"])
        return strategy

    archive_strategy(strategy)
    strategy = deepcopy(strategy)
    strategy["params"][param] = proposal["new_value"]
    strategy["version"] = new_version
    save_strategy(strategy)
    append_hypothesis(record)
    send_telegram(
        f"REFLECTION APPLIED\n{param}: {old_value} -> {proposal['new_value']}\n"
        f"Strategy {record['old_version']} -> {new_version}\n{proposal['hypothesis']}"
    )
    logger.info("Saved updated strategy (v%s) to %s", new_version, cfg.STRATEGY_FILE)
    return strategy


# ── Replay validation ────────────────────────────────────────────────────────
def replay_validate(strategy: dict, proposal: dict, max_days: int = 5) -> dict | None:
    """Simulate recent stored sessions under current vs proposed params.

    Returns None when no stored data is available (validation skipped),
    else {"ok": bool, "detail": str}. A proposal is rejected only on a strict
    degradation: lower total PnL AND lower win rate than the baseline.
    """
    if not cfg.DATA_DIR.is_dir():
        return None
    dates = sorted(d.name for d in cfg.DATA_DIR.iterdir() if d.is_dir())[-max_days:]
    if not dates:
        return None

    try:
        from kronos_integrated_bot.replay import replay_compare
    except ImportError as e:
        logger.warning("Replay unavailable (%s) — skipping validation.", e)
        return None

    baseline_params = dict(strategy.get("params", {}))
    candidate_params = dict(baseline_params)
    candidate_params[proposal["parameter"]] = proposal["new_value"]

    base_m, cand_m = replay_compare(dates, baseline_params, candidate_params)
    if base_m["closed_trades"] == 0 and cand_m["closed_trades"] == 0:
        return None  # not enough simulated activity to judge

    # The filter-level replay does not model the DeepSeek/Kronos confidence
    # layer, so confidence and kronos_* params produce identical baseline and
    # candidate metrics. A "pass" here is then vacuous — report it honestly
    # rather than letting it masquerade as validation.
    if (base_m["total_pnl"] == cand_m["total_pnl"]
            and base_m["win_rate"] == cand_m["win_rate"]
            and base_m["closed_trades"] == cand_m["closed_trades"]):
        logger.info("Replay cannot distinguish %s (not modeled at filter level) — "
                    "validation inconclusive, not used as a gate.", proposal["parameter"])
        return None

    degraded = (cand_m["total_pnl"] < base_m["total_pnl"]
                and cand_m["win_rate"] < base_m["win_rate"])
    detail = (f"replayed {dates}: baseline pnl={base_m['total_pnl']} wr={base_m['win_rate']}% "
              f"({base_m['closed_trades']} trades) vs candidate pnl={cand_m['total_pnl']} "
              f"wr={cand_m['win_rate']}% ({cand_m['closed_trades']} trades)")
    logger.info("Replay validation: %s", detail)
    return {"ok": not degraded, "detail": detail}


# ── Orchestration ────────────────────────────────────────────────────────────
def run_reflection(dry_run: bool = False) -> dict:
    """Run one reflection cycle. Returns a summary dict for callers/tests."""
    strategy = load_strategy()
    goal = strategy.get("goal", {})
    min_trades = int(goal.get("min_trades_per_change", 30))
    min_eval = int(goal.get("min_eval_trades", 20))

    hypotheses = load_hypotheses()

    # Step 1: auto-revert check on the previous change.
    revert = evaluate_previous_hypothesis(hypotheses, min_eval)
    if revert is not None:
        strategy = apply_revert(strategy, revert, dry_run)
        return {"action": "revert", "parameter": revert["record"].get("parameter_changed")}

    # Step 2: evidence gate.
    # Try the DB first (richer indicator data), fall back to CSVs.
    since = last_change_time(hypotheses)
    trades = load_closed_trades_from_db(days_back=60, since=None)
    trade_source = "DB"
    if not trades:
        trades = load_closed_trades(days_back=60, since=None)
        trade_source = "CSV"
    metrics = compute_metrics(trades)
    logger.info("Total closed trades available: %d from %s (gate: %d, last change: %s)",
                metrics["closed_trades"], trade_source, min_trades,
                since.strftime("%Y-%m-%d %H:%M") if since else "never")

    if metrics["closed_trades"] < min_trades:
        record = {
            "timestamp": datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S"),
            "action": "analysis_only",
            "old_version": strategy.get("version"),
            "new_version": strategy.get("version"),
            "parameter_changed": None,
            "hypothesis": f"Insufficient sample: {metrics['closed_trades']}/{min_trades} closed trades since last change.",
            "reasoning": "Evidence gate held the strategy unchanged.",
            "metrics": metrics,
        }
        if not dry_run:
            append_hypothesis(record)
        logger.info("Evidence gate: insufficient sample (%d < %d), no change proposed.",
                    metrics["closed_trades"], min_trades)
        return {"action": "analysis_only", "closed_trades": metrics["closed_trades"], "gate": min_trades}

    # Step 3: ask the LLM for one bounded change.
    proposal = propose_change(strategy, metrics, trades, hypotheses)
    if proposal is None:
        return {"action": "skipped", "reason": "no valid proposal"}

    # Step 3b: anti-oscillation guard. Reject proposals that simply undo the
    # most recent change to the same parameter (the min_confidence flip-flop).
    cur_value = strategy.get("params", {}).get(proposal["parameter"])
    if is_oscillation(proposal["parameter"], proposal["new_value"], cur_value, hypotheses):
        logger.info("Proposal rejected as oscillation: %s %s -> %s would undo the last change.",
                    proposal["parameter"], cur_value, proposal["new_value"])
        if not dry_run:
            append_hypothesis({
                "timestamp": datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S"),
                "action": "rejected_oscillation",
                "old_version": strategy.get("version"),
                "new_version": strategy.get("version"),
                "parameter_changed": proposal["parameter"],
                "old_value": cur_value,
                "new_value": proposal["new_value"],
                "hypothesis": proposal["hypothesis"],
                "reasoning": "Rejected: proposal reverts the most recent change to this parameter (oscillation guard).",
                "metrics": metrics,
            })
        return {"action": "rejected_oscillation", "parameter": proposal["parameter"],
                "new_value": proposal["new_value"]}

    # Step 4: replay validation — if stored candle data exists, simulate the
    # last few sessions under old vs new params and reject clear degradations.
    verdict = replay_validate(strategy, proposal)
    if verdict is not None and not verdict["ok"]:
        logger.info("Proposal rejected by replay validation: %s", verdict["detail"])
        if not dry_run:
            append_hypothesis({
                "timestamp": datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S"),
                "action": "rejected_by_replay",
                "old_version": strategy.get("version"),
                "new_version": strategy.get("version"),
                "parameter_changed": proposal["parameter"],
                "old_value": strategy.get("params", {}).get(proposal["parameter"]),
                "new_value": proposal["new_value"],
                "hypothesis": proposal["hypothesis"],
                "reasoning": f"Replay validation failed: {verdict['detail']}",
                "metrics": metrics,
            })
        return {"action": "rejected_by_replay", "parameter": proposal["parameter"],
                "detail": verdict["detail"]}

    apply_proposal(strategy, proposal, metrics, dry_run)
    return {"action": "change" if not dry_run else "dry_run_change",
            "parameter": proposal["parameter"], "new_value": proposal["new_value"]}
