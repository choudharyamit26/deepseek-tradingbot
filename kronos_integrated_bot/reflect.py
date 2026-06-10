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
  4. Infra-failure filtering — trades that died on order/API errors (mode
     contains FAILED, or "ORDER FAILED" in reasoning) are excluded from
     win-rate / PnL metrics so strategy decisions aren't polluted.
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

# Markers of trades that failed for infrastructure reasons, not strategy.
_INFRA_FAILURE_MARKERS = ("ORDER FAILED", "Invalid IP", "DH-9")


# ── Trade loading & metrics ──────────────────────────────────────────────────
def load_closed_trades(days_back: int = 14, since: datetime | None = None) -> list[dict]:
    """Load closed trades from daily signal CSVs, excluding infra failures."""
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
                    exit_str = (row.get("exit_price") or "").strip()
                    if not pnl_str or not exit_str:
                        continue  # open position / entry row
                    mode = (row.get("mode") or "").upper()
                    reasoning = row.get("reasoning") or ""
                    if "FAILED" in mode or any(m in reasoning for m in _INFRA_FAILURE_MARKERS):
                        continue  # guardrail 4: infra failure, not a strategy outcome
                    try:
                        ts = datetime.strptime(row["timestamp"], "%Y-%m-%d %H:%M:%S").replace(tzinfo=IST)
                    except (ValueError, KeyError):
                        ts = None
                    if since is not None and ts is not None and ts <= since:
                        continue
                    try:
                        trades.append({
                            "timestamp": row.get("timestamp", ""),
                            "symbol": row["symbol"],
                            "direction": row.get("direction", ""),
                            "confidence": int(float(row.get("confidence") or 0)),
                            "pnl": float(pnl_str),
                            "entry": float(row.get("entry_price") or 0),
                            "exit": float(exit_str),
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


def propose_change(strategy: dict, metrics: dict, trades: list[dict],
                   hypotheses: list[dict]) -> dict | None:
    """Ask DeepSeek for exactly one parameter change. Returns validated dict
    {parameter, new_value, hypothesis, analysis} or None."""
    params = strategy.get("params", {})
    tunable_lines = []
    for p, (lo, hi, step) in PARAM_BOUNDS.items():
        if step == 0:
            continue  # locked params not offered to the LLM
        cur = params.get(p, "unset")
        tunable_lines.append(f"- {p}: current={cur}, allowed range [{lo}, {hi}], max change per cycle {step}")

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
        "- Prefer parameters that were never explored, but only with a plausible causal hypothesis.\n"
        "- If win rate is healthy but trade count is low, consider relaxing a filter; "
        "if win rate is poor, consider tightening one."
    )
    user_prompt = (
        f"CURRENT STRATEGY VERSION: {strategy.get('version')}\n\n"
        f"PERFORMANCE SINCE LAST CHANGE ({metrics['closed_trades']} closed trades, infra failures excluded):\n"
        f"{json.dumps(metrics, indent=2)}\n\n"
        f"BY REGIME: {json.dumps(regime)}\n"
        f"BY DIRECTION: {json.dumps(direction)}\n\n"
        f"CHANGE HISTORY WITH MEASURED OUTCOMES:\n{_hypotheses_with_outcomes(hypotheses)}\n\n"
        f"TUNABLE PARAMETERS:\n" + "\n".join(tunable_lines) +
        f"\n\nGOALS: {json.dumps(strategy.get('goal', {}))}"
    )

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
    since = last_change_time(hypotheses)
    trades = load_closed_trades(days_back=30, since=since)
    metrics = compute_metrics(trades)
    logger.info("Closed trades since last change (%s): %d (gate: %d)",
                since.strftime("%Y-%m-%d %H:%M") if since else "ever", metrics["closed_trades"], min_trades)

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

    apply_proposal(strategy, proposal, metrics, dry_run)
    return {"action": "change" if not dry_run else "dry_run_change",
            "parameter": proposal["parameter"], "new_value": proposal["new_value"]}
