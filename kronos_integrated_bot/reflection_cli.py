#!/usr/bin/env python3
"""JSON tool surface over the reflection agent, for external operator agents.

This is the bridge that lets an external agent (e.g. Hermes Agent, running in a
separate process) drive the bot's nightly self-improvement loop *as a set of
tools* without importing the trading code or touching the live trading path.

Every subcommand prints a single JSON object to stdout and nothing else (all
logging goes to stderr), so the calling agent can parse the result directly.

Design rule — the guardrails are authoritative, not advisory:
    The operator agent decides WHICH parameter to move and to WHAT value, using
    the same evidence the in-house DeepSeek proposer sees (`metrics`/`context`).
    But every write (`apply`, `revert`) is re-validated here through the exact
    same gates `run_reflection` uses — PARAM_BOUNDS clamp, locked-param veto,
    evidence gate, anti-oscillation guard, replay validation. The external agent
    CANNOT bypass them; a refused write returns {"ok": false, "refused": "..."}.

Subcommands
    state            current strategy, version, tunable params + bounds/flags,
                     evidence-gate status (open/closed, trades since last change)
    metrics          performance since last change: aggregate + by regime /
                     direction + indicator-level buckets (the proposer's evidence)
    hypotheses [-n]  recent reflection decisions with measured outcomes
    context          the exact system+user prompt the in-house proposer would get
    propose P V      DRY validation of moving param P to value V (no write):
                     bounds/step clamp, oscillation check, replay verdict
    apply P V        validate AND apply (write) — refused if the evidence gate is
                     closed, the param is locked, it oscillates, or replay degrades
    revert           run the auto-revert check on the last applied change
    schema           read-only schema + per-column fill rates of analog_history.db
    query SQL        run a single read-only SELECT/WITH against analog_history.db
    run [--dry-run]  the full in-house autonomous cycle (DeepSeek picks the move)

Usage
    python -m kronos_integrated_bot.reflection_cli state
    python -m kronos_integrated_bot.reflection_cli metrics
    python -m kronos_integrated_bot.reflection_cli propose min_adx_trending 24
    python -m kronos_integrated_bot.reflection_cli apply min_adx_trending 24 \
        --reason "ADX<15 bucket bleeding; tighten trend gate" --confirm
"""

from __future__ import annotations

import argparse
import json
import logging
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from kronos_integrated_bot import config as cfg
from kronos_integrated_bot import reflect


def _load_context():
    """Load the shared state every subcommand reasons over: strategy, gate
    config, hypotheses, the cutoff for "since last change", and the closed
    trades + metrics in that window (the same window the proposer uses)."""
    strategy = reflect.load_strategy()
    goal = strategy.get("goal", {})
    min_trades = int(goal.get("min_trades_per_change", 30))
    min_eval = int(goal.get("min_eval_trades", 20))
    hypotheses = reflect.load_hypotheses()
    since = reflect.last_change_time(hypotheses)

    trades = reflect.load_closed_trades_from_db(days_back=60, since=since)
    source = "DB"
    if not trades:
        trades = reflect.load_closed_trades(days_back=60, since=since)
        source = "CSV"
    metrics = reflect.compute_metrics(trades)
    return {
        "strategy": strategy,
        "min_trades": min_trades,
        "min_eval": min_eval,
        "hypotheses": hypotheses,
        "since": since,
        "trades": trades,
        "trade_source": source,
        "metrics": metrics,
    }


def _tunable_params(strategy: dict, hypotheses: list[dict]) -> list[dict]:
    params = strategy.get("params", {})
    avoid = set(reflect.recently_adjusted_params(hypotheses))
    out = []
    for p, (lo, hi, step) in reflect.PARAM_BOUNDS.items():
        out.append({
            "param": p,
            "current": params.get(p),
            "min": lo,
            "max": hi,
            "max_step": step,
            "locked": step == 0,
            "recently_adjusted": p in avoid,
            "replay_blind": p in reflect.REPLAY_BLIND_PARAMS,
        })
    return out


# ── Subcommand handlers ──────────────────────────────────────────────────────
def cmd_state(args) -> dict:
    ctx = _load_context()
    gate_open = ctx["metrics"]["closed_trades"] >= ctx["min_trades"]
    since = ctx["since"]
    return {
        "ok": True,
        "version": ctx["strategy"].get("version"),
        "goal": ctx["strategy"].get("goal", {}),
        "evidence_gate": {
            "open": gate_open,
            "closed_trades_since_last_change": ctx["metrics"]["closed_trades"],
            "min_trades_per_change": ctx["min_trades"],
            "last_change": since.strftime("%Y-%m-%d %H:%M:%S") if since else None,
            "trade_source": ctx["trade_source"],
        },
        "tunable_params": _tunable_params(ctx["strategy"], ctx["hypotheses"]),
    }


def cmd_metrics(args) -> dict:
    ctx = _load_context()
    return {
        "ok": True,
        "version": ctx["strategy"].get("version"),
        "window": "since_last_change",
        "trade_source": ctx["trade_source"],
        "metrics": ctx["metrics"],
        "by_regime": reflect.breakdown_by(ctx["trades"], "market_regime"),
        "by_direction": reflect.breakdown_by(ctx["trades"], "direction"),
        "indicator_analysis": reflect.indicator_analysis(ctx["trades"]),
    }


def cmd_hypotheses(args) -> dict:
    hypotheses = reflect.load_hypotheses()
    n = args.n
    selected = hypotheses[-n:] if n and n > 0 else hypotheses
    return {"ok": True, "count": len(hypotheses), "returned": len(selected),
            "hypotheses": selected}


def cmd_context(args) -> dict:
    ctx = _load_context()
    system, user = reflect.build_proposal_prompts(
        ctx["strategy"], ctx["metrics"], ctx["trades"], ctx["hypotheses"])
    return {"ok": True, "system_prompt": system, "user_prompt": user,
            "evidence_gate_open": ctx["metrics"]["closed_trades"] >= ctx["min_trades"]}


def _evaluate_candidate(ctx: dict, param: str, value, run_replay: bool = True) -> dict:
    """Run the read-only gates (bounds/step clamp, oscillation, replay) for a
    candidate (param, value) and return the verdicts. No write.

    `run_replay` is expensive (full pandas replay over recent sessions, twice),
    so callers exploring many candidates can skip it; the write path (`apply`)
    always runs it.
    """
    strategy = ctx["strategy"]
    params = strategy.get("params", {})

    validated = reflect.validate_proposal(
        {"parameter": param, "new_value": value,
         "hypothesis": "", "analysis": ""}, params)
    if validated is None:
        lo_hi = reflect.PARAM_BOUNDS.get(param)
        return {"valid": False,
                "reason": "rejected by validate_proposal (unknown/locked param, "
                          "non-numeric, or no-op after clamp)",
                "bounds": lo_hi}

    clamped = validated["new_value"]
    cur = params.get(param)
    oscillation = reflect.is_oscillation(param, clamped, cur, ctx["hypotheses"])
    replay = reflect.replay_validate(strategy, validated) if run_replay else None
    return {
        "valid": True,
        "param": param,
        "requested_value": value,
        "clamped_value": clamped,
        "current_value": cur,
        "was_clamped": clamped != value,
        "oscillation": oscillation,
        # replay: None = skipped or not validatable; else {"ok": bool, "detail": str}
        "replay": replay,
        "replay_ran": run_replay,
        "validated": validated,
    }


def cmd_propose(args) -> dict:
    ctx = _load_context()
    verdict = _evaluate_candidate(ctx, args.param, args.value, run_replay=args.replay)
    gate_open = ctx["metrics"]["closed_trades"] >= ctx["min_trades"]
    verdict.pop("validated", None)
    return {"ok": True, "evidence_gate_open": gate_open,
            "closed_trades_since_last_change": ctx["metrics"]["closed_trades"],
            "min_trades_per_change": ctx["min_trades"], **verdict}


def cmd_apply(args) -> dict:
    ctx = _load_context()
    gate_open = ctx["metrics"]["closed_trades"] >= ctx["min_trades"]
    if not gate_open:
        return {"ok": False, "refused": "evidence_gate_closed",
                "closed_trades_since_last_change": ctx["metrics"]["closed_trades"],
                "min_trades_per_change": ctx["min_trades"],
                "detail": "Not enough closed trades since the last change to justify "
                          "a new change. The gate is working as intended — wait for "
                          "more outcomes."}

    verdict = _evaluate_candidate(ctx, args.param, args.value, run_replay=True)
    if not verdict["valid"]:
        return {"ok": False, "refused": "invalid_proposal", **verdict}
    if verdict["oscillation"]:
        return {"ok": False, "refused": "oscillation",
                "detail": "Proposal undoes the most recent change to this parameter.",
                **{k: verdict[k] for k in ("param", "current_value", "clamped_value")}}
    replay = verdict["replay"]
    if replay is not None and not replay["ok"]:
        return {"ok": False, "refused": "replay_degraded",
                "detail": replay["detail"],
                **{k: verdict[k] for k in ("param", "current_value", "clamped_value")}}

    proposal = dict(verdict["validated"])
    proposal["hypothesis"] = args.reason or f"Operator change: {args.param}={proposal['new_value']}"
    proposal["analysis"] = args.reason or "Applied via reflection_cli by external operator agent."

    if not args.confirm:
        return {"ok": True, "applied": False, "would_apply": True,
                "detail": "Passed all gates. Re-run with --confirm to write.",
                "param": proposal["parameter"], "new_value": proposal["new_value"],
                "current_value": verdict["current_value"]}

    reflect.apply_proposal(ctx["strategy"], proposal, ctx["metrics"], dry_run=False)
    new_strategy = reflect.load_strategy()
    return {"ok": True, "applied": True, "param": proposal["parameter"],
            "old_value": verdict["current_value"], "new_value": proposal["new_value"],
            "new_version": new_strategy.get("version")}


def cmd_revert(args) -> dict:
    hypotheses = reflect.load_hypotheses()
    strategy = reflect.load_strategy()
    min_eval = int(strategy.get("goal", {}).get("min_eval_trades", 20))
    decision = reflect.evaluate_previous_hypothesis(hypotheses, min_eval)
    if decision is None:
        return {"ok": True, "reverted": False,
                "detail": "No applied change is both fully evaluated and degraded."}
    if not args.confirm:
        return {"ok": True, "reverted": False, "would_revert": True,
                "param": decision["record"].get("parameter_changed"),
                "post_metrics": decision["post_metrics"],
                "detail": "Re-run with --confirm to write the revert."}
    reflect.apply_revert(strategy, decision, dry_run=False)
    return {"ok": True, "reverted": True,
            "param": decision["record"].get("parameter_changed"),
            "new_version": reflect.load_strategy().get("version")}


# ── Read-only analog_history.db query layer ──────────────────────────────────
# This lets an operator agent DIAGNOSE structural problems (time-of-day decay,
# multi-timeframe alignment, sector regime, late entry) by querying the real
# fill/PnL ground truth — not just tune parameters. It is read-only by three
# independent guarantees: the connection is opened mode=ro, an authorizer denies
# every action except SELECT/READ/FUNCTION, and the SQL must be a single
# SELECT/WITH statement. Any write attempt fails rather than mutating the DB.

DEFAULT_ROW_LIMIT = 100
MAX_ROW_LIMIT = 2000

_ALLOWED_AUTH = {
    sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ,
    sqlite3.SQLITE_FUNCTION, sqlite3.SQLITE_RECURSIVE,
}


def _ro_authorizer(action, arg1, arg2, db_name, trigger):
    return sqlite3.SQLITE_OK if action in _ALLOWED_AUTH else sqlite3.SQLITE_DENY


def _open_ro_db(restrict: bool = False) -> sqlite3.Connection:
    """Open analog_history.db read-only (mode=ro — writes physically fail).

    `restrict=True` additionally installs an authorizer that denies everything
    except SELECT/READ/FUNCTION — used for untrusted, agent-supplied SQL in
    `query`. `schema` runs only trusted introspection (PRAGMA), so it leaves the
    authorizer off but is still read-only via mode=ro.
    """
    if not reflect.ANALOG_DB.exists():
        raise FileNotFoundError(f"analog_history.db not found at {reflect.ANALOG_DB}")
    uri = f"{reflect.ANALOG_DB.resolve().as_uri()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    if restrict:
        conn.set_authorizer(_ro_authorizer)
    return conn


def _validate_select(sql: str) -> str:
    s = sql.strip().rstrip(";").strip()
    if not s:
        raise ValueError("empty query")
    if ";" in s:
        raise ValueError("only a single statement is allowed (no ';')")
    low = s.lower()
    if not (low.startswith("select") or low.startswith("with")):
        raise ValueError("only read-only SELECT/WITH queries are allowed")
    return s


def cmd_query(args) -> dict:
    limit = max(1, min(int(args.limit), MAX_ROW_LIMIT))
    try:
        sql = _validate_select(args.sql)
    except ValueError as e:
        return {"ok": False, "error": str(e)}

    conn = _open_ro_db(restrict=True)
    try:
        cur = conn.execute(sql)
        cols = [d[0] for d in cur.description] if cur.description else []
        rows = cur.fetchmany(limit + 1)
    except sqlite3.DatabaseError as e:
        # Authorizer denials and write attempts surface here.
        return {"ok": False, "error": f"query rejected: {e}"}
    finally:
        conn.close()

    truncated = len(rows) > limit
    rows = rows[:limit]
    return {
        "ok": True,
        "columns": cols,
        "row_count": len(rows),
        "truncated": truncated,
        "limit": limit,
        "rows": [dict(r) for r in rows],
    }


def cmd_schema(args) -> dict:
    conn = _open_ro_db()
    try:
        tables = [r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "AND name NOT LIKE 'sqlite_%' ORDER BY name")]
        out_tables = {}
        for t in tables:
            cols = [(r["name"], r["type"]) for r in conn.execute(f'PRAGMA table_info("{t}")')]
            total = conn.execute(f'SELECT COUNT(*) FROM "{t}"').fetchone()[0]
            # Fill rate per column: non-null AND non-empty-string. Tells the agent
            # which columns are actually populated (e.g. sector_trend is empty).
            fill = {}
            if total and cols:
                exprs = ", ".join(
                    f'SUM(CASE WHEN "{c}" IS NOT NULL AND "{c}" != \'\' THEN 1 ELSE 0 END)'
                    for c, _ in cols)
                vals = conn.execute(f'SELECT {exprs} FROM "{t}"').fetchone()
                for (c, _), v in zip(cols, vals):
                    fill[c] = {"filled": int(v or 0),
                               "pct": round((v or 0) / total * 100, 1)}
            out_tables[t] = {
                "row_count": total,
                "columns": [{"name": c, "type": ty,
                             "filled": fill.get(c, {}).get("filled"),
                             "fill_pct": fill.get(c, {}).get("pct")}
                            for c, ty in cols],
            }

        extras = {}
        if "setups" in tables:
            extras["ts_range"] = list(conn.execute(
                "SELECT MIN(ts), MAX(ts) FROM setups").fetchone())
            extras["by_outcome"] = {r[0]: r[1] for r in conn.execute(
                "SELECT outcome, COUNT(*) FROM setups GROUP BY outcome")}
            extras["by_signal_type"] = {r[0]: r[1] for r in conn.execute(
                "SELECT signal_type, COUNT(*) FROM setups GROUP BY signal_type")}
    finally:
        conn.close()
    return {"ok": True, "db": str(reflect.ANALOG_DB), "tables": out_tables,
            "setups_summary": extras,
            "note": "Read-only. Diagnose with `query \"SELECT ...\"`. Columns with low "
                    "fill_pct (e.g. sector_trend) are unpopulated — don't rely on them."}


def cmd_run(args) -> dict:
    result = reflect.run_reflection(dry_run=args.dry_run)
    return {"ok": True, "dry_run": args.dry_run, "result": result}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="reflection_cli",
        description="JSON tool surface over the reflection agent for operator agents.")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("state", help="current strategy + evidence-gate status").set_defaults(fn=cmd_state)
    sub.add_parser("metrics", help="performance evidence since last change").set_defaults(fn=cmd_metrics)

    h = sub.add_parser("hypotheses", help="recent reflection decisions")
    h.add_argument("-n", type=int, default=10, help="last N records (0 = all)")
    h.set_defaults(fn=cmd_hypotheses)

    sub.add_parser("context", help="exact proposer system+user prompt").set_defaults(fn=cmd_context)

    pr = sub.add_parser("propose", help="DRY validation of a candidate change")
    pr.add_argument("param")
    pr.add_argument("value", type=float)
    pr.add_argument("--replay", action="store_true",
                    help="also run the (slow) replay validation; off by default for fast exploration")
    pr.set_defaults(fn=cmd_propose)

    ap = sub.add_parser("apply", help="validate AND write a change (gated)")
    ap.add_argument("param")
    ap.add_argument("value", type=float)
    ap.add_argument("--reason", default="", help="hypothesis/rationale recorded with the change")
    ap.add_argument("--confirm", action="store_true", help="actually write (else dry preview)")
    ap.set_defaults(fn=cmd_apply)

    rv = sub.add_parser("revert", help="auto-revert the last change if it degraded")
    rv.add_argument("--confirm", action="store_true", help="actually write the revert")
    rv.set_defaults(fn=cmd_revert)

    sub.add_parser("schema", help="read-only schema + fill rates of analog_history.db").set_defaults(fn=cmd_schema)

    q = sub.add_parser("query", help="run a read-only SELECT against analog_history.db")
    q.add_argument("sql", help="a single SELECT/WITH statement")
    q.add_argument("--limit", type=int, default=DEFAULT_ROW_LIMIT,
                   help=f"max rows returned (default {DEFAULT_ROW_LIMIT}, cap {MAX_ROW_LIMIT})")
    q.set_defaults(fn=cmd_query)

    rn = sub.add_parser("run", help="full in-house autonomous reflection cycle")
    rn.add_argument("--dry-run", action="store_true")
    rn.set_defaults(fn=cmd_run)
    return p


def main(argv=None) -> int:
    # Keep stdout pure JSON; route all logging to stderr.
    logging.basicConfig(level=logging.WARNING,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                        stream=sys.stderr)
    args = build_parser().parse_args(argv)
    try:
        result = args.fn(args)
    except Exception as exc:  # surface failures as JSON so the agent can react
        logging.getLogger("reflection_cli").exception("subcommand failed")
        print(json.dumps({"ok": False, "error": str(exc),
                          "error_type": type(exc).__name__}))
        return 1
    print(json.dumps(result, default=str, indent=2))
    return 0 if result.get("ok", False) else 2


if __name__ == "__main__":
    raise SystemExit(main())
