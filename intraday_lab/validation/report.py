"""Final report: IS-optimized params -> single OOS run + walk-forward, ranked,
with the fixed survivor criteria applied."""
import json
import logging

import pandas as pd

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as cfg
from engine import backtester, metrics, optimizer
from validation import splits

logger = logging.getLogger(__name__)


def validate_strategy(strategy, data, ctx):
    (is_s, is_e), (oos_s, oos_e) = splits.is_oos()
    best, best_row, all_rows = optimizer.optimize(strategy, data, ctx, is_s, is_e)
    out = {"strategy": strategy.name, "best_params": best}
    if best is None:
        out.update({"status": "NO-QUALIFIER",
                    "note": f"no combo reached {cfg.MIN_IS_TRADES} IS trades with score"})
        return out, None

    out["is"] = {k: best_row[k] for k in
                 ("trades", "net", "wr", "pf", "sharpe", "maxdd_pct", "exp_pct")}
    out["plateau"] = best_row.get("plateau")

    # single OOS shot with frozen params
    oos_trades = backtester.run_strategy(strategy, data, best, ctx, oos_s, oos_e)
    m_oos = metrics.compute(oos_trades)
    out["oos"] = m_oos

    # walk-forward: re-optimize per fold train, score fold test
    wf_trades = []
    wf_prof = 0
    folds = splits.walk_forward_folds()
    for tr_s, tr_e, te_s, te_e in folds:
        p, _, _ = optimizer.optimize(strategy, data, ctx, tr_s, tr_e,
                                     min_trades=max(20, cfg.MIN_IS_TRADES // 3))
        if p is None:
            continue
        t = backtester.run_strategy(strategy, data, p, ctx, te_s, te_e)
        if len(t):
            wf_trades.append(t)
            wf_prof += 1 if t["net"].sum() > 0 else 0
    wf_all = pd.concat(wf_trades, ignore_index=True) if wf_trades else metrics.empty_trades()
    m_wf = metrics.compute(wf_all)
    out["wf"] = {**m_wf, "folds": len(folds), "profitable_folds": wf_prof}

    # survivor verdict (criteria fixed in config before any results were seen)
    is_sh = out["is"]["sharpe"]
    decay_ok = is_sh > 0 and (m_oos["sharpe"] / is_sh) >= (1 - cfg.SURVIVOR["max_decay"])
    out["survivor"] = bool(
        m_oos["pf"] >= cfg.SURVIVOR["oos_pf"]
        and m_oos["sharpe"] >= cfg.SURVIVOR["oos_sharpe"]
        and m_oos["trades"] >= cfg.SURVIVOR["oos_trades"]
        and decay_ok
        and m_wf["pf"] > cfg.SURVIVOR.get("wf_pf", 0)
        and wf_prof >= cfg.SURVIVOR.get("wf_folds_min", 0))
    out["status"] = "SURVIVOR" if out["survivor"] else "REJECTED"
    return out, oos_trades


def run_all(data, ctx, registry, prefix="validation"):
    results = []
    for strat in registry:
        logger.info("validating %s (%d combos)", strat.name,
                    len(list(optimizer.grid(strat.space))))
        try:
            out, _ = validate_strategy(strat, data, ctx)
        except Exception as exc:
            logger.exception("%s failed: %s", strat.name, exc)
            out = {"strategy": strat.name, "status": "ERROR", "note": str(exc)}
        results.append(out)
        (cfg.RESULTS / f"{prefix}.json").write_text(json.dumps(results, indent=2, default=str))
    write_report(results, prefix)
    return results


def write_report(results, prefix="validation"):
    lines = ["# Intraday Lab — IS/OOS + Walk-Forward Report", "",
             f"Window {cfg.START}..{cfg.END} | IS ends {cfg.IS_END} | "
             f"{cfg.N_STOCKS} high-beta stocks | 5-min | net of Dhan costs + slippage", "",
             "| strategy | status | IS shp | IS pf | IS n | OOS shp | OOS pf | OOS n | OOS net | WF pf | WF folds+ |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    def key(r):
        return r.get("oos", {}).get("sharpe", -99)
    for r in sorted(results, key=key, reverse=True):
        i, o, w = r.get("is", {}), r.get("oos", {}), r.get("wf", {})
        lines.append(
            f"| {r['strategy']} | {r['status']} | {i.get('sharpe','-')} | {i.get('pf','-')} | "
            f"{i.get('trades','-')} | {o.get('sharpe','-')} | {o.get('pf','-')} | "
            f"{o.get('trades','-')} | {o.get('net','-')} | {w.get('pf','-')} | "
            f"{w.get('profitable_folds','-')}/{w.get('folds','-')} |")
    surv = [r["strategy"] for r in results if r.get("survivor")]
    lines += ["", f"**Survivors ({len(surv)}):** {', '.join(surv) if surv else 'NONE'}", "",
              "Criteria (fixed up front): OOS PF>=1.2, OOS Sharpe>=1.0, OOS trades>=30, "
              "IS->OOS Sharpe decay<50%."]
    (cfg.RESULTS / f"report_{prefix}.md").write_text("\n".join(lines), encoding="utf-8")
    logger.info("report written to %s", cfg.RESULTS / f"report_{prefix}.md")
