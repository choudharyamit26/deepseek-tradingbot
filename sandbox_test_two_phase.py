"""Integration test: two-phase partial exit against the REAL Dhan sandbox API.

Sandbox facts (measured 2026-07-03, https://sandbox.dhan.co/v2):
  - POST /orders WORKS (order accepted, id returned, status TRANSIT)
  - /super/orders is NOT implemented (404) — so the happy-path leg cancel can't
    run here; it stays covered by unit tests + the first live qty>=2 trade
  - every GET (orders, positions, trades, holdings, fundlimit) returns HTTP 500
    -> broker-side state can't be read back; reads below are best-effort

What IS verified end-to-end against real HTTP:

  SCENARIO A — adopted-position partial (order_id "DHAN-*", no broker legs):
    real entry order accepted -> guardian tick -> REAL reduce_position partial
    accepted by the sandbox (orderId captured) -> trade-dict state correct
    (phase armed, qty halved, breakeven stop, partial PnL, ratchet-only trail).

  SCENARIO B — super-order legs uncancellable (the safety fallback):
    trade carries a super-order-style id; cancel_super_order 404s on sandbox —
    exactly the failure the guardian must survive: partial aborted 3 polls in a
    row (NO reduce order ever sent while "legs" stand), then tp2_partial_blocked
    arms phase 2 WITHOUT the partial.

Credentials in .env: DHAN_SANDBOX_CLIENT_ID / DHAN_SANDBOX_ACCESS_TOKEN
(separate signup at https://developer.dhanhq.co).

Usage:  python sandbox_test_two_phase.py
"""
import asyncio
import os
import sys
import logging
from pathlib import Path

from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")
logger = logging.getLogger("sandbox-test")

SANDBOX_BASE_URL = "https://sandbox.dhan.co/v2"
SYM_A, SYM_B = "HDFCBANK", "TCS"
QTY = 2

results = []


def check(name, cond, detail=""):
    results.append((name, bool(cond)))
    logger.info("%s  %s %s", "PASS" if cond else "FAIL", name, detail)


def note(name, detail=""):
    logger.info("SKIP  %s %s", name, detail)


def main():
    load_dotenv(Path(__file__).parent / ".env")
    cid = os.getenv("DHAN_SANDBOX_CLIENT_ID", "")
    token = os.getenv("DHAN_SANDBOX_ACCESS_TOKEN", "")
    if not cid or not token:
        logger.error("Missing DHAN_SANDBOX_CLIENT_ID / DHAN_SANDBOX_ACCESS_TOKEN in .env")
        sys.exit(1)

    # dhan_integration calls load_dotenv(override=True) at import (stomps env
    # with the LIVE .env creds), so don't fight it via os.environ — construct
    # the bot, then swap credentials + base URL directly on the shared DhanHTTP.
    from dhan_integration import DhanStockTradingBot
    bot_api = DhanStockTradingBot()
    http = bot_api._dhan_ctx.dhan_http
    http.base_url = SANDBOX_BASE_URL
    http.client_id = cid
    http.access_token = token
    http.header["access-token"] = token
    http.header["client-id"] = cid
    bot_api.client_id = cid
    bot_api.access_token = token
    logger.info("SDK pointed at %s (client %s)", SANDBOX_BASE_URL, cid)

    # Record every reduce_position call/response while keeping the real method
    reduce_log = []
    real_reduce = bot_api.reduce_position

    def recording_reduce(security_id, transaction_type, quantity, product_type="INTRADAY"):
        resp = real_reduce(security_id, transaction_type, quantity, product_type)
        reduce_log.append((security_id, transaction_type, quantity, resp))
        return resp

    bot_api.reduce_position = recording_reduce

    from kronos_integrated_bot.enhanced_guardian import KronosExitGuardian
    from kronos_integrated_bot import config as cfg

    class Shim:
        def __init__(self):
            self.active_trades = {}
            self._dhan_sem = asyncio.Semaphore(1)

    shim = Shim()
    guardian = KronosExitGuardian(bot_api, shim, kronos=None, dry_run=False)
    sid_a = bot_api.security_ids[SYM_A]
    sid_b = bot_api.security_ids[SYM_B]

    def tick(sym, trade, price, pct):
        asyncio.run(guardian._two_phase_tick(sym, trade, price, pct))

    def place_entry(sid, sym):
        resp = bot_api.place_equity_order(sid, bot_api.dhan.BUY, QTY)
        data = resp.get("data") if isinstance(resp, dict) else None
        oid = str((data or {}).get("orderId") or "") if isinstance(data, dict) else ""
        ok = bool(oid) and resp.get("status") == "success"
        check(f"{sym}: entry order accepted by sandbox", ok, f"orderId={oid or resp}")
        return oid

    def read_positions():
        resp = bot_api.dhan.get_positions()
        if isinstance(resp, dict) and resp.get("status") == "success":
            return {str(p.get("securityId")): p.get("netQty") for p in resp.get("data") or []}
        return None

    entry = 100.0  # sandbox simulated fill price

    # ══ SCENARIO A: adopted position — partial happy path ═══════════════════
    print("\n" + "=" * 64 + "\nSCENARIO A: adopted-position partial (real reduce order)\n" + "=" * 64)
    oid_a = place_entry(sid_a, SYM_A)
    trade = {"symbol": SYM_A, "security_id": sid_a, "entry_price": entry,
             "quantity": QTY, "transaction_type": bot_api.dhan.BUY,
             "order_id": f"DHAN-{SYM_A}",   # adopted position: no broker legs
             "trailing_sl": 0, "sl_price": 0, "atr_value": 1.0}
    shim.active_trades[SYM_A] = trade

    arm = entry * (1 + cfg.TWO_PHASE_PARTIAL_AT_PCT / 100)
    tick(SYM_A, trade, arm, cfg.TWO_PHASE_PARTIAL_AT_PCT)

    check("A: phase 2 armed", trade.get("tp2_active") is True)
    check("A: runner qty halved in trade", trade.get("quantity") == QTY // 2)
    check("A: breakeven stop set", abs(trade.get("trailing_sl", 0) - entry) < 0.01,
          f"sl={trade.get('trailing_sl')}")
    check("A: partial pnl accumulated",
          abs(trade.get("realized_partial_pnl", 0) - (arm - entry) * (QTY // 2)) < 0.01,
          f"pnl={trade.get('realized_partial_pnl', 0):+.2f}")
    a_reduces = [r for r in reduce_log if r[0] == sid_a]
    a_ok = (len(a_reduces) == 1 and a_reduces[0][1] == bot_api.dhan.SELL
            and a_reduces[0][2] == QTY // 2
            and isinstance(a_reduces[0][3], dict) and a_reduces[0][3].get("status") == "success")
    check("A: REAL partial reduce order accepted by sandbox", a_ok,
          f"resp={a_reduces[0][3].get('data') if a_reduces else 'none'}")

    pos = read_positions()
    if pos is None:
        note("A: broker netQty verification", "(sandbox GET /positions returns 500 — placement-level verification only)")
    else:
        check("A: broker net qty reduced", pos.get(str(sid_a)) == QTY // 2, f"netQty={pos.get(str(sid_a))}")

    tick(SYM_A, trade, entry * 1.01, 1.0)
    sl_hi = trade["trailing_sl"]
    check("A: trail ratchets with new high", sl_hi > entry, f"sl={sl_hi:.2f}")
    tick(SYM_A, trade, entry * 1.005, 0.5)
    check("A: trail never loosens", trade["trailing_sl"] == sl_hi)

    # ══ SCENARIO B: super-order legs uncancellable — safe fallback ══════════
    print("\n" + "=" * 64 + "\nSCENARIO B: leg-cancel failure fallback (real sandbox 404s)\n" + "=" * 64)
    oid_b = place_entry(sid_b, SYM_B)
    trade_b = {"symbol": SYM_B, "security_id": sid_b, "entry_price": entry,
               "quantity": QTY, "transaction_type": bot_api.dhan.BUY,
               "order_id": oid_b or "SO-SANDBOX",  # super-order-style id -> leg cancel attempted
               "trailing_sl": 0, "sl_price": 0, "atr_value": 1.0}
    shim.active_trades[SYM_B] = trade_b
    n_reduce_before = len(reduce_log)

    for i in range(3):
        tick(SYM_B, trade_b, arm, cfg.TWO_PHASE_PARTIAL_AT_PCT)
        logger.info("  poll %d: tp2_active=%s leg_cancel_fails=%s", i + 1,
                    trade_b.get("tp2_active"), trade_b.get("tp2_leg_cancel_fails"))
    check("B: partial aborted while legs stand", not trade_b.get("tp2_active"))
    check("B: 3 failures recorded", trade_b.get("tp2_leg_cancel_fails") == 3)
    check("B: partial permanently blocked", trade_b.get("tp2_partial_blocked") is True)
    check("B: NO reduce order sent during failures", len(reduce_log) == n_reduce_before)

    tick(SYM_B, trade_b, arm, cfg.TWO_PHASE_PARTIAL_AT_PCT)  # 4th poll
    check("B: phase 2 arms without partial", trade_b.get("tp2_active") is True)
    check("B: full qty kept", trade_b.get("quantity") == QTY)
    check("B: breakeven stop still armed", abs(trade_b.get("trailing_sl", 0) - entry) < 0.01)
    check("B: exit path will still cancel legs later", not trade_b.get("super_legs_cancelled"))

    # ══ cleanup (best-effort; sandbox capital resets daily anyway) ═══════════
    print("\ncleanup: closing test positions (best-effort) ...")
    for sid, qty in ((sid_a, QTY // 2), (sid_b, QTY)):
        try:
            real_reduce(sid, bot_api.dhan.SELL, qty)
        except Exception as exc:
            logger.warning("cleanup close failed for %s: %s", sid, exc)

    passed = sum(1 for _, ok in results if ok)
    print(f"\n{'='*64}\nSANDBOX TWO-PHASE TEST: {passed}/{len(results)} checks passed "
          f"-> {'PASS' if passed == len(results) else 'FAIL'}\n{'='*64}")
    sys.exit(0 if passed == len(results) else 1)


if __name__ == "__main__":
    main()
