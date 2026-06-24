"""
DhanMCP — Model Context Protocol server for DhanHQ trading.

Exposes live Dhan account capabilities as MCP tools so any MCP-compatible
client (Claude Desktop, Cursor, VS Code, custom) can trade, manage portfolio,
and fetch market data through natural language.

Usage:
    python server.py                   # stdio transport (Claude Desktop / Cursor)
    python server.py --transport http  # HTTP transport

Auth: set DHAN_CLIENT_ID and DHAN_ACCESS_TOKEN in environment (or .env file).
"""

from __future__ import annotations

import os
import sys
import time
import logging
from datetime import datetime, timedelta
from typing import Optional

from dotenv import load_dotenv

load_dotenv(override=True)

# ── SDK ─────────────────────────────────────────────────────────────────────
try:
    from dhanhq import dhanhq, DhanContext
except ImportError:
    sys.exit("dhanhq package not found. Run: pip install dhanhq")

try:
    from mcp.server.fastmcp import FastMCP
except ImportError:
    sys.exit("mcp package not found. Run: pip install mcp")

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("dhan_mcp")

# ── Dhan client ──────────────────────────────────────────────────────────────
_CLIENT_ID = os.getenv("DHAN_CLIENT_ID", "")
_ACCESS_TOKEN = os.getenv("DHAN_ACCESS_TOKEN", "")

if not _CLIENT_ID or not _ACCESS_TOKEN:
    log.warning(
        "DHAN_CLIENT_ID / DHAN_ACCESS_TOKEN not set — "
        "all tools will return authentication errors."
    )

_ctx = DhanContext(_CLIENT_ID, _ACCESS_TOKEN)
dhan = dhanhq(_ctx)

# ── MCP server ───────────────────────────────────────────────────────────────
mcp = FastMCP(
    name="DhanMCP",
    instructions=(
        "You are connected to a live Dhan trading account via the DhanHQ API. "
        "You can place orders, check portfolio/positions, fetch live and historical "
        "market data, calculate margins, and manage alerts. "
        "Always confirm order details with the user before executing BUY or SELL trades. "
        "Default exchange is NSE_EQ. Default product type is INTRADAY (INTRA)."
    ),
)

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _ok(data: object) -> dict:
    return {"status": "success", "data": data}


def _err(msg: str) -> dict:
    return {"status": "error", "message": msg}


def _assert_auth() -> Optional[dict]:
    if not _CLIENT_ID or not _ACCESS_TOKEN:
        return _err("DHAN_CLIENT_ID and DHAN_ACCESS_TOKEN must be set in environment.")
    return None


EXCHANGE_MAP = {
    "NSE": dhan.NSE,
    "NSE_EQ": dhan.NSE,
    "BSE": dhan.BSE,
    "BSE_EQ": dhan.BSE,
    "MCX": dhan.MCX,
    "NSE_FNO": "NSE_FNO",
    "BSE_FNO": "BSE_FNO",
    "CUR": "NSE_CURRENCY",
}

PRODUCT_MAP = {
    "INTRA": dhan.INTRA,
    "INTRADAY": dhan.INTRA,
    "CNC": dhan.CNC,
    "DELIVERY": dhan.CNC,
    "MARGIN": dhan.MARGIN,
    "CO": dhan.CO,
    "BO": dhan.BO,
}

ORDER_TYPE_MAP = {
    "MARKET": dhan.MARKET,
    "LIMIT": dhan.LIMIT,
    "SL": dhan.SL,
    "SLM": dhan.SLM,
    "STOP_LOSS": dhan.SL,
    "STOP_LOSS_MARKET": dhan.SLM,
}

TRANSACTION_MAP = {
    "BUY": dhan.BUY,
    "SELL": dhan.SELL,
    "B": dhan.BUY,
    "S": dhan.SELL,
}


def _exchange(exchange: str) -> str:
    return EXCHANGE_MAP.get(exchange.upper(), dhan.NSE)


def _product(product: str) -> str:
    return PRODUCT_MAP.get(product.upper(), dhan.INTRA)


def _order_type(ot: str) -> str:
    return ORDER_TYPE_MAP.get(ot.upper(), dhan.MARKET)


def _txn(t: str) -> str:
    return TRANSACTION_MAP.get(t.upper(), dhan.BUY)


# ─────────────────────────────────────────────────────────────────────────────
# Portfolio Tools
# ─────────────────────────────────────────────────────────────────────────────

@mcp.tool()
def get_fund_limits() -> dict:
    """
    Return available balance, used margin, and fund summary for the Dhan account.
    Call this to check how much capital is available before placing orders.
    """
    auth_err = _assert_auth()
    if auth_err:
        return auth_err
    try:
        resp = dhan.get_fund_limits()
        if isinstance(resp, dict) and resp.get("status") == "success":
            d = resp["data"]
            return _ok({
                "available_balance": float(d.get("availabelBalance", 0)),
                "used_margin": float(d.get("utilizedAmount", 0)),
                "opening_balance": float(d.get("openingBalance", 0)),
                "payin_amount": float(d.get("payinAmount", 0)),
                "withdrawal_amount": float(d.get("withdrawalAmount", 0)),
            })
        return _err(f"API error: {resp}")
    except Exception as exc:
        return _err(str(exc))


@mcp.tool()
def get_positions() -> dict:
    """
    Return all open intraday and overnight positions with entry price,
    quantity, unrealized P&L, and product type.
    """
    auth_err = _assert_auth()
    if auth_err:
        return auth_err
    try:
        resp = dhan.get_positions()
        if not isinstance(resp, dict) or resp.get("status") != "success":
            return _err(f"API error: {resp}")
        positions = []
        for pos in resp.get("data", []):
            net_qty = int(pos.get("netQty", 0))
            if net_qty == 0:
                continue
            positions.append({
                "symbol": pos.get("tradingSymbol", ""),
                "security_id": pos.get("securityId", ""),
                "exchange": pos.get("exchangeSegment", ""),
                "product_type": pos.get("productType", ""),
                "net_quantity": net_qty,
                "buy_avg": float(pos.get("buyAvg", 0)),
                "sell_avg": float(pos.get("sellAvg", 0)),
                "net_avg": float(pos.get("netAvg", 0)),
                "unrealized_pnl": float(pos.get("unrealizedProfit", 0)),
                "realized_pnl": float(pos.get("realizedProfit", 0)),
                "day_buy_value": float(pos.get("dayBuyValue", 0)),
                "day_sell_value": float(pos.get("daySelValue", 0)),
            })
        return _ok({"count": len(positions), "positions": positions})
    except Exception as exc:
        return _err(str(exc))


@mcp.tool()
def get_holdings() -> dict:
    """
    Return long-term delivery holdings (CNC/DELIVERY positions) with
    average cost, current value, and total P&L.
    """
    auth_err = _assert_auth()
    if auth_err:
        return auth_err
    try:
        resp = dhan.get_holdings()
        if not isinstance(resp, dict) or resp.get("status") != "success":
            return _err(f"API error: {resp}")
        holdings = []
        for h in resp.get("data", []):
            holdings.append({
                "symbol": h.get("tradingSymbol", ""),
                "security_id": h.get("securityId", ""),
                "exchange": h.get("exchangeSegment", ""),
                "isin": h.get("isin", ""),
                "total_quantity": int(h.get("totalQty", 0)),
                "dp_quantity": int(h.get("dpQty", 0)),
                "t1_quantity": int(h.get("t1Qty", 0)),
                "avg_cost_price": float(h.get("avgCostPrice", 0)),
                "collateral_quantity": int(h.get("collateralQty", 0)),
            })
        return _ok({"count": len(holdings), "holdings": holdings})
    except Exception as exc:
        return _err(str(exc))


# ─────────────────────────────────────────────────────────────────────────────
# Order Management Tools
# ─────────────────────────────────────────────────────────────────────────────

@mcp.tool()
def get_order_list() -> dict:
    """
    Return today's full order book — all orders with their status
    (PENDING, TRADED, CANCELLED, REJECTED, etc.).
    """
    auth_err = _assert_auth()
    if auth_err:
        return auth_err
    try:
        resp = dhan.get_order_list()
        if not isinstance(resp, dict) or resp.get("status") != "success":
            return _err(f"API error: {resp}")
        orders = []
        for o in resp.get("data", []):
            orders.append({
                "order_id": o.get("orderId", ""),
                "symbol": o.get("tradingSymbol", ""),
                "security_id": o.get("securityId", ""),
                "exchange": o.get("exchangeSegment", ""),
                "transaction_type": o.get("transactionType", ""),
                "order_type": o.get("orderType", ""),
                "product_type": o.get("productType", ""),
                "quantity": int(o.get("quantity", 0)),
                "pending_quantity": int(o.get("pendingQuantity", 0)),
                "price": float(o.get("price", 0)),
                "trigger_price": float(o.get("triggerPrice", 0)),
                "average_price": float(o.get("averageTradedPrice", 0)),
                "status": o.get("orderStatus", ""),
                "order_time": o.get("createTime", ""),
                "update_time": o.get("updateTime", ""),
                "remarks": o.get("omsErrorDescription", ""),
            })
        return _ok({"count": len(orders), "orders": orders})
    except Exception as exc:
        return _err(str(exc))


@mcp.tool()
def get_order_by_id(order_id: str) -> dict:
    """
    Get full details of a specific order by its order_id.

    Args:
        order_id: The Dhan order ID string (e.g. "112111182198").
    """
    auth_err = _assert_auth()
    if auth_err:
        return auth_err
    try:
        resp = dhan.get_order_by_id(order_id)
        if not isinstance(resp, dict) or resp.get("status") != "success":
            return _err(f"API error: {resp}")
        return _ok(resp.get("data", {}))
    except Exception as exc:
        return _err(str(exc))


@mcp.tool()
def place_order(
    security_id: str,
    transaction_type: str,
    quantity: int,
    exchange_segment: str = "NSE_EQ",
    order_type: str = "MARKET",
    product_type: str = "INTRA",
    price: float = 0.0,
    trigger_price: float = 0.0,
    disclosed_quantity: int = 0,
    validity: str = "DAY",
    tag: str = "",
) -> dict:
    """
    Place a new equity or F&O order on Dhan.

    Args:
        security_id:       Dhan security ID (e.g. "1333" for HDFCBANK).
        transaction_type:  "BUY" or "SELL".
        quantity:          Number of shares / lots.
        exchange_segment:  "NSE_EQ" (default), "BSE_EQ", "MCX", "NSE_FNO", "BSE_FNO".
        order_type:        "MARKET" (default), "LIMIT", "SL", "SLM".
        product_type:      "INTRA" (default intraday), "CNC" (delivery), "MARGIN", "CO", "BO".
        price:             Limit price (required for LIMIT/SL orders; 0 for MARKET).
        trigger_price:     Trigger price (required for SL/SLM orders).
        disclosed_quantity: Iceberg quantity to disclose (0 = full).
        validity:          "DAY" (default) or "IOC".
        tag:               Optional order tag / correlation ID.

    Returns dict with order_id on success.
    """
    auth_err = _assert_auth()
    if auth_err:
        return auth_err
    try:
        resp = dhan.place_order(
            security_id=security_id,
            exchange_segment=_exchange(exchange_segment),
            transaction_type=_txn(transaction_type),
            quantity=int(quantity),
            order_type=_order_type(order_type),
            product_type=_product(product_type),
            price=float(price),
            trigger_price=float(trigger_price),
            disclosed_quantity=int(disclosed_quantity),
            validity=validity.upper(),
            tag=tag,
        )
        if not isinstance(resp, dict):
            return _err(f"Unexpected response: {resp}")
        if resp.get("status") == "success":
            data = resp.get("data", {})
            return _ok({
                "order_id": data.get("orderId", ""),
                "order_status": data.get("orderStatus", ""),
                "message": "Order placed successfully.",
            })
        return _err(f"Order failed: {resp.get('remarks', resp)}")
    except Exception as exc:
        return _err(str(exc))


@mcp.tool()
def modify_order(
    order_id: str,
    order_type: str,
    quantity: int,
    price: float = 0.0,
    trigger_price: float = 0.0,
    disclosed_quantity: int = 0,
    validity: str = "DAY",
    leg_name: str = "",
) -> dict:
    """
    Modify a pending order.

    Args:
        order_id:          Order ID to modify.
        order_type:        New order type ("MARKET", "LIMIT", "SL", "SLM").
        quantity:          New quantity.
        price:             New limit price (0 for MARKET).
        trigger_price:     New trigger price (for SL/SLM).
        disclosed_quantity: New iceberg quantity.
        validity:          "DAY" or "IOC".
        leg_name:          For bracket/cover orders — "ENTRY_LEG", "TARGET_LEG", "STOP_LOSS_LEG".
    """
    auth_err = _assert_auth()
    if auth_err:
        return auth_err
    try:
        resp = dhan.modify_order(
            order_id=order_id,
            order_type=_order_type(order_type),
            quantity=int(quantity),
            price=float(price),
            trigger_price=float(trigger_price),
            disclosed_quantity=int(disclosed_quantity),
            validity=validity.upper(),
            leg_name=leg_name,
        )
        if isinstance(resp, dict) and resp.get("status") == "success":
            return _ok({"order_id": order_id, "message": "Order modified."})
        return _err(f"Modify failed: {resp}")
    except Exception as exc:
        return _err(str(exc))


@mcp.tool()
def cancel_order(order_id: str) -> dict:
    """
    Cancel a pending order by order_id.

    Args:
        order_id: The Dhan order ID to cancel.
    """
    auth_err = _assert_auth()
    if auth_err:
        return auth_err
    try:
        resp = dhan.cancel_order(order_id=order_id)
        if isinstance(resp, dict) and resp.get("status") == "success":
            return _ok({"order_id": order_id, "message": "Order cancelled."})
        return _err(f"Cancel failed: {resp}")
    except Exception as exc:
        return _err(str(exc))


@mcp.tool()
def get_trade_book() -> dict:
    """
    Return today's executed trades (fills) with price, quantity, and time.
    """
    auth_err = _assert_auth()
    if auth_err:
        return auth_err
    try:
        resp = dhan.get_trade_book()
        if not isinstance(resp, dict) or resp.get("status") != "success":
            return _err(f"API error: {resp}")
        trades = []
        for t in resp.get("data", []):
            trades.append({
                "order_id": t.get("orderId", ""),
                "exchange_trade_id": t.get("exchangeTradeId", ""),
                "symbol": t.get("tradingSymbol", ""),
                "security_id": t.get("securityId", ""),
                "transaction_type": t.get("transactionType", ""),
                "product_type": t.get("productType", ""),
                "quantity": int(t.get("tradedQuantity", 0)),
                "trade_price": float(t.get("tradedPrice", 0)),
                "trade_time": t.get("createTime", ""),
            })
        return _ok({"count": len(trades), "trades": trades})
    except Exception as exc:
        return _err(str(exc))


# ─────────────────────────────────────────────────────────────────────────────
# Market Data Tools
# ─────────────────────────────────────────────────────────────────────────────

@mcp.tool()
def get_live_quote(security_ids: list[int], exchange_segment: str = "NSE_EQ") -> dict:
    """
    Get live quotes (LTP, OHLC, volume, market depth) for up to 100 securities.

    Args:
        security_ids:    List of integer security IDs (e.g. [1333, 2885]).
        exchange_segment: "NSE_EQ" (default), "BSE_EQ", "MCX", "NSE_FNO".

    Returns dict mapping security_id -> quote data.
    """
    auth_err = _assert_auth()
    if auth_err:
        return auth_err
    try:
        exch = _exchange(exchange_segment)
        resp = dhan.quote_data(securities={exch: [int(i) for i in security_ids]})
        if not isinstance(resp, dict) or resp.get("status") != "success":
            return _err(f"API error: {resp}")
        raw = resp.get("data", {}).get("data", {})
        result = {}
        for seg, seg_data in raw.items():
            for sid_str, quote in seg_data.items():
                ohlc = quote.get("ohlc", {})
                depth = quote.get("depth", {})
                result[sid_str] = {
                    "last_price": float(quote.get("last_price", 0)),
                    "open": float(ohlc.get("open", 0)),
                    "high": float(ohlc.get("high", 0)),
                    "low": float(ohlc.get("low", 0)),
                    "close": float(ohlc.get("close", 0)),
                    "volume": int(quote.get("volume", 0)),
                    "average_price": float(quote.get("average_price", 0)),
                    "bid": float(depth.get("buy", [{}])[0].get("price", 0)) if depth.get("buy") else 0,
                    "ask": float(depth.get("sell", [{}])[0].get("price", 0)) if depth.get("sell") else 0,
                    "timestamp": quote.get("exchange_timestamp", ""),
                }
        return _ok(result)
    except Exception as exc:
        return _err(str(exc))


@mcp.tool()
def get_intraday_candles(
    security_id: str,
    interval: int = 5,
    from_date: str = "",
    to_date: str = "",
    exchange_segment: str = "NSE_EQ",
) -> dict:
    """
    Fetch intraday OHLCV candles for a security.

    Args:
        security_id:     Dhan security ID string (e.g. "1333").
        interval:        Candle interval in minutes: 1, 3, 5, 10, 15, 30, or 60.
        from_date:       Start date "YYYY-MM-DD" (default: today).
        to_date:         End date "YYYY-MM-DD" (default: today).
        exchange_segment: "NSE_EQ" (default), "BSE_EQ", etc.

    Returns list of OHLCV candles with timestamps (IST).
    """
    auth_err = _assert_auth()
    if auth_err:
        return auth_err
    today = datetime.now().strftime("%Y-%m-%d")
    from_date = from_date or today
    to_date = to_date or today
    exch = _exchange(exchange_segment)
    try:
        resp = dhan.intraday_minute_data(
            security_id=security_id,
            exchange_segment=exch,
            instrument_type="EQUITY",
            from_date=from_date,
            to_date=to_date,
            interval=int(interval),
        )
        if not isinstance(resp, dict) or resp.get("status") != "success":
            return _err(f"API error: {resp}")
        data = resp.get("data", {})
        if not data or not data.get("open"):
            return _ok({"count": 0, "candles": []})
        timestamps = data.get("timestamp", [])
        opens = data["open"]
        candles = []
        for i, o in enumerate(opens):
            ts = ""
            if i < len(timestamps):
                dt = datetime.utcfromtimestamp(timestamps[i]) + timedelta(hours=5, minutes=30)
                ts = dt.strftime("%Y-%m-%d %H:%M")
            candles.append({
                "timestamp": ts,
                "open": float(o),
                "high": float(data["high"][i]) if i < len(data.get("high", [])) else 0,
                "low": float(data["low"][i]) if i < len(data.get("low", [])) else 0,
                "close": float(data["close"][i]) if i < len(data.get("close", [])) else 0,
                "volume": int(data["volume"][i]) if i < len(data.get("volume", [])) else 0,
            })
        return _ok({"count": len(candles), "candles": candles})
    except Exception as exc:
        return _err(str(exc))


@mcp.tool()
def get_daily_candles(
    security_id: str,
    from_date: str = "",
    to_date: str = "",
    exchange_segment: str = "NSE_EQ",
) -> dict:
    """
    Fetch daily OHLCV candles (up to ~5 years) for a security.

    Args:
        security_id:     Dhan security ID string (e.g. "1333").
        from_date:       Start date "YYYY-MM-DD" (default: 365 days ago).
        to_date:         End date "YYYY-MM-DD" (default: today).
        exchange_segment: "NSE_EQ" (default), "BSE_EQ", etc.
    """
    auth_err = _assert_auth()
    if auth_err:
        return auth_err
    today = datetime.now()
    to_date = to_date or today.strftime("%Y-%m-%d")
    from_date = from_date or (today - timedelta(days=365)).strftime("%Y-%m-%d")
    exch = _exchange(exchange_segment)
    try:
        resp = dhan.historical_daily_data(
            security_id=str(security_id),
            exchange_segment=exch,
            instrument_type="EQUITY",
            from_date=from_date,
            to_date=to_date,
        )
        if not isinstance(resp, dict) or resp.get("status") != "success":
            return _err(f"API error: {resp}")
        data = resp.get("data", {})
        if not data or not data.get("open"):
            return _ok({"count": 0, "candles": []})
        timestamps = data.get("timestamp", [])
        opens = data["open"]
        candles = []
        for i, o in enumerate(opens):
            ts = ""
            if i < len(timestamps):
                dt = datetime.utcfromtimestamp(timestamps[i]) + timedelta(hours=5, minutes=30)
                ts = dt.strftime("%Y-%m-%d")
            candles.append({
                "date": ts,
                "open": float(o),
                "high": float(data["high"][i]) if i < len(data.get("high", [])) else 0,
                "low": float(data["low"][i]) if i < len(data.get("low", [])) else 0,
                "close": float(data["close"][i]) if i < len(data.get("close", [])) else 0,
                "volume": int(data["volume"][i]) if i < len(data.get("volume", [])) else 0,
            })
        return _ok({"count": len(candles), "candles": candles})
    except Exception as exc:
        return _err(str(exc))


@mcp.tool()
def get_option_chain(
    underlying_scrip: int,
    underlying_seg: str,
    expiry: str,
) -> dict:
    """
    Fetch full option chain with Greeks for a given underlying and expiry.

    Args:
        underlying_scrip: Security ID of the underlying (e.g. 13 for Nifty, 25 for BankNifty).
        underlying_seg:   Segment — "IDX_I" for index, "NSE_EQ" for stock.
        expiry:           Expiry date "YYYY-MM-DD".

    Returns option chain with calls and puts, strike prices, IV, delta, theta, vega.
    """
    auth_err = _assert_auth()
    if auth_err:
        return auth_err
    try:
        resp = dhan.get_option_chain(
            optionchain_scripid=str(underlying_scrip),
            exchange_segment=underlying_seg,
            expiry=expiry,
        )
        if not isinstance(resp, dict) or resp.get("status") != "success":
            return _err(f"API error: {resp}")
        return _ok(resp.get("data", {}))
    except Exception as exc:
        return _err(str(exc))


@mcp.tool()
def get_expiry_list(
    underlying_scrip: int,
    underlying_seg: str,
) -> dict:
    """
    Get list of available expiry dates for an F&O underlying.

    Args:
        underlying_scrip: Security ID of the underlying (e.g. 13 for Nifty).
        underlying_seg:   "IDX_I" for index, "NSE_EQ" for stock.
    """
    auth_err = _assert_auth()
    if auth_err:
        return auth_err
    try:
        resp = dhan.expiry_list(
            optionchain_scripid=str(underlying_scrip),
            exchange_segment=underlying_seg,
        )
        if not isinstance(resp, dict) or resp.get("status") != "success":
            return _err(f"API error: {resp}")
        return _ok(resp.get("data", {}))
    except Exception as exc:
        return _err(str(exc))


# ─────────────────────────────────────────────────────────────────────────────
# Margin Tools
# ─────────────────────────────────────────────────────────────────────────────

@mcp.tool()
def calculate_margin(
    security_id: str,
    transaction_type: str,
    quantity: int,
    exchange_segment: str = "NSE_EQ",
    product_type: str = "INTRA",
    price: float = 0.0,
    trade_type: str = "INTRADAY",
) -> dict:
    """
    Calculate margin required before placing an order — use this to check
    if you have enough funds for a trade.

    Args:
        security_id:     Dhan security ID.
        transaction_type: "BUY" or "SELL".
        quantity:         Number of shares.
        exchange_segment: "NSE_EQ" (default), "BSE_EQ", "NSE_FNO", etc.
        product_type:     "INTRA" or "CNC".
        price:            Expected entry price (0 uses current market price).
        trade_type:       "INTRADAY" or "POSITIONAL".

    Returns required margin and charges breakdown.
    """
    auth_err = _assert_auth()
    if auth_err:
        return auth_err
    try:
        resp = dhan.margin_calculator(
            security_id=security_id,
            exchange_segment=_exchange(exchange_segment),
            transaction_type=_txn(transaction_type),
            quantity=int(quantity),
            product_type=_product(product_type),
            price=float(price),
            trade_type=trade_type.upper(),
        )
        if not isinstance(resp, dict) or resp.get("status") != "success":
            return _err(f"API error: {resp}")
        d = resp.get("data", {})
        return _ok({
            "total_margin": float(d.get("totalMargin", 0)),
            "span_margin": float(d.get("spanMargin", 0)),
            "exposure_margin": float(d.get("exposureMargin", 0)),
            "available_balance": float(d.get("availableBalance", 0)),
            "required_margin": float(d.get("requiredMargin", 0)),
            "brokerage": float(d.get("brokerage", 0)),
        })
    except Exception as exc:
        return _err(str(exc))


# ─────────────────────────────────────────────────────────────────────────────
# Super Order (Bracket Order with trailing SL)
# ─────────────────────────────────────────────────────────────────────────────

@mcp.tool()
def place_super_order(
    security_id: str,
    transaction_type: str,
    quantity: int,
    entry_price: float,
    stop_loss_price: float,
    target_price: float,
    trailing_jump: float = 0.0,
    exchange_segment: str = "NSE_EQ",
) -> dict:
    """
    Place a Super Order — a market entry with a built-in SL and target.
    Super Orders are intraday only and auto-square off at market close.

    Args:
        security_id:      Dhan security ID.
        transaction_type: "BUY" or "SELL".
        quantity:         Number of shares.
        entry_price:      Expected entry price (used to validate SL/target).
        stop_loss_price:  Absolute SL price (e.g. 490.0 for a 500 buy).
        target_price:     Absolute target price (e.g. 515.0 for a 500 buy).
        trailing_jump:    Trailing SL step in rupees (0 = fixed SL).
        exchange_segment: "NSE_EQ" (default), "BSE_EQ".
    """
    auth_err = _assert_auth()
    if auth_err:
        return auth_err
    try:
        payload = {
            "transactionType": _txn(transaction_type).upper(),
            "exchangeSegment": _exchange(exchange_segment).upper(),
            "productType": "INTRADAY",
            "orderType": "MARKET",
            "securityId": str(security_id),
            "quantity": int(quantity),
            "price": None,
            "targetPrice": float(target_price),
            "stopLossPrice": float(stop_loss_price),
            "trailingJump": float(trailing_jump),
        }
        resp = dhan.dhan_http.post("/super/orders", payload)
        if isinstance(resp, dict) and resp.get("status") == "success":
            data = resp.get("data", {})
            return _ok({
                "order_id": data.get("orderId", ""),
                "status": data.get("orderStatus", ""),
                "message": "Super order placed.",
            })
        return _err(f"Super order failed: {resp}")
    except Exception as exc:
        return _err(str(exc))


# ─────────────────────────────────────────────────────────────────────────────
# Utility Tools
# ─────────────────────────────────────────────────────────────────────────────

@mcp.tool()
def lookup_security_id(symbol: str) -> dict:
    """
    Look up the Dhan security ID for a stock symbol from the built-in universe.
    Covers Nifty 500 stocks, ETFs, and common F&O names.

    Args:
        symbol: NSE trading symbol (e.g. "HDFCBANK", "RELIANCE", "NIFTYBEES").

    Returns security_id string and whether it was found.
    """
    UNIVERSE: dict[str, str] = {
        "HDFCBANK": "1333", "RELIANCE": "2885", "TCS": "11536",
        "ICICIBANK": "4963", "KOTAKBANK": "1922", "AXISBANK": "5900",
        "SBIN": "3045", "INFY": "1594", "WIPRO": "3787",
        "HCLTECH": "7229", "SUNPHARMA": "3351", "DRREDDY": "881",
        "MARUTI": "10999", "ASIANPAINT": "236", "TITAN": "3506",
        "BAJFINANCE": "317", "BPCL": "526", "NTPC": "11630",
        "ONGC": "2475", "COALINDIA": "20374", "TATASTEEL": "3499",
        "JSWSTEEL": "11723", "HINDALCO": "1363", "VEDL": "3063",
        "ADANIPORTS": "15083", "LTIM": "17818", "TECHM": "13538",
        "POWERGRID": "14977", "M&M": "2031", "BAJAJFINSV": "16675",
        "NESTLEIND": "17963", "COLPAL": "15141", "DABUR": "772",
        "MARICO": "4067", "PIDILITIND": "2664", "ASIANPAINT": "236",
        "BRITANNIA": "513", "ITC": "1669", "HINDUNILVR": "1394",
        "TATACONSUM": "11346", "BHARTIARTL": "10604",
        "NIFTYBEES": "10576", "BANKBEES": "11439", "GOLDBEES": "14428",
        "ITBEES": "19084", "JUNIORBEES": "10939",
        "ZOMATO": "21296", "PAYTM": "21267", "DMART": "19913",
        "TRENT": "1964", "DIXON": "21690", "HAVELLS": "9819",
        "POLYCAB": "9590", "AUROPHARMA": "275", "LUPIN": "10440",
        "CIPLA": "694", "BIOCON": "11373", "PERSISTENT": "18365",
        "MPHASIS": "4503", "COFORGE": "11543", "TATATECH": "20293",
        "HAL": "2303", "BEL": "534", "ADANIGREEN": "3563",
        "ADANIPOWER": "17388", "TATAPOWER": "3426",
        "IRFC": "543257", "BDL": "541143", "TVSMOTOR": "532343",
    }
    sym = symbol.upper().strip()
    sid = UNIVERSE.get(sym)
    if sid:
        return _ok({"symbol": sym, "security_id": sid, "found": True})
    return _ok({
        "symbol": sym,
        "security_id": None,
        "found": False,
        "hint": (
            "Symbol not in built-in universe. Download the Dhan scrip master CSV from "
            "https://images.dhan.co/api-data/api-scrip-master.csv to look up any security."
        ),
    })


@mcp.tool()
def market_status() -> dict:
    """
    Return whether the NSE market is currently open based on IST time.
    Market hours: Mon–Fri 09:15–15:30 IST (pre-open 09:00–09:15).
    """
    now_ist = datetime.utcnow() + timedelta(hours=5, minutes=30)
    weekday = now_ist.weekday()  # 0=Mon … 6=Sun
    hour, minute = now_ist.hour, now_ist.minute
    current_minutes = hour * 60 + minute

    if weekday >= 5:
        return _ok({"is_open": False, "reason": "Weekend", "ist_time": now_ist.strftime("%Y-%m-%d %H:%M")})

    pre_open_start = 9 * 60
    market_open = 9 * 60 + 15
    market_close = 15 * 60 + 30

    if current_minutes < pre_open_start:
        phase = "Pre-market (closed)"
        is_open = False
    elif current_minutes < market_open:
        phase = "Pre-open session"
        is_open = False
    elif current_minutes <= market_close:
        phase = "Market open"
        is_open = True
    else:
        phase = "After hours (closed)"
        is_open = False

    minutes_to_open = max(0, market_open - current_minutes)
    minutes_to_close = max(0, market_close - current_minutes)

    return _ok({
        "is_open": is_open,
        "phase": phase,
        "ist_time": now_ist.strftime("%Y-%m-%d %H:%M"),
        "minutes_to_open": minutes_to_open if not is_open else 0,
        "minutes_to_close": minutes_to_close if is_open else 0,
    })


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="DhanMCP — MCP server for DhanHQ trading")
    parser.add_argument(
        "--transport",
        choices=["stdio", "http"],
        default="stdio",
        help="Transport mode: 'stdio' for Claude Desktop/Cursor (default), 'http' for web clients",
    )
    parser.add_argument("--host", default="0.0.0.0", help="Host for HTTP transport")
    parser.add_argument("--port", type=int, default=8765, help="Port for HTTP transport")
    args = parser.parse_args()

    log.info("Starting DhanMCP server (transport=%s)", args.transport)

    if args.transport == "http":
        mcp.run(transport="streamable-http", host=args.host, port=args.port)
    else:
        mcp.run(transport="stdio")
