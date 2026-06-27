"""Sector → stock mapping for the momentum bot.

Only stocks that appear here participate in sector scoring. Any stock in the
Dhan watchlist that is NOT mapped is simply ignored by the scanner.
"""

SECTORS: dict[str, list[str]] = {
    "BANKING": [
        "HDFCBANK", "ICICIBANK", "KOTAKBANK", "AXISBANK", "SBIN",
        "INDUSINDBK", "PNB", "BANKBARODA", "CANBK", "IDFCFIRSTB",
        "FEDERALBNK", "BANDHANBNK", "YESBANK", "AUBANK", "BANKINDIA",
    ],
    "FINANCIALS": [
        "BAJFINANCE", "BAJAJFINSV", "SHRIRAMFIN", "CHOLAFIN",
        "ICICIPRULI", "HDFCLIFE", "SBILIFE", "ICICIGI", "ABCAPITAL",
        "HDFCAMC", "MANAPPURAM", "KFINTECH", "POLICYBZR", "MCX",
    ],
    "IT": [
        "TCS", "INFY", "WIPRO", "HCLTECH", "TECHM",
        "LTIM", "PERSISTENT", "COFORGE", "TATAELXSI", "MPHASIS",
        "NAUKRI",
    ],
    "PHARMA": [
        "SUNPHARMA", "DRREDDY", "CIPLA", "DIVISLAB",
        "LUPIN", "ZYDUSLIFE", "BIOCON", "MANKIND", "AUROPHARMA",
        "APOLLOHOSP", "PPLPHARMA",
    ],
    "AUTO": [
        "MARUTI", "TATAMOTORS", "BAJAJ-AUTO", "EICHERMOT",
        "HEROMOTOCO", "M&M", "MOTHERSON", "BHARATFORG",
        "BOSCHLTD", "TATATECH", "TVSMOTOR",   # TVSMOTOR needs Dhan watchlist entry
    ],
    "METALS": [
        "TATASTEEL", "JSWSTEEL", "HINDALCO", "VEDL",
        "COALINDIA", "HINDZINC", "SAIL",
    ],
    "ENERGY": [
        "ONGC", "BPCL", "IOC", "GAIL", "RELIANCE",
        "TATAPOWER", "ADANIPOWER", "ADANIGREEN", "ADANIENT",
        "NTPC", "POWERGRID", "TORNTPOWER",
        "NHPC", "WAAREEENER", "IEX",
    ],
    "FMCG": [
        "ITC", "HINDUNILVR", "NESTLEIND", "BRITANNIA",
        "DABUR", "MARICO", "COLPAL", "GODREJCP", "TATACONSUM",
    ],
    "CEMENT_INFRA": [
        "LT", "AMBUJACEM", "GRASIM", "ULTRACEMCO", "ADANIPORTS",
        "NBCC", "HUDCO", "RVNL",
    ],
    "CAPGOODS": [
        "SIEMENS", "CUMMINSIND", "POLYCAB", "HAVELLS", "DIXON",
        "BHEL",
    ],
    "CONSUMER_DISC": [
        "TITAN", "TRENT", "DMART", "JUBLFOOD",
        "ZOMATO", "ASIANPAINT", "ASTRAL",
        "INDIGO", "ETERNAL", "DELHIVERY",
    ],
    "TELECOM": [
        "BHARTIARTL",
    ],
    # ── New sectors ────────────────────────────────────────────────────────────
    "DEFENCE": [
        "HAL",      # Hindustan Aeronautics
        "BEL",      # Bharat Electronics
        "KAYNES",   # Kaynes Technology (defence electronics)
        # BDL was 1W top mover (+16.29%) and IRFC was 1D mover (+1.40%)
        # per sector performance screen — add to Dhan watchlist to activate:
        "BDL",      # Bharat Dynamics Limited — needs Dhan watchlist entry
        "IRFC",     # Indian Railway Finance Corp — needs Dhan watchlist entry
    ],
    "CHEMICAL": [
        "UPL",        # agrochemicals
        "PIDILITIND", # specialty adhesives / chemicals
    ],
}

# Reverse lookup: symbol → sector name
SYMBOL_TO_SECTOR: dict[str, str] = {
    sym: sector
    for sector, symbols in SECTORS.items()
    for sym in symbols
}
