FNO_UNIVERSE = {
    "RELIANCE": "2885",
    "TCS": "11536",
    "HDFCBANK": "1333",
    "INFY": "1594",
    "ICICIBANK": "4963",
    "KOTAKBANK": "1922",
    "AXISBANK": "5900",
    "SBIN": "3045",
    "BAJFINANCE": "317",
    "TATASTEEL": "3499",
    "WIPRO": "3787",
    "HCLTECH": "7229",
    "SUNPHARMA": "3351",
    "DRREDDY": "881",
    "MARUTI": "10999",
    "ASIANPAINT": "236",
    "TITAN": "3506",
    "ADANIPORTS": "15083",
    "LTIM": "17818",
    "TECHM": "13538",
    "POWERGRID": "14977",
    "NTPC": "11630",
    "ONGC": "2475",
    "COALINDIA": "20374",
    "JSWSTEEL": "11723",
    "HINDALCO": "1363",
    "VEDL": "3063",
    "M&M": "2031",
    "BAJAJFINSV": "16675",
}


# ═══════════════════════════════════════════════════════════════════════════════
# ── ETF UNIVERSE (from scrip-master_NSE_EQ.csv) ────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════
# Format: "DISPLAY_NAME": "SECURITY_ID"

ETF_UNIVERSE = {
    # ── Index ETFs (Nifty 50 / Sensex) ─────────────────────────────────────
    "NIFTYBEES": "10576",  # Nippon Nifty 50 ETF
    "SBI NIFTY 50": "10176",  # SBI Nifty 50 ETF
    "KOTAK NIFTY 50": "18102",  # Kotak Nifty 50 ETF
    "MIRAE NIFTY 50": "6353",  # Mirae Asset Nifty 50 ETF
    "MOT OSWAL NIFTY 50": "19289",  # Motilal Oswal Nifty 50 ETF
    "QUANTUM NIFTY 50": "16819",  # Quantum Nifty 50 ETF
    "TATA NIFTY 50": "7838",  # Tata Nifty 50 ETF
    "INVESCO NIFTY 50": "24217",  # Invesco India Nifty 50 ETF
    "ICICI NIFTY 50": "29553",  # ICICI Pru Nifty 50 ETF
    "BIRLA NIFTY 50": "24781",  # Aditya Birla Nifty 50 ETF
    "BAJAJ NIFTY 50": "21959",  # Bajaj Finserv Nifty 50 ETF
    "ICICI SENSEX": "4378",  # ICICI Pru Sensex ETF
    "DSP SENSEX": "17613",  # DSP Sensex ETF
    "MIRAE SENSEX": "19224",  # Mirae Asset Sensex ETF
    "BIRLA SENSEX": "5957",  # Aditya Birla Sensex ETF
    # ── Nifty Next 50 / Nifty 100 ─────────────────────────────────────────
    "JUNIORBEES": "10939",  # Nippon Nifty Next 50 Junior Bees
    "SBI NIFTY NEXT 50": "7353",  # SBI Nifty Next 50 ETF
    "NIF100BEES": "29577",  # Nippon India Nifty 100 ETF
    "ICICI NIFTY 100": "30392",  # ICICI Pru Nifty 100 ETF
    # ── Bank ETFs ──────────────────────────────────────────────────────────
    "BANKBEES": "11439",  # Nippon Nifty Bank ETF
    "SBI BANK": "7361",  # SBI Nifty Bank ETF
    "KOTAK BANK": "5851",  # Kotak Nifty Bank ETF
    "MIRAE BANK": "17419",  # Mirae Asset Nifty Bank ETF
    "AXIS BANK ETF": "1044",  # Axis Nifty Bank ETF
    "BAJAJ BANK": "21986",  # Bajaj Finserv Nifty Bank ETF
    "BIRLA BANK": "13987",  # Aditya Birla Nifty Bank ETF
    "BARODA BANK": "24117",  # Baroda BNP Paribas Nifty Bank ETF
    "PSUBNKBEES": "15032",  # Nippon Nifty PSU Bank ETF
    "DSP PSU BANK": "17616",  # DSP Nifty PSU Bank ETF
    "SBI PVT BANK": "722",  # SBI Nifty Private Bank ETF
    "DSP PVT BANK": "17576",  # DSP Nifty Private Bank ETF
    # ── IT / Tech ETFs ─────────────────────────────────────────────────────
    "ITBEES": "19084",  # Nippon Nifty IT ETF
    "MIRAE IT": "19633",  # Mirae Asset Nifty IT ETF
    "SBI IT": "740",  # SBI Nifty IT ETF
    "AXIS IT": "3010",  # Axis Nifty IT ETF
    "TATA DIGITAL": "8882",  # Tata Nifty India Digital ETF
    # ── Pharma / Healthcare ────────────────────────────────────────────────
    "PHARMABEES": "4973",  # Nippon Pharma ETF
    "AXIS HEALTHCARE": "3608",  # Axis Nifty Healthcare ETF
    # ── Infrastructure / Auto / Manufacturing ──────────────────────────────
    "INFRABEES": "20072",  # Nippon Nifty Infra ETF
    "AUTOBEES": "7880",  # Nippon Auto ETF
    "MIRAE MFG": "7979",  # Mirae Asset Nifty India Manufacturing ETF
    "ICICI EV AUTO": "755881",  # ICICI Pru Nifty EV & New Age Auto ETF
    # ── Metals / Oil & Gas ─────────────────────────────────────────────────
    "ICICI METAL": "24861",  # ICICI Pru Nifty Metal ETF
    "ICICI OIL GAS": "24533",  # ICICI Pru Nifty Oil & Gas ETF
    # ── Financial Services / Consumption ───────────────────────────────────
    "MIRAE FINSERV": "5220",  # Mirae Asset Nifty Financial Service ETF
    "CONSUMBEES": "2435",  # Nippon Nifty Consumption ETF
    "SBI CONSUMPTION": "5168",  # SBI Nifty Consumption ETF
    "AXIS CONSUMPTION": "5732",  # Axis Nifty Consumption ETF
    # ── Midcap ETFs ────────────────────────────────────────────────────────
    "MID150BEES": "8506",  # Nippon Nifty Midcap 150 ETF
    "MIRAE MIDCAP 150": "8413",  # Mirae Asset Nifty Midcap 150 ETF
    "ICICI MIDCAP": "17702",  # ICICI Pru BSE Midcap Select ETF
    "MOT OSWAL MIDCAP": "21423",  # Motilal Oswal Midcap 100 ETF
    "DSP MIDCAP Q50": "7456",  # DSP Nifty Midcap 150 Quality 50 ETF
    # ── Factor / Smart Beta ETFs ───────────────────────────────────────────
    "NV20BEES": "9847",  # Nippon Nifty 50 Value 20 ETF
    "KOTAK NV20": "11457",  # Kotak Nifty 50 Value 20 ETF
    "ICICI NV20": "17475",  # ICICI Pru Nifty 50 Value 20 ETF
    "ICICI LOW VOL 30": "21254",  # ICICI Pru Nifty 100 Low Vol 30 ETF
    "ICICI VAL 30": "25851",  # ICICI Pru Nifty 200 Value 30 ETF
    "MIRAE ALPHA 30": "19640",  # Mirae Asset Nifty 200 Alpha 30 ETF
    "SBI QUALITY 30": "7218",  # SBI Nifty 200 Quality 30 ETF
    "SBI EQ WEIGHT": "24524",  # SBI Nifty 50 Equal Weight ETF
    "DSP EQ WEIGHT": "6606",  # DSP Nifty 50 Equal Weight ETF
    "DIVOPPBEES": "2636",  # Nippon Nifty 50 Dividend Opp ETF
    "ICICI TOP 15": "757781",  # ICICI Pru Nifty Top 15 Equal Weight ETF
    "MIRAE ESG": "1200",  # Mirae Asset Nifty 100 ESG Sector Leader
    "SHARIAH BEES": "17044",  # Nippon India Nifty 50 Shariah ETF
    "CPSE ETF": "2328",  # CPSE ETF
    # ── Gold ETFs ──────────────────────────────────────────────────────────
    "GOLDBEES": "14428",  # Nippon Gold ETF
    "SBI GOLD": "17272",  # SBI Gold ETF
    "HDFC GOLD": "19543",  # HDFC Gold ETF
    "KOTAK GOLD": "14858",  # Kotak Gold ETF
    "ICICI GOLD": "19679",  # ICICI Pru Gold ETF
    "MIRAE GOLD": "14286",  # Mirae Asset Gold ETF
    "AXIS GOLD": "20532",  # Axis Gold ETF
    "BIRLA GOLD": "23804",  # Aditya Birla Gold ETF
    "INVESCO GOLD": "18292",  # Invesco India Gold ETF
    "LIC GOLD": "25640",  # LIC Gold ETF
    "WEALTH GOLD": "760482",  # The Wealth Company Gold ETF
    # ── Silver ETFs ────────────────────────────────────────────────────────
    "SILVERBEES": "8080",  # Nippon Silver ETF
    "DSP SILVER": "10761",  # DSP Silver ETF
    # ── Debt / Bond ETFs ───────────────────────────────────────────────────
    "LIQUIDBEES": "11006",  # Nippon India Nifty Liquid ETF
    "DSP LIQUID": "1927",  # DSP Nifty Liquid ETF
    "BAJAJ LIQUID": "23915",  # Bajaj Finserv Nifty 1D Rate Liquid ETF
    "SBI LIQUID": "758496",  # SBI Nifty 1D Rate Liquid ETF
    "ICICI LIQUID": "30139",  # ICICI Pru BSE Liquid Rate ETF
    "LTGILTBEES": "17700",  # Nippon 8-13 Year G-Sec ETF
    "MIRAE GSEC": "14938",  # Mirae Asset Nifty 8-13 Year G-Sec ETF
    "SBI 10Y GILT": "17395",  # SBI Nifty 10 Year G-Sec ETF
    "GILT5YBEES": "3172",  # Nippon 5 Year G-Sec ETF
    "SDL26BEES": "3022",  # Nippon Nifty SDL26 T20 ETF
    "AXIS BOND SDL": "3530",  # Axis Nifty AAA Bond + SDL Apr26 ETF
    "BBOND APR30": "16253",  # Edelweiss Bharat Bond April 2030 ETF
    "BBOND APR31": "22239",  # Edelweiss Bharat Bond April 2031 ETF
    "BBOND APR32": "7196",  # Edelweiss Bharat Bond April 2032 ETF
    "BBOND APR33": "13139",  # Edelweiss Bharat Bond April 2033 ETF
    # ── International ETFs ─────────────────────────────────────────────────
    "NASDAQ100 ETF": "22739",  # Motilal Oswal Nasdaq 100 ETF
    "HNGSNGBEES": "18284",  # Nippon Hang Seng ETF
}


# ── Curated: Most Liquid ETFs (high volume, tight spreads) ─────────────────
# Use this subset for intraday backtesting — these have the best liquidity
ETF_LIQUID = {
    "NIFTYBEES": "10576",  # Nippon Nifty 50 ETF
    "BANKBEES": "11439",  # Nippon Nifty Bank ETF
    "ITBEES": "19084",  # Nippon Nifty IT ETF
    "GOLDBEES": "14428",  # Nippon Gold ETF
    "SILVERBEES": "8080",  # Nippon Silver ETF
    "JUNIORBEES": "10939",  # Nippon Nifty Next 50
    "PSUBNKBEES": "15032",  # Nippon PSU Bank ETF
    "INFRABEES": "20072",  # Nippon Nifty Infra ETF
    "PHARMABEES": "4973",  # Nippon Pharma ETF
    "AUTOBEES": "7880",  # Nippon Auto ETF
    "CONSUMBEES": "2435",  # Nippon Consumption ETF
    "MID150BEES": "8506",  # Nippon Midcap 150 ETF
    "NV20BEES": "9847",  # Nippon Value 20 ETF
    "NIF100BEES": "29577",  # Nippon Nifty 100 ETF
    "CPSE ETF": "2328",  # CPSE ETF
    "NASDAQ100 ETF": "22739",  # Motilal Oswal Nasdaq 100 ETF
    "LIQUIDBEES": "11006",  # Nippon Liquid ETF
    "DIVOPPBEES": "2636",  # Nippon Dividend Opp ETF
    "KOTAK NIFTY 50": "18102",  # Kotak Nifty 50 ETF
    "SBI NIFTY 50": "10176",  # SBI Nifty 50 ETF
}

FILTERED_FNO_UNIVERSE = {
    # ── Strong symbols (PF > 1.3) ─────────────────────────────────────────
    "RELIANCE": "2885",  # PF 2.29 · WR 67% · +0.83%
    "TCS": "11536",  # PF 1.52 · WR 66% · -0.90%
    "INFY": "1594",  # PF 1.68 · WR 60% · -1.11%
    "HCLTECH": "7229",  # PF 1.87 · WR 67% · -0.07%
    "DRREDDY": "881",  # PF 1.69 · WR 66% · -0.57%
    "MARUTI": "10999",  # PF 1.95 · WR 61% · +0.28%
    "TATASTEEL": "3499",  # PF 1.49 · WR 55% · -2.55%
    "BAJFINANCE": "317",  # PF 1.42 · WR 52% · -2.47%
    "SUNPHARMA": "3351",  # PF 1.35 · WR 60% · -1.28%
    "POWERGRID": "14977",  # PF 1.45 · WR 58% · -1.75%
    # ── Borderline symbols (PF 0.9–1.3) — monitor after each run ─────────
    "BAJAJFINSV": "16675",  # PF 1.28 · WR 61% · -5.81%  — high trade count, watch DD
    "JSWSTEEL": "11723",  # PF 1.36 · WR 54% · -4.08%  — steel trending ok
    "ADANIPORTS": "15083",  # PF 1.17 · WR 57% · -5.16%  — improving, watch
    "WIPRO": "3787",  # PF 1.14 · WR 58% · -3.60%  — IT, close to strong
    "M&M": "2031",  # PF 1.12 · WR 54% · -4.86%  — auto, acceptable
    "SBIN": "3045",  # PF 1.03 · WR 53% · -2.20%  — lowest DD, marginal
    "ONGC": "2475",  # PF 0.99 · WR 48% · -6.37%  — on the edge, review
}
