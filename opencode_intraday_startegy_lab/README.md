# Intraday Strategy Lab

Python research lab for downloading Dhan historical data, selecting 20 high-beta NSE stocks, testing 20 intraday strategies, optimizing parameters on in-sample data, and validating frozen parameters on out-of-sample data.

## Quick Start

```powershell
cd opencode_intraday_startegy_lab
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```

Add your Dhan credentials to `.env`:

```text
DHAN_CLIENT_ID=your_client_id
DHAN_ACCESS_TOKEN=your_access_token
```

Download one year of data into the dedicated Dhan folder:

```powershell
python -m intraday_strategy_lab download-data
```

Run optimization and in-sample/out-of-sample testing:

```powershell
python -m intraday_strategy_lab run
```

Run rolling walk-forward optimization and validation:

```powershell
python -m intraday_strategy_lab walk-forward
```

Run a local synthetic-data smoke test without Dhan credentials:

```powershell
python -m intraday_strategy_lab smoke-test
```

## Outputs

- Raw Dhan responses: `dhan_historical_data/raw/`
- Clean candles: `dhan_historical_data/processed/`
- Universe metadata: `dhan_historical_data/metadata/`
- Optimization results: `results/optimization/`
- In-sample reports: `results/in_sample/`
- Out-of-sample reports: `results/out_of_sample/`
- Combined reports: `results/combined_reports/`
- Walk-forward reports: `results/walk_forward/`

## Research Rules

- Signals use only completed candles.
- Entries execute on the next candle open.
- All positions are squared off intraday.
- In-sample data is used for optimization only.
- Out-of-sample data is evaluated once with frozen parameters.
- Transaction costs and slippage are included.
- If stop and target both touch in the same candle, the backtester assumes the adverse fill first.

## Strategy Count

The registry contains 40 strategies: the original 20 plus a second 20-strategy `b3_*` research batch. List them with:

```powershell
python -m intraday_strategy_lab list-strategies
```

## Notes

The Dhan historical API requires valid credentials and instrument mappings. The downloader stores raw API JSON separately from cleaned CSV candles so results are auditable and repeatable.
