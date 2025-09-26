
# BTC Intelligence Dashboard

A lightweight Flask app to visualize your collected datasets (OHLCV, derivatives, on‑chain, Fear & Greed) with an elegant blue‑gradient UI.

## Quick start

```bash
# from this folder
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Point to your data directory (defaults to ./data)
export DATA_DIR="/Users/rexching/app/service/btc/data"

# run
python app.py
# then open http://localhost:8000
```

## Endpoints

- `/` – Dashboard UI
- `/api/summary` – Presence + latest timestamps
- `/api/ohlcv?symbol=BTCUSDT&last_n=365` – Close series
- `/api/funding?exchange=OKX&instrument=BTCUSDT-PERP` – Funding rate series
- `/api/open_interest?exchange=OKX&instrument=BTCUSDT-PERP` – OI series
- `/api/onchain?source=onchain_blockchair.parquet&metric=transactions` – On‑chain series
- `/api/bitnodes` – Bitnodes series
- `/api/fng` – Fear & Greed series

## Notes

- Make sure your parquet files exist in `DATA_DIR` (see `/api/summary` in the UI).
- Order book depth tile currently shows a placeholder plot; you can extend the backend to parse your `orderbook_snapshots.parquet` schema once finalized.


## Configure with `.env`
Create a `.env` file in the project root (or set `ENV_FILE=/path/to/.env`) with:
```
DATA_DIR=/Users/rexching/app/service/btc/data
PORT=8000
DEBUG=1
```
Then simply run:
```bash
python app.py
```
