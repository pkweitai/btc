
# BTC Full Collector + ML

This is the **full** data collector (not minimal) and ML bundle:
- Multi-exchange **OHLCV**: Binance, Coinbase, Kraken, Bitstamp
- **Order book snapshots (L2)**: best bid/ask, spread, top-10 depth sums (per exchange)
- **Derivatives**:
  - Binance: funding history, open interest history
  - Bybit: funding history, open interest (1d interval)
- **On-chain proxies** (free): blockchain.com charts (txn count, hash rate, difficulty)
- **On-chain (optional)**: Glassnode endpoints (with `GLASSNODE_API_KEY`)
- **Sentiment**: Alternative.me Fear & Greed, optional Google Trends
- Full **ML pipeline** (GBDT + GRU + regime + backtest + daily inference)

## Quick start
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp config.example.env .env   # set DATA_DIR; add GLASSNODE_API_KEY if you have one

# Collect full daily dataset (creates Parquets under ./data)
python -m data_pipeline.daily_update

# Train / validate / save models
python -m ml.train_walkforward

# Produce today's signal
python -m ml.predict_daily
```
