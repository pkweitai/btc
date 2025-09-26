# save as scripts/backtest_predictions.py (run: python scripts/backtest_predictions.py)
import pandas as pd
from backtest import backtest

pred = pd.read_csv("data/predictions_daily.csv", parse_dates=["asof_timestamp"])
pred = pred.drop_duplicates(subset=["asof_timestamp"], keep="last").set_index("asof_timestamp")

# Load realized next-day return from OHLCV
ohlcv = pd.read_parquet("data/ohlcv.parquet")
ohlcv = ohlcv[ohlcv["symbol"]=="BTC/USDT"].copy()
ohlcv["ts"] = pd.to_datetime(ohlcv["timestamp"]).dt.floor("D")
ohlcv = ohlcv.sort_values("ts").set_index("ts")
ohlcv["ret_next"] = ohlcv["close"].pct_change().shift(-1)

# Signal: simple scaled prediction → position in [-1,1]
S = pred["predicted_next_ret"].clip(-0.02, 0.02) / 0.02  # scale by 2% band
r = ohlcv["ret_next"]

# Align and backtest
S, r = S.align(r, join="inner")
metrics, pnl, cum = backtest(S, r, cost_bps=3.0)
print(metrics)
