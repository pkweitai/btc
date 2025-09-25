from datetime import datetime, timezone
import pandas as pd, ccxt
from .utils import append_parquet
from .config import USE_BINANCE_US_SPOT

# in sources_market_full.py
EXCHANGES = {
    "binanceus": lambda: ccxt.binanceus(),   # <— instead of ccxt.binance()
    "coinbase":  lambda: ccxt.coinbase(),
    "kraken":    lambda: ccxt.kraken(),
    "bitstamp":  lambda: ccxt.bitstamp(),
}
SYMBOLS = {
    "binanceus": "BTC/USDT",  # or BTC/USD depending on availability
    "coinbase":  "BTC/USD",
    "kraken":    "BTC/USD",
    "bitstamp":  "BTC/USD",
}


def fetch_multi_ohlcv_daily(limit=1500):
    now = datetime.now(timezone.utc)
    rows = []
    for name, factory in EXCHANGES.items():
        try:
            ex = factory()
            sym = SYMBOLS[name]
            ohlcv = ex.fetch_ohlcv(sym, timeframe="1d", limit=limit)
            for ts,o,h,l,c,v in ohlcv:
                rows.append({"timestamp": pd.to_datetime(ts, unit="ms", utc=True),
                             "exchange": name, "symbol": sym, "open": float(o),
                             "high": float(h), "low": float(l), "close": float(c),
                             "volume": float(v), "ingested_at": now})
        except Exception as e:
            print(f"{name} OHLCV error:", e)
    df = pd.DataFrame(rows)
    if not df.empty:
        append_parquet(df, "ohlcv.parquet", ["timestamp","exchange","symbol"])

def fetch_orderbook_snapshots(depth=10):
    """Capture a *daily snapshot* of L2 order book per exchange (top N levels).
    Stores: best bid/ask, spread, sum top-N bid/ask sizes.
    """
    now = datetime.now(timezone.utc)
    rows = []
    for name, factory in EXCHANGES.items():
        try:
            ex = factory()
            sym = SYMBOLS[name]
            ob = ex.fetch_order_book(sym, limit=depth)
            bids = ob.get("bids", [])[:depth]
            asks = ob.get("asks", [])[:depth]
            best_bid = bids[0][0] if bids else None
            best_ask = asks[0][0] if asks else None
            spread = (best_ask - best_bid) if (best_ask and best_bid) else None
            sum_bid_size = sum([b[1] for b in bids]) if bids else 0.0
            sum_ask_size = sum([a[1] for a in asks]) if asks else 0.0
            rows.append({
                "timestamp": now,
                "exchange": name, "symbol": sym,
                "best_bid": best_bid, "best_ask": best_ask, "spread": spread,
                "sum_bid_size_topN": sum_bid_size, "sum_ask_size_topN": sum_ask_size,
                "levels": depth, "ingested_at": now
            })
        except Exception as e:
            print(f"{name} orderbook error:", e)
    df = pd.DataFrame(rows)
    if not df.empty:
        append_parquet(df, "orderbook_snapshots.parquet", ["timestamp","exchange","symbol"])
