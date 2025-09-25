from datetime import datetime, timezone
import pandas as pd, requests

from .utils import append_parquet
from .config import SKIP_BINANCE_DERIVATIVES, USE_OKX_DERIVATIVES, SKIP_BYBIT_DERIVATIVES

# -------- BYBIT (optional) --------
BYBIT_API = "https://api.bybit.com"
BYBIT_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; DataCollector/1.0)"}

def fetch_bybit_funding(symbol: str = "BTCUSDT"):
    if SKIP_BYBIT_DERIVATIVES:
        print("Skipping Bybit funding (SKIP_BYBIT_DERIVATIVES=1).")
        return
    now = datetime.now(timezone.utc)
    url = f"{BYBIT_API}/v5/market/funding/history"
    params = {"category":"linear","symbol":symbol, "limit":200}
    r = requests.get(url, params=params, headers=BYBIT_HEADERS, timeout=30)
    if r.status_code == 403:
        print("Bybit funding 403; set SKIP_BYBIT_DERIVATIVES=1 to silence.")
        r.raise_for_status()
    r.raise_for_status()
    data = r.json().get("result", {}).get("list", [])
    rows = []
    for it in data:
        ts = pd.to_datetime(int(it["fundingRateTimestamp"]), unit="ms", utc=True)
        rows.append({"timestamp": ts, "venue":"bybit", "symbol": symbol,
                     "funding_rate": float(it["fundingRate"]), "ingested_at": now})
    df = pd.DataFrame(rows)
    if not df.empty:
        append_parquet(df, "derivatives_funding.parquet", ["timestamp","venue","symbol"])

def fetch_bybit_open_interest(symbol: str = "BTCUSDT", interval: str = "1d", limit: int = 200):
    if SKIP_BYBIT_DERIVATIVES:
        print("Skipping Bybit open interest (SKIP_BYBIT_DERIVATIVES=1).")
        return
    now = datetime.now(timezone.utc)
    url = f"{BYBIT_API}/v5/market/open-interest"
    params = {"category":"linear","symbol":symbol,"interval":interval,"limit":limit}
    r = requests.get(url, params=params, headers=BYBIT_HEADERS, timeout=30)
    if r.status_code == 403:
        print("Bybit open interest 403; set SKIP_BYBIT_DERIVATIVES=1 to silence.")
        r.raise_for_status()
    r.raise_for_status()
    data = r.json().get("result", {}).get("list", [])
    rows = []
    for it in data:
        ts = pd.to_datetime(int(it["timestamp"]), unit="ms", utc=True)
        rows.append({"timestamp": ts, "venue":"bybit", "symbol": symbol,
                     "period": interval, "open_interest": float(it["openInterest"]),
                     "ingested_at": now})
    df = pd.DataFrame(rows)
    if not df.empty:
        append_parquet(df, "derivatives_open_interest.parquet", ["timestamp","venue","symbol","period"])

# -------- OKX (preferred fallback) --------
OKX_API = "https://www.okx.com/api/v5"

def fetch_okx_funding(instId: str = "BTC-USDT-SWAP", limit: int = 100):
    if not USE_OKX_DERIVATIVES:
        print("Skipping OKX funding (USE_OKX_DERIVATIVES=0).")
        return
    now = datetime.now(timezone.utc)
    url = f"{OKX_API}/public/funding-rate-history"
    params = {"instId": instId, "limit": limit}
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    data = r.json().get("data", [])
    rows = []
    for it in data:
        # OKX returns 'fundingTime' in ms; NOT 'ts'
        # e.g. {'instId':'BTC-USDT-SWAP','fundingRate':'0.0001','fundingTime':'1727222400000'}
        ft = it.get("fundingTime") or it.get("fundingRateTimestamp") or it.get("ts")
        if ft is None:
            continue
        ts = pd.to_datetime(int(ft), unit="ms", utc=True)
        rows.append({"timestamp": ts, "venue":"okx", "symbol": instId,
                     "funding_rate": float(it["fundingRate"]), "ingested_at": now})
    df = pd.DataFrame(rows)
    if not df.empty:
        append_parquet(df, "derivatives_funding.parquet", ["timestamp","venue","symbol"])

def fetch_okx_open_interest(instType: str = "SWAP", instId: str = "BTC-USDT-SWAP"):
    if not USE_OKX_DERIVATIVES:
        print("Skipping OKX open interest (USE_OKX_DERIVATIVES=0).")
        return
    now = datetime.now(timezone.utc)
    url = f"{OKX_API}/public/open-interest"
    params = {"instType": instType, "instId": instId}
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    data = r.json().get("data", [])
    rows = []
    for it in data:
        # OKX OI uses 'ts' (ms) and 'oi'
        if "oi" not in it or "ts" not in it:
            continue
        ts = pd.to_datetime(int(it["ts"]), unit="ms", utc=True)
        rows.append({"timestamp": ts, "venue":"okx", "symbol": instId,
                     "period": "1d", "open_interest": float(it["oi"]),
                     "ingested_at": now})
    df = pd.DataFrame(rows)
    if not df.empty:
        append_parquet(df, "derivatives_open_interest.parquet", ["timestamp","venue","symbol","period"])

# -------- BINANCE (disabled by default) --------
def fetch_binance_funding(*args, **kwargs):
    if SKIP_BINANCE_DERIVATIVES:
        print("Skipping Binance funding (SKIP_BINANCE_DERIVATIVES=1).")
        return

def fetch_binance_open_interest(*args, **kwargs):
    if SKIP_BINANCE_DERIVATIVES:
        print("Skipping Binance OI (SKIP_BINANCE_DERIVATIVES=1).")
        return
