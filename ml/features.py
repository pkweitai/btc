
import os
import pandas as pd
import numpy as np
from datetime import datetime
from .utils import ema, rsi, macd, zscore

def _normalize_symbol_column(df: pd.DataFrame) -> pd.DataFrame:
    if "symbol" not in df.columns:
        df["symbol"] = "ASSET"
    return df

def load_ohlcv(data_dir: str, symbol: str | None = None) -> pd.DataFrame:
    path = os.path.join(data_dir, "ohlcv.parquet")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing {path}")
    df = pd.read_parquet(path)
    if "timestamp" not in df.columns:
        raise ValueError("ohlcv.parquet must have 'timestamp' column.")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp")
    df = _normalize_symbol_column(df)
    if symbol is not None and "symbol" in df.columns:
        df = df[df["symbol"] == symbol]
    if "close" not in df.columns:
        raise ValueError("ohlcv.parquet must have 'close' column.")
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df.dropna(subset=["close"])
    return df

def build_basic_features(ohlcv: pd.DataFrame) -> pd.DataFrame:
    work = ohlcv.copy()
    work["ret_1"] = work["close"].pct_change(1)
    work["ret_5"] = work["close"].pct_change(5)
    work["ret_10"] = work["close"].pct_change(10)
    work["roll_mean_5"] = work["close"].rolling(5, min_periods=2).mean()
    work["roll_std_5"] = work["close"].rolling(5, min_periods=2).std()
    work["roll_mean_20"] = work["close"].rolling(20, min_periods=5).mean()
    work["roll_std_20"] = work["close"].rolling(20, min_periods=5).std()
    work["ema_10"] = ema(work["close"], 10)
    work["ema_20"] = ema(work["close"], 20)
    work["ema_50"] = ema(work["close"], 50)
    work["dist_ema20_z"] = (work["close"] - work["ema_20"]) / (work["ema_20"].rolling(20, min_periods=10).std(ddof=0)+1e-12)
    work["rsi_14"] = rsi(work["close"], 14)
    macd_line, sig, hist = macd(work["close"])
    work["macd"] = macd_line
    work["macd_sig"] = sig
    work["macd_hist"] = hist
    work["target_next_ret"] = work["close"].pct_change(1).shift(-1)
    cols = ["timestamp", "symbol", "close",
            "ret_1","ret_5","ret_10",
            "roll_mean_5","roll_std_5","roll_mean_20","roll_std_20",
            "ema_10","ema_20","ema_50","dist_ema20_z","rsi_14","macd","macd_sig","macd_hist",
            "target_next_ret"]
    return work[cols].dropna(subset=["ret_10"])
