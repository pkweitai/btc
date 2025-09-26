
import numpy as np
import pandas as pd

def sharpe(returns, eps: float = 1e-12, periods_per_year: int = 252) -> float:
    r = np.asarray(returns, dtype=float)
    if r.size == 0:
        return 0.0
    mu = r.mean()
    sd = r.std(ddof=1) + eps
    return float((mu / sd) * np.sqrt(periods_per_year))

def max_drawdown(cum_equity: np.ndarray) -> float:
    peak = -1e18
    mdd = 0.0
    for v in np.asarray(cum_equity, dtype=float):
        peak = max(peak, v)
        mdd = min(mdd, v - peak)
    return float(mdd)

def ema(x: pd.Series, span: int) -> pd.Series:
    return pd.Series(x, index=x.index, dtype=float).ewm(span=span, adjust=False).mean()

def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    c = pd.Series(close, index=close.index, dtype=float)
    delta = c.diff()
    gain = (delta.clip(lower=0)).ewm(alpha=1/period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/period, adjust=False).mean()
    rs = gain / (loss + 1e-12)
    return 100 - (100 / (1 + rs))

def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    fast_ema = ema(close, fast)
    slow_ema = ema(close, slow)
    macd_line = fast_ema - slow_ema
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def zscore(x: pd.Series, win: int) -> pd.Series:
    r = pd.Series(x, index=x.index, dtype=float)
    m = r.rolling(win, min_periods=win//2).mean()
    s = r.rolling(win, min_periods=win//2).std(ddof=0)
    return (r - m) / (s + 1e-12)

def rolling_vol(ret: pd.Series, win: int = 20) -> pd.Series:
    return pd.Series(ret, index=ret.index, dtype=float).rolling(win, min_periods=win//2).std(ddof=0)
