
import pandas as pd
import numpy as np

def infer_regime(feats: pd.DataFrame) -> pd.DataFrame:
    ema20 = feats.get("ema_20")
    ema50 = feats.get("ema_50")
    if ema20 is None or ema50 is None:
        # build simple EMAs if missing
        close = feats["close"].astype(float)
        ema20 = close.ewm(span=20, adjust=False).mean()
        ema50 = close.ewm(span=50, adjust=False).mean()
    slope = ema20 - ema20.shift(5)
    up = (slope > 0) & (feats["close"] > ema50)
    down = (slope < 0) & (feats["close"] < ema50)
    p_up = (up.rolling(5, min_periods=1).mean()).clip(0,1)
    p_down = (down.rolling(5, min_periods=1).mean()).clip(0,1)
    p_chop = (1 - (p_up + p_down)).clip(0,1)
    return pd.DataFrame({"p_up": p_up, "p_down": p_down, "p_chop": p_chop}, index=feats.index)
