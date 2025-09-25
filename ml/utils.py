import numpy as np
def sharpe(returns, eps=1e-12):
    r = np.asarray(returns)
    if r.size == 0: return 0.0
    mu, sd = r.mean(), r.std(ddof=1)+eps
    return (mu/sd)*np.sqrt(252.0)
def max_drawdown(cum):
    peak=-1e18; mdd=0.0
    for v in cum:
        peak=max(peak,v); mdd=min(mdd, v-peak)
    return mdd
def to_logret(s): s=s.astype(float); import numpy as np; return np.log(s/s.shift(1))
def zscore(x, win): return (x - x.rolling(win).mean()) / (x.rolling(win).std(ddof=0)+1e-12)
def rolling_vol(r, win=20): return r.rolling(win).std(ddof=0)
def ema(x, span): return x.ewm(span=span, adjust=False).mean()
def rsi(close, period=14):
    d=close.diff(); up=d.clip(lower=0); dn=-d.clip(upper=0)
    ma_up=up.ewm(alpha=1/period, adjust=False).mean()
    ma_dn=dn.ewm(alpha=1/period, adjust=False).mean()
    rs=ma_up/(ma_dn+1e-12); return 100-(100/(1+rs))
def macd(close, fast=12, slow=26, signal=9):
    ef=ema(close, fast); es=ema(close, slow)
    ml=ef-es; sig=ema(ml, signal); hist=ml-sig
    return ml, sig, hist
