import numpy as np, pandas as pd
def infer_regime(feats):
    df=feats.copy()
    ema20=df['ema20']; slope=ema20-ema20.shift(5)
    up=((slope>0)&(df['close']>df['ema50'])).astype(float)
    down=((slope<0)&(df['close']<df['ema50'])).astype(float)
    chop=1.0-np.maximum(up,down)
    p_up=0.7*up+0.15*chop; p_down=0.7*down+0.15*chop; p_chop=1.0-(p_up+p_down)
    return pd.DataFrame({'p_up':p_up,'p_down':p_down,'p_chop':p_chop}, index=df.index).clip(0,1)
