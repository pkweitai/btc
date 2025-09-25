import numpy as np
from .utils import sharpe, max_drawdown
def backtest(signals, returns, cost_bps=3.0):
    s=np.asarray(signals); r=np.asarray(returns)
    s_prev=np.concatenate([[0.0], s[:-1]]); turnover=np.abs(s-s_prev)
    costs=(cost_bps/1e4)*turnover; pnl=s*r - costs; cum=np.cumsum(pnl)
    return {'sharpe':sharpe(pnl),'return':float(np.sum(pnl)),'avg_daily':float(np.mean(pnl)),
            'stdev_daily':float(np.std(pnl,ddof=1)),'max_dd':float(max_drawdown(cum)),'turnover':float(np.mean(turnover))}, pnl, cum
