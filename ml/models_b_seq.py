
# Placeholder GRU-free sequence model to avoid torch dependency.
# We emulate a sequence learner by adding rolling-window aggregated features
# and training a simple gradient boosting on those "sequence summaries".
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.metrics import mean_squared_error, log_loss

def make_sequence_summaries(X_df: pd.DataFrame, win: int = 60) -> pd.DataFrame:
    # rolling mean and std over window for each feature
    out = X_df.copy()
    for col in X_df.columns:
        out[f"{col}_mean{win}"] = X_df[col].rolling(win, min_periods=win//2).mean()
        out[f"{col}_std{win}"]  = X_df[col].rolling(win, min_periods=win//2).std()
    return out.dropna()

def train_seq(X_tr_df, y_tr, c_tr, X_va_df, y_va, c_va, win: int = 60):
    X_tr_s = make_sequence_summaries(pd.DataFrame(X_tr_df, columns=X_tr_df.columns if hasattr(X_tr_df,'columns') else None), win)
    valid_cut = len(X_tr_df) - len(X_tr_s)
    y_tr = y_tr[valid_cut:]
    c_tr = c_tr[valid_cut:]
    X_va_s = make_sequence_summaries(pd.DataFrame(X_va_df, columns=X_va_df.columns if hasattr(X_va_df,'columns') else None), win)
    reg = GradientBoostingRegressor(random_state=1).fit(X_tr_s, y_tr)
    clf = GradientBoostingClassifier(random_state=1).fit(X_tr_s, c_tr.astype(int))
    # eval
    rmse = float(np.sqrt(mean_squared_error(y_va[-len(X_va_s):], reg.predict(X_va_s))))
    logloss = float(log_loss(c_va[-len(X_va_s):].astype(int), clf.predict_proba(X_va_s)[:,1]))
    return {"reg": reg, "clf": clf, "win": win, "rmse": rmse, "logloss": logloss}
