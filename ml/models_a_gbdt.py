
from __future__ import annotations
import numpy as np
from typing import Tuple, List, Dict, Any
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import mean_squared_error, log_loss

def train_gbdt(X_tr, y_tr, c_tr, X_va, y_va, c_va) -> Tuple[Any, Any, IsotonicRegression, Dict[str,float]]:
    reg = GradientBoostingRegressor(random_state=42)
    clf = GradientBoostingClassifier(random_state=42)
    reg.fit(X_tr, y_tr)
    clf.fit(X_tr, c_tr.astype(int))
    # calibrate
    p_va_raw = clf.predict_proba(X_va)[:,1]
    cal = IsotonicRegression(out_of_bounds="clip").fit(p_va_raw, c_va.astype(int))
    # metrics
    rmse = float(np.sqrt(mean_squared_error(y_va, reg.predict(X_va))))
    logloss = float(log_loss(c_va.astype(int), cal.predict(p_va_raw).clip(1e-6,1-1e-6)))
    return reg, clf, cal, {"rmse": rmse, "logloss": logloss}
