import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import log_loss, mean_squared_error
from lightgbm import LGBMRegressor, LGBMClassifier
def train_gbdt(X_tr,y_tr,c_tr,X_va,y_va,c_va,params_reg=None,params_clf=None):
    params_reg=params_reg or dict(n_estimators=700,learning_rate=0.03,max_depth=-1,subsample=0.8,colsample_bytree=0.9)
    params_clf=params_clf or dict(n_estimators=700,learning_rate=0.03,max_depth=-1,subsample=0.8,colsample_bytree=0.9)
    reg=LGBMRegressor(**params_reg); clf=LGBMClassifier(**params_clf)
    reg.fit(X_tr,y_tr); clf.fit(X_tr,c_tr)
    yhat=reg.predict(X_va); ph_raw=clf.predict_proba(X_va)[:,1]
    calibrator=IsotonicRegression(out_of_bounds='clip'); calibrator.fit(ph_raw, c_va); ph=calibrator.transform(ph_raw)
    metrics={'rmse':float(np.sqrt(mean_squared_error(y_va,yhat))),'logloss':float(log_loss(c_va,ph,eps=1e-9))}
    return reg,clf,calibrator,metrics
