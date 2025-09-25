import os, json, joblib, numpy as np, pandas as pd, torch
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from lightgbm import Booster
from .features import build_feature_table
from .models_a_gbdt import train_gbdt
from .models_b_seq import train_gru, make_sequences, GRUHead
from .models_c_regime import infer_regime
from .ensemble import combine
from .backtest import backtest
def main():
    X,y,c,feats=build_feature_table(lag_onchain_days=2)
    tss=TimeSeriesSplit(n_splits=6)
    results=[]
    for i,(tr,va) in enumerate(tss.split(X),1):
        Xtr,Xva=X.iloc[tr],X.iloc[va]; ytr,yva=y.iloc[tr],y.iloc[va]; ctr,cva=c.iloc[tr],c.iloc[va]
        scaler=StandardScaler().fit(Xtr); Xtr_s=pd.DataFrame(scaler.transform(Xtr),index=Xtr.index,columns=X.columns); Xva_s=pd.DataFrame(scaler.transform(Xva),index=Xva.index,columns=X.columns)
        reg,clf,cal,metrA=train_gbdt(Xtr_s,ytr,ctr,Xva_s,yva,cva)
        yA=reg.predict(Xva_s); pA=cal.transform(clf.predict_proba(Xva_s)[:,1])
        modelB,metrB,win=train_gru(Xtr_s,ytr,ctr,Xva_s,yva,cva,win=60,epochs=12)
        Xall=pd.concat([Xtr_s,Xva_s]); vaX,_,_=make_sequences(Xall, pd.concat([ytr,yva]), pd.concat([ctr,cva]), win)
        seq_idx=np.arange(win,len(Xall)); mask_va=seq_idx>=len(Xtr_s); vaX_seq=vaX[mask_va]
        yB=[]; pB=[]
        for j in range(len(vaX_seq)):
            with torch.no_grad():
                yhat,phat=modelB(torch.tensor(vaX_seq[j:j+1],dtype=torch.float32))
            yB.append(float(yhat.squeeze())); pB.append(float(phat.squeeze()))
        yB=np.array(yB); pB=np.array(pB)
        regime=infer_regime(feats.loc[Xva.index])
        S,y_bl,p_bl=combine(yA,pA,yB,pB,feats.loc[Xva.index,'vol20'].values, regime)
        metrics,pnl,cum=backtest(S, yva.values, cost_bps=3.0)
        results.append({'fold':i,'A':metrA,'B':metrB,'bt':metrics})
    summary={'cv_sharpe_mean':float(np.mean([r['bt']['sharpe'] for r in results])),'cv_sharpe_std':float(np.std([r['bt']['sharpe'] for r in results])),'folds':results}
    date_str=datetime.utcnow().strftime('%Y-%m-%d'); outdir=os.path.join('model_registry',date_str); os.makedirs(outdir, exist_ok=True)
    scaler=StandardScaler().fit(X); Xs=pd.DataFrame(scaler.transform(X), index=X.index, columns=X.columns)
    reg,clf,cal,_=train_gbdt(Xs,y,c,Xs,y,c)
    joblib.dump(scaler, os.path.join(outdir,'scaler.pkl'))
    reg.booster_.save_model(os.path.join(outdir,'lgbm_regressor.txt')); clf.booster_.save_model(os.path.join(outdir,'lgbm_classifier.txt'))
    joblib.dump(cal, os.path.join(outdir,'calibrator.pkl'))
    modelB,_,win=train_gru(Xs,y,c,Xs,y,c,win=60,epochs=12); torch.save(modelB.state_dict(), os.path.join(outdir,'gru_model.pt'))
    with open(os.path.join(outdir,'sequence_spec.json'),'w') as f: json.dump({'win':win,'feature_order':list(X.columns)}, f, indent=2)
    with open(os.path.join(outdir,'ensemble.json'),'w') as f: json.dump({'weights':{'A':0.55,'B':0.45},'alpha':0.5,'threshold':0.15}, f, indent=2)
    with open(os.path.join(outdir,'cv_summary.json'),'w') as f: json.dump(summary, f, indent=2)
    try:
        live=os.path.join('model_registry','live')
        if os.path.islink(live) or os.path.exists(live): os.remove(live)
        os.symlink(date_str, live, target_is_directory=True)
    except Exception as e:
        print('Symlink update failed:', e)
    print(json.dumps(summary, indent=2))
if __name__=='__main__': main()
