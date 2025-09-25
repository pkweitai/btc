import os, json, joblib, torch, numpy as np, pandas as pd
from datetime import datetime
from lightgbm import Booster
from .features import build_feature_table
from .models_b_seq import GRUHead
from .models_c_regime import infer_regime
from .ensemble import combine
def main():
    X,y,c,feats=build_feature_table(lag_onchain_days=2)
    live=os.path.join('model_registry','live')
    if not os.path.exists(live):
        dirs=sorted([d for d in os.listdir('model_registry') if d!='live'])
        if not dirs: raise FileNotFoundError('No trained models found. Run training first.')
        live=os.path.join('model_registry', dirs[-1])
    scaler=joblib.load(os.path.join(live,'scaler.pkl')); Xs=pd.DataFrame(scaler.transform(X), index=X.index, columns=X.columns)
    reg=Booster(model_file=os.path.join(live,'lgbm_regressor.txt')); clf=Booster(model_file=os.path.join(live,'lgbm_classifier.txt')); cal=joblib.load(os.path.join(live,'calibrator.pkl'))
    yA=reg.predict(Xs.iloc[[-1]]); pA=cal.transform(clf.predict(Xs.iloc[[-1]], pred_contrib=False, raw_score=False))
    spec=json.load(open(os.path.join(live,'sequence_spec.json'))); win=spec['win']; in_dim=len(spec['feature_order'])
    modelB=GRUHead(in_dim=in_dim); modelB.load_state_dict(torch.load(os.path.join(live,'gru_model.pt'), map_location='cpu'))
    Xwin=Xs.iloc[-win:].values
    with torch.no_grad(): yB,pB=modelB(torch.tensor(Xwin[None,...], dtype=torch.float32))
    yB=float(yB.squeeze().numpy()); pB=float(pB.squeeze().numpy())
    regime=infer_regime(feats.iloc[[-1]])
    S,y_bl,p_bl=combine(np.array([yA[0]]), np.array([pA[0]]), np.array([yB]), np.array([pB]), feats.iloc[-1:]['vol20'].values, regime.tail(1))
    out={'as_of':X.index[-1].isoformat(),'signal_S':float(S[0]),'y_blend':float(y_bl[0]),'p_blend':float(p_bl[0]),'regime':regime.tail(1).to_dict(orient='records')[0]}
    os.makedirs('signals', exist_ok=True); fname=os.path.join('signals', f"{datetime.utcnow().strftime('%Y-%m-%d')}_signal.json")
    json.dump(out, open(fname,'w'), indent=2); print(json.dumps(out, indent=2))
if __name__=='__main__': main()
