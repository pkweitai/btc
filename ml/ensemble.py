import numpy as np
def combine(yA,pA,yB,pB,vol20,regime_probs,wA=0.55,wB=0.45,alpha=0.5,threshold=0.15):
    y_bl=wA*yA+wB*yB; p_bl=np.clip(wA*pA+wB*pB,1e-4,1-1e-4)
    zret=y_bl/(vol20+1e-12); score=alpha*(2*p_bl-1.0)+(1-alpha)*zret
    g=(1.0-0.6*regime_probs['p_chop'].values); S=g*score
    S[np.abs(S)<threshold]=0.0; S=np.clip(S,-1.0,1.0)
    return S,y_bl,p_bl
