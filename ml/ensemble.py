
import numpy as np
import pandas as pd

def combine(yA, pA, yB, pB, vol20, regime_probs, wA=0.55, wB=0.45, alpha=0.5, threshold=0.15):
    yA = np.asarray(yA, dtype=float)
    yB = np.asarray(yB, dtype=float)
    pA = np.asarray(pA, dtype=float)
    pB = np.asarray(pB, dtype=float)
    vol = np.asarray(vol20, dtype=float)
    p_chop = np.asarray(regime_probs.get("p_chop"), dtype=float)
    y_bl = wA * yA + wB * yB
    p_bl = np.clip(wA * pA + wB * pB, 1e-4, 1-1e-4)
    zret = y_bl / (vol + 1e-12)
    score = alpha * (2*p_bl - 1.0) + (1 - alpha) * zret
    gate = (1.0 - 0.6 * p_chop)
    S = gate * score
    S[np.abs(S) < threshold] = 0.0
    S = np.clip(S, -1.0, 1.0)
    return S, y_bl, p_bl
