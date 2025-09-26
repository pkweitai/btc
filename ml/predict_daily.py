# predict_daily.py

from __future__ import annotations

# --- bootstrap so this file can be run as a script or as a module ---
if __package__ in (None, "",):
    import os, sys
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
# --------------------------------------------------------------------

import os
import argparse
from datetime import datetime, timezone
import pickle

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

try:
    from dotenv import load_dotenv
except Exception:
    load_dotenv = None

# relative imports work when run as a package; the bootstrap above helps when run as a script
from .features import load_ohlcv, build_basic_features
from .models_c_regime import infer_regime
from .ensemble import combine


def prepare_X_for_model(model, X_row: pd.DataFrame, trained_feats: list | None):
    """
    Returns a 2D numpy array with the exact number of features the model expects.
    - If trained_feats provided: reindex X_row to those names (add missing=0.0) and return .values
    - Else: pad/trim X_row.values to model.n_features_in_
    """
    import numpy as np
    n_req = getattr(model, "n_features_in_", None)

    if trained_feats:
        # add any missing columns with 0.0
        for col in trained_feats:
            if col not in X_row.columns:
                X_row[col] = 0.0
        # order exactly as trained; ignore extras
        X_ordered = X_row[trained_feats]
        arr = X_ordered.to_numpy(dtype=float)
        if n_req is not None and arr.shape[1] != n_req:
            # pad/trim to satisfy the estimator (handles cases where trained_feats list is stale)
            if arr.shape[1] < n_req:
                pad = np.zeros((arr.shape[0], n_req - arr.shape[1]), dtype=float)
                arr = np.concatenate([arr, pad], axis=1)
            else:
                arr = arr[:, :n_req]
        return arr

    # No trained feature names available — fall back to numeric shape compliance
    arr = X_row.to_numpy(dtype=float)
    if n_req is not None and arr.shape[1] != n_req:
        if arr.shape[1] < n_req:
            pad = np.zeros((arr.shape[0], n_req - arr.shape[1]), dtype=float)
            arr = np.concatenate([arr, pad], axis=1)
        else:
            arr = arr[:, :n_req]
    return arr



def log(msg: str) -> None:
    print(f"[predict_daily] {msg}")


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def latest_model_path(reg_dir: str) -> str | None:
    """Pick the newest timestamped folder under registry that contains model.pkl"""
    if not reg_dir or not os.path.exists(reg_dir):
        return None
    subs = [d for d in os.listdir(reg_dir) if os.path.isdir(os.path.join(reg_dir, d))]
    if not subs:
        return None
    subs.sort()  # assumes YYYYMMDD_HHMMSS lexicographic order
    cand = os.path.join(reg_dir, subs[-1], "model.pkl")
    return cand if os.path.exists(cand) else None


def save_prediction(data_dir: str, symbol: str, y_pred: float, model_path: str, asof_ts: str, generated_at: str) -> str:
    out = os.path.join(data_dir, "predictions_daily.csv")
    row = {
        "generated_at_utc": generated_at,
        "symbol": symbol,
        "asof_timestamp": asof_ts,
        "predicted_next_ret": y_pred,
        "model_path": model_path,
    }
    df = pd.DataFrame([row])
    if os.path.exists(out):
        df.to_csv(out, mode="a", header=False, index=False)
    else:
        ensure_dir(os.path.dirname(out) or ".")
        df.to_csv(out, index=False)
    log(f"Appended prediction -> {out}")
    return out


def feature_engineer_latest(data_dir: str, symbol: str | None = None) -> pd.DataFrame:
    ohlcv = load_ohlcv(data_dir, symbol)
    feat = build_basic_features(ohlcv)
    return feat


def build_X_latest(feat_df: pd.DataFrame):
    # prefer snake-case names commonly used in our baseline
    drop_cols = {"timestamp", "symbol", "target_next_ret"}
    feat_cols = [c for c in feat_df.columns if c not in drop_cols]
    latest = feat_df.dropna().tail(1).copy()
    if latest.empty:
        raise ValueError("No valid feature row available for prediction (insufficient history).")
    X_row = latest[feat_cols].copy()  # keep as DataFrame for easy reindexing later
    meta = {
        "timestamp": pd.to_datetime(latest["timestamp"].iloc[0]),
        "symbol": str(latest.get("symbol", pd.Series(["NA"])).iloc[0]),
        "close": float(latest.get("close", pd.Series([np.nan])).iloc[0]),
    }
    return X_row, feat_cols, meta


def auto_train_and_register(feat_df: pd.DataFrame, reg_dir: str) -> str:
    drop_cols = {"timestamp", "symbol", "target_next_ret"}
    feat_cols = [c for c in feat_df.columns if c not in drop_cols]
    hist = feat_df.dropna()
    if hist.empty:
        raise ValueError("Not enough rows to auto-train (all-NaN after feature building).")
    X = hist[feat_cols].values
    y = hist["target_next_ret"].values
    model = RandomForestRegressor(n_estimators=200, max_depth=6, random_state=42, n_jobs=-1)
    model.fit(X, y)

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    ver_dir = ensure_dir(os.path.join(reg_dir, ts))
    model_path = os.path.join(ver_dir, "model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump({"model": model, "features": feat_cols}, f)
    log(f"Auto-trained baseline model -> {model_path}")
    return model_path


def parse_args():
    ap = argparse.ArgumentParser(description="Daily prediction using latest model in model registry.")
    ap.add_argument("--data-dir", default=None, help="Override DATA_DIR (.env)")
    ap.add_argument("--registry", default=None, help="Override model registry dir (default ./model_registry or $MODEL_REGISTRY)")
    ap.add_argument("--symbol", default=None, help="Specific symbol (if present in OHLCV)")
    ap.add_argument("--auto-train", action="store_true", help="If no model found, auto-train a quick baseline model.")
    ap.add_argument("--with-ensemble", action="store_true", help="Also compute a regime-gated score using a simple self-ensemble.")
    ap.add_argument("--model-path", default=None, help="Direct path to a model.pkl (overrides --registry if provided).")
    return ap.parse_args()


def _resolve_model_path(args, reg_dir: str) -> str | None:
    """Respect --model-path first, else pick latest from registry."""
    if args.model_path:
        return args.model_path if os.path.exists(args.model_path) else None
    return latest_model_path(reg_dir)


def main():
    if load_dotenv:
        load_dotenv(os.environ.get("ENV_FILE", ".env"))

    args = parse_args()
    data_dir = args.data_dir or os.environ.get("DATA_DIR", "./data")
    reg_dir = args.registry or os.environ.get("MODEL_REGISTRY", "./model_registry")
    symbol = args.symbol or os.environ.get("SYMBOL", None)

    log(f"DATA_DIR={data_dir}")
    log(f"MODEL_REGISTRY={reg_dir}")

    # Build latest features
    feat = feature_engineer_latest(data_dir, symbol)
    X_row, feat_cols, meta = build_X_latest(feat)

    # Resolve or auto-train model
    model_path = _resolve_model_path(args, reg_dir)
    if model_path is None and args.auto_train:
        ensure_dir(reg_dir)
        model_path = auto_train_and_register(feat, reg_dir)
    if model_path is None:
        raise FileNotFoundError("No model found. Pass --auto-train or provide --model-path or --registry with at least one model.")

    with open(model_path, "rb") as f:
        payload = pickle.load(f)
    model = payload["model"]
    trained_feats = payload.get("features")

    # --- Robust feature parity: add missing cols (0.0), order exactly as trained, drop extras ---
    if trained_feats:
        for col in trained_feats:
            if col not in X_row.columns:
                X_row[col] = 0.0
        # order to match training; extra cols are ignored
        X_row = X_row[trained_feats]
    # ---------------------------------------------------------------------------------------------

    # Predict
    asof_ts_iso = pd.to_datetime(meta["timestamp"]).strftime("%Y-%m-%dT%H:%M:%SZ")
    sym_val = str(meta.get("symbol", "NA"))
    X_mat = prepare_X_for_model(model, X_row, trained_feats)
    y_pred = float(model.predict(X_mat)[0])

    now_iso = utcnow_iso()
    log(f"PREDICTION for {sym_val} @ {asof_ts_iso} -> next-day return ~ {y_pred:.6f}")
    save_prediction(data_dir, sym_val, y_pred, model_path, asof_ts_iso, now_iso)

    # Optional: ensemble/regime-gated score for context
    if args.with_ensemble:
        hist = feat.dropna().copy()
        if hist.empty:
            log("Ensemble skipped: not enough history.")
            return

        # Choose a return series for volatility gating
        if "ret_1" in hist.columns:
            ret_series = hist["ret_1"]
        elif "ret1" in hist.columns:
            ret_series = hist["ret1"]
        else:
            # fallback: simple daily return from close
            ret_series = hist["close"].pct_change().fillna(0.0)

        # build X_hist to match the model's expected shape
        usable_feats = [c for c in hist.columns if c not in {"timestamp","symbol","target_next_ret"}]
        X_hist_df = hist[trained_feats] if trained_feats else hist[usable_feats]
        # Reuse the same preparation for the primary model's shape
        X_hist = prepare_X_for_model(model, X_hist_df, trained_feats)

        y_hist = hist["target_next_ret"].values
        m2 = RandomForestRegressor(n_estimators=120, max_depth=4, random_state=7, n_jobs=-1).fit(X_hist, y_hist)

        yA = model.predict(X_hist)
        yB = m2.predict(X_hist)

        # crude probabilities from sign (mapped into [0.3, 0.7] band to avoid overconfidence)
        pA = (np.sign(yA) + 1) / 2 * 0.4 + 0.3
        pB = (np.sign(yB) + 1) / 2 * 0.4 + 0.3

        # realized vol proxy + regime for gating
        from .utils import rolling_vol
        vol20 = rolling_vol(ret_series, 20).values
        reg = infer_regime(hist)

        S, y_bl, p_bl = combine(yA, pA, yB, pB, vol20, reg)
        last_S = float(S[-1])
        log(f"Ensemble score (regime-gated) for {sym_val} as of {asof_ts_iso}: {last_S:.4f}")


if __name__ == "__main__":
    main()
