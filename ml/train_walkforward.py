# ml/train_walkforward.py

from __future__ import annotations

import os
import argparse
import warnings
import pickle
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

try:
    from dotenv import load_dotenv
except Exception:
    load_dotenv = None

from .features import load_ohlcv, build_basic_features


def log(msg: str) -> None:
    print(f"[train_walkforward] {msg}")


def ensure_parent_dir(path: str) -> None:
    """Create the parent directory of a file path if it doesn't exist."""
    d = os.path.dirname(os.path.abspath(path))
    if d:
        os.makedirs(d, exist_ok=True)


def feature_engineer(
    data_dir: str,
    symbol: str | None = None,
    start: str | None = None,
    end: str | None = None,
    max_samples: int | None = None,
) -> pd.DataFrame:
    df = load_ohlcv(data_dir, symbol)
    if start:
        df = df[df["timestamp"] >= pd.to_datetime(start, utc=True)]
    if end:
        df = df[df["timestamp"] <= pd.to_datetime(end, utc=True)]
    if max_samples:
        df = df.tail(max_samples)

    feat = build_basic_features(df)
    # keep a tidy frame (timestamp, symbol, close, features..., target)
    drop_cols = {"timestamp", "symbol", "target_next_ret"}
    feats = [c for c in feat.columns if c not in drop_cols]
    keep_cols = ["timestamp", "symbol", "close"] + feats + ["target_next_ret"]
    return feat[keep_cols].dropna(subset=["target_next_ret"])


def build_xy(df_feat: pd.DataFrame):
    feat_cols = [c for c in df_feat.columns if c not in ("timestamp", "symbol", "target_next_ret")]
    X = df_feat[feat_cols].to_numpy()
    y = df_feat["target_next_ret"].to_numpy()
    return X, y, feat_cols


def parse_args():
    ap = argparse.ArgumentParser(description="Walk-forward training for BTC models.")
    ap.add_argument("--data-dir", default=None, help="Override DATA_DIR (.env)")
    ap.add_argument("--symbol", default=None, help="Symbol to train on (e.g., BTC/USDT)")
    ap.add_argument("--start", default=None, help="ISO start date (UTC)")
    ap.add_argument("--end", default=None, help="ISO end date (UTC)")
    ap.add_argument("--n-splits", type=int, default=7, help="TimeSeriesSplit folds (auto-reduced if needed)")
    ap.add_argument("--max-samples", type=int, default=None, help="Optional cap on rows for speed")
    ap.add_argument("--save-model", action="store_true", help="Save final model pickle to --out-path or DATA_DIR.")
    ap.add_argument("--out-path", default=None, help="Explicit output path for model.pkl (overrides default).")
    return ap.parse_args()


def main():
    if load_dotenv:
        load_dotenv(os.environ.get("ENV_FILE", ".env"))

    args = parse_args()
    data_dir = args.data_dir or os.environ.get("DATA_DIR", "./data")
    symbol = args.symbol or os.environ.get("SYMBOL", None)
    n_splits = int(args.n_splits)

    log(f"DATA_DIR={data_dir}")
    log(f"SYMBOL={symbol}")

    df_feat = feature_engineer(data_dir, symbol, args.start, args.end, args.max_samples)
    if len(df_feat) < (n_splits + 20):
        # auto-adjust if dataset is small
        n_splits = max(3, min(5, len(df_feat) // 50)) or 3
        log(f"Auto-adjusted n_splits -> {n_splits}")

    X, y, feat_cols = build_xy(df_feat)

    tss = TimeSeriesSplit(n_splits=n_splits)
    model = RandomForestRegressor(n_estimators=400, max_depth=8, random_state=42, n_jobs=-1)

    rmses, r2s = [], []
    done_any_fold = False
    for fold, (tr_idx, va_idx) in enumerate(tss.split(X), 1):
        done_any_fold = True
        Xtr, Xva = X[tr_idx], X[va_idx]
        ytr, yva = y[tr_idx], y[va_idx]
        model.fit(Xtr, ytr)
        pred = model.predict(Xva)
        rmse = float(np.sqrt(mean_squared_error(yva, pred)))
        r2 = r2_score(yva, pred) if len(np.unique(yva)) > 1 else np.nan
        rmses.append(rmse)
        r2s.append(r2)
        log(f"Fold {fold}/{n_splits}: RMSE={rmse:.6f} R2={r2:.4f}")

    if not done_any_fold:
        raise RuntimeError("TimeSeriesSplit produced no folds. Insufficient data?")

    log(f"AVG RMSE={np.nanmean(rmses):.6f}, AVG R2={np.nanmean(r2s):.4f}")

    # Train final model on all data
    model.fit(X, y)
    log("Trained final model on all samples.")

    # --- Safe feature-importance export ---
    try:
        fi = getattr(model, "feature_importances_", None)
        if fi is not None:
            fi = np.asarray(fi).ravel()
            n = min(len(feat_cols), len(fi))
            if n > 0:
                df_fi = pd.DataFrame({"feature": feat_cols[:n], "importance": fi[:n]}).sort_values(
                    "importance", ascending=False
                )
                out_csv = os.path.join(data_dir, "model_feature_importance.csv")
                df_fi.to_csv(out_csv, index=False)
            else:
                warnings.warn("Feature importances exist but length is zero.")
        else:
            warnings.warn("Model has no feature_importances_. Skipping export.")
    except Exception as e:
        warnings.warn(f"Could not export feature importances: {e}")
    # --------------------------------------

    if args.save_model:
        payload = {"model": model, "features": feat_cols}
        out_path = args.out_path or os.path.join(data_dir, "model.pkl")
        ensure_parent_dir(out_path)
        with open(out_path, "wb") as f:
            pickle.dump(payload, f)
        log(f"Saved model -> {out_path}")


if __name__ == "__main__":
    main()
