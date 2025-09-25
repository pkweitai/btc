# ml/features.py
import os
import numpy as np
import pandas as pd

from data_pipeline.config import DATA_DIR
from .utils import to_logret, zscore, rolling_vol, rsi, macd, ema

def _read_or_none(path):
    return pd.read_parquet(path) if os.path.exists(path) else None

def load_all():
    """Load all data parquet files if present."""
    fns = {
        # Market / derivatives / orderbook
        "ohlcv":                 "ohlcv.parquet",
        "fund":                  "derivatives_funding.parquet",
        "oi":                    "derivatives_open_interest.parquet",
        "orderbook":             "orderbook_snapshots.parquet",

        # On-chain (no-key)
        "onchain_bc":            "onchain_blockchain_dot_com.parquet",
        "onchain_mempool":       "onchain_mempool_space.parquet",
        "onchain_blockchair":    "onchain_blockchair.parquet",
        "onchain_bitnodes":      "onchain_bitnodes.parquet",

        # On-chain (optional, key)
        "onchain_gn":            "onchain_glassnode.parquet",

        # Sentiment
        "fng":                   "sentiment_fng.parquet",
        "gt":                    "google_trends.parquet",
    }
    dfs = {}
    for k, fn in fns.items():
        dfs[k] = _read_or_none(os.path.join(DATA_DIR, fn))
    return dfs

def build_feature_table(lag_onchain_days=2):
    """
    Build the aligned daily feature table and targets.
    - Anchor: BinanceUS/Coinbase/Kraken/Bitstamp merged set, but we *anchor* on BinanceUS/Coinbase price.
    - Critical leakage guards:
      * on-chain series are lagged (default 2 days).
      * daily snapshots (mempool, blockchair, bitnodes) lagged 1–2 days.
    Returns
    -------
    X : pd.DataFrame
    y : pd.Series (next-day log return)
    c : pd.Series (next-day direction, {0,1})
    feats : pd.DataFrame (full feature table including targets)
    """
    dfs = load_all()

    # ---------- Price anchor ----------
    if dfs["ohlcv"] is None or dfs["ohlcv"].empty:
        raise FileNotFoundError("Missing ohlcv.parquet. Run data_pipeline.daily_update first.")

    # Prefer binanceus if present, else coinbase, else any BTC/* market with max coverage.
    px_all = dfs["ohlcv"].copy()
    px_all = px_all.drop_duplicates().sort_values("timestamp")
    # Pick one venue-symbol pair with longest history as anchor
    anchor = (
        px_all.groupby(["exchange", "symbol"])["timestamp"]
              .count()
              .sort_values(ascending=False)
              .index[0]
    )
    anchor_ex, anchor_sym = anchor
    px = px_all.query("exchange == @anchor_ex and symbol == @anchor_sym").copy()
    px = px[["timestamp", "close", "open", "high", "low", "volume"]]
    px = px.set_index("timestamp").sort_index().asfreq("D")

    # ---------- Technicals ----------
    px["ret1"] = to_logret(px["close"])
    for k in [2, 3, 5, 10, 20]:
        px[f"ret{k}"] = np.log(px["close"] / px["close"].shift(k))
    px["vol20"] = rolling_vol(px["ret1"], 20)
    px["rsi14"] = rsi(px["close"], 14)
    m, s, h = macd(px["close"])
    px["macd"] = m; px["macd_sig"] = s; px["macd_hist"] = h
    for span in [10, 20, 50]:
        px[f"ema{span}"] = ema(px["close"], span)
        denom = px[f"ema{span}"].rolling(20).std(ddof=0) + 1e-12
        px[f"z_price_ema{span}"] = (px["close"] - px[f"ema{span}"]) / denom
    px["vol_z20"] = zscore(px["volume"], 20)

    feats = px.copy()

    # ---------- Derivatives (venue-averaged) ----------
    if dfs["fund"] is not None and not dfs["fund"].empty:
        f = dfs["fund"].set_index("timestamp").sort_index()
        # Combine venues (bybit/okx) by daily mean; safe if one venue only
        f_d = f.resample("D").mean(numeric_only=True)
        feats["funding"] = f_d["funding_rate"]
        feats["funding_z20"] = zscore(feats["funding"], 20)
        feats["funding_chg"] = feats["funding"].diff()

    if dfs["oi"] is not None and not dfs["oi"].empty:
        oi = dfs["oi"].set_index("timestamp").sort_index()
        # Keep daily interval if present, then venue-mean per day
        if "period" in oi.columns:
            oi = oi[oi["period"].str.lower().isin(["1d", "1day", "1dtrading", "1d "])]
        oi_d = oi.resample("D").mean(numeric_only=True)
        if "open_interest" in oi_d.columns:
            feats["oi"] = oi_d["open_interest"]
            feats["oi_z20"] = zscore(feats["oi"], 20)
            feats["oi_chg"] = feats["oi"].pct_change()

    # ---------- Orderbook snapshot ----------
    if dfs["orderbook"] is not None and not dfs["orderbook"].empty:
        ob = dfs["orderbook"].set_index("timestamp").sort_index()
        # mean across exchanges per day
        ob_d = ob.resample("D").mean(numeric_only=True)
        for col in ["spread", "sum_bid_size_topN", "sum_ask_size_topN"]:
            if col in ob_d.columns:
                feats[f"ob_{col}"] = ob_d[col]
        # basic normalization on sizes
        if "ob_sum_bid_size_topN" in feats:
            feats["ob_bid_top_z30"] = zscore(feats["ob_sum_bid_size_topN"], 30)
        if "ob_sum_ask_size_topN" in feats:
            feats["ob_ask_top_z30"] = zscore(feats["ob_sum_ask_size_topN"], 30)

    # ---------- On-chain: Blockchain.com (lagged) ----------
    if dfs["onchain_bc"] is not None and not dfs["onchain_bc"].empty:
        bc = dfs["onchain_bc"].pivot_table(index="timestamp", columns="metric", values="value")
        bc = bc.sort_index().resample("D").last().shift(lag_onchain_days)
        for c in bc.columns:
            feats[f"bc_{c}"] = bc[c]
            feats[f"bc_{c}_z30"] = zscore(bc[c], 30)
            feats[f"bc_{c}_d"] = bc[c].diff()

    # ---------- On-chain: mempool.space (lag 1 day) ----------
    if dfs["onchain_mempool"] is not None and not dfs["onchain_mempool"].empty:
        ms = dfs["onchain_mempool"].pivot_table(index="timestamp", columns="metric", values="value")
        ms = ms.sort_index().resample("D").last().shift(1)
        for col in ms.columns:
            # keep a few common ones directly; zscore all
            feats[f"mps_{col}"] = ms[col]
            feats[f"mps_{col}_z30"] = zscore(ms[col], 30)

    # ---------- On-chain: Blockchair (lag 1 day) ----------
    if dfs["onchain_blockchair"] is not None and not dfs["onchain_blockchair"].empty:
        bc2 = dfs["onchain_blockchair"].pivot_table(index="timestamp", columns="metric", values="value")
        bc2 = bc2.sort_index().resample("D").last().shift(1)
        for col in bc2.columns:
            feats[f"bch_{col}"] = bc2[col]
            feats[f"bch_{col}_z30"] = zscore(bc2[col], 30)

    # ---------- On-chain: Bitnodes (lag 2 days) ----------
    if dfs["onchain_bitnodes"] is not None and not dfs["onchain_bitnodes"].empty:
        bn = dfs["onchain_bitnodes"].pivot_table(index="timestamp", columns="metric", values="value")
        bn = bn.sort_index().resample("D").last().shift(2)
        if "reachable_nodes" in bn.columns:
            feats["nodes_reachable"] = bn["reachable_nodes"]
            feats["nodes_reachable_z30"] = zscore(bn["reachable_nodes"], 30)

    # ---------- On-chain: Glassnode (optional; keep lag) ----------
    if dfs["onchain_gn"] is not None and not dfs["onchain_gn"].empty:
        gn = dfs["onchain_gn"].pivot_table(index="timestamp", columns="metric", values="value")
        gn = gn.sort_index().resample("D").last().shift(lag_onchain_days)
        # Add a subset to avoid huge dimensionality; zscore a few
        for c in gn.columns[:12]:
            feats[f"gn_{c}"] = gn[c]
            feats[f"gn_{c}_z30"] = zscore(gn[c], 30)

    # ---------- Sentiment ----------
    if dfs["fng"] is not None and not dfs["fng"].empty:
        f = dfs["fng"].set_index("timestamp").sort_index().resample("D").last()
        if "value" in f.columns:
            feats["fng"] = f["value"]
            feats["fng_chg7"] = f["value"].diff(7)
            feats["fng_z30"] = zscore(f["value"], 30)

    if dfs["gt"] is not None and not dfs["gt"].empty:
        gt = dfs["gt"].copy()
        if "keyword" in gt.columns:
            gt = gt[gt["keyword"] == "bitcoin"]
        gt = gt.set_index("timestamp").sort_index().resample("D").last()
        if "value" in gt.columns:
            feats["gt"] = gt["value"]
            feats["gt_z30"] = zscore(gt["value"], 30)
            feats["gt_chg7"] = gt["value"].diff(7)

    # Align to a clean daily index, drop rows with missing anchors
    feats = feats.sort_index()

    # ---------- Targets ----------
    feats["y_next"] = feats["ret1"].shift(-1)               # next-day log return
    feats["c_next"] = (feats["y_next"] > 0).astype(float)   # next-day direction

    # Drop rows with any NaNs (after engineering/lagging)
    feats = feats.dropna().copy()

    # Split
    y = feats["y_next"].copy()
    c = feats["c_next"].copy()
    X = feats.drop(columns=["y_next", "c_next"])

    return X, y, c, feats
