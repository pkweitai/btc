
import os
import json
from datetime import datetime, timezone
from functools import lru_cache
from typing import Optional

from dotenv import load_dotenv

from flask import Flask, jsonify, render_template, request
import pandas as pd

# ---- Config ----
DATA_DIR = os.environ.get("DATA_DIR", os.path.abspath(os.path.join(os.getcwd(), "data")))
DATE_FMT = "%Y-%m-%dT%H:%M:%SZ"

load_dotenv(os.environ.get("ENV_FILE", ".env"))

app = Flask(__name__, template_folder="templates", static_folder="static")


def _ts_to_iso(ts):
    if pd.isna(ts):
        return None
    if isinstance(ts, (int, float)):
        # assume unix seconds
        return datetime.fromtimestamp(ts, tz=timezone.utc).strftime(DATE_FMT)
    if isinstance(ts, pd.Timestamp):
        return ts.tz_convert("UTC").strftime(DATE_FMT) if ts.tzinfo else ts.tz_localize("UTC").strftime(DATE_FMT)
    if isinstance(ts, datetime):
        return ts.astimezone(timezone.utc).strftime(DATE_FMT)
    return str(ts)


def _file(path):
    p = os.path.join(DATA_DIR, path)
    return p if os.path.exists(p) else None


@lru_cache(maxsize=16)
def _load_parquet(fname: str) -> Optional[pd.DataFrame]:
    p = _file(fname)
    if not p:
        return None
    try:
        df = pd.read_parquet(p)
        # normalize timestamp column(s)
        for col in ["timestamp", "time", "date"]:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")
        return df
    except Exception as e:
        print(f"Failed reading {fname}: {e}")
        return None


def _slice_timerange(df: pd.DataFrame, col="timestamp"):
    start = request.args.get("start")
    end = request.args.get("end")
    if start:
        start = pd.to_datetime(start, utc=True, errors="coerce")
        df = df[df[col] >= start]
    if end:
        end = pd.to_datetime(end, utc=True, errors="coerce")
        df = df[df[col] <= end]
    return df


@app.route("/")
def index():
    # quick presence check for files
    files = {
        "ohlcv": bool(_file("ohlcv.parquet")),
        "orderbook": bool(_file("orderbook_snapshots.parquet")),
        "funding": bool(_file("derivatives_funding.parquet")),
        "open_interest": bool(_file("derivatives_open_interest.parquet")),
        "blockchain": bool(_file("onchain_blockchain_dot_com.parquet")),
        "blockchair": bool(_file("onchain_blockchair.parquet")),
        "mempool": bool(_file("onchain_mempool_space.parquet")),
        "bitnodes": bool(_file("onchain_bitnodes.parquet")),
        "fng": bool(_file("sentiment_fng.parquet")),
    }
    return render_template("index.html", files=files, data_dir=DATA_DIR)



@app.route("/api/summary")
def api_summary():
    summary = {}

    def add(name, fname, groupby=None):
        df = _load_parquet(fname)
        if df is None:
            summary[name] = {"present": False}
            return
        # compute last_ts
        last_ts = None
        for c in ["timestamp", "time", "date"]:
            if c in df.columns:
                last_ts = _ts_to_iso(pd.to_datetime(df[c], utc=True, errors="coerce").max())
                break
        meta = {"present": True, "rows": int(len(df)), "last_ts": last_ts}
        if groupby:
            safe = [g for g in groupby if g in df.columns]
            if len(safe) > 0:
                vals = []
                for tup, _sub in df.groupby(safe):
                    key = tup if isinstance(tup, tuple) else (tup,)
                    vals.append(tuple("" if pd.isna(x) else str(x) for x in key))
                meta["groups"] = sorted(list(set(vals)))
        summary[name] = meta

    # Prepare groupby candidates safely
    ohlcv_df = _load_parquet("ohlcv.parquet")
    ob_df = _load_parquet("orderbook_snapshots.parquet")
    fund_df = _load_parquet("derivatives_funding.parquet")
    oi_df = _load_parquet("derivatives_open_interest.parquet")
    bcdc_df = _load_parquet("onchain_blockchain_dot_com.parquet")
    bc_df = _load_parquet("onchain_blockchair.parquet")
    mem_df = _load_parquet("onchain_mempool_space.parquet")

    add("ohlcv", "ohlcv.parquet", groupby=["symbol"] if (ohlcv_df is not None and "symbol" in ohlcv_df.columns) else None)
    add("orderbook", "orderbook_snapshots.parquet", groupby=["exchange","symbol"] if (ob_df is not None) else None)
    add("funding", "derivatives_funding.parquet", groupby=["exchange","instrument","symbol"])
    add("open_interest", "derivatives_open_interest.parquet", groupby=["exchange","instrument","symbol"])
    add("blockchain", "onchain_blockchain_dot_com.parquet", groupby=["metric"] if (bcdc_df is not None and "metric" in bcdc_df.columns) else None)
    add("blockchair", "onchain_blockchair.parquet", groupby=["metric"] if (bc_df is not None and "metric" in bc_df.columns) else None)
    add("mempool", "onchain_mempool_space.parquet", groupby=["metric"] if (mem_df is not None and "metric" in mem_df.columns) else None)
    add("bitnodes", "onchain_bitnodes.parquet")
    add("fng", "sentiment_fng.parquet")
    return jsonify(summary)


@app.route("/api/ohlcv")
def api_ohlcv():
    df = _load_parquet("ohlcv.parquet")
    if df is None:
        return jsonify({"error": "ohlcv.parquet not found in DATA_DIR"}), 404
    # optional filters
    for col in ["symbol", "exchange"]:
        val = request.args.get(col)
        if val and col in df.columns:
            df = df[df[col] == val]
    df = _slice_timerange(df, col="timestamp" if "timestamp" in df.columns else df.columns[0])
    # default to last 365 days if daily data
    if request.args.get("last_n"):
        n = int(request.args["last_n"])
        df = df.sort_values("timestamp").tail(n)
    out = []
    cols = df.columns
    for _, r in df.sort_values("timestamp").iterrows():
        item = {"t": _ts_to_iso(r.get("timestamp", None))}
        for k in ["open","high","low","close","volume"]:
            if k in cols:
                item[k] = None if pd.isna(r[k]) else float(r[k])
        if "symbol" in cols: item["symbol"] = r["symbol"]
        if "exchange" in cols: item["exchange"] = r["exchange"]
        out.append(item)
    return jsonify(out)


@app.route("/api/funding")
def api_funding():
    df = _load_parquet("derivatives_funding.parquet")
    if df is None:
        return jsonify({"error": "derivatives_funding.parquet not found"}), 404
    for col in ["exchange", "instrument", "symbol"]:
        v = request.args.get(col)
        if v and col in df.columns:
            df = df[df[col] == v]
    df = _slice_timerange(df)
    # guess funding column
    fund_col = next((c for c in df.columns if "fund" in c.lower() and "rate" in c.lower()), None)
    out = []
    for _, r in df.sort_values("timestamp").iterrows():
        out.append({
            "t": _ts_to_iso(r.get("timestamp")),
            "exchange": r.get("exchange"),
            "instrument": r.get("instrument", r.get("symbol")),
            "funding_rate": None if fund_col is None or pd.isna(r.get(fund_col)) else float(r.get(fund_col))
        })
    return jsonify(out)


@app.route("/api/open_interest")
def api_open_interest():
    df = _load_parquet("derivatives_open_interest.parquet")
    if df is None:
        return jsonify({"error": "derivatives_open_interest.parquet not found"}), 404
    for col in ["exchange", "instrument", "symbol"]:
        v = request.args.get(col)
        if v and col in df.columns:
            df = df[df[col] == v]
    df = _slice_timerange(df)
    # pick oi column
    oi_col = next((c for c in df.columns if "open" in c.lower() and "interest" in c.lower()), None)
    out = [{
        "t": _ts_to_iso(r.get("timestamp")),
        "exchange": r.get("exchange"),
        "instrument": r.get("instrument", r.get("symbol")),
        "open_interest": None if oi_col is None or pd.isna(r.get(oi_col)) else float(r.get(oi_col))
    } for _, r in df.sort_values("timestamp").iterrows()]
    return jsonify(out)


@app.route("/api/onchain")
def api_onchain():
    # supports: blockchain.com, blockchair, mempool.space via combined shape metric/time/value
    fname = request.args.get("source", "onchain_blockchain_dot_com.parquet")
    df = _load_parquet(fname)
    if df is None:
        return jsonify({"error": f"{fname} not found"}), 404
    metric = request.args.get("metric")
    if metric and "metric" in df.columns:
        df = df[df["metric"] == metric]
    df = _slice_timerange(df)
    # find value column
    val_col = next((c for c in ["value","val","v"] if c in df.columns), None)
    out = [{
        "t": _ts_to_iso(r.get("timestamp", r.get("time", r.get("date")))),
        "metric": r.get("metric"),
        "value": None if val_col is None or pd.isna(r.get(val_col)) else float(r.get(val_col))
    } for _, r in df.sort_values(by=[c for c in ["timestamp","time","date"] if c in df.columns][0]).iterrows()]
    return jsonify(out)


@app.route("/api/bitnodes")
def api_bitnodes():
    df = _load_parquet("onchain_bitnodes.parquet")
    if df is None:
        return jsonify({"error": "onchain_bitnodes.parquet not found"}), 404
    df = _slice_timerange(df)
    # try to find node count column
    node_col = next((c for c in df.columns if "node" in c.lower() and "total" in c.lower()), None)
    if node_col is None:
        node_col = next((c for c in df.columns if "node" in c.lower()), None)
    out = [{
        "t": _ts_to_iso(r.get("timestamp", r.get("time", r.get("date")))),
        "total_nodes": None if node_col is None or pd.isna(r.get(node_col)) else float(r.get(node_col))
    } for _, r in df.sort_values(by=[c for c in ["timestamp","time","date"] if c in df.columns][0]).iterrows()]
    return jsonify(out)


@app.route("/api/fng")
def api_fng():
    df = _load_parquet("sentiment_fng.parquet")
    if df is None:
        return jsonify({"error": "sentiment_fng.parquet not found"}), 404
    df = _slice_timerange(df, col="timestamp" if "timestamp" in df.columns else df.columns[0])
    value_col = next((c for c in df.columns if c.lower() in ["value","fng_value","fear_greed","score"]), None)
    out = [{
        "t": _ts_to_iso(r.get("timestamp", r.get("time", r.get("date")))),
        "value": None if value_col is None or pd.isna(r.get(value_col)) else float(r.get(value_col)),
        "label": r.get("classification", r.get("label"))
    } for _, r in df.sort_values(by=[c for c in ["timestamp","time","date"] if c in df.columns][0]).iterrows()]
    return jsonify(out)


@app.route("/health")
def health():
    present = {f: bool(_file(f)) for f in [
        "ohlcv.parquet",
        "orderbook_snapshots.parquet",
        "derivatives_funding.parquet",
        "derivatives_open_interest.parquet",
        "onchain_blockchain_dot_com.parquet",
        "onchain_blockchair.parquet",
        "onchain_mempool_space.parquet",
        "onchain_bitnodes.parquet",
        "sentiment_fng.parquet",
    ]}
    return jsonify({"ok": True, "data_dir": DATA_DIR, "files": present})


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    debug = os.getenv("DEBUG", "0") in ("1","true","True","yes","on")
    app.run(host="0.0.0.0", port=port, debug=debug)
