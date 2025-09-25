# data_pipeline/sources_onchain_full.py
from datetime import datetime, timezone
import os
import requests
import pandas as pd

from .utils import append_parquet
from .config import GLASSNODE_API_KEY  # optional; can be empty

# -----------------------------
# Blockchain.com charts (no key)
# -----------------------------
BLOCKCHAIN_COM = "https://api.blockchain.info/charts"
# Extend freely available metrics here:
BC_METRICS = [
    ("n-transactions", "n-transactions"),
    ("hash-rate", "hash-rate"),
    ("difficulty", "difficulty"),
    ("miners-revenue", "miners-revenue"),
    ("avg-block-size", "avg-block-size"),
    ("mempool-size", "mempool-size"),
    ("market-price", "market-price"),
]

def fetch_blockchain_com_charts(timespan: str = "5years", rolling_average: str = "8hours") -> None:
    """
    Fetch multiple blockchain.com chart metrics (no API key required).
    Writes/updates data/data_pipeline/onchain_blockchain_dot_com.parquet
    """
    now = datetime.now(timezone.utc)
    rows = []
    for metric, path in BC_METRICS:
        try:
            url = f"{BLOCKCHAIN_COM}/{path}"
            r = requests.get(
                url,
                params={"timespan": timespan, "format": "json", "rollingAverage": rolling_average},
                timeout=30,
            )
            r.raise_for_status()
            js = r.json()
            for p in js.get("values", []):
                rows.append(
                    {
                        "timestamp": pd.to_datetime(p["x"], unit="s", utc=True),
                        "source": "blockchain.com",
                        "metric": metric,
                        "value": float(p["y"]),
                        "ingested_at": now,
                    }
                )
        except Exception as e:
            print(f"blockchain.com error for {metric}: {e}")

    if rows:
        df = pd.DataFrame(rows)
        append_parquet(df, "onchain_blockchain_dot_com.parquet", ["timestamp", "metric", "source"])


# -----------------------------
# mempool.space (no key)
# -----------------------------
MEMPOOL_API = "https://mempool.space/api"

def fetch_mempool_space() -> None:
    """
    Daily snapshot of mempool size and recommended fees (sats/vB).
    Writes/updates data/onchain_mempool_space.parquet
    """
    now = datetime.now(timezone.utc)
    rows = []
    try:
        r1 = requests.get(f"{MEMPOOL_API}/mempool", timeout=30)
        r1.raise_for_status()
        mp = r1.json()  # {'count': ..., 'vsize': ...}
        rows.extend(
            [
                {
                    "timestamp": now,
                    "source": "mempool.space",
                    "metric": "mempool_count",
                    "value": float(mp.get("count", 0.0)),
                    "ingested_at": now,
                },
                {
                    "timestamp": now,
                    "source": "mempool.space",
                    "metric": "mempool_vsize",
                    "value": float(mp.get("vsize", 0.0)),
                    "ingested_at": now,
                },
            ]
        )
    except Exception as e:
        print("mempool.space /mempool error:", e)

    try:
        r2 = requests.get(f"{MEMPOOL_API}/v1/fees/recommended", timeout=30)
        r2.raise_for_status()
        fees = r2.json()  # {'fastestFee','halfHourFee','hourFee','economyFee','minimumFee'}
        for k, v in fees.items():
            rows.append(
                {
                    "timestamp": now,
                    "source": "mempool.space",
                    "metric": f"fee_{k}",
                    "value": float(v),
                    "ingested_at": now,
                }
            )
    except Exception as e:
        print("mempool.space fees error:", e)

    if rows:
        df = pd.DataFrame(rows)
        append_parquet(df, "onchain_mempool_space.parquet", ["timestamp", "source", "metric"])


# -----------------------------
# Blockchair (no key)
# -----------------------------
BLOCKCHAIR = "https://api.blockchair.com/bitcoin/stats"

def fetch_blockchair_stats() -> None:
    """
    Snapshot of broad network stats from Blockchair (no key).
    Writes/updates data/onchain_blockchair.parquet
    """
    now = datetime.now(timezone.utc)
    try:
        r = requests.get(BLOCKCHAIR, timeout=30)
        r.raise_for_status()
        data = r.json().get("data", {})
    except Exception as e:
        print("Blockchair stats error:", e)
        return

    wanted = [
        "transactions",
        "transactions_24h",
        "mempool_transactions",
        "mempool_size",
        "mempool_tps",
        "hashrate_24h",
        "difficulty",
        "average_block_size_24h",
        "market_price_usd",
    ]

    rows = []
    for key in wanted:
        if key in data and isinstance(data[key], (int, float)):
            rows.append(
                {
                    "timestamp": now,
                    "source": "blockchair",
                    "metric": key,
                    "value": float(data[key]),
                    "ingested_at": now,
                }
            )

    if rows:
        df = pd.DataFrame(rows)
        append_parquet(df, "onchain_blockchair.parquet", ["timestamp", "source", "metric"])


# -----------------------------
# Bitnodes (no key)
# -----------------------------
BITNODES = "https://bitnodes.io/api/v1/snapshots/?limit=1"

def fetch_bitnodes_snapshot() -> None:
    """
    Latest reachable node count from Bitnodes (no key).
    Writes/updates data/onchain_bitnodes.parquet
    """
    now = datetime.now(timezone.utc)
    try:
        r = requests.get(BITNODES, timeout=30)
        r.raise_for_status()
        js = r.json()
        total = None
        if "results" in js and js["results"]:
            total = js["results"][0].get("total_nodes")
        if total is None:
            total = js.get("total_nodes")
        if total is None:
            return
        df = pd.DataFrame(
            [
                {
                    "timestamp": now,
                    "source": "bitnodes",
                    "metric": "reachable_nodes",
                    "value": float(total),
                    "ingested_at": now,
                }
            ]
        )
        append_parquet(df, "onchain_bitnodes.parquet", ["timestamp", "source", "metric"])
    except Exception as e:
        print("Bitnodes snapshot error:", e)


# ------------------------------------
# Optional: Glassnode (requires a key)
# ------------------------------------
GLASSNODE_BASE = "https://api.glassnode.com/v1/metrics"

def fetch_glassnode(metric_path: str, asset: str = "BTC", key: str | None = None) -> pd.DataFrame:
    """
    Helper to request a Glassnode metric. Returns empty DataFrame if no key.
    """
    key = key or GLASSNODE_API_KEY
    if not key:
        return pd.DataFrame()
    try:
        url = f"{GLASSNODE_BASE}/{metric_path}"
        params = {"api_key": key, "a": asset, "i": "1d"}
        r = requests.get(url, params=params, timeout=30)
        r.raise_for_status()
        data = r.json()
        rows = []
        for it in data:
            ts = pd.to_datetime(it["t"], unit="s", utc=True)
            v = it.get("v", None)
            if isinstance(v, dict):
                for k2, v2 in v.items():
                    rows.append(
                        {
                            "timestamp": ts,
                            "source": "glassnode",
                            "metric": f"{metric_path}:{k2}",
                            "value": float(v2),
                        }
                    )
            else:
                rows.append(
                    {
                        "timestamp": ts,
                        "source": "glassnode",
                        "metric": metric_path,
                        "value": float(v),
                    }
                )
        return pd.DataFrame(rows)
    except Exception as e:
        print(f"Glassnode error for {metric_path}:", e)
        return pd.DataFrame()

def fetch_glassnode_selected() -> None:
    """
    Pull a small set of daily Glassnode metrics if GLASSNODE_API_KEY is set.
    Writes/updates data/onchain_glassnode.parquet
    """
    if not GLASSNODE_API_KEY:
        # Silently skip if no key provided
        return

    metrics = [
        "addresses/new_addresses",
        "supply/active_1d_1w",
        "distribution/balance_exchanges_relative",
    ]
    frames = []
    for m in metrics:
        df = fetch_glassnode(m)
        if not df.empty:
            frames.append(df)

    if frames:
        df_all = pd.concat(frames, ignore_index=True)
        # Add ingestion time and sort
        df_all["ingested_at"] = datetime.now(timezone.utc)
        df_all = df_all.sort_values("timestamp")
        append_parquet(df_all, "onchain_glassnode.parquet", ["timestamp", "metric", "source"])
