# data_pipeline/daily_update.py
from datetime import datetime, timezone
import os
import glob
import logging
import pandas as pd

from .config import DATA_DIR
from .sources_market_full import fetch_multi_ohlcv_daily, fetch_orderbook_snapshots
from .sources_derivatives_full import (
    fetch_bybit_funding, fetch_bybit_open_interest,
    fetch_okx_funding, fetch_okx_open_interest,
)
from .sources_onchain_full import (
    fetch_blockchain_com_charts,
    fetch_mempool_space,
    fetch_blockchair_stats,
    fetch_bitnodes_snapshot,
    fetch_glassnode_selected,   # safe no-op if no key
)
from .sources_sentiment_full import fetch_fng, fetch_google_trends

# ---------- logging ----------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger("daily_update")

def _run_step(name, fn, *args, **kwargs):
    log.info("START: %s", name)
    try:
        fn(*args, **kwargs)
        log.info("DONE : %s", name)
    except Exception as e:
        log.exception("FAIL : %s -> %s", name, e)

def _summarize_data_dir():
    log.info("==== SUMMARY of data files in %s ====", DATA_DIR)
    if not os.path.isdir(DATA_DIR):
        log.warning("Data dir does not exist: %s", DATA_DIR)
        return
    files = sorted(glob.glob(os.path.join(DATA_DIR, "*.parquet")))
    if not files:
        log.warning("No parquet files found.")
        return
    for fp in files:
        try:
            df = pd.read_parquet(fp)
            n = len(df)
            last_ts = None
            if "timestamp" in df.columns and n > 0:
                last_ts = pd.to_datetime(df["timestamp"]).max()
            log.info("File: %-40s | rows=%-7d | last_ts=%s", os.path.basename(fp), n, last_ts)
        except Exception as e:
            log.warning("File: %-40s | could not read (%s)", os.path.basename(fp), e)

def main():
    start = datetime.now(timezone.utc)
    log.info("=== Daily update started @ %s ===", start.isoformat())

    # ----- Market -----
    _run_step("OHLCV (multi-exchange)", fetch_multi_ohlcv_daily, limit=1500)
    _run_step("Orderbook snapshots (top-10 depth)", fetch_orderbook_snapshots, depth=10)

    # ----- Derivatives (Bybit + OKX; no Binance futures) -----
    _run_step("Bybit funding", fetch_bybit_funding, "BTCUSDT")
    _run_step("Bybit open interest", fetch_bybit_open_interest, "BTCUSDT", "1d", 200)
    _run_step("OKX funding", fetch_okx_funding, "BTC-USDT-SWAP", 100)
    _run_step("OKX open interest", fetch_okx_open_interest, "SWAP", "BTC-USDT-SWAP")

    # ----- On-chain (no key) -----
    _run_step("Blockchain.com charts", fetch_blockchain_com_charts, "5years", "8hours")
    _run_step("mempool.space snapshot", fetch_mempool_space)
    _run_step("Blockchair stats", fetch_blockchair_stats)
    _run_step("Bitnodes snapshot", fetch_bitnodes_snapshot)

    # ----- Optional Glassnode (no-op if no key) -----
    _run_step("Glassnode selected (optional)", fetch_glassnode_selected)

    # ----- Sentiment -----
    _run_step("Fear & Greed", fetch_fng)
    _run_step("Google Trends (optional)", fetch_google_trends, "bitcoin", "")

    _summarize_data_dir()

    end = datetime.now(timezone.utc)
    log.info("=== Daily update finished @ %s (elapsed %.1fs) ===", end.isoformat(), (end - start).total_seconds())

if __name__ == "__main__":
    main()

