from datetime import datetime, timezone
import pandas as pd, requests
from .utils import append_parquet
from .config import ENABLE_GOOGLE_TRENDS

def fetch_fng():
    now = datetime.now(timezone.utc)
    url = "https://api.alternative.me/fng/?limit=0&format=json"
    r = requests.get(url, timeout=30); r.raise_for_status()
    rows = []
    for it in r.json().get("data", []):
        rows.append({"timestamp": pd.to_datetime(int(it["timestamp"]), unit="s", utc=True),
                     "source": "alternative.me", "value": float(it["value"]),
                     "classification": it.get("value_classification",""), "ingested_at": now})
    df = pd.DataFrame(rows)
    if not df.empty: append_parquet(df, "sentiment_fng.parquet", ["timestamp","source"])

def fetch_google_trends(keyword="bitcoin", geo=""):
    if not ENABLE_GOOGLE_TRENDS: return
    from pytrends.request import TrendReq
    now = datetime.now(timezone.utc)
    pytrends = TrendReq(hl="en-US", tz=0)
    pytrends.build_payload([keyword], timeframe="today 5-y", geo=geo)
    df = pytrends.interest_over_time()
    if df.empty: return
    df = df.reset_index().rename(columns={"date":"timestamp", keyword:"value"})
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["source"] = "google_trends"; df["keyword"] = keyword; df["ingested_at"] = now
    append_parquet(df[["timestamp","source","keyword","value","ingested_at"]], "google_trends.parquet", ["timestamp","source","keyword"])
