import os
from typing import Optional, List
import pandas as pd
from .config import DATA_DIR
def _ensure_dt(df: pd.DataFrame, col: str = "timestamp") -> pd.DataFrame:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], utc=True)
        df = df.sort_values(col)
    return df
def append_parquet(df: pd.DataFrame, filename: str, key_cols: Optional[List[str]] = None) -> None:
    path = os.path.join(DATA_DIR, filename)
    df = _ensure_dt(df)
    if os.path.exists(path):
        old = pd.read_parquet(path)
        all_df = pd.concat([old, df], ignore_index=True)
    else:
        all_df = df.copy()
    if key_cols:
        all_df = all_df.drop_duplicates(subset=key_cols, keep="last")
    else:
        all_df = all_df.drop_duplicates(keep="last")
    if "timestamp" in all_df.columns:
        all_df = all_df.sort_values("timestamp")
    all_df.to_parquet(path, index=False)
