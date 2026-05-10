"""横截面 rank / zscore 工具."""
from __future__ import annotations
import numpy as np
import pandas as pd


def cross_section_rank(series: pd.Series, ascending: bool = False) -> pd.Series:
    """返回 0~1 的横截面 rank.
    ascending=False（默认）: 数值越大 rank 越小（0=最强，1=最弱）。
    NaN 保留为 NaN。
    """
    if series is None or series.empty:
        return series
    s = series.astype(float)
    n = s.notna().sum()
    if n <= 1:
        return pd.Series(np.where(s.notna(), 0.5, np.nan), index=s.index)
    ranks = s.rank(ascending=ascending, method="average", na_option="keep")
    return (ranks - 1) / (n - 1)


def zscore(series: pd.Series) -> pd.Series:
    if series is None or series.empty:
        return series
    s = series.astype(float)
    mu, sd = s.mean(), s.std(ddof=0)
    if sd == 0 or pd.isna(sd):
        return pd.Series(0.0, index=s.index)
    return (s - mu) / sd


def percentile(value: float, history: pd.Series) -> float | None:
    """value 在 history 中的分位（0~1）。history 越长越准。"""
    if history is None or history.empty or pd.isna(value):
        return None
    arr = history.dropna().astype(float).to_numpy()
    if arr.size == 0:
        return None
    return float((arr <= value).mean())
