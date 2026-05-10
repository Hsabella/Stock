"""市场环境 / 行业相对强度维度.

简化版（个股 sector 接口经常空）:
- index_regime: 上证 MA20/MA60 → bull/bear/sideways
- north_total_inflow_pos: 当日北向是否净流入
- stock_rs_20d: 个股 20d 涨跌 − 沪深300 同期涨跌（个股相对强度）
- stock_rs_60d: 同上 60d
- 行业维度: sector 缺时退化为个股 vs 沪深300
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from engine.ranker import cross_section_rank, zscore
from .data import get_index_kline, get_csi300_kline, get_north_total_inflow


def get_index_regime() -> str:
    """bull / bear / sideways."""
    df = get_index_kline("sh000001")
    if df.empty or len(df) < 60:
        return "sideways"
    close = df["close"].astype(float)
    ma20 = close.rolling(20).mean()
    ma60 = close.rolling(60).mean()
    c, m20, m60 = close.iloc[-1], ma20.iloc[-1], ma60.iloc[-1]
    if c > m20 > m60:
        return "bull"
    if c < m20 < m60:
        return "bear"
    return "sideways"


def _ret(close: pd.Series, n: int) -> float | None:
    if len(close) <= n:
        return None
    return float(close.iloc[-1] / close.iloc[-1 - n] - 1)


def compute_one(symbol: str, name: str = "", kline: pd.DataFrame | None = None,
                csi300: pd.DataFrame | None = None) -> dict:
    out = {"symbol": symbol, "name": name}
    if kline is None or kline.empty:
        out.update({"stock_rs_20d": None, "stock_rs_60d": None, "regime_raw": 0.0})
        return out
    if csi300 is None or csi300.empty:
        csi300 = get_csi300_kline()
    s_close = kline["close"].astype(float)
    i_close = csi300["close"].astype(float) if not csi300.empty else pd.Series(dtype=float)

    s20 = _ret(s_close, 20)
    s60 = _ret(s_close, 60)
    i20 = _ret(i_close, 20) if len(i_close) else None
    i60 = _ret(i_close, 60) if len(i_close) else None
    rs20 = (s20 - i20) if (s20 is not None and i20 is not None) else None
    rs60 = (s60 - i60) if (s60 is not None and i60 is not None) else None

    out["stock_rs_20d"] = rs20
    out["stock_rs_60d"] = rs60
    return out


def compose_dim(rows: list[dict], regime: str = "sideways", north_pos: bool = True) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if df.empty:
        return df

    rs20 = pd.to_numeric(df["stock_rs_20d"], errors="coerce")
    rs60 = pd.to_numeric(df["stock_rs_60d"], errors="coerce")
    z20 = zscore(rs20.fillna(rs20.median() if rs20.notna().any() else 0.0))
    z60 = zscore(rs60.fillna(rs60.median() if rs60.notna().any() else 0.0))

    raw = 0.6 * z20 + 0.4 * z60
    # 大盘环境调整: bull → +0.3 整体加成, bear → -0.5 压制
    if regime == "bull":
        raw = raw + 0.3
    elif regime == "bear":
        raw = raw - 0.5
    if north_pos:
        raw = raw + 0.1
    df["regime_raw"] = raw
    df["regime_rank"] = cross_section_rank(raw)
    df["index_regime"] = regime
    return df
