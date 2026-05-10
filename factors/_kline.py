"""共享 K 线获取 + 缓存（新浪 stock_zh_a_daily，列最全）.

供 liquidity / technical / market_action 等多个维度复用。
- 新浪源直接含 volume / amount / outstanding_share / turnover（小数，已转为 %）
- 缓存目录: cache/kline/{symbol}_daily_qfq.csv
"""
from __future__ import annotations
import time
from pathlib import Path
import pandas as pd

try:
    import akshare as ak
except ImportError:
    ak = None

CACHE_DIR = Path(__file__).resolve().parents[1] / "cache" / "kline"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
CACHE_TTL_HOURS = 6


def _is_fresh(p: Path, ttl_hours: int = CACHE_TTL_HOURS) -> bool:
    if not p.exists():
        return False
    return (time.time() - p.stat().st_mtime) / 3600 < ttl_hours


def _sina_symbol(symbol: str) -> str:
    return f"sh{symbol}" if symbol.startswith("6") else f"sz{symbol}"


def get_kline(symbol: str, days: int = 250, adjust: str = "qfq") -> pd.DataFrame:
    """日线 K 线（前复权）, 列: date / open / high / low / close / volume / amount /
    outstanding_share / turnover (单位: %).
    """
    p = CACHE_DIR / f"{symbol}_daily_{adjust}.csv"
    if _is_fresh(p):
        try:
            df = pd.read_csv(p, parse_dates=["date"])
            return df.tail(days).reset_index(drop=True)
        except Exception:
            pass

    if ak is None:
        return pd.DataFrame()

    end = pd.Timestamp.today()
    start = end - pd.Timedelta(days=int(days * 1.6))
    s = start.strftime("%Y%m%d")
    e = end.strftime("%Y%m%d")
    try:
        df = ak.stock_zh_a_daily(
            symbol=_sina_symbol(symbol), start_date=s, end_date=e, adjust=adjust
        )
    except Exception as ex:
        print(f"  [kline sina] {symbol} error: {ex}")
        return pd.DataFrame()

    if df is None or df.empty:
        return pd.DataFrame()

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    for c in ("open", "high", "low", "close", "volume", "amount", "outstanding_share", "turnover"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "turnover" in df.columns:
        df["turnover"] = df["turnover"] * 100.0  # 0.0038 → 0.38 (%)
    df.to_csv(p, index=False)
    return df.tail(days).reset_index(drop=True)
